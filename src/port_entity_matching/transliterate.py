"""Frozen LLM transliteration of Hangul vessel names to Latin n-best lists."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from port_entity_matching.config import MatchConfig
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.records import QueryRecord

__all__ = [
    "SYSTEM_PROMPT",
    "CachedTransliterator",
    "TranslitFn",
    "Transliterator",
    "TransliteratorUnavailable",
    "attach_alt_names",
    "select_fewshot",
]

SYSTEM_PROMPT: str = (
    "Korean port-call records often write a ship's name in Hangul, transliterated from its "
    "registered name (usually English words or romanized Chinese/Japanese/Korean). "
    "Recover the registered ship name in Latin letters. Keep numbers. "
    "Answer with the ship name only."
)

TranslitFn = Callable[[Sequence[str]], Mapping[str, tuple[str, ...]]]

logger = logging.getLogger(__name__)


class TransliteratorUnavailable(RuntimeError):  # noqa: N818
    """Raised when torch/transformers, CUDA, or the weights directory is unavailable."""


def select_fewshot(train: pd.DataFrame, catalog: pd.DataFrame, count: int, seed: int) -> tuple[tuple[str, str], ...]:
    """Select deterministic (Hangul name, catalog name) few-shot pairs from train Hangul rows."""
    if count <= 0:
        return ()
    if "vessel_name" not in train.columns or "target_vessel_id" not in train.columns:
        return ()
    hangul_positions: list[int] = []
    hangul_names: list[str] = []
    hangul_targets: list[str] = []
    for position in range(len(train)):
        name = train["vessel_name"].iloc[position]
        if is_hangul_name(name):
            hangul_positions.append(position)
            hangul_names.append(str(name))
            hangul_targets.append(str(train["target_vessel_id"].iloc[position]))
    if not hangul_positions:
        return ()
    vessels = sorted(set(hangul_targets))
    take = min(count, len(vessels))
    chosen = set(np.random.default_rng(seed).choice(vessels, take, replace=False).tolist())
    catalog_lookup: dict[str, str] = {}
    if "vessel_id" in catalog.columns and "vessel_name" in catalog.columns:
        for vessel_id, vessel_name in zip(catalog["vessel_id"].tolist(), catalog["vessel_name"].tolist(), strict=False):
            key = str(vessel_id)
            if key not in catalog_lookup:
                catalog_lookup[key] = str(vessel_name)
    first_seen: dict[str, tuple[int, str, str]] = {}
    for position, name, target in zip(hangul_positions, hangul_names, hangul_targets, strict=True):
        if target in chosen and target not in first_seen and target in catalog_lookup:
            first_seen[target] = (position, name, catalog_lookup[target])
    ordered = sorted(first_seen.values(), key=lambda item: item[0])
    return tuple((name, latin) for _, name, latin in ordered)


def attach_alt_names(records: Sequence[QueryRecord], table: Mapping[str, tuple[str, ...]]) -> list[QueryRecord]:
    """Return records with alt_names filled from table for matching Hangul names."""
    out: list[QueryRecord] = []
    for record in records:
        if is_hangul_name(record.vessel_name) and record.vessel_name in table:
            out.append(dataclasses.replace(record, alt_names=tuple(table[record.vessel_name])))
        else:
            out.append(record)
    return out


def _validate_llm_config(config: MatchConfig) -> None:
    if config.llm_fewshot < 0:
        raise ValueError("llm_fewshot must be non-negative")
    if config.llm_num_beams < 1:
        raise ValueError("llm_num_beams must be at least 1")
    if config.llm_max_new_tokens < 1:
        raise ValueError("llm_max_new_tokens must be at least 1")
    if not 0 <= config.llm_retrieval_beams <= config.llm_num_beams:
        raise ValueError("llm_retrieval_beams must satisfy 0 <= llm_retrieval_beams <= llm_num_beams")
    if config.llm_max_name_chars < 1:
        raise ValueError("llm_max_name_chars must be positive")


class Transliterator:
    """Frozen causal LM mapping one Hangul vessel name to Latin n-best names."""

    def __init__(self, model: Any, tokenizer: Any, shots: Sequence[tuple[str, str]], config: MatchConfig) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._shots = tuple(shots)
        self._config = config

    @classmethod
    def load(cls, weights_dir: str | Path, shots: Sequence[tuple[str, str]], config: MatchConfig) -> Transliterator:
        """Load tokenizer and model on CUDA in bfloat16 from the local directory only."""
        _validate_llm_config(config)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise TransliteratorUnavailable(f"transliteration stack unavailable: {error}") from error
        if not torch.cuda.is_available():
            raise TransliteratorUnavailable("CUDA is unavailable")
        resolved = Path(weights_dir)
        if not (resolved / "config.json").is_file():
            raise TransliteratorUnavailable(f"weights config.json missing in {resolved}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                str(resolved), local_files_only=True, trust_remote_code=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(resolved),
                dtype=torch.bfloat16,
                device_map="cuda",
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as error:
            raise TransliteratorUnavailable(f"failed to load transliteration weights: {error}") from error
        model.eval()  # type: ignore[no-untyped-call]
        tokenizer.padding_side = "left"
        return cls(model, tokenizer, shots, config)

    def _prompt_for(self, name: str) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for hangul, latin in self._shots:
            messages.append({"role": "user", "content": hangul[: self._config.llm_max_name_chars]})
            messages.append({"role": "assistant", "content": latin})
        messages.append({"role": "user", "content": name[: self._config.llm_max_name_chars]})
        rendered: Any = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return str(rendered)

    def translate(self, names: Sequence[str]) -> dict[str, tuple[str, ...]]:
        """Map each distinct Hangul name to its Latin n-best list in sorted order."""
        import torch

        distinct = sorted({name for name in names if is_hangul_name(name)})
        out: dict[str, tuple[str, ...]] = {}
        generated = 0
        started = time.perf_counter()
        for name in distinct:
            prompt = self._prompt_for(name)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self._model.device)
            attention = inputs.get("attention_mask")
            if attention is not None:
                attention = attention.to(self._model.device)
            input_len = int(input_ids.shape[1])
            with torch.inference_mode():
                outputs = self._model.generate(
                    input_ids,
                    attention_mask=attention,
                    num_beams=self._config.llm_num_beams,
                    num_return_sequences=self._config.llm_num_beams,
                    do_sample=False,
                    length_penalty=0.0,
                    max_new_tokens=self._config.llm_max_new_tokens,
                )
            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            beams: list[str] = []
            for beam_index in range(sequences.shape[0]):
                text = self._tokenizer.decode(sequences[beam_index][input_len:], skip_special_tokens=True)
                first_line = text.splitlines()[0] if text.splitlines() else ""
                cleaned = first_line.strip()
                if cleaned and cleaned not in beams:
                    beams.append(cleaned)
            out[name] = tuple(beams)
            generated += 1
        elapsed = time.perf_counter() - started
        per_name = elapsed / generated if generated else 0.0
        logger.info(
            "[ALGO] translit names=%d generated=%d seconds=%.2f seconds_per_name=%.3f",
            len(distinct),
            generated,
            elapsed,
            per_name,
        )
        return out


def _translit_fingerprint(shots: Sequence[tuple[str, str]], config: MatchConfig, weights_dir: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(SYSTEM_PROMPT.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(json.dumps([list(pair) for pair in shots], ensure_ascii=False, sort_keys=True).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(
        json.dumps({
            "beams": config.llm_num_beams,
            "new_tokens": config.llm_max_new_tokens,
            "max_name_chars": config.llm_max_name_chars,
        }).encode()
    )
    digest.update(b"\x1f")
    config_path = Path(weights_dir) / "config.json"
    try:
        digest.update(config_path.read_bytes())
    except OSError as error:
        raise TransliteratorUnavailable("weights config.json is unavailable") from error
    return digest.hexdigest()


class CachedTransliterator:
    """Dev/training wrapper persisting translate() results in a JSON-lines file."""

    def __init__(
        self,
        cache_dir: str | Path,
        weights_dir: str | Path,
        shots: Sequence[tuple[str, str]],
        config: MatchConfig,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._weights_dir = Path(weights_dir)
        self._shots = tuple(shots)
        self._config = config
        self._fingerprint = _translit_fingerprint(self._shots, config, self._weights_dir)
        self._cache_file = self._cache_dir / f"{self._fingerprint}.jsonl"
        self._memory: dict[str, tuple[str, ...]] = {}
        self._loaded_file = False
        self._model: Transliterator | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded_file:
            return
        self._loaded_file = True
        if not self._cache_file.is_file():
            return
        for line in self._cache_file.read_bytes().splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
                name = payload["name"]
                alts = payload["alts"]
                if (
                    not isinstance(name, str)
                    or not isinstance(alts, list)
                    or not all(isinstance(alt, str) for alt in alts)
                ):
                    continue
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            self._memory[name] = tuple(alts)

    def translate(self, names: Sequence[str]) -> dict[str, tuple[str, ...]]:
        """Return cached results, generating and appending only the misses."""
        self._ensure_loaded()
        distinct = list(dict.fromkeys(name for name in names if is_hangul_name(name)))
        misses = [name for name in distinct if name not in self._memory]
        if misses:
            if self._model is None:
                self._model = Transliterator.load(self._weights_dir, self._shots, self._config)
            fresh = self._model.translate(misses)
            self._memory.update(fresh)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, "ab+") as handle:
                if handle.tell():
                    handle.seek(-1, 2)
                    if handle.read(1) != b"\n":
                        handle.write(b"\n")
                for name in misses:
                    if name in fresh:
                        line = json.dumps({"name": name, "alts": list(fresh[name])}, ensure_ascii=False) + "\n"
                        handle.write(line.encode("utf-8"))
        return {name: self._memory[name] for name in distinct if name in self._memory}
