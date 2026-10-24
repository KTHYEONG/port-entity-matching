"""Twin evaluation: foreign vessels written in Hangul.

Builds a reusable twin map from Latin record rows and exposes the twin
perturbation scenarios for cross-validation.
"""

from __future__ import annotations

import dataclasses
import logging
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from port_entity_matching.evaluation.hangulize import random_style, transliterate_latin
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.records import QueryRecord
from port_entity_matching.train import RecordPerturbation

__all__ = ["build_twin", "reverse_llm", "twin_scenarios"]

logger = logging.getLogger(__name__)

REVERSE_SYSTEM = (
    "Korean port-call records write foreign ship names in Hangul as a Korean clerk would pronounce them "
    "(Korean loanword transliteration; romanized Korean/Chinese/Japanese words are written in their original reading). "
    "Keep numbers. Answer with the Hangul name only."
)


def build_twin(
    train: pd.DataFrame,
    n_rows: int,
    seed: int,
    reverse: Callable[[Sequence[str]], Mapping[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    """Map sampled Latin sample_ids to {"src", "r", "q"} Hangul renderings; "q" is present only when reverse is given."""
    latin = train[~train["vessel_name"].map(is_hangul_name)]
    if n_rows > len(latin):
        raise ValueError(f"Cannot sample {n_rows} Latin rows from {len(latin)} available rows")
    sampled = latin.sample(n_rows, random_state=seed) if n_rows else latin.iloc[0:0]
    twin: dict[str, dict[str, str]] = {}
    for sample_id, vessel_name in zip(sampled["sample_id"].tolist(), sampled["vessel_name"].tolist(), strict=True):
        sid = str(sample_id)
        src = str(vessel_name)
        rng = np.random.default_rng(zlib.crc32(f"test|{sid}".encode()))
        style = random_style(rng)
        twin[sid] = {"src": src, "r": transliterate_latin(src, rng, style)}
    if reverse is not None:
        names = sorted({entry["src"] for entry in twin.values()})
        mapping = reverse(names)
        for entry in twin.values():
            entry["q"] = str(mapping[entry["src"]])
    return twin


def reverse_llm(
    weights_dir: str | Path,
    shots: Sequence[tuple[str, str]],
    batch_size: int = 16,
) -> Callable[[Sequence[str]], Mapping[str, str]]:
    """Greedy Latin→Hangul reverse prompt (shots swapped) returning the first output line per name."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(f"reverse transliteration stack unavailable: {error}") from error
    resolved = Path(weights_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(resolved), local_files_only=True, trust_remote_code=False)  # type: ignore[no-untyped-call]
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved), dtype=torch.bfloat16, device_map="cuda", local_files_only=True, trust_remote_code=False
    )
    model.eval()  # type: ignore[no-untyped-call]
    prefix: list[dict[str, str]] = [{"role": "system", "content": REVERSE_SYSTEM}]
    for hangul, latin in shots:
        prefix.append({"role": "user", "content": latin})
        prefix.append({"role": "assistant", "content": hangul})

    def _prompt_for(name: str) -> str:
        messages: list[dict[str, str]] = [*prefix, {"role": "user", "content": name}]
        try:
            rendered: Any = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return str(rendered)

    def reverse(names: Sequence[str]) -> Mapping[str, str]:
        distinct = sorted(set(names))
        prompts = [_prompt_for(name) for name in distinct]
        order = np.argsort([len(prompt) for prompt in prompts], kind="stable")
        out: dict[str, str] = {}
        for start in range(0, len(prompts), batch_size):
            chunk = order[start : start + batch_size]
            batch_prompts = [prompts[int(i)] for i in chunk]
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            device = str(getattr(model, "device", "cuda"))
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_len = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=32, do_sample=False, num_beams=1)
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            texts = tokenizer.batch_decode(sequences[:, input_len:], skip_special_tokens=True)
            for index, text in zip(chunk.tolist(), texts, strict=True):
                first = str(text).strip().split("\n")[0] if str(text).strip() else ""
                out[distinct[int(index)]] = first.strip()
        return out

    return reverse


def twin_scenarios(twin: Mapping[str, Mapping[str, str]]) -> dict[str, RecordPerturbation]:
    """Scenarios kor_nt, r2k_wt, r2k_nt and (when present) q2k_wt, q2k_nt."""
    scenarios: dict[str, RecordPerturbation] = {
        "kor_nt": _kor_nt,
        "r2k_wt": _renamed(twin, "r", blank=False),
        "r2k_nt": _renamed(twin, "r", blank=True),
    }
    if any("q" in entry for entry in twin.values()):
        scenarios["q2k_wt"] = _renamed(twin, "q", blank=False)
        scenarios["q2k_nt"] = _renamed(twin, "q", blank=True)
    return scenarios


def _kor_nt(record: QueryRecord) -> QueryRecord | None:
    if not is_hangul_name(record.vessel_name):
        return None
    return dataclasses.replace(record, gross_tonnage=None, international_gross_tonnage=None)


def _renamed(twin: Mapping[str, Mapping[str, str]], key: str, *, blank: bool) -> RecordPerturbation:
    def perturb(record: QueryRecord) -> QueryRecord | None:
        entry = twin.get(record.sample_id)
        if entry is None:
            return None
        rendering = entry.get(key)
        if rendering is None or not is_hangul_name(rendering):
            return None
        replaced = dataclasses.replace(record, vessel_name=rendering)
        if blank:
            return dataclasses.replace(replaced, gross_tonnage=None, international_gross_tonnage=None)
        return replaced

    return perturb
