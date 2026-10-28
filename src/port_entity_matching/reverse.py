"""Reverse channel: Hangul query vs LLM Hangul renderings of catalog names (jamo similarity).

Renderings are catalog-side precomputes (one JSONL file per rendering model, ``{"name", "hangul"}`` where
``hangul`` is a string or a list of variants). A vessel's similarity is the max over all of its variants.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from port_entity_matching.candidates import CandidateSet
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.records import QueryRecord

__all__ = ["REVERSE_FEATURE_NAMES", "ReverseChannel", "jamo_string"]

REVERSE_FEATURE_NAMES: tuple[str, ...] = ("rev_jamo_sim", "rev_jamo_rank", "rev_ton_rank")

_SYLLABLE_BASE = 0xAC00
_SYLLABLE_COUNT = 11172
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_WHITESPACE_RE = re.compile(r"\s+")


def jamo_string(value: str) -> str:
    """Whitespace-free compatibility-jamo decomposition; non-syllable characters are upper-cased."""
    out: list[str] = []
    for char in _WHITESPACE_RE.sub("", str(value)):
        code = ord(char) - _SYLLABLE_BASE
        if 0 <= code < _SYLLABLE_COUNT:
            final = _JONG[code % 28] if code % 28 else ""
            out.append(_CHO[code // 588] + _JUNG[(code % 588) // 28] + final)
        else:
            out.append(char.upper())
    return "".join(out)


@dataclass(eq=False)
class ReverseChannel:
    """Catalog renderings indexed by jamo; candidates and features for Hangul records."""

    files: tuple[Path, ...]
    variants: list[list[str]]
    flat: list[str]
    owner: list[int]
    gt: NDArray[np.float64]
    _memo: dict[tuple[str, float | None], tuple[list[int], dict[int, int]]] = field(
        init=False, repr=False, default_factory=dict
    )

    @classmethod
    def load(cls, index: CatalogIndex, paths: Sequence[str | Path]) -> ReverseChannel:
        """Read rendering JSONL files; FileNotFoundError if one is missing, ValueError if none is given."""
        files = tuple(Path(path) for path in paths)
        if not files:
            raise ValueError("ReverseChannel needs at least one rendering file")
        per_position: list[set[str]] = [set() for _ in index.names]
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(f"Rendering file not found: {path}")
            rendering: dict[str, str | list[str]] = {}
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    entry = json.loads(line)
                    rendering[entry["name"]] = entry["hangul"]
            for position, name in enumerate(index.names):
                # A name absent from a file contributes the empty variant; kept for fidelity with fitted models.
                hangul = rendering.get(name, "")
                per_position[position].update(
                    jamo_string(v) for v in (hangul if isinstance(hangul, list) else [hangul])
                )
        flat: list[str] = []
        owner: list[int] = []
        for position, keys in enumerate(per_position):
            for key in sorted(keys):
                flat.append(key)
                owner.append(position)
        return cls(
            files=files,
            variants=[sorted(keys) for keys in per_position],
            flat=flat,
            owner=owner,
            gt=np.asarray(index.gt, dtype=np.float64),
        )

    def lookup(self, record: QueryRecord, config: MatchConfig) -> tuple[list[int], dict[int, int]]:
        """Jamo top-k positions and {position: rank} among tonnage-compatible jamo hits for one record."""
        tonnage = record.international_gross_tonnage or record.gross_tonnage
        key = (record.vessel_name, tonnage)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        hits = process.extract(
            jamo_string(record.vessel_name),
            self.flat,
            scorer=Levenshtein.normalized_similarity,
            limit=config.reverse_pool * len(self.files),
        )
        order = list(dict.fromkeys(self.owner[hit[2]] for hit in hits))[: config.reverse_pool]
        compatible = [
            position
            for position in order
            if tonnage
            and self.gt[position] > 0
            and abs(math.log(self.gt[position] / tonnage)) <= config.reverse_tonnage_window
        ]
        result = (order[: config.reverse_top_k], {position: rank for rank, position in enumerate(compatible)})
        self._memo[key] = result
        return result

    def extend(
        self, records: Sequence[QueryRecord], candidate_sets: Sequence[CandidateSet], config: MatchConfig
    ) -> list[CandidateSet]:
        """Add reverse-channel positions (source bits 0) to Hangul records' sets; Latin sets pass through."""
        out: list[CandidateSet] = []
        for record, candidate_set in zip(records, candidate_sets, strict=True):
            if not is_hangul_name(record.vessel_name):
                out.append(candidate_set)
                continue
            top, tonnage_rank = self.lookup(record, config)
            sources = dict(zip(candidate_set.positions.tolist(), candidate_set.sources.tolist(), strict=True))
            extra = [position for position, rank in tonnage_rank.items() if rank < config.reverse_tonnage_top_k]
            for position in top + extra:
                sources.setdefault(int(position), 0)
            keys = sorted(sources)
            out.append(
                CandidateSet(
                    np.asarray(keys, dtype=np.int64),
                    np.asarray([sources[key] for key in keys], dtype=np.uint8),
                )
            )
        return out

    def columns(
        self, record: QueryRecord, positions: NDArray[np.int64], config: MatchConfig, hangul: bool
    ) -> dict[str, NDArray[np.float64]]:
        """REVERSE_FEATURE_NAMES columns for one record's candidates; all NaN for Latin records."""
        size = len(positions)
        if not hangul:
            return {name: np.full(size, np.nan) for name in REVERSE_FEATURE_NAMES}
        query = jamo_string(record.vessel_name)
        sims = np.array(
            [max(Levenshtein.normalized_similarity(query, key) for key in self.variants[p]) for p in positions]
        )
        order = (-sims).argsort(kind="stable")
        rank = np.empty(size)
        rank[order] = np.arange(size)
        _, tonnage_rank = self.lookup(record, config)
        return {
            "rev_jamo_sim": sims,
            "rev_jamo_rank": rank,
            "rev_ton_rank": np.array([tonnage_rank.get(int(p), np.nan) for p in positions], dtype=float),
        }
