"""Row-independent candidate retrieval with per-source evidence flags."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein, Levenshtein

from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.memory import CallSignMemory
from port_entity_matching.normalize import (
    canonical_latin,
    hangul_phonetic_key_v2,
    hangul_variants,
    is_hangul_name,
    latin_tokens,
)
from port_entity_matching.records import QueryRecord

__all__ = [
    "SRC_FALLBACK",
    "SRC_MEMORY",
    "SRC_NAME",
    "SRC_TOKEN",
    "SRC_TONNAGE",
    "CandidateSet",
    "generate_candidates",
]

Positions = NDArray[np.int64]
Scores = NDArray[np.float64]

SRC_NAME: int = 1
SRC_TONNAGE: int = 2
SRC_TOKEN: int = 4
SRC_MEMORY: int = 8
SRC_FALLBACK: int = 16


@dataclass(frozen=True, eq=False)
class CandidateSet:
    """Candidate catalog positions for one record with per-candidate source bit flags."""

    positions: NDArray[np.int64]
    sources: NDArray[np.uint8]


@dataclass(frozen=True)
class _Prepared:
    record: QueryRecord
    is_hangul: bool
    latin_key: str
    query_tokens: tuple[str, ...]
    cons_keys: tuple[str, ...]
    vowel_keys: tuple[str, ...]
    tonnages: tuple[float, ...]


def generate_candidates(
    records: Sequence[QueryRecord],
    index: CatalogIndex,
    memory: CallSignMemory | None,
    config: MatchConfig,
) -> list[CandidateSet]:
    """Return one non-empty CandidateSet per record, in input order."""
    if config.query_chunk < 1 or config.max_candidates < 1:
        raise ValueError("query_chunk and max_candidates must be positive")
    if min(config.name_top_k, config.hangul_key_top_k, config.hangul_window_keep, config.latin_exact_gt_cap) < 0:
        raise ValueError("Retrieval limits must be non-negative")
    if config.hangul_tonnage_window < 0:
        raise ValueError("hangul_tonnage_window must be non-negative")
    results: list[CandidateSet] = []
    step = config.query_chunk
    for start in range(0, len(records), step):
        results.extend(_process_chunk(records[start : start + step], index, memory, config))
    return results


def _process_chunk(
    chunk: Sequence[QueryRecord],
    index: CatalogIndex,
    memory: CallSignMemory | None,
    config: MatchConfig,
) -> list[CandidateSet]:
    prepared = [_prepare(record) for record in chunk]
    latin_names = _latin_name_scores(prepared, index, config)
    hangul_names = _hangul_name_scores(prepared, index, config)
    results = []
    for position, item in enumerate(prepared):
        if item.is_hangul:
            name_positions, name_scores = hangul_names.get(position, _empty_ranking())
            tonnage_positions, exact_positions = _hangul_tonnage(item, index, config)
            token_positions: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        else:
            name_positions, name_scores = latin_names.get(position, _empty_ranking())
            tonnage_positions, exact_positions = _latin_tonnage(item, index, config)
            token_positions = _token_candidates(item, index, config)
        results.append(
            _assemble(item, index, memory, config, name_positions, name_scores, tonnage_positions, token_positions, exact_positions)
        )
    return results


def _prepare(record: QueryRecord) -> _Prepared:
    tonnages = tuple(value for value in (record.international_gross_tonnage, record.gross_tonnage) if value is not None)
    if is_hangul_name(record.vessel_name):
        variants = hangul_variants(record.vessel_name)
        return _Prepared(
            record=record,
            is_hangul=True,
            latin_key="",
            query_tokens=(),
            cons_keys=tuple(hangul_phonetic_key_v2(variant, False) for variant in variants),
            vowel_keys=tuple(hangul_phonetic_key_v2(variant, True) for variant in variants),
            tonnages=tonnages,
        )
    return _Prepared(
        record=record,
        is_hangul=False,
        latin_key=canonical_latin(record.vessel_name),
        query_tokens=latin_tokens(record.vessel_name),
        cons_keys=(),
        vowel_keys=(),
        tonnages=tonnages,
    )


def _empty_ranking() -> tuple[Positions, Scores]:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)


def _latin_name_scores(
    prepared: Sequence[_Prepared],
    index: CatalogIndex,
    config: MatchConfig,
) -> dict[int, tuple[Positions, Scores]]:
    keys = {position: item.latin_key for position, item in enumerate(prepared) if not item.is_hangul and item.latin_key}
    if not keys:
        return {}
    unique = sorted(set(keys.values()))
    matrix = process.cdist(
        unique,
        index.canonical,
        scorer=DamerauLevenshtein.normalized_similarity,
        workers=config.num_threads,
        dtype=np.float32,
    )
    row_of = {key: row for row, key in enumerate(unique)}
    scores: dict[int, tuple[Positions, Scores]] = {}
    for position, key in keys.items():
        row = matrix[row_of[key]]
        order = np.argsort(-row, kind="stable")[: config.name_top_k]
        scores[position] = (order.astype(np.int64), row[order].astype(np.float64))
    return scores


def _hangul_name_scores(
    prepared: Sequence[_Prepared],
    index: CatalogIndex,
    config: MatchConfig,
) -> dict[int, tuple[Positions, Scores]]:
    records = [(position, item) for position, item in enumerate(prepared) if item.is_hangul and item.cons_keys]
    if not records:
        return {}
    unique = sorted({key for _, item in records for key in item.cons_keys})
    key_scores: dict[str, tuple[Positions, Scores]] = {}
    for start in range(0, len(unique), config.query_chunk):
        keys = unique[start : start + config.query_chunk]
        matrix = process.cdist(
            keys,
            index.key2_cons,
            scorer=Levenshtein.normalized_similarity,
            workers=config.num_threads,
            dtype=np.float32,
        )
        for row_number, key in enumerate(keys):
            order = np.argsort(-matrix[row_number], kind="stable")[: config.hangul_key_top_k].copy()
            key_scores[key] = (order.astype(np.int64), matrix[row_number, order].astype(np.float64))
        del matrix

    scores: dict[int, tuple[Positions, Scores]] = {}
    for position, item in records:
        best: dict[int, float] = {}
        # Top-k of a pointwise maximum is contained in the union of each key's top-k.
        for key in item.cons_keys:
            positions, similarities = key_scores[key]
            for candidate, similarity in zip(positions.tolist(), similarities.tolist(), strict=True):
                best[candidate] = max(best.get(candidate, 0.0), similarity)
        selected = sorted(best, key=lambda candidate: (-best[candidate], candidate))[: config.hangul_key_top_k]
        scores[position] = (np.asarray(selected, dtype=np.int64), np.asarray([best[p] for p in selected], dtype=np.float64))
    return scores


def _latin_tonnage(item: _Prepared, index: CatalogIndex, config: MatchConfig) -> tuple[Positions, Positions]:
    exact: list[Positions] = []
    capped: list[Positions] = []
    for tonnage in item.tonnages:
        positions = index.gt_exact(tonnage)
        exact.append(positions)
        capped.append(positions[: config.latin_exact_gt_cap])
    return _union(capped), _union(exact)


def _hangul_tonnage(item: _Prepared, index: CatalogIndex, config: MatchConfig) -> tuple[Positions, Positions]:
    ranges = [
        index.gt_range(tonnage * (1.0 - config.hangul_tonnage_window), tonnage * (1.0 + config.hangul_tonnage_window))
        for tonnage in item.tonnages
    ]
    window = _union(ranges)
    exact = _union([index.gt_exact(tonnage) for tonnage in item.tonnages])
    if len(window) == 0 or not item.cons_keys:
        return _union([window, exact]), exact

    def score(position: int) -> float:
        return max(
            0.5 * (
                Levenshtein.normalized_similarity(cons_key, index.key2_cons[position])
                + Levenshtein.normalized_similarity(vowel_key, index.key2_vow[position])
            )
            for cons_key, vowel_key in zip(item.cons_keys, item.vowel_keys, strict=True)
        )

    selected = sorted(window.tolist(), key=lambda position: (-score(position), position))[: config.hangul_window_keep]
    return _union([np.asarray(selected, dtype=np.int64), exact]), exact


def _token_candidates(item: _Prepared, index: CatalogIndex, config: MatchConfig) -> Positions:
    if not item.tonnages:
        return np.empty(0, dtype=np.int64)
    found: list[Positions] = []
    for token in item.query_tokens:
        if len(token) < config.rare_token_min_len or token.isdigit():
            continue
        document_frequency = index.token_df.get(token, 0)
        if document_frequency < 1 or document_frequency > config.rare_token_max_df:
            continue
        postings = index.token_postings(token)
        catalog_tonnage = index.gt[postings]
        mask = np.zeros(len(postings), dtype=bool)
        for tonnage in item.tonnages:
            mask |= (catalog_tonnage >= tonnage * (1.0 - config.tonnage_window)) & (
                catalog_tonnage <= tonnage * (1.0 + config.tonnage_window)
            )
        found.append(postings[mask])
    return _union(found)


def _assemble(
    item: _Prepared,
    index: CatalogIndex,
    memory: CallSignMemory | None,
    config: MatchConfig,
    name_positions: Positions,
    name_scores: Scores,
    tonnage_positions: Positions,
    token_positions: Positions,
    exact_positions: Positions,
) -> CandidateSet:
    masks: dict[int, int] = {}

    def add(positions: Positions, bit: int) -> None:
        for position in positions.tolist():
            masks[int(position)] = masks.get(int(position), 0) | bit

    add(name_positions, SRC_NAME)
    add(tonnage_positions, SRC_TONNAGE)
    add(token_positions, SRC_TOKEN)
    memory_position = memory.lookup(item.record.call_sign) if memory is not None else None
    if memory_position is not None:
        add(np.asarray([memory_position], dtype=np.int64), SRC_MEMORY)

    if not masks:
        return CandidateSet(np.asarray([0], dtype=np.int64), np.asarray([SRC_FALLBACK], dtype=np.uint8))

    score_of = dict(zip(name_positions.tolist(), name_scores.tolist(), strict=True))
    if len(masks) > config.max_candidates:
        for position in masks:
            if position not in score_of:
                if item.is_hangul:
                    score = max(Levenshtein.normalized_similarity(key, index.key2_cons[position]) for key in item.cons_keys)
                else:
                    score = DamerauLevenshtein.normalized_similarity(item.latin_key, index.canonical[position])
                score_of[position] = float(np.float32(score))
    exact_set = set(exact_positions.tolist())

    def sort_key(position: int) -> tuple[int, float, int]:
        if memory_position is not None and position == memory_position:
            tier = 0
        elif position in exact_set:
            tier = 1
        else:
            tier = 2
        return tier, -float(score_of.get(position, 0.0)), position

    selected = sorted(masks, key=sort_key)[: config.max_candidates]
    ordered = sorted(selected)
    positions = np.asarray(ordered, dtype=np.int64)
    sources = np.asarray([masks[position] for position in ordered], dtype=np.uint8)
    return CandidateSet(positions, sources)


def _union(arrays: Sequence[Positions]) -> Positions:
    usable = [array.astype(np.int64) for array in arrays if len(array) > 0]
    if not usable:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(usable)).astype(np.int64)
