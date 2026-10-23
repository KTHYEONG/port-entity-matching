"""Row-independent candidate retrieval with per-source evidence flags."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import process
from rapidfuzz.distance import OSA, DamerauLevenshtein, Levenshtein

from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.memory import CallSignMemory
from port_entity_matching.normalize import (
    canonical_latin,
    hangul_phonetic_key_v2,
    hangul_token_keys,
    hangul_variants,
    is_hangul_name,
    latin_tokens,
    sorted_token_key,
)
from port_entity_matching.records import QueryRecord

__all__ = [
    "SRC_FALLBACK",
    "SRC_LLM",
    "SRC_MEMORY",
    "SRC_NAME",
    "SRC_SORTED",
    "SRC_TOKEN",
    "SRC_TOKEN_INDEX",
    "SRC_TONNAGE",
    "CandidateSet",
    "_top_k_desc",
    "generate_candidates",
]

Positions = NDArray[np.int64]
Scores = NDArray[np.float64]

SRC_NAME: int = 1
SRC_TONNAGE: int = 2
SRC_TOKEN: int = 4
SRC_MEMORY: int = 8
SRC_FALLBACK: int = 16
SRC_SORTED: int = 32
SRC_TOKEN_INDEX: int = 64
SRC_LLM: int = 128


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
    sorted_latin: str
    sorted_cons: tuple[str, ...]


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
    if config.latin_prefilter_k < 0:
        raise ValueError("latin_prefilter_k must be non-negative")
    if config.hangul_tonnage_window < 0:
        raise ValueError("hangul_tonnage_window must be non-negative")
    if config.token_top_k < 0 or config.token_df_cap < 1:
        raise ValueError("token_top_k and token_df_cap must satisfy token_top_k >= 0 and token_df_cap >= 1")
    if config.llm_retrieval_beams < 0:
        raise ValueError("llm_retrieval_beams must be non-negative")
    if not 0.0 < config.token_match_threshold <= 1.0:
        raise ValueError("token_match_threshold must be in (0, 1]")
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
    pseudo_prepared: list[_Prepared] = []
    pseudo_owner: list[int] = []
    pseudo_by_owner: dict[int, list[_Prepared]] = {}
    for position, item in enumerate(prepared):
        if not item.is_hangul:
            continue
        alts = _llm_kept_alts(item.record, config)
        if not alts:
            continue
        owned: list[_Prepared] = []
        for alt in alts:
            pseudo = _prepare(dataclasses.replace(item.record, vessel_name=alt, alt_names=()))
            pseudo_prepared.append(pseudo)
            pseudo_owner.append(position)
            owned.append(pseudo)
        pseudo_by_owner[position] = owned
    if pseudo_prepared:
        extended = [*prepared, *pseudo_prepared]
        latin_names = _latin_name_scores(extended, index, config)
        latin_sorted = _latin_sorted_scores(extended, index, config)
        token_index = _token_index_scores(extended, index, config)
    else:
        extended = list(prepared)
        latin_names = _latin_name_scores(prepared, index, config)
        hangul_names = _hangul_name_scores(prepared, index, config)
        latin_sorted = _latin_sorted_scores(prepared, index, config)
        hangul_sorted = _hangul_sorted_scores(prepared, index, config)
        token_index = _token_index_scores(prepared, index, config)
        results = []
        for position, item in enumerate(prepared):
            if item.is_hangul:
                name_positions, name_scores = hangul_names.get(position, _empty_ranking())
                sorted_positions, sorted_scores = hangul_sorted.get(position, _empty_ranking())
                tonnage_positions, exact_positions = _hangul_tonnage(item, index, config)
                token_positions: NDArray[np.int64] = np.empty(0, dtype=np.int64)
            else:
                name_positions, name_scores = latin_names.get(position, _empty_ranking())
                sorted_positions, sorted_scores = latin_sorted.get(position, _empty_ranking())
                tonnage_positions, exact_positions = _latin_tonnage(item, index, config)
                token_positions = _token_candidates(item, index, config)
            tidx_positions, tidx_scores = token_index.get(position, _empty_ranking())
            results.append(
                _assemble(
                    item,
                    index,
                    memory,
                    config,
                    name_positions,
                    name_scores,
                    sorted_positions,
                    sorted_scores,
                    tonnage_positions,
                    token_positions,
                    tidx_positions,
                    tidx_scores,
                    exact_positions,
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.float64),
                )
            )
        return results
    base = len(prepared)
    owner_slots: dict[int, list[int]] = {}
    for slot, owner in enumerate(pseudo_owner):
        owner_slots.setdefault(owner, []).append(base + slot)
    hangul_names = _hangul_name_scores(prepared, index, config)
    hangul_sorted = _hangul_sorted_scores(prepared, index, config)
    results = []
    for position, item in enumerate(prepared):
        if item.is_hangul:
            name_positions, name_scores = hangul_names.get(position, _empty_ranking())
            sorted_positions, sorted_scores = hangul_sorted.get(position, _empty_ranking())
            tonnage_positions, exact_positions = _hangul_tonnage(item, index, config)
            token_positions = np.empty(0, dtype=np.int64)
            llm_positions, llm_scores = _llm_ranking(
                owner_slots.get(position, []), latin_names, latin_sorted, token_index, extended, index, config
            )
        else:
            name_positions, name_scores = latin_names.get(position, _empty_ranking())
            sorted_positions, sorted_scores = latin_sorted.get(position, _empty_ranking())
            tonnage_positions, exact_positions = _latin_tonnage(item, index, config)
            token_positions = _token_candidates(item, index, config)
            llm_positions, llm_scores = _empty_ranking()
            tidx_positions, tidx_scores = token_index.get(position, _empty_ranking())
            results.append(
                _assemble(
                    item,
                    index,
                    memory,
                    config,
                    name_positions,
                    name_scores,
                    sorted_positions,
                    sorted_scores,
                    tonnage_positions,
                    token_positions,
                    tidx_positions,
                    tidx_scores,
                    exact_positions,
                    llm_positions,
                    llm_scores,
                )
            )
            continue
        tidx_positions, tidx_scores = token_index.get(position, _empty_ranking())
        results.append(
            _assemble(
                item,
                index,
                memory,
                config,
                name_positions,
                name_scores,
                sorted_positions,
                sorted_scores,
                tonnage_positions,
                token_positions,
                tidx_positions,
                tidx_scores,
                exact_positions,
                llm_positions,
                llm_scores,
                _llm_pseudo=pseudo_by_owner.get(position, ()),
            )
        )
    return results


def _llm_kept_alts(record: QueryRecord, config: MatchConfig) -> list[str]:
    if config.llm_retrieval_beams <= 0:
        return []
    seen: dict[str, None] = {}
    for alt in record.alt_names:
        key = canonical_latin(alt)
        if not key or key in seen:
            continue
        seen[key] = None
        if len(seen) >= config.llm_retrieval_beams:
            break
    if not seen:
        return []
    keys = list(seen)
    ordered: list[str] = []
    for alt in record.alt_names:
        key = canonical_latin(alt)
        if key in seen and key in keys:
            keys.remove(key)
            ordered.append(alt)
            if not keys:
                break
    return ordered


def _llm_ranking(
    slots: Sequence[int],
    latin_names: dict[int, tuple[Positions, Scores]],
    latin_sorted: dict[int, tuple[Positions, Scores]],
    token_index: dict[int, tuple[Positions, Scores]],
    extended: Sequence[_Prepared],
    index: CatalogIndex,
    config: MatchConfig,
) -> tuple[Positions, Scores]:
    if not slots:
        return _empty_ranking()
    best: dict[int, float] = {}
    union: list[Positions] = []
    for slot in slots:
        name_positions, name_scores = latin_names.get(slot, _empty_ranking())
        sorted_positions, sorted_scores = latin_sorted.get(slot, _empty_ranking())
        tidx_positions, tidx_scores = token_index.get(slot, _empty_ranking())
        token_positions = _token_candidates(extended[slot], index, config)
        union.extend([name_positions, sorted_positions, tidx_positions, token_positions])
        for position, similarity in zip(name_positions.tolist(), name_scores.tolist(), strict=True):
            best[int(position)] = max(best.get(int(position), 0.0), float(similarity))
        for position, similarity in zip(sorted_positions.tolist(), sorted_scores.tolist(), strict=True):
            best[int(position)] = max(best.get(int(position), 0.0), float(similarity))
        for position, similarity in zip(tidx_positions.tolist(), tidx_scores.tolist(), strict=True):
            best[int(position)] = max(best.get(int(position), 0.0), float(similarity))
    positions = _union(union)
    scores = np.asarray([best.get(int(position), 0.0) for position in positions.tolist()], dtype=np.float64)
    return positions, scores


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
            sorted_latin="",
            sorted_cons=tuple(sorted_token_key(hangul_token_keys(variant)) for variant in variants),
        )
    tokens = latin_tokens(record.vessel_name)
    return _Prepared(
        record=record,
        is_hangul=False,
        latin_key=canonical_latin(record.vessel_name),
        query_tokens=tokens,
        cons_keys=(),
        vowel_keys=(),
        tonnages=tonnages,
        sorted_latin=sorted_token_key(tokens),
        sorted_cons=(),
    )


def _empty_ranking() -> tuple[Positions, Scores]:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)


def _top_k_desc(scores: NDArray[np.floating[Any]], k: int) -> NDArray[np.int64]:
    """Indices of the k largest scores ordered by (-score, index); all indices when k >= len(scores)."""
    n = int(scores.shape[0])
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.argsort(-scores, kind="stable").astype(np.int64)
    threshold = float(np.partition(scores, n - k)[n - k])
    mask = scores >= threshold
    cand = np.flatnonzero(mask)
    order = np.argsort(-scores[cand], kind="stable")[:k]
    selected: NDArray[np.int64] = cand[order].astype(np.int64)
    return selected


def _dl_top_k_for_keys(
    unique: list[str],
    choices: Sequence[str],
    config: MatchConfig,
) -> dict[str, tuple[Positions, Scores]]:
    """Top name_top_k DL hits per key, optionally via an OSA prefilter.

    When latin_prefilter_k == 0 the full DL cdist is used (ablation switch).
    Otherwise per key the top max(prefilter, name_top_k) OSA hits are rescored
    with DL and the top name_top_k rescored hits are returned ordered by
    (-DL score, catalog position). Returned scores are DL similarities.
    """
    if config.latin_prefilter_k == 0:
        matrix = process.cdist(
            unique,
            choices,
            scorer=DamerauLevenshtein.normalized_similarity,
            workers=config.num_threads,
            dtype=np.float32,
        )
        ranked: dict[str, tuple[Positions, Scores]] = {}
        for row_number, key in enumerate(unique):
            row = matrix[row_number]
            order = _top_k_desc(row, config.name_top_k)
            ranked[key] = (order.astype(np.int64), row[order].astype(np.float64))
        return ranked
    osa_matrix = process.cdist(
        unique,
        choices,
        scorer=OSA.normalized_similarity,
        workers=config.num_threads,
        dtype=np.float32,
    )
    choice_list = list(choices)
    pre_k = max(config.latin_prefilter_k, config.name_top_k)
    ranked = {}
    for row_number, key in enumerate(unique):
        osa_row = osa_matrix[row_number]
        preselected = _top_k_desc(osa_row, pre_k)
        subset = [choice_list[int(position)] for position in preselected.tolist()]
        dl_row = process.cdist(
            [key],
            subset,
            scorer=DamerauLevenshtein.normalized_similarity,
            workers=1,
            dtype=np.float32,
        )[0]
        catalog_positions = preselected.astype(np.int64)
        rank_order = np.lexsort((catalog_positions, -dl_row.astype(np.float64)))
        if config.name_top_k < len(rank_order):
            rank_order = rank_order[: config.name_top_k]
        final_positions = catalog_positions[rank_order].astype(np.int64)
        final_scores = dl_row[rank_order].astype(np.float64)
        ranked[key] = (final_positions, final_scores)
    return ranked


def _latin_name_scores(
    prepared: Sequence[_Prepared],
    index: CatalogIndex,
    config: MatchConfig,
) -> dict[int, tuple[Positions, Scores]]:
    keys = {position: item.latin_key for position, item in enumerate(prepared) if not item.is_hangul and item.latin_key}
    if not keys:
        return {}
    unique = sorted(set(keys.values()))
    ranked = _dl_top_k_for_keys(unique, index.canonical, config)
    return {position: ranked[key] for position, key in keys.items()}


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
            order = _top_k_desc(matrix[row_number], config.hangul_key_top_k).copy()
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
        scores[position] = (
            np.asarray(selected, dtype=np.int64),
            np.asarray([best[p] for p in selected], dtype=np.float64),
        )
    return scores


def _latin_sorted_scores(
    prepared: Sequence[_Prepared],
    index: CatalogIndex,
    config: MatchConfig,
) -> dict[int, tuple[Positions, Scores]]:
    keys = {
        position: item.sorted_latin
        for position, item in enumerate(prepared)
        if not item.is_hangul and item.sorted_latin
    }
    if not keys:
        return {}
    unique = sorted(set(keys.values()))
    ranked = _dl_top_k_for_keys(unique, index.sorted_canonical, config)
    return {position: ranked[key] for position, key in keys.items()}


def _hangul_sorted_scores(
    prepared: Sequence[_Prepared],
    index: CatalogIndex,
    config: MatchConfig,
) -> dict[int, tuple[Positions, Scores]]:
    records = [(position, item) for position, item in enumerate(prepared) if item.is_hangul and any(item.sorted_cons)]
    if not records:
        return {}
    unique = sorted({key for _, item in records for key in item.sorted_cons if key})
    key_scores: dict[str, tuple[Positions, Scores]] = {}
    for start in range(0, len(unique), config.query_chunk):
        keys = unique[start : start + config.query_chunk]
        matrix = process.cdist(
            keys,
            index.sorted_key2_cons,
            scorer=Levenshtein.normalized_similarity,
            workers=config.num_threads,
            dtype=np.float32,
        )
        for row_number, key in enumerate(keys):
            order = _top_k_desc(matrix[row_number], config.hangul_key_top_k).copy()
            key_scores[key] = (order.astype(np.int64), matrix[row_number, order].astype(np.float64))
        del matrix
    scores: dict[int, tuple[Positions, Scores]] = {}
    for position, item in records:
        best: dict[int, float] = {}
        for key in item.sorted_cons:
            if not key:
                continue
            positions, similarities = key_scores[key]
            for candidate, similarity in zip(positions.tolist(), similarities.tolist(), strict=True):
                best[candidate] = max(best.get(candidate, 0.0), similarity)
        selected = sorted(best, key=lambda candidate: (-best[candidate], candidate))[: config.hangul_key_top_k]
        scores[position] = (
            np.asarray(selected, dtype=np.int64),
            np.asarray([best[p] for p in selected], dtype=np.float64),
        )
    return scores


def _token_index_scores(
    prepared: Sequence[_Prepared],
    index: CatalogIndex,
    config: MatchConfig,
) -> dict[int, tuple[Positions, Scores]]:
    scores: dict[int, tuple[Positions, Scores]] = {}
    if config.token_top_k == 0:
        return scores
    for position, item in enumerate(prepared):
        if item.is_hangul:
            seen: dict[str, None] = {}
            for variant in hangul_variants(item.record.vessel_name):
                for key in hangul_token_keys(variant):
                    if key not in seen:
                        seen[key] = None
            units = tuple(seen)
            if not units:
                continue
            idfs = [index.word_key_idf(unit) for unit in units]
            denom = float(sum(idfs))
            if denom <= 0.0:
                continue
            eligible: list[tuple[Positions, float]] = []
            for unit, idf in zip(units, idfs, strict=True):
                df = index.word_key_df.get(unit, 0)
                if 1 <= df <= config.token_df_cap:
                    eligible.append((index.word_key_postings(unit), idf))
        else:
            seen_tokens: dict[str, None] = {}
            for token in item.query_tokens:
                if token not in seen_tokens:
                    seen_tokens[token] = None
            units = tuple(seen_tokens)
            if not units:
                continue
            idfs = [index.token_idf(unit) for unit in units]
            denom = float(sum(idfs))
            if denom <= 0.0:
                continue
            eligible = []
            for unit, idf in zip(units, idfs, strict=True):
                df = index.token_df.get(unit, 0)
                if 1 <= df <= config.token_df_cap:
                    eligible.append((index.token_postings(unit), idf))
        if not eligible:
            continue
        postings_list = [postings for postings, _ in eligible]
        weights_list = [idf for (postings, idf) in eligible]
        concat = np.concatenate(postings_list).astype(np.int64)
        weights = np.concatenate(
            [
                np.full(len(postings), idf, dtype=np.float64)
                for postings, idf in zip(postings_list, weights_list, strict=True)
            ]
        )
        uniq, inverse = np.unique(concat, return_inverse=True)
        sums = np.bincount(inverse, weights=weights)
        cand_scores = sums / denom
        mask = cand_scores > 0.0
        if not bool(mask.any()):
            continue
        cand_pos = uniq[mask]
        cand_vals = cand_scores[mask]
        order = np.lexsort((cand_pos, -cand_vals))
        top = order[: config.token_top_k]
        scores[position] = (cand_pos[top].astype(np.int64), cand_vals[top].astype(np.float64))
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

    window_list = window.tolist()
    window_cons = [index.key2_cons[position] for position in window_list]
    window_vow = [index.key2_vow[position] for position in window_list]
    window_sorted = [index.sorted_key2_cons[position] for position in window_list]
    total = np.zeros(len(window_list), dtype=np.float64)
    for cons_key, vowel_key in zip(item.cons_keys, item.vowel_keys, strict=True):
        cons_sim = process.cdist(
            [cons_key], window_cons, scorer=Levenshtein.normalized_similarity, workers=1, dtype=np.float64
        )[0]
        vow_sim = process.cdist(
            [vowel_key], window_vow, scorer=Levenshtein.normalized_similarity, workers=1, dtype=np.float64
        )[0]
        total = np.maximum(total, 0.5 * (cons_sim + vow_sim))
    sorted_keys = [key for key in item.sorted_cons if key]
    if sorted_keys:
        sorted_best = np.zeros(len(window_list), dtype=np.float64)
        for sorted_key in sorted_keys:
            sim = process.cdist(
                [sorted_key], window_sorted, scorer=Levenshtein.normalized_similarity, workers=1, dtype=np.float64
            )[0]
            sorted_best = np.maximum(sorted_best, sim)
        total = np.maximum(total, sorted_best)

    selected = sorted(range(len(window_list)), key=lambda i: (-float(total[i]), int(window_list[i])))[
        : config.hangul_window_keep
    ]
    return _union([np.asarray([window_list[i] for i in selected], dtype=np.int64), exact]), exact


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


def _name_similarity(item: _Prepared, index: CatalogIndex, position: int) -> float:
    if item.is_hangul:
        if not item.cons_keys:
            return 0.0
        return float(
            np.float32(max(Levenshtein.normalized_similarity(key, index.key2_cons[position]) for key in item.cons_keys))
        )
    if not item.latin_key:
        return 0.0
    return float(np.float32(DamerauLevenshtein.normalized_similarity(item.latin_key, index.canonical[position])))


def _sorted_similarity(item: _Prepared, index: CatalogIndex, position: int) -> float:
    if item.is_hangul:
        keys = [key for key in item.sorted_cons if key]
        if not keys:
            return 0.0
        return float(
            np.float32(max(Levenshtein.normalized_similarity(key, index.sorted_key2_cons[position]) for key in keys))
        )
    if not item.sorted_latin:
        return 0.0
    return float(
        np.float32(DamerauLevenshtein.normalized_similarity(item.sorted_latin, index.sorted_canonical[position]))
    )


def _token_index_similarity(item: _Prepared, index: CatalogIndex, config: MatchConfig, position: int) -> float:
    if item.is_hangul:
        seen: dict[str, None] = {}
        for variant in hangul_variants(item.record.vessel_name):
            for key in hangul_token_keys(variant):
                if key not in seen:
                    seen[key] = None
        units = tuple(seen)
        if not units:
            return 0.0
        idfs = {unit: index.word_key_idf(unit) for unit in units}
        denom = float(sum(idfs.values()))
        if denom <= 0.0:
            return 0.0
        candidate = set(index.word_keys[position])
        numerator = sum(
            idf
            for unit, idf in idfs.items()
            if unit in candidate and 1 <= index.word_key_df.get(unit, 0) <= config.token_df_cap
        )
        return float(numerator / denom)
    seen_tokens: dict[str, None] = {}
    for token in item.query_tokens:
        if token not in seen_tokens:
            seen_tokens[token] = None
    units = tuple(seen_tokens)
    if not units:
        return 0.0
    idfs = {unit: index.token_idf(unit) for unit in units}
    denom = float(sum(idfs.values()))
    if denom <= 0.0:
        return 0.0
    candidate = index.tokens[position]
    numerator = sum(
        idf
        for unit, idf in idfs.items()
        if unit in candidate and 1 <= index.token_df.get(unit, 0) <= config.token_df_cap
    )
    return float(numerator / denom)


def _llm_pseudo_similarity(
    pseudo_items: Sequence[_Prepared], index: CatalogIndex, config: MatchConfig, position: int
) -> float:
    best = 0.0
    for pseudo in pseudo_items:
        best = max(best, _name_similarity(pseudo, index, position))
        best = max(best, _sorted_similarity(pseudo, index, position))
        best = max(best, _token_index_similarity(pseudo, index, config, position))
    return float(np.float32(best))


def _assemble(
    item: _Prepared,
    index: CatalogIndex,
    memory: CallSignMemory | None,
    config: MatchConfig,
    name_positions: Positions,
    name_scores: Scores,
    sorted_positions: Positions,
    sorted_scores: Scores,
    tonnage_positions: Positions,
    token_positions: Positions,
    tidx_positions: Positions,
    tidx_scores: Scores,
    exact_positions: Positions,
    llm_positions: Positions | None = None,
    llm_scores: Scores | None = None,
    _llm_pseudo: Sequence[_Prepared] = (),
) -> CandidateSet:
    masks: dict[int, int] = {}

    def add(positions: Positions, bit: int) -> None:
        for position in positions.tolist():
            masks[int(position)] = masks.get(int(position), 0) | bit

    add(name_positions, SRC_NAME)
    add(sorted_positions, SRC_SORTED)
    add(tonnage_positions, SRC_TONNAGE)
    add(token_positions, SRC_TOKEN)
    add(tidx_positions, SRC_TOKEN_INDEX)
    if llm_positions is not None:
        add(llm_positions, SRC_LLM)
    memory_position = memory.lookup(item.record.call_sign) if memory is not None else None
    if memory_position is not None:
        add(np.asarray([memory_position], dtype=np.int64), SRC_MEMORY)

    if not masks:
        return CandidateSet(np.asarray([0], dtype=np.int64), np.asarray([SRC_FALLBACK], dtype=np.uint8))

    score_of: dict[int, float] = {}
    for position, similarity in zip(name_positions.tolist(), name_scores.tolist(), strict=True):
        score_of[int(position)] = max(score_of.get(int(position), 0.0), float(similarity))
    for position, similarity in zip(sorted_positions.tolist(), sorted_scores.tolist(), strict=True):
        score_of[int(position)] = max(score_of.get(int(position), 0.0), float(similarity))
    for position, similarity in zip(tidx_positions.tolist(), tidx_scores.tolist(), strict=True):
        score_of[int(position)] = max(score_of.get(int(position), 0.0), float(similarity))
    if llm_positions is not None and llm_scores is not None:
        for position, similarity in zip(llm_positions.tolist(), llm_scores.tolist(), strict=True):
            score_of[int(position)] = max(score_of.get(int(position), 0.0), float(similarity))
    name_set = {int(position) for position in name_positions.tolist()}
    sorted_set = {int(position) for position in sorted_positions.tolist()}
    tidx_set = {int(position) for position in tidx_positions.tolist()}
    if len(masks) > config.max_candidates:
        for position in masks:
            best = score_of.get(position, 0.0)
            if position not in name_set:
                best = max(best, _name_similarity(item, index, position))
            if position not in sorted_set:
                best = max(best, _sorted_similarity(item, index, position))
            if position not in tidx_set:
                best = max(best, _token_index_similarity(item, index, config, position))
            if _llm_pseudo:
                best = max(best, _llm_pseudo_similarity(_llm_pseudo, index, config, position))
            score_of[position] = float(np.float32(best))
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
