"""Pairwise feature matrix consumed by the LightGBM ranker.

Every feature is computed from a single record and its own candidate set, so a
record's rows never depend on the other records passed to :func:`build_features`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import fuzz, process
from rapidfuzz.distance import DamerauLevenshtein, Indel, JaroWinkler, Levenshtein

from port_entity_matching.candidates import (
    SRC_LLM,
    SRC_NAME,
    SRC_SORTED,
    SRC_TOKEN,
    SRC_TOKEN_INDEX,
    SRC_TONNAGE,
    CandidateSet,
)
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.normalize import (
    canonical_latin,
    decode_letter_acronym,
    digit_tokens,
    hangul_phonetic_key,
    hangul_phonetic_key_v2,
    hangul_token_keys,
    hangul_variants,
    is_hangul_name,
    latin_tokens,
    normalize_text,
    sorted_token_key,
)
from port_entity_matching.records import QueryRecord

__all__ = ["FEATURE_NAMES", "build_features"]

FEATURE_NAMES: tuple[str, ...] = (
    "is_hangul",
    "lat_dl_sim",
    "lat_lev_sim",
    "lat_jw_sim",
    "lat_token_set",
    "lat_partial",
    "lat_prefix",
    "lat_exact",
    "rare_token_overlap",
    "kor_cons_sim",
    "kor_vow_sim",
    "kor_cons_jw",
    "kor_len_ratio",
    "kor_acronym_hit",
    "digits_equal",
    "digits_xor",
    "logratio_igt",
    "exact_igt",
    "band_igt",
    "near_igt",
    "logratio_gt",
    "exact_gt",
    "band_gt",
    "near_gt",
    "min_abs_logratio",
    "any_exact",
    "query_tonnage_missing",
    "catalog_gt_missing",
    "log_same_gt_count",
    "type_logprob",
    "flag_logprob",
    "type_logprior",
    "primary_rank",
    "primary_gap",
    "n_exact_in_row",
    "log_n_candidates",
    "src_name",
    "src_tonnage",
    "src_token",
    "kor2_cons_sim",
    "kor2_vow_sim",
    "kor2_vow_jw",
    "kor2_vow_partial",
    "kor2_vow_indel",
    "kor2_primary_gap",
    "kor2_primary_rank",
    "name_sorted_sim",
    "name_token_cover_query",
    "name_token_cover_cand",
    "name_token_fuzzy_cover",
    "token_count_diff",
    "src_sorted",
    "src_token_index",
    "llm_dl_sim",
    "llm_token_set",
    "llm_sorted_sim",
    "llm_exact",
    "llm_top1_dl",
    "llm_best_rank",
    "src_llm",
)

_LATIN_NAMES: tuple[str, ...] = (
    "lat_dl_sim",
    "lat_lev_sim",
    "lat_jw_sim",
    "lat_token_set",
    "lat_partial",
    "lat_prefix",
    "lat_exact",
)
_HANGUL_NAMES: tuple[str, ...] = (
    "kor_cons_sim",
    "kor_vow_sim",
    "kor_cons_jw",
    "kor_len_ratio",
    "kor_acronym_hit",
)
_HANGUL_V2_NAMES: tuple[str, ...] = (
    "kor2_cons_sim",
    "kor2_vow_sim",
    "kor2_vow_jw",
    "kor2_vow_partial",
    "kor2_vow_indel",
    "kor2_primary_gap",
    "kor2_primary_rank",
)

_LLM_NAMES: tuple[str, ...] = (
    "llm_dl_sim",
    "llm_token_set",
    "llm_sorted_sim",
    "llm_exact",
    "llm_top1_dl",
    "llm_best_rank",
)

Scorer = Callable[[str, str], float]


def build_features(
    records: Sequence[QueryRecord],
    candidate_sets: Sequence[CandidateSet],
    index: CatalogIndex,
    maps: AttributeMaps,
    config: MatchConfig,
) -> NDArray[np.float32]:
    """Return the float32 feature matrix; rows follow the records and each set's position order."""
    if len(records) != len(candidate_sets):
        raise ValueError("records and candidate_sets must have the same length")
    sizes = [len(candidate_set.positions) for candidate_set in candidate_sets]
    matrix = np.full((sum(sizes), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    offset = 0
    for record, candidate_set in zip(records, candidate_sets, strict=True):
        size = len(candidate_set.positions)
        if size:
            matrix[offset : offset + size] = _record_features(record, candidate_set, index, maps, config)
        offset += size
    return matrix


def _record_features(
    record: QueryRecord,
    candidate_set: CandidateSet,
    index: CatalogIndex,
    maps: AttributeMaps,
    config: MatchConfig,
) -> NDArray[np.float32]:
    positions = candidate_set.positions
    size = len(positions)
    columns: dict[str, NDArray[np.float64]] = {}
    hangul = is_hangul_name(record.vessel_name)
    columns["is_hangul"] = np.full(size, 1.0 if hangul else 0.0)

    _latin_columns(record, positions, index, columns, hangul)
    _hangul_columns(record, positions, index, columns, hangul)
    _order_invariant_columns(record, positions, index, columns, config, hangul)
    _hangul_v2_columns(record, positions, index, columns, hangul)
    _llm_columns(record, positions, index, columns, hangul)
    columns["rare_token_overlap"] = _rare_token_overlap(record, positions, index, config, hangul)
    _digit_columns(record, positions, index, columns)
    _tonnage_columns(record, positions, index, columns, config)
    _attribute_columns(record, positions, index, maps, columns)
    _context_columns(columns, positions, index, candidate_set, hangul)

    matrix = np.empty((size, len(FEATURE_NAMES)), dtype=np.float32)
    for column, name in enumerate(FEATURE_NAMES):
        matrix[:, column] = columns[name]
    return matrix


def _latin_columns(
    record: QueryRecord,
    positions: NDArray[np.int64],
    index: CatalogIndex,
    columns: dict[str, NDArray[np.float64]],
    hangul: bool,
) -> None:
    if hangul:
        for name in _LATIN_NAMES:
            columns[name] = np.full(len(positions), np.nan)
        return
    query = canonical_latin(record.vessel_name)
    canonical = [index.canonical[position] for position in positions]
    columns["lat_dl_sim"] = _cdist(query, canonical, DamerauLevenshtein.normalized_similarity)
    columns["lat_lev_sim"] = _cdist(query, canonical, Levenshtein.normalized_similarity)
    columns["lat_jw_sim"] = _cdist(query, canonical, JaroWinkler.normalized_similarity)
    query_tokens = " ".join(latin_tokens(record.vessel_name))
    candidate_tokens = [" ".join(sorted(index.tokens[position])) for position in positions]
    columns["lat_token_set"] = _cdist(query_tokens, candidate_tokens, fuzz.token_set_ratio) / 100.0
    columns["lat_partial"] = _cdist(query, canonical, fuzz.partial_ratio) / 100.0
    columns["lat_prefix"] = np.array([_prefix_ratio(query, candidate) for candidate in canonical])
    columns["lat_exact"] = np.array([1.0 if query == candidate else 0.0 for candidate in canonical])


def _hangul_columns(
    record: QueryRecord,
    positions: NDArray[np.int64],
    index: CatalogIndex,
    columns: dict[str, NDArray[np.float64]],
    hangul: bool,
) -> None:
    if not hangul:
        for name in _HANGUL_NAMES:
            columns[name] = np.full(len(positions), np.nan)
        return
    variants = hangul_variants(record.vessel_name)
    candidate_cons = [index.key_cons[position] for position in positions]
    candidate_vow = [index.key_vow[position] for position in positions]
    cons_keys = [hangul_phonetic_key(variant, False) for variant in variants]
    vowel_keys = [hangul_phonetic_key(variant, True) for variant in variants]
    cons_matrix = np.vstack([_cdist(key, candidate_cons, Levenshtein.normalized_similarity) for key in cons_keys])
    vowel_matrix = np.vstack([_cdist(key, candidate_vow, Levenshtein.normalized_similarity) for key in vowel_keys])
    jw_matrix = np.vstack([_cdist(key, candidate_cons, JaroWinkler.normalized_similarity) for key in cons_keys])

    columns["kor_cons_sim"] = cons_matrix.max(axis=0)
    columns["kor_vow_sim"] = vowel_matrix.max(axis=0)
    columns["kor_cons_jw"] = jw_matrix.max(axis=0)

    best = np.argmax(cons_matrix, axis=0)
    ratios = np.full(len(positions), np.nan)
    for offset in range(len(positions)):
        query_length = len(cons_keys[int(best[offset])])
        candidate_length = len(candidate_cons[offset])
        longest = max(query_length, candidate_length)
        if longest:
            ratios[offset] = min(query_length, candidate_length) / longest
    columns["kor_len_ratio"] = ratios

    acronyms = _acronym_tokens(record.vessel_name)
    if acronyms:
        columns["kor_acronym_hit"] = np.array(
            [
                sum(1 for acronym in acronyms if acronym in index.tokens[position]) / len(acronyms)
                for position in positions
            ]
        )
    else:
        columns["kor_acronym_hit"] = np.full(len(positions), np.nan)


def _hangul_v2_columns(
    record: QueryRecord,
    positions: NDArray[np.int64],
    index: CatalogIndex,
    columns: dict[str, NDArray[np.float64]],
    hangul: bool,
) -> None:
    if not hangul:
        for name in _HANGUL_V2_NAMES:
            columns[name] = np.full(len(positions), np.nan)
        return
    variants = hangul_variants(record.vessel_name)
    candidate_cons = [index.key2_cons[position] for position in positions]
    candidate_vow = [index.key2_vow[position] for position in positions]
    cons_keys = [hangul_phonetic_key_v2(variant, False) for variant in variants]
    vowel_keys = [hangul_phonetic_key_v2(variant, True) for variant in variants]
    cons_matrix = np.vstack([_cdist(key, candidate_cons, Levenshtein.normalized_similarity) for key in cons_keys])
    vowel_lev = np.vstack([_cdist(key, candidate_vow, Levenshtein.normalized_similarity) for key in vowel_keys])
    vowel_jw = np.vstack([_cdist(key, candidate_vow, JaroWinkler.normalized_similarity) for key in vowel_keys])
    vowel_partial = np.vstack([_cdist(key, candidate_vow, fuzz.partial_ratio) for key in vowel_keys]) / 100.0
    vowel_indel = np.vstack([_cdist(key, candidate_vow, Indel.normalized_similarity) for key in vowel_keys])

    columns["kor2_cons_sim"] = cons_matrix.max(axis=0)
    columns["kor2_vow_sim"] = vowel_lev.max(axis=0)
    columns["kor2_vow_jw"] = vowel_jw.max(axis=0)
    columns["kor2_vow_partial"] = vowel_partial.max(axis=0)
    columns["kor2_vow_indel"] = vowel_indel.max(axis=0)

    primary = (columns["kor2_cons_sim"] + columns["kor2_vow_sim"]) / 2.0
    primary = np.maximum(primary, columns["name_sorted_sim"])
    columns["kor2_primary_gap"] = primary.max() - primary
    order = np.lexsort((positions, -primary))
    rank = np.empty(len(positions), dtype=np.float64)
    rank[order] = np.arange(1, len(positions) + 1)
    columns["kor2_primary_rank"] = rank


def _llm_columns(
    record: QueryRecord,
    positions: NDArray[np.int64],
    index: CatalogIndex,
    columns: dict[str, NDArray[np.float64]],
    hangul: bool,
) -> None:
    if not hangul:
        for name in _LLM_NAMES:
            columns[name] = np.full(len(positions), np.nan)
        return
    cleaned: list[str] = []
    canonicals: list[str] = []
    for alt in record.alt_names:
        key = canonical_latin(alt)
        if not key or key in canonicals:
            continue
        canonicals.append(key)
        cleaned.append(alt)
    if not cleaned:
        for name in _LLM_NAMES:
            columns[name] = np.full(len(positions), np.nan)
        return
    candidate_canonical = [index.canonical[position] for position in positions]
    candidate_sorted = [index.sorted_canonical[position] for position in positions]
    candidate_token_str = [" ".join(sorted(index.tokens[position])) for position in positions]
    query_sorted = [sorted_token_key(latin_tokens(alt)) for alt in cleaned]
    query_token_str = [" ".join(latin_tokens(alt)) for alt in cleaned]
    dl_matrix = np.vstack([_cdist(key, candidate_canonical, DamerauLevenshtein.normalized_similarity) for key in canonicals])
    sorted_matrix = np.vstack(
        [_cdist(key, candidate_sorted, DamerauLevenshtein.normalized_similarity) for key in query_sorted]
    )
    token_matrix = np.vstack([_cdist(query, candidate_token_str, fuzz.token_set_ratio) for query in query_token_str]) / 100.0
    columns["llm_dl_sim"] = dl_matrix.max(axis=0)
    columns["llm_sorted_sim"] = sorted_matrix.max(axis=0)
    columns["llm_token_set"] = token_matrix.max(axis=0)
    columns["llm_top1_dl"] = dl_matrix[0]
    columns["llm_exact"] = np.array(
        [1.0 if any(key == candidate for key in canonicals) else 0.0 for candidate in candidate_canonical]
    )
    best = columns["llm_dl_sim"]
    ranks = np.full(len(positions), np.nan)
    for offset in range(len(positions)):
        for rank, row in enumerate(dl_matrix):
            if row[offset] == best[offset]:
                ranks[offset] = float(rank)
                break
    columns["llm_best_rank"] = ranks


def _order_invariant_columns(
    record: QueryRecord,
    positions: NDArray[np.int64],
    index: CatalogIndex,
    columns: dict[str, NDArray[np.float64]],
    config: MatchConfig,
    hangul: bool,
) -> None:
    size = len(positions)
    threshold = config.token_match_threshold
    if hangul:
        variants = hangul_variants(record.vessel_name)
        variant_keys = [hangul_token_keys(variant) for variant in variants]
        query_sorted = [sorted_token_key(keys) for keys in variant_keys]
        candidate_sorted = [index.sorted_key2_cons[position] for position in positions]
        if variant_keys and any(query_sorted):
            matrices = [
                _cdist(key, candidate_sorted, Levenshtein.normalized_similarity) if key else np.zeros(size)
                for key in query_sorted
            ]
            columns["name_sorted_sim"] = np.vstack(matrices).max(axis=0)
        else:
            columns["name_sorted_sim"] = np.zeros(size)
        cover_q = np.zeros(size)
        cover_c = np.zeros(size)
        fuzzy = np.full(size, np.nan)
        if variants and any(variant_keys):
            has_units = False
            for keys in variant_keys:
                if not keys:
                    continue
                has_units = True
                q_idf = {unit: index.word_key_idf(unit) for unit in keys}
                q_sum = float(sum(q_idf.values()))
                for offset, position in enumerate(positions):
                    cand_set = set(index.word_keys[position])
                    shared = [unit for unit in q_idf if unit in cand_set]
                    numerator = sum(q_idf[unit] for unit in shared)
                    cand_sum = float(sum(index.word_key_idf(unit) for unit in sorted(cand_set)))
                    q_value = numerator / q_sum if q_sum > 0.0 else 0.0
                    c_value = numerator / cand_sum if cand_sum > 0.0 else 0.0
                    cover_q[offset] = max(cover_q[offset], q_value)
                    cover_c[offset] = max(cover_c[offset], c_value)
            if has_units:
                fuzzy_vals = np.zeros(size)
                for keys in variant_keys:
                    if not keys:
                        continue
                    total_weight = float(sum(len(unit) for unit in keys))
                    for offset, position in enumerate(positions):
                        cand_units = tuple(set(index.word_keys[position]))
                        if not cand_units:
                            value = 0.0
                        else:
                            matched = 0.0
                            for query_unit in keys:
                                best = 0.0
                                for cand_unit in cand_units:
                                    similarity = Levenshtein.normalized_similarity(query_unit, cand_unit)
                                    if similarity > best:
                                        best = similarity
                                        if best >= threshold:
                                            break
                                if best >= threshold:
                                    matched += len(query_unit)
                            value = matched / total_weight
                        fuzzy_vals[offset] = max(fuzzy_vals[offset], value)
                fuzzy = fuzzy_vals
        columns["name_token_cover_query"] = cover_q
        columns["name_token_cover_cand"] = cover_c
        columns["name_token_fuzzy_cover"] = fuzzy
        query_count = len(variant_keys[0]) if variants and variant_keys else 0
        columns["token_count_diff"] = np.array(
            [abs(query_count - len(index.word_keys[position])) for position in positions], dtype=np.float64
        )
    else:
        query_tokens = latin_tokens(record.vessel_name)
        query_sorted_key = sorted_token_key(query_tokens)
        candidate_sorted = [index.sorted_canonical[position] for position in positions]
        columns["name_sorted_sim"] = _cdist(
            query_sorted_key, candidate_sorted, DamerauLevenshtein.normalized_similarity
        )
        if not query_tokens:
            columns["name_token_cover_query"] = np.zeros(size)
            columns["name_token_cover_cand"] = np.zeros(size)
            columns["name_token_fuzzy_cover"] = np.full(size, np.nan)
            columns["token_count_diff"] = np.array(
                [len(latin_tokens(index.names[position])) for position in positions], dtype=np.float64
            )
            return
        q_idf = {unit: index.token_idf(unit) for unit in query_tokens}
        q_sum = float(sum(q_idf.values()))
        total_weight = float(sum(len(unit) for unit in query_tokens))
        cover_q = np.zeros(size)
        cover_c = np.zeros(size)
        fuzzy = np.zeros(size)
        for offset, position in enumerate(positions):
            cand_set = index.tokens[position]
            shared = [unit for unit in q_idf if unit in cand_set]
            numerator = sum(q_idf[unit] for unit in shared)
            cover_q[offset] = numerator / q_sum if q_sum > 0.0 else 0.0
            cand_sum = float(sum(index.token_idf(unit) for unit in sorted(cand_set)))
            cover_c[offset] = numerator / cand_sum if cand_sum > 0.0 else 0.0
            if not cand_set:
                fuzzy[offset] = 0.0
            else:
                matched = 0.0
                for query_unit in query_tokens:
                    best = 0.0
                    for cand_unit in cand_set:
                        similarity = DamerauLevenshtein.normalized_similarity(query_unit, cand_unit)
                        if similarity > best:
                            best = similarity
                            if best >= threshold:
                                break
                    if best >= threshold:
                        matched += len(query_unit)
                fuzzy[offset] = matched / total_weight if total_weight > 0.0 else 0.0
        columns["name_token_cover_query"] = cover_q
        columns["name_token_cover_cand"] = cover_c
        columns["name_token_fuzzy_cover"] = fuzzy
        columns["token_count_diff"] = np.array(
            [abs(len(query_tokens) - len(latin_tokens(index.names[position]))) for position in positions], dtype=np.float64
        )


def _rare_token_overlap(
    record: QueryRecord,
    positions: NDArray[np.int64],
    index: CatalogIndex,
    config: MatchConfig,
    hangul: bool,
) -> NDArray[np.float64]:
    if hangul:
        tokens: list[str] = []
        for variant in hangul_variants(record.vessel_name):
            tokens.extend(latin_tokens(variant))
    else:
        tokens = list(latin_tokens(record.vessel_name))
    rare = {token for token in tokens if index.token_df.get(token, 0) <= config.rare_token_max_df}
    if not rare:
        return np.zeros(len(positions))
    return np.array([sum(1 for token in rare if token in index.tokens[position]) for position in positions])


def _digit_columns(
    record: QueryRecord,
    positions: NDArray[np.int64],
    index: CatalogIndex,
    columns: dict[str, NDArray[np.float64]],
) -> None:
    query_digits = digit_tokens(record.vessel_name)
    equal = np.full(len(positions), np.nan)
    exclusive = np.zeros(len(positions))
    for offset, position in enumerate(positions):
        candidate_digits = index.digits[position]
        if query_digits and candidate_digits:
            equal[offset] = 1.0 if query_digits == candidate_digits else 0.0
        exclusive[offset] = 1.0 if bool(query_digits) != bool(candidate_digits) else 0.0
    columns["digits_equal"] = equal
    columns["digits_xor"] = exclusive


def _tonnage_columns(
    record: QueryRecord,
    positions: NDArray[np.int64],
    index: CatalogIndex,
    columns: dict[str, NDArray[np.float64]],
    config: MatchConfig,
) -> None:
    catalog_tonnage = index.gt[positions]
    queries = (
        ("igt", record.international_gross_tonnage),
        ("gt", record.gross_tonnage),
    )
    for key, query in queries:
        logratio = np.full(len(positions), np.nan)
        exact = np.full(len(positions), np.nan)
        band = np.full(len(positions), np.nan)
        near = np.full(len(positions), np.nan)
        if query is not None:
            for offset, candidate in enumerate(catalog_tonnage):
                if not np.isfinite(candidate):
                    continue
                ratio = query / float(candidate)
                difference = abs(ratio - 1.0)
                logratio[offset] = (
                    math.log(ratio) if 0.0 < ratio < math.inf else math.log(query) - math.log(float(candidate))
                )
                exact[offset] = 1.0 if query == float(candidate) else 0.0
                band[offset] = 1.0 if config.band_low <= difference <= config.band_high else 0.0
                near[offset] = 1.0 if 0.0 < difference <= config.near_tol else 0.0
        columns[f"logratio_{key}"] = logratio
        columns[f"exact_{key}"] = exact
        columns[f"band_{key}"] = band
        columns[f"near_{key}"] = near

    absolute = np.abs(np.vstack([columns["logratio_igt"], columns["logratio_gt"]]))
    comparable = np.isfinite(absolute).any(axis=0)
    minimum = np.full(len(positions), np.nan)
    if bool(comparable.any()):
        masked = np.where(np.isfinite(absolute), absolute, np.inf)
        minimum[comparable] = masked[:, comparable].min(axis=0)
    columns["min_abs_logratio"] = minimum

    exact_flags = [columns["exact_igt"], columns["exact_gt"]]
    comparable = np.isfinite(np.vstack(exact_flags)).any(axis=0)
    any_exact = np.zeros(len(positions))
    for value in exact_flags:
        any_exact = np.where(np.nan_to_num(value, nan=0.0) == 1.0, 1.0, any_exact)
    columns["any_exact"] = np.where(comparable, any_exact, np.nan)

    missing = 1.0 if record.international_gross_tonnage is None and record.gross_tonnage is None else 0.0
    columns["query_tonnage_missing"] = np.full(len(positions), missing)
    columns["catalog_gt_missing"] = np.array([0.0 if np.isfinite(value) else 1.0 for value in catalog_tonnage])
    columns["log_same_gt_count"] = np.log1p(index.same_gt_count[positions].astype(np.float64))


def _attribute_columns(
    record: QueryRecord,
    positions: NDArray[np.int64],
    index: CatalogIndex,
    maps: AttributeMaps,
    columns: dict[str, NDArray[np.float64]],
) -> None:
    columns["type_logprob"] = np.array(
        [maps.type_logprob(record.vessel_type, index.vessel_type[position]) for position in positions]
    )
    columns["flag_logprob"] = np.array(
        [maps.flag_logprob(record.flag_name, index.flag[position]) for position in positions]
    )
    columns["type_logprior"] = np.array([maps.type_logprior(index.vessel_type[position]) for position in positions])


def _context_columns(
    columns: dict[str, NDArray[np.float64]],
    positions: NDArray[np.int64],
    index: CatalogIndex,
    candidate_set: CandidateSet,
    hangul: bool,
) -> None:
    size = len(positions)
    if hangul:
        primary = (columns["kor_cons_sim"] + columns["kor_vow_sim"]) / 2.0
    else:
        primary = np.maximum(columns["lat_dl_sim"], columns["name_sorted_sim"])
    finite = np.isfinite(primary)
    sortable = np.where(finite, primary, -np.inf)
    order = np.lexsort((positions, -sortable))
    rank = np.empty(size, dtype=np.float64)
    rank[order] = np.arange(1, size + 1)
    columns["primary_rank"] = rank

    best = float(np.max(primary[finite]))
    columns["primary_gap"] = best - primary

    exact_count = int(np.count_nonzero(columns["any_exact"] == 1.0))
    columns["n_exact_in_row"] = np.full(size, float(exact_count))
    columns["log_n_candidates"] = np.full(size, math.log(size) if size else np.nan)

    sources = candidate_set.sources
    columns["src_name"] = ((sources & SRC_NAME) > 0).astype(np.float64)
    columns["src_tonnage"] = ((sources & SRC_TONNAGE) > 0).astype(np.float64)
    columns["src_token"] = ((sources & SRC_TOKEN) > 0).astype(np.float64)
    columns["src_sorted"] = ((sources & SRC_SORTED) > 0).astype(np.float64)
    columns["src_token_index"] = ((sources & SRC_TOKEN_INDEX) > 0).astype(np.float64)
    columns["src_llm"] = ((sources & SRC_LLM) > 0).astype(np.float64)


def _cdist(query: str, choices: Sequence[str], scorer: Scorer) -> NDArray[np.float64]:
    matrix = process.cdist([query], list(choices), scorer=cast("Any", scorer), dtype=np.float32)
    return np.asarray(matrix[0], dtype=np.float64)


def _prefix_ratio(query: str, candidate: str) -> float:
    longest = max(len(query), len(candidate))
    if longest == 0:
        return math.nan
    shared = 0
    for left, right in zip(query, candidate, strict=False):
        if left != right:
            break
        shared += 1
    return shared / longest


def _acronym_tokens(name: str) -> tuple[str, ...]:
    tokens = normalize_text(name).split()
    decoded = [decode_letter_acronym(token) for token in tokens]
    return tuple(dict.fromkeys(token for token in decoded if token))
