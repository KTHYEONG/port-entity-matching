"""Contract tests for the pair feature matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel, JaroWinkler, Levenshtein

from port_entity_matching.candidates import SRC_NAME, SRC_TOKEN, CandidateSet
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.features import FEATURE_NAMES, build_features
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.normalize import hangul_phonetic_key_v2, hangul_variants
from port_entity_matching.records import QueryRecord


def _index() -> CatalogIndex:
    frame = pd.DataFrame(
        [
            {
                "vessel_id": "V0",
                "vessel_name": "GRAND ACE 11",
                "gross_tonnage": 100,
                "vessel_type": "Container",
                "flag_name": "Panama",
            },
            {
                "vessel_id": "V1",
                "vessel_name": "Grand Ace1",
                "gross_tonnage": 100,
                "vessel_type": "Container",
                "flag_name": "Panama",
            },
            {
                "vessel_id": "V2",
                "vessel_name": "Grand Ace 11",
                "gross_tonnage": 101,
                "vessel_type": "Bulk",
                "flag_name": "Liberia",
            },
            {
                "vessel_id": "V3",
                "vessel_name": "Glovis Spirit",
                "gross_tonnage": 5000,
                "vessel_type": "Tanker",
                "flag_name": "Marshall Islands",
            },
        ]
    )
    return CatalogIndex.build(frame)


def _maps(index: CatalogIndex) -> AttributeMaps:
    train = pd.DataFrame(
        [
            {"vessel_type": "컨테이너선", "flag_name": "파나마", "target_vessel_id": "V0"},
            {"vessel_type": "컨테이너선", "flag_name": "파나마", "target_vessel_id": "V1"},
            {"vessel_type": "벌크선", "flag_name": "라이베리아", "target_vessel_id": "V2"},
        ]
    )
    return AttributeMaps.fit(train, index)


def _record(
    name: str,
    *,
    igt: float | None = None,
    gt: float | None = None,
    vessel_type: str | None = "컨테이너선",
    flag: str | None = "파나마",
) -> QueryRecord:
    return QueryRecord(
        sample_id="S",
        vessel_name=name,
        call_sign=None,
        gross_tonnage=gt,
        international_gross_tonnage=igt,
        flag_name=flag,
        vessel_type=vessel_type,
        arrival_year=2020,
    )


def _candidate_set(positions: list[int], sources: int = SRC_NAME) -> CandidateSet:
    return CandidateSet(
        np.asarray(positions, dtype=np.int64),
        np.full(len(positions), sources, dtype=np.uint8),
    )


def _column(name: str) -> int:
    return FEATURE_NAMES.index(name)


_PREVIOUS_FEATURE_NAMES: tuple[str, ...] = (
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
)

_KEY_V2_NAMES: tuple[str, ...] = (
    "kor2_cons_sim",
    "kor2_vow_sim",
    "kor2_vow_jw",
    "kor2_vow_partial",
    "kor2_vow_indel",
    "kor2_primary_gap",
    "kor2_primary_rank",
)

_KEY_V2_SIMILARITY_NAMES: tuple[str, ...] = _KEY_V2_NAMES[:5]


def _korean_index() -> CatalogIndex:
    return CatalogIndex.build(
        pd.DataFrame(
            [
                {
                    "vessel_id": "H0",
                    "vessel_name": "HEUNG A",
                    "gross_tonnage": 100,
                    "vessel_type": "Tanker",
                    "flag_name": "Panama",
                },
                {
                    "vessel_id": "H1",
                    "vessel_name": "OTHER SHIP",
                    "gross_tonnage": 200,
                    "vessel_type": "Tanker",
                    "flag_name": "Panama",
                },
                {
                    "vessel_id": "H2",
                    "vessel_name": "HEUNG A",
                    "gross_tonnage": 300,
                    "vessel_type": "Tanker",
                    "flag_name": "Panama",
                },
            ]
        )
    )


def _build(
    records: list[QueryRecord],
    sets: list[CandidateSet],
    index: CatalogIndex | None = None,
) -> np.ndarray:
    catalog_index = index or _index()
    return build_features(records, sets, catalog_index, _maps(catalog_index), MatchConfig())


def test_shape_and_names() -> None:
    records = [_record("GRAND ACE 11", igt=100), _record("글로비스 스피릿", igt=5000)]
    sets = [_candidate_set([0, 1, 2]), _candidate_set([3, 0])]
    matrix = _build(records, sets)
    assert matrix.shape == (5, len(FEATURE_NAMES))
    assert matrix.dtype == np.float32


def test_script_gated_nans() -> None:
    hangul_rows = _build([_record("글로비스 스피릿", igt=5000)], [_candidate_set([3, 0])])
    for name in ("lat_dl_sim", "lat_lev_sim", "lat_jw_sim", "lat_token_set", "lat_partial", "lat_prefix", "lat_exact"):
        assert np.all(np.isnan(hangul_rows[:, _column(name)]))
    assert np.all(np.isfinite(hangul_rows[:, _column("kor_cons_sim")]))
    assert hangul_rows[0, _column("is_hangul")] == 1.0

    latin_rows = _build([_record("GRAND ACE 11", igt=100)], [_candidate_set([0, 1])])
    assert np.all(np.isnan(latin_rows[:, _column("kor_cons_sim")]))
    assert np.all(np.isnan(latin_rows[:, _column("kor_vow_sim")]))
    assert np.all(np.isnan(latin_rows[:, _column("kor_cons_jw")]))
    assert np.all(np.isfinite(latin_rows[:, _column("lat_dl_sim")]))
    assert latin_rows[0, _column("is_hangul")] == 0.0


def test_tonnage_band_flags() -> None:
    perturbed = _build([_record("GRAND ACE 11", igt=104)], [_candidate_set([0])])
    assert perturbed[0, _column("band_igt")] == 1.0
    assert perturbed[0, _column("exact_igt")] == 0.0
    assert perturbed[0, _column("near_igt")] == 0.0

    exact = _build([_record("GRAND ACE 11", igt=100)], [_candidate_set([0])])
    assert exact[0, _column("exact_igt")] == 1.0
    assert exact[0, _column("band_igt")] == 0.0


def test_missing_tonnage_is_nan() -> None:
    matrix = _build([_record("GRAND ACE 11")], [_candidate_set([0])])
    assert np.isnan(matrix[0, _column("logratio_igt")])
    assert np.isnan(matrix[0, _column("logratio_gt")])
    assert matrix[0, _column("query_tonnage_missing")] == 1.0


def test_digit_features() -> None:
    matrix = _build([_record("GRAND ACE 11")], [_candidate_set([1, 2])])
    assert matrix[0, _column("digits_equal")] == 0.0
    assert matrix[1, _column("digits_equal")] == 1.0
    assert np.all(matrix[:, _column("digits_xor")] == 0.0)


def test_row_local_rank() -> None:
    matrix = _build([_record("GRAND ACE 11", igt=100)], [_candidate_set([0, 1, 2])])
    ranks = matrix[:, _column("primary_rank")]
    assert int(np.count_nonzero(ranks == 1)) == 1
    best = int(np.flatnonzero(ranks == 1)[0])
    assert matrix[best, _column("primary_gap")] == pytest.approx(0.0)
    assert matrix[:, _column("log_n_candidates")] == pytest.approx(np.log(3))


def test_row_independence() -> None:
    first = _record("GRAND ACE 11", igt=100)
    second = _record("글로비스 스피릿", igt=5000)
    sets = [_candidate_set([0, 1, 2]), _candidate_set([3, 0])]
    together = _build([first, second], sets)
    alone = _build([first], sets[:1])
    assert np.array_equal(together[:3], alone, equal_nan=True)


def test_source_flags() -> None:
    index = _index()
    records = [_record("GRAND ACE 11", igt=100)]
    sets = [_candidate_set([0, 1], sources=SRC_NAME | SRC_TOKEN)]
    matrix = _build(records, sets, index)
    assert np.all(matrix[:, _column("src_name")] == 1.0)
    assert np.all(matrix[:, _column("src_token")] == 1.0)
    assert np.all(matrix[:, _column("src_tonnage")] == 0.0)


def test_missing_record_attributes_are_nan() -> None:
    matrix = _build(
        [_record("GRAND ACE 11", igt=100, vessel_type=None, flag=None)],
        [_candidate_set([0])],
    )
    assert np.isnan(matrix[0, _column("type_logprob")])
    assert np.isnan(matrix[0, _column("flag_logprob")])
    assert np.isfinite(matrix[0, _column("type_logprior")])


def test_mismatched_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        _build([_record("GRAND ACE 11")], [])


def test_hangul_acronym_hit() -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            [
                {
                    "vessel_id": "K0",
                    "vessel_name": "KMTC VENTURE",
                    "gross_tonnage": 100,
                    "vessel_type": "Tanker",
                    "flag_name": "Panama",
                },
                {
                    "vessel_id": "K1",
                    "vessel_name": "OTHER SHIP",
                    "gross_tonnage": 100,
                    "vessel_type": "Tanker",
                    "flag_name": "Panama",
                },
            ]
        )
    )
    matrix = _build([_record("케이엠티씨", igt=100)], [_candidate_set([0, 1])], index)
    assert matrix[0, _column("kor_acronym_hit")] == pytest.approx(1.0)
    assert matrix[1, _column("kor_acronym_hit")] == pytest.approx(0.0)


def test_missing_catalog_tonnage_is_nan() -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            [
                {
                    "vessel_id": "G0",
                    "vessel_name": "Alpha",
                    "gross_tonnage": None,
                    "vessel_type": "Tanker",
                    "flag_name": "Panama",
                },
            ]
        )
    )
    matrix = _build([_record("ALPHA", igt=100)], [_candidate_set([0])], index)
    assert np.isnan(matrix[0, _column("logratio_igt")])
    assert matrix[0, _column("catalog_gt_missing")] == 1.0
    assert np.isfinite(matrix[0, _column("log_same_gt_count")])


def test_empty_canonical_names_yield_nan_prefix() -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            [
                {
                    "vessel_id": "E0",
                    "vessel_name": "###",
                    "gross_tonnage": 100,
                    "vessel_type": "Tanker",
                    "flag_name": "Panama",
                },
            ]
        )
    )
    matrix = _build([_record("###")], [_candidate_set([0])], index)
    assert np.isnan(matrix[0, _column("lat_prefix")])


def test_missing_catalog_tonnage_does_not_become_a_mismatch() -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            {
                "vessel_id": ["A", "B"],
                "vessel_name": ["Alpha", "Beta"],
                "gross_tonnage": [100, np.nan],
            }
        )
    )
    matrix = _build([_record("Alpha", igt=100, gt=100)], [_candidate_set([0, 1])], index)
    assert matrix[0, _column("any_exact")] == 1
    assert np.isnan(matrix[1, _column("any_exact")])
    assert np.isnan(matrix[1, _column("min_abs_logratio")])
    assert np.all(matrix[:, _column("n_exact_in_row")] == 1)


def test_primary_ties_use_catalog_position_not_array_offset() -> None:
    matrix = _build([_record("GRAND ACE 11")], [_candidate_set([2, 0])])
    assert matrix[:, _column("primary_rank")].tolist() == [2, 1]


@pytest.mark.parametrize("name", ["GRAND GRAND ACE 11", "케이엠티씨 희망호"])
def test_rare_overlap_counts_shared_tokens_once(name: str) -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            {
                "vessel_id": ["A"],
                "vessel_name": ["GRAND ACE 11 KMTC"],
                "gross_tonnage": [100],
            }
        )
    )
    matrix = _build([_record(name)], [_candidate_set([0])], index)
    expected = 3 if name.startswith("GRAND") else 1
    assert matrix[0, _column("rare_token_overlap")] == expected


def test_shuffling_records_does_not_change_features() -> None:
    records = [_record("GRAND ACE 11", igt=100), _record("글로비스 스피릿", igt=5000)]
    sets = [_candidate_set([0, 1]), _candidate_set([3, 0])]
    original = _build(records, sets)
    shuffled = _build(records[::-1], sets[::-1])
    np.testing.assert_array_equal(original[:2], shuffled[2:])
    np.testing.assert_array_equal(original[2:], shuffled[:2])


@pytest.mark.parametrize(("query", "catalog"), [(1e-300, 1e300), (1e300, 1e-300)])
def test_log_tonnage_ratio_remains_finite_at_float_extremes(query: float, catalog: float) -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            {
                "vessel_id": ["A"],
                "vessel_name": ["Alpha"],
                "gross_tonnage": [catalog],
            }
        )
    )
    matrix = _build([_record("Alpha", igt=query)], [_candidate_set([0])], index)
    assert np.isfinite(matrix[0, _column("logratio_igt")])
    assert matrix[0, _column("logratio_igt")] == pytest.approx(np.log(query) - np.log(catalog), rel=1e-6)


def test_feature_width_grows_to_fifty_three() -> None:
    matrix = _build([_record("GRAND ACE 11", igt=100)], [_candidate_set([0, 1, 2])])
    assert matrix.shape[1] == len(FEATURE_NAMES) == 53
    assert FEATURE_NAMES[: len(_PREVIOUS_FEATURE_NAMES)] == _PREVIOUS_FEATURE_NAMES
    assert (
        FEATURE_NAMES[len(_PREVIOUS_FEATURE_NAMES) : len(_PREVIOUS_FEATURE_NAMES) + len(_KEY_V2_NAMES)] == _KEY_V2_NAMES
    )
    assert FEATURE_NAMES[len(_PREVIOUS_FEATURE_NAMES) + len(_KEY_V2_NAMES) :] == (
        "name_sorted_sim",
        "name_token_cover_query",
        "name_token_cover_cand",
        "name_token_fuzzy_cover",
        "token_count_diff",
        "src_sorted",
        "src_token_index",
    )


def test_latin_rows_have_nan_key_v2_columns() -> None:
    matrix = _build([_record("GRAND ACE 11", igt=100)], [_candidate_set([0, 1])])
    for name in _KEY_V2_NAMES:
        assert np.all(np.isnan(matrix[:, _column(name)]))


def test_key_v2_exact_phonetic_match_scores_one() -> None:
    matrix = _build([_record("흥아", igt=100)], [_candidate_set([0, 1])], _korean_index())
    assert matrix[0, _column("kor2_cons_sim")] == pytest.approx(1.0)
    assert matrix[0, _column("kor2_primary_rank")] == 1.0
    assert matrix[0, _column("kor2_primary_gap")] == pytest.approx(0.0)
    assert matrix[1, _column("kor2_primary_gap")] > 0.0


def test_key_v2_rank_ties_use_catalog_position() -> None:
    matrix = _build([_record("흥아", igt=100)], [_candidate_set([2, 0])], _korean_index())
    assert matrix[:, _column("kor2_primary_rank")].tolist() == [2.0, 1.0]


def test_key_v2_gap_is_non_negative_with_zero_at_best() -> None:
    matrix = _build([_record("흥아", igt=100)], [_candidate_set([0, 1, 2])], _korean_index())
    gaps = matrix[:, _column("kor2_primary_gap")]
    assert np.all(gaps >= 0.0)
    assert float(np.min(gaps)) == pytest.approx(0.0)


def test_key_v2_similarities_are_bounded() -> None:
    matrix = _build([_record("흥아", igt=100)], [_candidate_set([0, 1, 2])], _korean_index())
    for name in _KEY_V2_SIMILARITY_NAMES:
        values = matrix[:, _column(name)]
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)


def test_key_v2_columns_keep_row_independence() -> None:
    index = _korean_index()
    records = [
        _record("흥아", igt=100),
        _record("GRAND ACE 11", igt=100),
        _record("글로비스 스피릿", igt=5000),
    ]
    sets = [_candidate_set([0, 1]), _candidate_set([1, 2]), _candidate_set([0, 1, 2])]
    together = _build(records, sets, index)
    for offset, (record, candidate_set) in enumerate(zip(records, sets, strict=True)):
        alone = _build([record], [candidate_set], index)
        start = sum(len(item.positions) for item in sets[:offset])
        np.testing.assert_array_equal(together[start : start + len(candidate_set.positions)], alone)


@pytest.mark.parametrize("query", ["흥아호", "으", "흥아 KMTC 12"])
def test_key_v2_similarities_match_scalar_scorers(query: str) -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            {
                "vessel_id": ["A", "B", "C", "D"],
                "vessel_name": ["HEUNG A", "HEUNG A HO", "###", "AEIOU"],
                "gross_tonnage": [100] * 4,
            }
        )
    )
    positions = [3, 2, 0, 1]
    matrix = _build([_record(query)], [_candidate_set(positions)], index)
    variants = hangul_variants(query)
    scorers = (
        ("kor2_cons_sim", False, Levenshtein.normalized_similarity, 1.0),
        ("kor2_vow_sim", True, Levenshtein.normalized_similarity, 1.0),
        ("kor2_vow_jw", True, JaroWinkler.normalized_similarity, 1.0),
        ("kor2_vow_partial", True, fuzz.partial_ratio, 100.0),
        ("kor2_vow_indel", True, Indel.normalized_similarity, 1.0),
    )
    for name, with_vowels, scorer, scale in scorers:
        choices = index.key2_vow if with_vowels else index.key2_cons
        expected = [
            max(scorer(hangul_phonetic_key_v2(variant, with_vowels), choices[p]) for variant in variants) / scale
            for p in positions
        ]
        np.testing.assert_allclose(matrix[:, _column(name)], expected, rtol=1e-6, atol=1e-7)


def test_key_v2_context_does_not_replace_v1_context() -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            {
                "vessel_id": ["A", "B"],
                "vessel_name": ["EUNG A", "HEUNG A"],
                "gross_tonnage": [100, 100],
            }
        )
    )
    matrix = _build([_record("흥아")], [_candidate_set([0, 1])], index)
    np.testing.assert_array_equal(matrix[:, _column("primary_rank")], [1, 2])
    np.testing.assert_array_equal(matrix[:, _column("primary_gap")], [0, 0])
    np.testing.assert_array_equal(matrix[:, _column("kor2_primary_rank")], [2, 1])
    assert matrix[0, _column("kor2_primary_gap")] > 0
    assert matrix[1, _column("kor2_primary_gap")] == 0


def _order_index() -> CatalogIndex:
    return CatalogIndex.build(
        pd.DataFrame(
            [
                {
                    "vessel_id": "T",
                    "vessel_name": "GOLDEN OCEAN PIONEER",
                    "gross_tonnage": 1000,
                    "vessel_type": "Container",
                    "flag_name": "Panama",
                },
                {
                    "vessel_id": "X",
                    "vessel_name": "OTHER SHIP",
                    "gross_tonnage": 2000,
                    "vessel_type": "Container",
                    "flag_name": "Panama",
                },
            ]
        )
    )


def test_order_invariant_similarity_and_coverage() -> None:
    index = _order_index()
    matrix = _build([_record("PIONEER OCEAN GOLDEN")], [_candidate_set([0])], index)
    assert matrix[0, _column("name_sorted_sim")] == pytest.approx(1.0)
    assert matrix[0, _column("name_token_cover_query")] == pytest.approx(1.0)
    assert matrix[0, _column("token_count_diff")] == pytest.approx(0.0)
    assert matrix[0, _column("lat_dl_sim")] < 0.6


def test_dropped_token_coverage() -> None:
    index = _order_index()
    matrix = _build([_record("GOLDEN PIONEER")], [_candidate_set([0])], index)
    assert matrix[0, _column("name_token_cover_query")] == pytest.approx(1.0)
    assert matrix[0, _column("name_token_cover_cand")] < 1.0
    assert matrix[0, _column("token_count_diff")] == pytest.approx(1.0)


def test_fuzzy_cover_tolerates_typos() -> None:
    index = _order_index()
    matrix = _build([_record("PIONEERR GOLDEN")], [_candidate_set([0])], index)
    assert matrix[0, _column("name_token_fuzzy_cover")] == pytest.approx(1.0)


def test_hangul_sorted_similarity_is_one() -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            [
                {
                    "vessel_id": "G",
                    "vessel_name": "GLOBIS SUPREME",
                    "gross_tonnage": 1000,
                    "vessel_type": "Container",
                    "flag_name": "Panama",
                }
            ]
        )
    )
    matrix = _build([_record("슈프림 글로비스")], [_candidate_set([0])], index)
    assert matrix[0, _column("name_sorted_sim")] == pytest.approx(1.0)


def test_new_columns_have_no_unexpected_nan() -> None:
    index = _order_index()
    matrix = _build([_record("GOLDEN PIONEER")], [_candidate_set([0])], index)
    assert matrix.shape[1] == len(FEATURE_NAMES)
    for name in (
        "name_sorted_sim",
        "name_token_cover_query",
        "name_token_cover_cand",
        "token_count_diff",
        "src_sorted",
        "src_token_index",
    ):
        assert np.all(np.isfinite(matrix[:, _column(name)]))
    unitless = _build([_record("###")], [_candidate_set([0])], index)
    assert np.all(np.isnan(unitless[:, _column("name_token_fuzzy_cover")]))


def test_mixed_empty_variant_keys_are_skipped() -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            [{"vessel_id": "A", "vessel_name": "GLORY", "gross_tonnage": 100, "vessel_type": "T", "flag_name": "P"}]
        )
    )
    matrix = _build([_record("아아호")], [_candidate_set([0])], index)
    assert np.all(np.isfinite(matrix[:, _column("name_sorted_sim")]))


def test_latin_fuzzy_against_unitless_candidate_is_zero() -> None:
    index = CatalogIndex.build(
        pd.DataFrame(
            [{"vessel_id": "E0", "vessel_name": "###", "gross_tonnage": 100, "vessel_type": "T", "flag_name": "P"}]
        )
    )
    matrix = _build([_record("GOLDEN PIONEER")], [_candidate_set([0])], index)
    assert matrix[0, _column("name_token_fuzzy_cover")] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("query", "candidate"),
    [("BLUE BLUE STAR", "BLUE BLUE STAR"), ("슈프림 슈프림 글로비스", "SUPREME SUPREME GLOBIS")],
)
def test_repeated_tokens_preserved_in_sorted_similarity_and_counts(query: str, candidate: str) -> None:
    index = CatalogIndex.build(pd.DataFrame({
        "vessel_id": ["A", "B"], "vessel_name": [candidate, "OTHER SHIP"], "gross_tonnage": [100, 200],
    }))
    matrix = _build([_record(query)], [_candidate_set([0])], index)
    for name in ("name_sorted_sim", "name_token_cover_query", "name_token_cover_cand", "name_token_fuzzy_cover"):
        assert matrix[0, _column(name)] == pytest.approx(1.0)
    assert matrix[0, _column("token_count_diff")] == 0
    shorter = "BLUE STAR" if query.startswith("BLUE") else "슈프림 글로비스"
    dropped = _build([_record(shorter)], [_candidate_set([0])], index)
    assert dropped[0, _column("token_count_diff")] == 1
    assert dropped[0, _column("name_sorted_sim")] < 1


def test_fuzzy_cover_weights_repeated_query_occurrences() -> None:
    index = _order_index()
    matrix = _build([_record("GOLDEN GOLDEN UNKNOWN")], [_candidate_set([0])], index)
    assert matrix[0, _column("name_token_fuzzy_cover")] == pytest.approx(12 / 19)
