"""Contract tests for candidate generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rapidfuzz.distance import Levenshtein

from port_entity_matching.candidates import (
    SRC_FALLBACK,
    SRC_MEMORY,
    SRC_NAME,
    SRC_SORTED,
    SRC_TOKEN,
    SRC_TOKEN_INDEX,
    SRC_TONNAGE,
    generate_candidates,
)
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.memory import CallSignMemory
from port_entity_matching.records import QueryRecord
from port_entity_matching.normalize import hangul_phonetic_key_v2, hangul_token_keys, hangul_variants, sorted_token_key


def _catalog_frame(entries: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "vessel_id": vessel_id,
                "vessel_name": name,
                "gross_tonnage": gross_tonnage,
                "vessel_type": "Container",
                "flag_name": "Panama",
            }
            for vessel_id, name, gross_tonnage in entries
        ]
    )


def _record(
    sample_id: str,
    name: str,
    call_sign: str | None = None,
    gross_tonnage: float | None = None,
    international_gross_tonnage: float | None = None,
) -> QueryRecord:
    return QueryRecord(
        sample_id=sample_id,
        vessel_name=name,
        call_sign=call_sign,
        gross_tonnage=gross_tonnage,
        international_gross_tonnage=international_gross_tonnage,
        flag_name=None,
        vessel_type=None,
        arrival_year=None,
    )


def _sources_of(candidate_set, position: int) -> int:
    matches = candidate_set.positions.tolist()
    if position not in matches:
        return 0
    return int(candidate_set.sources[matches.index(position)])


def test_typo_retrieves_target() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Stena Promise", 15000.0),
                ("V1", "Stena Premium", 16000.0),
                ("V2", "Hanjin Seattle", 17000.0),
            ]
        )
    )
    result = generate_candidates([_record("s1", "STENA PROMIISE")], index, None, MatchConfig(name_top_k=1))[0]
    assert 0 in result.positions.tolist()
    assert _sources_of(result, 0) & SRC_NAME


def test_hangul_loanword_retrieves_target() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Reflection", 30000.0),
                ("V1", "Meridian", 31000.0),
                ("V2", "Northern Star", 32000.0),
            ]
        )
    )
    result = generate_candidates([_record("s1", "리플렉션")], index, None, MatchConfig(hangul_key_top_k=1))[0]
    assert 0 in result.positions.tolist()
    assert _sources_of(result, 0) & SRC_NAME


def test_hangul_tonnage_window() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Glovis Spirit", 64650.0),
                ("V1", "Hyundai Fortune", 90000.0),
            ]
        )
    )
    record = _record("s1", "글로비스 스피릿", international_gross_tonnage=64824.0)
    result = generate_candidates([record], index, None, MatchConfig())[0]
    assert 0 in result.positions.tolist()
    assert _sources_of(result, 0) & SRC_TONNAGE


def test_rename_via_rare_token() -> None:
    entries = [(f"F{position}", f"Hafnia {position}", 1000.0 + position) for position in range(40)]
    entries.append(("WREN", "BW Wren", 29768.0))
    index = CatalogIndex.build(_catalog_frame(entries))
    config = MatchConfig(name_top_k=1)
    record = _record("s1", "HAFNIA WREN", gross_tonnage=29765.0)
    result = generate_candidates([record], index, None, config)[0]
    assert 40 in result.positions.tolist()
    assert _sources_of(result, 40) & SRC_TOKEN
    assert not _sources_of(result, 40) & SRC_NAME


def test_exact_tonnage_source() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Glovis Spirit", 1000.0),
                ("V1", "Other Ship", 5000.0),
            ]
        )
    )
    record = _record("s1", "ZZZ UNRELATED", gross_tonnage=1000.0)
    result = generate_candidates([record], index, None, MatchConfig())[0]
    assert 0 in result.positions.tolist()
    assert _sources_of(result, 0) & SRC_TONNAGE


def test_memory_source() -> None:
    index = CatalogIndex.build(
        _catalog_frame([(f"V{position}", f"Vessel {position}", 1000.0 + position) for position in range(9)])
    )
    train = pd.DataFrame([{"call_sign": "CS_1", "target_vessel_id": "V7"}])
    memory = CallSignMemory.fit(train, index)
    result = generate_candidates([_record("s1", "ANY NAME", call_sign="CS_1")], index, memory, MatchConfig())[0]
    assert 7 in result.positions.tolist()
    assert _sources_of(result, 7) & SRC_MEMORY


def test_fallback_never_empty() -> None:
    index = CatalogIndex.build(_catalog_frame([("V0", "Alpha", 1000.0)]))
    result = generate_candidates([_record("s1", "")], index, None, MatchConfig())[0]
    assert result.positions.tolist() == [0]
    assert result.sources.tolist() == [SRC_FALLBACK]


def test_row_independence() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Stena Promise", 15000.0),
                ("V1", "Glovis Spirit", 64650.0),
                ("V2", "BW Wren", 29768.0),
                ("V3", "Reflection", 30000.0),
            ]
        )
    )
    records = [
        _record("s1", "STENA PROMIISE"),
        _record("s2", "글로비스 스피릿", international_gross_tonnage=64824.0),
        _record("s3", ""),
        _record("s4", "HAFNIA WREN", gross_tonnage=29765.0),
        _record("s5", "리플렉션"),
        _record("s6", "Other Ship", gross_tonnage=5000.0),
    ]
    config = MatchConfig()
    batch = generate_candidates(records, index, None, config)
    individual = [generate_candidates([record], index, None, config)[0] for record in records]
    reversed_batch = list(reversed(generate_candidates(list(reversed(records)), index, None, config)))
    for position in range(len(records)):
        expected_positions = batch[position].positions.tolist()
        expected_sources = batch[position].sources.tolist()
        assert individual[position].positions.tolist() == expected_positions
        assert individual[position].sources.tolist() == expected_sources
        assert reversed_batch[position].positions.tolist() == expected_positions
        assert reversed_batch[position].sources.tolist() == expected_sources


def test_hangul_name_variants_are_combined() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Glovis", 64650.0),
                ("V1", "Other Ship", 90000.0),
            ]
        )
    )
    record = _record("s1", "글로비스호")
    result = generate_candidates([record], index, None, MatchConfig())[0]
    assert 0 in result.positions.tolist()
    assert _sources_of(result, 0) & SRC_NAME


def test_hangul_batches_are_split_by_key_budget() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Glovis", 64650.0),
                ("V1", "Meridian", 31000.0),
                ("V2", "Northern Star", 32000.0),
            ]
        )
    )
    records = [_record("s1", "글로비스호"), _record("s2", "메리디안호"), _record("s3", "노스스타호")]
    batch = generate_candidates(records, index, None, MatchConfig(query_chunk=2))
    individual = [generate_candidates([record], index, None, MatchConfig(query_chunk=2))[0] for record in records]
    for position in range(len(records)):
        assert batch[position].positions.tolist() == individual[position].positions.tolist()
        assert batch[position].sources.tolist() == individual[position].sources.tolist()


def test_exact_tonnage_matches_are_capped() -> None:
    entries = [(f"S{position}", f"Ship {position}", 1000.0) for position in range(60)]
    index = CatalogIndex.build(_catalog_frame(entries))
    config = MatchConfig(latin_exact_gt_cap=50)
    record = _record("s1", "SOME NAME", gross_tonnage=1000.0)
    result = generate_candidates([record], index, None, config)[0]
    tonnage_positions = [
        position for position in result.positions.tolist() if _sources_of(result, position) & SRC_TONNAGE
    ]
    assert len(tonnage_positions) == 50


def test_short_and_unknown_tokens_are_ignored() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Ship Alpha", 1000.0),
                ("V1", "Ship Beta", 2000.0),
            ]
        )
    )
    record = _record("s1", "AB UNLISTED", gross_tonnage=1000.0)
    result = generate_candidates([record], index, None, MatchConfig())[0]
    assert not any(_sources_of(result, position) & SRC_TOKEN for position in result.positions.tolist())


def test_candidate_cap_respected() -> None:
    index = CatalogIndex.build(
        _catalog_frame([(f"V{position}", f"Vessel {position}", 1000.0 + position) for position in range(20)])
    )
    train = pd.DataFrame([{"call_sign": "CS_1", "target_vessel_id": "V7"}])
    memory = CallSignMemory.fit(train, index)
    config = MatchConfig(max_candidates=5)
    result = generate_candidates([_record("s1", "ANY NAME", call_sign="CS_1")], index, memory, config)[0]
    assert len(result.positions) <= 5
    assert 7 in result.positions.tolist()
    assert result.positions.dtype == np.int64
    assert result.sources.dtype == np.uint8


@pytest.mark.parametrize("name", ["ALPHA", "알파호"])
def test_cap_scores_candidates_outside_name_top_k(name: str) -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "ZZZZZZ", 1000.0),
                ("V1", "Alpha X", 1000.0),
                ("V2", "Alpha", 9000.0),
            ]
        )
    )
    config = MatchConfig(name_top_k=0, hangul_key_top_k=0, max_candidates=1, rare_token_max_df=0, token_top_k=0)
    result = generate_candidates([_record("s", name, gross_tonnage=1000)], index, None, config)[0]
    assert result.positions.tolist() == [1]
    assert result.sources.tolist() == [SRC_TONNAGE]


def test_cap_scores_hangul_window_hits_outside_name_top_k() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "ZZZZZZ", 1001.0),
                ("V1", "Alpha", 1002.0),
            ]
        )
    )
    config = MatchConfig(hangul_key_top_k=0, max_candidates=1)
    result = generate_candidates([_record("s", "알파", gross_tonnage=1000)], index, None, config)[0]
    assert result.positions.tolist() == [1]


def test_cap_scores_rare_token_hits_outside_name_top_k() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Alpha ZZZZZZZ", 1001.0),
                ("V1", "Alpha Wren", 1002.0),
            ]
        )
    )
    config = MatchConfig(name_top_k=0, max_candidates=1, token_top_k=0)
    result = generate_candidates([_record("s", "Alpha Wren", gross_tonnage=1000)], index, None, config)[0]
    assert result.positions.tolist() == [1]
    assert result.sources.tolist() == [SRC_TOKEN]


def test_exact_priority_applies_to_name_hits_beyond_latin_gt_source_cap() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "ZZZZZZ", 1000.0),
                ("V1", "Alpha X", 1000.0),
                ("V2", "Alpha", 2000.0),
            ]
        )
    )
    config = MatchConfig(name_top_k=2, latin_exact_gt_cap=1, max_candidates=1)
    result = generate_candidates([_record("s", "Alpha", gross_tonnage=1000)], index, None, config)[0]
    assert result.positions.tolist() == [1]
    assert not result.sources[0] & SRC_TONNAGE


def test_source_union_and_stable_ties() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Alpha", 1000.0),
                ("V1", "Alpha", 1000.0),
                ("V2", "Alpha", 1000.0),
            ]
        )
    )
    memory = CallSignMemory.fit(pd.DataFrame([{"call_sign": "CS", "target_vessel_id": "V1"}]), index)
    config = MatchConfig(name_top_k=2, max_candidates=2)
    result = generate_candidates([_record("s", "Alpha", "CS", 1000)], index, memory, config)[0]
    assert result.positions.tolist() == [0, 1]
    assert result.sources.tolist() == [
        SRC_NAME | SRC_TONNAGE | SRC_TOKEN | SRC_SORTED,
        SRC_NAME | SRC_TONNAGE | SRC_TOKEN | SRC_SORTED | SRC_MEMORY,
    ]


def test_hangul_exact_hits_survive_window_reduction() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "ZZZZZZ", 1000.0),
                ("V1", "Alpha", 1001.0),
                ("V2", "ZZZZZZ", 1000.0),
            ]
        )
    )
    result = generate_candidates(
        [_record("s", "알파", gross_tonnage=1000)],
        index,
        None,
        MatchConfig(hangul_key_top_k=0, hangul_window_keep=1, token_top_k=0),
    )[0]
    assert result.positions.tolist() == [0, 1, 2]
    assert result.sources.tolist() == [SRC_TONNAGE] * 3


def test_hangul_variants_match_full_score_reference_for_all_chunk_sizes() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Glovis", 100.0),
                ("V1", "Reflection", 200.0),
                ("V2", "Glovis Ho", 300.0),
                ("V3", "Meridian", 400.0),
                ("V4", "Meridian Ho", 500.0),
                ("V5", "Other", 600.0),
            ]
        )
    )
    records = [_record("s0", "글로비스호"), _record("s1", "리플렉션호"), _record("s2", "메리디안호")]
    for chunk_size in (1, 2, 3, 256):
        config = MatchConfig(query_chunk=chunk_size, hangul_key_top_k=2)
        batch = generate_candidates(records, index, None, config)
        for record, result in zip(records, batch, strict=True):
            keys = [hangul_phonetic_key_v2(v, False) for v in hangul_variants(record.vessel_name)]
            scores = [
                max(np.float32(Levenshtein.normalized_similarity(key, name)) for key in keys)
                for name in index.key2_cons
            ]
            expected = sorted(sorted(range(len(index)), key=lambda p: (-scores[p], p))[:2])
            assert set(expected) <= set(result.positions.tolist())
            for position in expected:
                matches = result.positions.tolist()
                assert result.sources[matches.index(position)] & SRC_NAME
        reversed_batch = generate_candidates(records[::-1], index, None, config)[::-1]
        subset = generate_candidates(records[::2], index, None, config)
        for original, reordered in zip(batch, reversed_batch, strict=True):
            np.testing.assert_array_equal(original.positions, reordered.positions)
            np.testing.assert_array_equal(original.sources, reordered.sources)
        for original, selected in zip(batch[::2], subset, strict=True):
            np.testing.assert_array_equal(original.positions, selected.positions)
            np.testing.assert_array_equal(original.sources, selected.sources)


def test_hangul_window_ranking_matches_variant_mean_reference() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Other", 1010.0),
                ("V1", "Glovis", 1020.0),
                ("V2", "Glovis Ho", 1000.0),
                ("V3", "Glovis X", 990.0),
                ("V4", "ZZZZZZ", 2000.0),
            ]
        )
    )
    record = _record("s", "글로비스호", gross_tonnage=1000)
    config = MatchConfig(hangul_key_top_k=0, hangul_window_keep=1, token_top_k=0)
    pairs = [
        (hangul_phonetic_key_v2(v, False), hangul_phonetic_key_v2(v, True)) for v in hangul_variants(record.vessel_name)
    ]
    window = index.gt_range(900.0, 1100.0).tolist()
    scores = {
        p: max(
            max(
                (
                    Levenshtein.normalized_similarity(kc, index.key2_cons[p])
                    + Levenshtein.normalized_similarity(kv, index.key2_vow[p])
                )
                / 2
                for kc, kv in pairs
            ),
            max(
                Levenshtein.normalized_similarity(sorted_key, index.sorted_key2_cons[p])
                for sorted_key in [
                    sorted_token_key(hangul_token_keys(variant)) for variant in hangul_variants(record.vessel_name)
                ]
                if sorted_key
            )
            if any(sorted_token_key(hangul_token_keys(variant)) for variant in hangul_variants(record.vessel_name))
            else 0.0,
        )
        for p in window
    }
    best = min(window, key=lambda p: (-scores[p], p))
    result = generate_candidates([record], index, None, config)[0]
    assert result.positions.tolist() == sorted({best, 2})
    assert result.sources.tolist() == [SRC_TONNAGE] * len(result.positions)


def test_empty_input_and_worker_count_independence() -> None:
    index = CatalogIndex.build(_catalog_frame([("V0", "Alpha", 1000.0), ("V1", "Other", 2000.0)]))
    assert generate_candidates([], index, None, MatchConfig()) == []
    records = [_record("s0", "Alpha"), _record("s1", "알파호")]
    serial = generate_candidates(records, index, None, MatchConfig(num_threads=1))
    parallel = generate_candidates(records, index, None, MatchConfig(num_threads=4))
    for first, second in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.sources, second.sources)


@pytest.mark.parametrize(
    "config", [MatchConfig(query_chunk=0), MatchConfig(max_candidates=0), MatchConfig(name_top_k=-1)]
)
def test_invalid_retrieval_limits_rejected(config: MatchConfig) -> None:
    index = CatalogIndex.build(_catalog_frame([("V0", "Alpha", 1000.0)]))
    with pytest.raises(ValueError, match="must be"):
        generate_candidates([_record("s", "Alpha")], index, None, config)


def test_negative_hangul_tonnage_window_rejected() -> None:
    index = CatalogIndex.build(_catalog_frame([("V0", "Alpha", 1000.0)]))
    with pytest.raises(ValueError, match="must be"):
        generate_candidates([_record("s", "알파")], index, None, MatchConfig(hangul_tonnage_window=-0.01))


def test_hangul_tonnage_window_is_ten_percent() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Alpha", 1000.0),
                ("V1", "Unrelated Ship", 1090.0),
            ]
        )
    )
    record = _record("s1", "글로비스호", gross_tonnage=1000.0)

    widened = generate_candidates([record], index, None, MatchConfig())[0]
    narrowed = generate_candidates([record], index, None, MatchConfig(hangul_tonnage_window=0.06))[0]

    assert _sources_of(widened, 1) & SRC_TONNAGE
    assert not _sources_of(narrowed, 1) & SRC_TONNAGE


def test_latin_rare_token_filter_keeps_six_percent_window() -> None:
    entries = [(f"F{position}", f"Hafnia {position}", 1000.0 + position) for position in range(40)]
    entries.append(("WREN", "BW Wren", 1090.0))
    index = CatalogIndex.build(_catalog_frame(entries))
    record = _record("s1", "HAFNIA WREN", gross_tonnage=1000.0)

    result = generate_candidates([record], index, None, MatchConfig())[0]

    assert not _sources_of(result, 40) & SRC_TOKEN


def test_hangul_retrieval_uses_key_v2() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Meker Jaya", 50000.0),
                ("V1", "MARSGRACHT", 60000.0),
            ]
        )
    )
    result = generate_candidates([_record("s1", "마스그라치")], index, None, MatchConfig(hangul_key_top_k=1))[0]

    assert _sources_of(result, 1) & SRC_NAME
    assert result.positions.tolist() == [1]


def test_latin_candidates_are_unchanged_by_hangul_key_switch() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Stena Promise", 15000.0),
                ("V1", "Glovis Spirit", 64650.0),
                ("V2", "BW Wren", 29768.0),
                ("V3", "Other Ship", 5000.0),
            ]
        )
    )
    records = [
        _record("s1", "STENA PROMIISE", gross_tonnage=15000.0),
        _record("s2", "HAFNIA WREN", gross_tonnage=29765.0),
        _record("s3", "ZZZ UNRELATED", gross_tonnage=5000.0),
    ]

    narrow = generate_candidates(records, index, None, MatchConfig(name_top_k=1))
    default = generate_candidates(records, index, None, MatchConfig())

    assert [candidate.positions.tolist() for candidate in narrow] == [[0], [2], [0, 2, 3]]
    assert [candidate.positions.tolist() for candidate in default] == [[0, 1, 2, 3]] * 3
    latin_only = generate_candidates(records, index, None, MatchConfig(hangul_key_top_k=1))
    latin_switched = generate_candidates(records, index, None, MatchConfig(hangul_key_top_k=5))
    assert [c.positions.tolist() for c in latin_only] == [c.positions.tolist() for c in latin_switched]
    assert [c.sources.tolist() for c in latin_only] == [c.sources.tolist() for c in latin_switched]


def test_candidate_generation_is_row_independent_with_key_v2() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Stena Promise", 15000.0),
                ("V1", "Glovis Spirit", 64650.0),
                ("V2", "MARSGRACHT", 60000.0),
            ]
        )
    )
    records = [
        _record("s1", "STENA PROMIISE", gross_tonnage=15000.0),
        _record("s2", "글로비스 스피릿", international_gross_tonnage=64824.0),
        _record("s3", "마스그라치"),
    ]
    config = MatchConfig()

    batch = generate_candidates(records, index, None, config)
    individual = [generate_candidates([record], index, None, config)[0] for record in records]

    for position in range(len(records)):
        assert batch[position].positions.tolist() == individual[position].positions.tolist()
        assert batch[position].sources.tolist() == individual[position].sources.tolist()


def test_reversed_latin_name_retrieved_via_sorted() -> None:
    entries = [("T", "GOLDEN OCEAN PIONEER", 1000.0)] + [
        (f"D{i}", f"DISTRACTOR VESSEL {i}", 2000.0 + i) for i in range(40)
    ]
    index = CatalogIndex.build(_catalog_frame(entries))
    result = generate_candidates([_record("s", "PIONEER OCEAN GOLDEN")], index, None, MatchConfig())[0]
    assert 0 in result.positions.tolist()
    assert _sources_of(result, 0) & SRC_SORTED


def test_dropped_token_retrieved_via_token_index_without_tonnage() -> None:
    entries = [("T", "GOLDEN OCEAN PIONEER", 1000.0)] + [
        (f"D{i}", f"DISTRACTOR VESSEL {i}", 2000.0 + i) for i in range(40)
    ]
    index = CatalogIndex.build(_catalog_frame(entries))
    result = generate_candidates([_record("s", "GOLDEN PIONEER")], index, None, MatchConfig())[0]
    assert 0 in result.positions.tolist()
    assert _sources_of(result, 0) & SRC_TOKEN_INDEX


def test_reversed_hangul_name_retrieved_via_sorted() -> None:
    index = CatalogIndex.build(_catalog_frame([("G", "GLOBIS SUPREME", 5000.0), ("X", "OTHER SHIP", 6000.0)]))
    result = generate_candidates([_record("s", "슈프림 글로비스")], index, None, MatchConfig())[0]
    assert 0 in result.positions.tolist()
    assert _sources_of(result, 0) & SRC_SORTED


def test_frequent_tokens_do_not_flood_token_index() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("A", "ALPHA ONE", 100.0),
                ("B", "ALPHA TWO", 200.0),
                ("C", "BETA THREE", 300.0),
            ]
        )
    )
    result = generate_candidates([_record("s", "ALPHA")], index, None, MatchConfig(token_df_cap=1))[0]
    assert not any(_sources_of(result, p) & SRC_TOKEN_INDEX for p in result.positions.tolist())


def test_token_index_candidates_are_row_independent() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("T", "GOLDEN OCEAN PIONEER", 1000.0),
                ("X", "OTHER SHIP", 2000.0),
                ("Y", "GOLDEN PIONEER", 3000.0),
            ]
            + [(f"D{i}", f"DISTRACTOR {i}", 4000.0 + i) for i in range(27)]
        )
    )
    records = [
        _record(f"s{i}", name)
        for i, name in enumerate(["PIONEER OCEAN GOLDEN", "GOLDEN PIONEER", "슈프림 글로비스"] * 10)
    ]
    config = MatchConfig(query_chunk=7, max_candidates=3)
    batch = generate_candidates(records, index, None, config)
    order = np.random.default_rng(42).permutation(len(records)).tolist()
    shuffled = generate_candidates([records[p] for p in order], index, None, config)
    for position, candidate_set in zip(order, shuffled, strict=True):
        np.testing.assert_array_equal(candidate_set.positions, batch[position].positions)
        np.testing.assert_array_equal(candidate_set.sources, batch[position].sources)
    for position, record in enumerate(records):
        alone = generate_candidates([record], index, None, config)[0]
        assert alone.positions.tolist() == batch[position].positions.tolist()
        assert alone.sources.tolist() == batch[position].sources.tolist()


def test_invalid_token_retrieval_limits_rejected() -> None:
    index = CatalogIndex.build(_catalog_frame([("V0", "Alpha", 1000.0)]))
    with pytest.raises(ValueError, match=r"token_top_k|token_df_cap|token_match_threshold"):
        generate_candidates([_record("s", "Alpha")], index, None, MatchConfig(token_top_k=-1))
    with pytest.raises(ValueError, match=r"token_top_k|token_df_cap|token_match_threshold"):
        generate_candidates([_record("s", "Alpha")], index, None, MatchConfig(token_df_cap=0))
    with pytest.raises(ValueError, match=r"token_top_k|token_df_cap|token_match_threshold"):
        generate_candidates([_record("s", "Alpha")], index, None, MatchConfig(token_match_threshold=0.0))
    with pytest.raises(ValueError, match=r"token_top_k|token_df_cap|token_match_threshold"):
        generate_candidates([_record("s", "Alpha")], index, None, MatchConfig(token_match_threshold=1.5))


def test_mixed_empty_sorted_variant_is_skipped() -> None:
    index = CatalogIndex.build(_catalog_frame([("A", "GLORY", 100.0), ("B", "OTHER", 200.0)]))
    result = generate_candidates([_record("s", "아아호")], index, None, MatchConfig())[0]
    assert len(result.positions) > 0


def test_hangul_without_token_units_skips_token_index() -> None:
    index = CatalogIndex.build(_catalog_frame([("A", "Alpha", 100.0), ("B", "Beta", 200.0)]))
    result = generate_candidates([_record("s", "아")], index, None, MatchConfig())[0]
    assert not any(_sources_of(result, p) & SRC_TOKEN_INDEX for p in result.positions.tolist())


def test_ubiquitous_hangul_units_skip_token_index() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("G", "GLOBIS SUPREME", 100.0),
                ("H", "GLOBIS SUPREME", 200.0),
            ]
        )
    )
    result = generate_candidates([_record("s", "슈프림 글로비스")], index, None, MatchConfig())[0]
    assert 0 in result.positions.tolist() or 1 in result.positions.tolist()


def test_zero_weight_token_index_yields_no_candidates() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("A", "ALPHA ONE", 100.0),
                ("B", "ALPHA TWO", 200.0),
            ]
        )
    )
    result = generate_candidates([_record("s", "ALPHA UNKNOWN")], index, None, MatchConfig(token_df_cap=2))[0]
    assert not any(_sources_of(result, p) & SRC_TOKEN_INDEX for p in result.positions.tolist())


def test_empty_key_similarities_are_zero() -> None:
    from port_entity_matching.candidates import _name_similarity, _prepare, _sorted_similarity

    index = CatalogIndex.build(_catalog_frame([("V0", "Alpha", 1000.0)]))
    empty_latin = _prepare(_record("s", ""))
    assert _name_similarity(empty_latin, index, 0) == 0.0
    assert _sorted_similarity(empty_latin, index, 0) == 0.0
    empty_hangul = _prepare(_record("s", "아"))
    assert _sorted_similarity(empty_hangul, index, 0) == 0.0


def test_hangul_tonnage_with_empty_sorted_key() -> None:
    index = CatalogIndex.build(_catalog_frame([("A", "Alpha", 100.0), ("B", "Beta", 200.0)]))
    result = generate_candidates([_record("s", "아", gross_tonnage=100)], index, None, MatchConfig())[0]
    assert len(result.positions) > 0


def test_empty_query_units_similarities_are_zero() -> None:
    from port_entity_matching.candidates import _Prepared, _name_similarity, _token_index_similarity

    index = CatalogIndex.build(_catalog_frame([("V0", "Alpha", 1000.0)]))
    record = _record("s", "Alpha")
    hangul_empty = _Prepared(
        record=_record("s", "아"),
        is_hangul=True,
        latin_key="",
        query_tokens=(),
        cons_keys=(),
        vowel_keys=(),
        tonnages=(),
        sorted_latin="",
        sorted_cons=(),
    )
    assert _name_similarity(hangul_empty, index, 0) == 0.0
    assert _token_index_similarity(hangul_empty, index, MatchConfig(), 0) == 0.0
    latin_empty = _Prepared(
        record=record,
        is_hangul=False,
        latin_key="",
        query_tokens=(),
        cons_keys=(),
        vowel_keys=(),
        tonnages=(),
        sorted_latin="",
        sorted_cons=(),
    )
    assert _token_index_similarity(latin_empty, index, MatchConfig(), 0) == 0.0


def test_zero_denom_token_index_similarity_is_zero() -> None:
    from port_entity_matching.candidates import _token_index_similarity

    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("G", "GLOBIS SUPREME", 100.0),
                ("H", "GLOBIS SUPREME", 200.0),
            ]
        )
    )
    record = _record("s", "슈프림 글로비스")
    from port_entity_matching.candidates import _prepare

    item = _prepare(record)
    assert _token_index_similarity(item, index, MatchConfig(), 0) == 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_top_k_desc_matches_stable_argsort(dtype) -> None:
    from port_entity_matching.candidates import _top_k_desc

    rng = np.random.default_rng(7)
    for _trial in range(200):
        length = int(rng.integers(1, 501))
        vector = rng.random(length).round(2).astype(dtype)
        for k in (0, 1, 5, 30, length, length + 3):
            expected = np.argsort(-vector, kind="stable")[:k]
            np.testing.assert_array_equal(_top_k_desc(vector, k), expected.astype(np.int64))


def test_top_k_desc_tie_at_boundary() -> None:
    from port_entity_matching.candidates import _top_k_desc

    vector = np.asarray([0.5, 0.9, 0.5, 0.5], dtype=np.float32)
    assert _top_k_desc(vector, 2).tolist() == [1, 0]


def test_hangul_window_selection_matches_scalar_reference() -> None:
    from rapidfuzz.distance import Levenshtein as _Lev

    from port_entity_matching.candidates import _hangul_tonnage, _prepare

    names = ["GLOVIS SPIRIT", "SPIRIT GLOVIS", "OTHER SHIP", "GLOVIS HO", "SHINING ROAD"]
    entries = [(f"V{i:04d}", names[i % len(names)], 1001.0 + i % 3) for i in range(300)]
    index = CatalogIndex.build(_catalog_frame(entries))
    record = _record("s", "글로비스 스피릿", gross_tonnage=1000.0)
    config = MatchConfig(hangul_window_keep=50)
    item = _prepare(record)
    window, _exact = _hangul_tonnage(item, index, config)
    assert len(window) == 50

    def _scalar(position: int) -> float:
        base = max(
            0.5
            * (
                _Lev.normalized_similarity(c, index.key2_cons[position])
                + _Lev.normalized_similarity(v, index.key2_vow[position])
            )
            for c, v in zip(item.cons_keys, item.vowel_keys, strict=True)
        )
        keys = [key for key in item.sorted_cons if key]
        sorted_sim = (
            max(_Lev.normalized_similarity(key, index.sorted_key2_cons[position]) for key in keys) if keys else 0.0
        )
        return max(base, sorted_sim)

    full_window = index.gt_range(900.0, 1100.0).tolist()
    expected = sorted(full_window, key=lambda p: (-_scalar(p), p))[:50]
    expected_set = set(expected) | set(index.gt_exact(1000.0).tolist())
    assert set(window.tolist()) == expected_set
    assert window.tolist() == sorted(expected_set)


def test_candidate_sets_match_scalar_reference_path() -> None:
    index = CatalogIndex.build(
        _catalog_frame(
            [
                ("V0", "Stena Promise", 15000.0),
                ("V1", "Stena Premium", 16000.0),
                ("V2", "Hanjin Seattle", 17000.0),
                ("V3", "Glovis Spirit", 64650.0),
            ]
        )
    )
    records = [_record("s1", "STENA PROMIISE"), _record("s2", "글로비스 스피릿", international_gross_tonnage=64824.0)]
    config = MatchConfig()
    actual = generate_candidates(records, index, None, config)
    expected = [([0, 1, 2, 3], [97, 97, 33, 33]), ([0, 1, 2, 3], [33, 33, 33, 99])]
    for result, (positions, sources) in zip(actual, expected, strict=True):
        assert result.positions.tolist() == positions
        assert result.sources.tolist() == sources
