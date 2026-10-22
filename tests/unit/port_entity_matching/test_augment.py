"""Contract tests for the training-only degraded copies of Hangul records."""

from __future__ import annotations

import numpy as np
import pytest

from port_entity_matching.augment import (
    char_edits,
    degraded_copies,
    drop_token,
    reverse_tokens,
)
from port_entity_matching.config import MatchConfig
from port_entity_matching.records import QueryRecord


def _record(sample_id: str, name: str, gross: float | None = 1000.0, international: float | None = None) -> QueryRecord:
    return QueryRecord(
        sample_id=sample_id,
        vessel_name=name,
        call_sign="CS_A",
        gross_tonnage=gross,
        international_gross_tonnage=international,
        flag_name="파나마",
        vessel_type="Tanker",
        arrival_year=2020,
    )


def _config(**overrides: float | int | bool) -> MatchConfig:
    defaults: dict[str, float | int | bool] = {"augment_tonnage_factor": 1.08, "augment_min_syllables": 3}
    defaults.update(overrides)
    return MatchConfig(**defaults)  # type: ignore[arg-type]


def test_three_copies_per_hangul_record_and_none_for_latin() -> None:
    records = [_record("S0", "한진부산"), _record("S1", "글로비스 스피릿"), _record("S2", "CALIDA")]
    copies, sources = degraded_copies(records, _config())
    assert len(copies) == 6
    assert sources == [0, 0, 0, 1, 1, 1]


def test_tonnage_copies() -> None:
    copies, _ = degraded_copies([_record("S0", "한진부산")], _config())
    assert copies[0].gross_tonnage == pytest.approx(1080.0, rel=1e-9)
    assert copies[0].international_gross_tonnage is None
    assert copies[1].gross_tonnage is None
    assert copies[1].international_gross_tonnage is None


def test_inflated_copy_keeps_missing_tonnage_missing() -> None:
    copies, _ = degraded_copies([_record("S0", "한진부산", gross=None, international=500.0)], _config())
    assert copies[0].gross_tonnage is None
    assert copies[0].international_gross_tonnage == pytest.approx(540.0, rel=1e-9)


def test_short_names_survive_syllable_deletion_unchanged() -> None:
    copies, _ = degraded_copies([_record("S0", "대한")], _config())
    assert copies[2].vessel_name == "대한"


def test_one_syllable_removed_from_long_names() -> None:
    copies, _ = degraded_copies([_record("S0", "한진부산")], _config())
    assert len(copies[2].vessel_name) == 3
    assert copies[2].vessel_name != "한진부산"


def test_deterministic_and_pure() -> None:
    records = [_record("S0", "한진부산"), _record("S1", "CALIDA")]
    first, first_sources = degraded_copies(records, _config())
    second, second_sources = degraded_copies(records, _config())
    assert [copy.vessel_name for copy in first] == [copy.vessel_name for copy in second]
    assert first_sources == second_sources
    assert records[0].vessel_name == "한진부산"
    assert records[0].gross_tonnage == 1000.0


def test_copy_ids_are_unique_and_suffixed() -> None:
    copies, _ = degraded_copies([_record("S0", "한진부산")], _config())
    assert [copy.sample_id for copy in copies] == ["S0#aug0", "S0#aug1", "S0#aug2"]


def test_other_fields_are_preserved() -> None:
    copies, _ = degraded_copies([_record("S0", "한진부산")], _config())
    for copy in copies:
        assert copy.call_sign == "CS_A"
        assert copy.flag_name == "파나마"
        assert copy.vessel_type == "Tanker"
        assert copy.arrival_year == 2020


@pytest.mark.parametrize("factor", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_factor_rejected(factor: float) -> None:
    with pytest.raises(ValueError, match="augment_tonnage_factor"):
        degraded_copies([_record("S0", "한진부산")], _config(augment_tonnage_factor=factor))


def test_short_names_do_not_consume_rng_and_only_raw_syllables_are_deleted() -> None:
    import numpy as np

    name = "A한-진 부산9"
    records = [_record("short", "대한"), _record("latin", "ALPHA"), _record("long", name)]
    copies, sources = degraded_copies(records, _config())
    positions = [i for i, char in enumerate(name) if "가" <= char <= "힣"]
    deleted = positions[int(np.random.default_rng(0).integers(len(positions)))]
    assert copies[-1].vessel_name == name[:deleted] + name[deleted + 1:]
    assert sources == [0, 0, 0, 2, 2, 2]
    assert copies == degraded_copies(records, _config())[0]


def test_invalid_min_syllables_rejected() -> None:
    with pytest.raises(ValueError, match="augment_min_syllables"):
        degraded_copies([_record("S0", "한진부산")], _config(augment_min_syllables=0))


def test_short_names_leave_the_generator_untouched() -> None:
    import numpy as np

    from port_entity_matching.augment import drop_hangul_syllable

    generator = np.random.default_rng(0)
    assert drop_hangul_syllable("대한", generator, 3) == "대한"
    fresh = np.random.default_rng(0)
    assert int(generator.integers(1000)) == int(fresh.integers(1000))


def test_dropped_syllable_is_a_single_hangul_character() -> None:
    import numpy as np

    from port_entity_matching.augment import drop_hangul_syllable

    name = "글로비스 스피릿"
    result = drop_hangul_syllable(name, np.random.default_rng(0), 3)
    assert len(result) == len(name) - 1
    assert result != name


def test_reverse_tokens_reverses_the_token_order() -> None:
    assert reverse_tokens("GLOBAL STAR PIONEER") == "PIONEER STAR GLOBAL"


def test_reverse_tokens_keeps_single_token_names_unchanged() -> None:
    assert reverse_tokens("PIONEER") == "PIONEER"


def test_reverse_tokens_keeps_the_characters_inside_a_token() -> None:
    assert reverse_tokens("OCEAN  PIONEER") == "PIONEER OCEAN"


def test_drop_token_keeps_the_remaining_tokens() -> None:
    original = ["A", "B", "C"]
    result = drop_token("A B C", np.random.default_rng(0)).split()
    assert len(result) == 2
    assert result == [token for token in original if token in result]
    assert result in (["A", "B"], ["A", "C"], ["B", "C"])


def test_drop_token_is_deterministic_for_the_same_seed() -> None:
    first = drop_token("A B C", np.random.default_rng(11))
    second = drop_token("A B C", np.random.default_rng(11))
    assert first == second


def test_drop_token_keeps_single_token_names_unchanged() -> None:
    generator = np.random.default_rng(0)
    assert drop_token("SOLO", generator) == "SOLO"
    fresh = np.random.default_rng(0)
    assert int(generator.integers(1000)) == int(fresh.integers(1000))


def _damerau_distance(left: str, right: str) -> int:
    """Return the optimal string alignment distance between two strings."""
    rows, columns = len(left), len(right)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        table[row][0] = row
    for column in range(columns + 1):
        table[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            cost = 0 if left[row - 1] == right[column - 1] else 1
            table[row][column] = min(
                table[row - 1][column] + 1,
                table[row][column - 1] + 1,
                table[row - 1][column - 1] + cost,
            )
            if row > 1 and column > 1 and left[row - 1] == right[column - 2] and left[row - 2] == right[column - 1]:
                table[row][column] = min(table[row][column], table[row - 2][column - 2] + 1)
    return table[rows][columns]


def test_char_edits_is_deterministic_and_bounded() -> None:
    name = "OCEAN PIONEER"
    first = char_edits(name, np.random.default_rng(7), 3)
    second = char_edits(name, np.random.default_rng(7), 3)
    assert first == second
    assert 1 <= _damerau_distance(name, first) <= 3
    assert first.count(" ") == name.count(" ")
    assert len(first.split()) == len(name.split())


def test_char_edits_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError, match="count"):
        char_edits("OCEAN PIONEER", np.random.default_rng(0), -1)


def test_char_edits_leaves_short_names_unchanged() -> None:
    assert char_edits("AB 1", np.random.default_rng(0), 2) == "AB 1"


@pytest.mark.parametrize(
    ("seed", "expected"),
    [(0, "ABdC"), (2, "ABCdd"), (3, "ABC"), (10, "ABCH")],
)
def test_char_edits_operation_contract(seed: int, expected: str) -> None:
    assert char_edits("ABCd", np.random.default_rng(seed), 1) == expected


def test_char_edits_stops_after_ascii_letters_drop_below_four() -> None:
    rng = np.random.default_rng(3)
    reference = np.random.default_rng(3)
    assert char_edits("ABCd", rng, 10) == char_edits("ABCd", reference, 1) == "ABC"
    assert rng.integers(1000) == reference.integers(1000)


def test_char_edits_short_ascii_name_does_not_draw() -> None:
    rng = np.random.default_rng(7)
    assert char_edits("ABé한 1", rng, 3) == "ABé한 1"
    assert rng.integers(1000) == np.random.default_rng(7).integers(1000)


def test_drop_token_draws_exactly_one_index() -> None:
    rng = np.random.default_rng(7)
    reference = np.random.default_rng(7)
    tokens = ["A-1", "B.2", "C_3"]
    dropped = int(reference.integers(len(tokens)))
    assert drop_token("  A-1\tB.2  C_3 ", rng) == " ".join(tokens[:dropped] + tokens[dropped + 1:])
    assert rng.integers(1000) == reference.integers(1000)


def test_char_edits_with_zero_count_draws_nothing() -> None:
    generator = np.random.default_rng(0)
    assert char_edits("OCEAN PIONEER", generator, 0) == "OCEAN PIONEER"
    fresh = np.random.default_rng(0)
    assert int(generator.integers(1000)) == int(fresh.integers(1000))


def test_char_edits_keeps_the_token_structure_over_many_seeds() -> None:
    name = "OCEAN PIONEER"
    for seed in range(30):
        result = char_edits(name, np.random.default_rng(seed), 3)
        assert result.count(" ") == name.count(" ")
        assert len(result.split()) == len(name.split())
        assert 1 <= _damerau_distance(name, result) <= 3


def test_degraded_copies_output_is_unchanged() -> None:
    records = [
        _record("S0", "한진부산", gross=1000.0, international=900.0),
        _record("S1", "글로비스 스피릿"),
        _record("S2", "CALIDA"),
    ]
    copies, sources = degraded_copies(records, _config())
    assert sources == [0, 0, 0, 1, 1, 1]
    assert [
        (copy.sample_id, copy.vessel_name, copy.gross_tonnage, copy.international_gross_tonnage)
        for copy in copies
    ] == [
        ("S0#aug0", "한진부산", 1080.0, 972.0000000000001),
        ("S0#aug1", "한진부산", None, None),
        ("S0#aug2", "한진부", 1000.0, 900.0),
        ("S1#aug0", "글로비스 스피릿", 1080.0, None),
        ("S1#aug1", "글로비스 스피릿", None, None),
        ("S1#aug2", "글로비스 피릿", 1000.0, None),
    ]
