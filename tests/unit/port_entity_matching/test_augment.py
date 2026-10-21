"""Contract tests for the training-only degraded copies of Hangul records."""

from __future__ import annotations

import pytest

from port_entity_matching.augment import degraded_copies
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
