"""Contract tests for the deterministic record perturbations."""

from __future__ import annotations

import dataclasses

import pytest

from port_entity_matching.evaluation.perturb import PERTURBATIONS, UNSEEN_PERTURBATIONS, perturb_record
from port_entity_matching.records import QueryRecord

_MEDIALS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_INITIALS = tuple("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_FINALS = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")
_SWAP = {"ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅓ": "ㅗ", "ㅗ": "ㅓ", "ㅜ": "ㅡ", "ㅡ": "ㅜ", "ㅣ": "ㅟ", "ㅟ": "ㅣ"}


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


def _jamo(char: str) -> tuple[str, str, str]:
    code = ord(char) - 0xAC00
    return _INITIALS[code // 588], _MEDIALS[(code % 588) // 28], _FINALS[code % 28]


def test_scenario_list_is_fixed() -> None:
    assert PERTURBATIONS == (
        "kor_tonnage_x1.08",
        "kor_tonnage_x1.12",
        "kor_tonnage_x0.90",
        "kor_tonnage_missing",
        "kor_syllable_deleted",
        "kor_syllable_inserted",
        "kor_vowel_swapped",
        "kor_spaces_removed",
        "kor_flag_type_missing",
        "lat_extra_typo",
        "lat_tonnage_missing",
        "kor_tokens_reversed",
        "kor_token_dropped",
        "kor_tokens_rotated",
        "kor_tonnage_noise_10_20",
        "lat_tokens_reversed",
        "lat_token_dropped",
        "lat_tokens_rotated",
        "lat_heavy_typo",
        "lat_tonnage_noise_10_20",
    )


@pytest.mark.parametrize(
    "scenario",
    [name for name in PERTURBATIONS if name.startswith("kor_")],
)
def test_korean_scenarios_ignore_latin_records(scenario: str) -> None:
    assert perturb_record(_record("S0", "CALIDA"), scenario) is None


@pytest.mark.parametrize(
    "scenario",
    [name for name in PERTURBATIONS if name.startswith("lat_")],
)
def test_latin_scenarios_ignore_hangul_records(scenario: str) -> None:
    assert perturb_record(_record("S0", "한진부산"), scenario) is None


def test_tonnage_scaling() -> None:
    perturbed = perturb_record(_record("S0", "한진부산", gross=1000.0), "kor_tonnage_x0.90")
    assert perturbed is not None
    assert perturbed.gross_tonnage == pytest.approx(900.0, rel=1e-9)
    assert perturbed.international_gross_tonnage is None


def test_tonnage_missing_clears_both_values() -> None:
    perturbed = perturb_record(_record("S0", "한진부산", gross=1000.0, international=900.0), "kor_tonnage_missing")
    assert perturbed is not None
    assert perturbed.gross_tonnage is None
    assert perturbed.international_gross_tonnage is None


def test_order_independence() -> None:
    first = _record("S0", "한진부산")
    second = _record("S1", "글로비스 스피릿")
    forward = {record.sample_id: perturb_record(record, "kor_syllable_deleted") for record in (first, second)}
    backward = {record.sample_id: perturb_record(record, "kor_syllable_deleted") for record in (second, first)}
    assert forward == backward


def test_seed_changes_the_draw() -> None:
    records = [_record(f"S{index}", "한진부산") for index in range(20)]
    outputs = [
        [perturb_record(record, "kor_syllable_deleted", seed) for record in records] for seed in (0, 1)
    ]
    assert outputs[0] != outputs[1]
    for record, first, second in zip(records, outputs[0], outputs[1], strict=True):
        for perturbed in (first, second):
            assert perturbed is not None
            assert len(perturbed.vessel_name) == len(record.vessel_name) - 1
            assert perturbed.vessel_name in {
                record.vessel_name[:position] + record.vessel_name[position + 1 :]
                for position in range(len(record.vessel_name))
            }


def test_syllable_insertion_duplicates_in_place() -> None:
    perturbed = perturb_record(_record("S0", "한진부산"), "kor_syllable_inserted")
    assert perturbed is not None
    assert len(perturbed.vessel_name) == 5
    assert any(
        perturbed.vessel_name[:position] + perturbed.vessel_name[position + 1 :] == "한진부산"
        for position in range(len(perturbed.vessel_name))
    )


def test_vowel_swap_keeps_consonants() -> None:
    perturbed = perturb_record(_record("S0", "대한"), "kor_vowel_swapped")
    assert perturbed is not None
    changed = [position for position, (a, b) in enumerate(zip("대한", perturbed.vessel_name, strict=True)) if a != b]
    assert len(changed) == 1
    position = changed[0]
    initial, medial, final = _jamo("대한"[position])
    new_initial, new_medial, new_final = _jamo(perturbed.vessel_name[position])
    assert (new_initial, new_final) == (initial, final)
    assert new_medial == _SWAP[medial]


def test_not_applicable_returns_none() -> None:
    assert perturb_record(_record("S0", "대한"), "kor_syllable_deleted") is None
    assert perturb_record(_record("S0", "한진부산"), "kor_spaces_removed") is None


def test_spaces_removed() -> None:
    perturbed = perturb_record(_record("S0", "글로비스 스피릿"), "kor_spaces_removed")
    assert perturbed is not None
    assert perturbed.vessel_name == "글로비스스피릿"


def test_flag_and_type_replaced() -> None:
    perturbed = perturb_record(_record("S0", "한진부산"), "kor_flag_type_missing")
    assert perturbed is not None
    assert perturbed.flag_name is None
    assert perturbed.vessel_type == "기타선"


def test_latin_typo_changes_one_letter() -> None:
    perturbed = perturb_record(_record("S0", "CALIDA"), "lat_extra_typo")
    assert perturbed is not None
    assert len(perturbed.vessel_name) in {len("CALIDA") - 1, len("CALIDA") + 1}


def test_latin_typo_needs_four_letters() -> None:
    assert perturb_record(_record("S0", "ABC"), "lat_extra_typo") is None


def test_latin_typo_deletes_and_duplicates() -> None:
    lengths = set()
    for index in range(20):
        perturbed = perturb_record(_record(f"S{index}", "CALIDA"), "lat_extra_typo")
        assert perturbed is not None
        lengths.add(len(perturbed.vessel_name))
    assert lengths == {len("CALIDA") - 1, len("CALIDA") + 1}


def test_vowel_swap_without_swappable_medial_returns_none() -> None:
    assert perturb_record(_record("S0", "가나"), "kor_vowel_swapped") is None


def test_latin_tonnage_missing() -> None:
    perturbed = perturb_record(_record("S0", "CALIDA", gross=1000.0, international=900.0), "lat_tonnage_missing")
    assert perturbed is not None
    assert perturbed.gross_tonnage is None
    assert perturbed.international_gross_tonnage is None


def test_unknown_scenario_rejected() -> None:
    with pytest.raises(ValueError, match=r"kor_tonnage_x1\.08"):
        perturb_record(_record("S0", "한진부산"), "bogus")


def test_sample_id_and_source_preserved() -> None:
    record = _record("S0", "한진부산")
    perturbed = perturb_record(record, "kor_tonnage_x1.08")
    assert perturbed is not None
    assert perturbed.sample_id == record.sample_id
    assert record.gross_tonnage == 1000.0
    assert record.vessel_name == "한진부산"


def test_result_is_a_copy() -> None:
    record = _record("S0", "한진부산")
    perturbed = perturb_record(record, "kor_tonnage_x1.08")
    assert perturbed is not None
    assert perturbed is not record
    assert dataclasses.replace(perturbed, gross_tonnage=record.gross_tonnage) == record


def test_rotation_needs_three_tokens() -> None:
    perturbed = perturb_record(_record("S0", "BLUE OCEAN STAR"), "lat_tokens_rotated")
    assert perturbed is not None
    assert perturbed.vessel_name == "OCEAN STAR BLUE"
    assert perturb_record(_record("S0", "BLUE STAR"), "lat_tokens_rotated") is None


def test_token_scenarios_respect_the_script() -> None:
    assert perturb_record(_record("S0", "상하이 보이저"), "lat_tokens_reversed") is None
    perturbed = perturb_record(_record("S0", "상하이 보이저"), "kor_tokens_reversed")
    assert perturbed is not None
    assert perturbed.vessel_name == "보이저 상하이"


def test_single_token_names_are_skipped() -> None:
    assert perturb_record(_record("S0", "신간"), "kor_token_dropped") is None
    assert perturb_record(_record("S0", "신간"), "kor_tokens_reversed") is None


def test_token_dropped_keeps_the_remaining_tokens() -> None:
    perturbed = perturb_record(_record("S0", "상하이 보이저"), "kor_token_dropped")
    assert perturbed is not None
    assert perturbed.vessel_name.split() in (["상하이"], ["보이저"])


def test_tonnage_noise_stays_inside_the_band() -> None:
    ratios: list[float] = []
    for index in range(50):
        perturbed = perturb_record(_record(f"S{index}", "한진 부산", gross=1000.0), "kor_tonnage_noise_10_20")
        assert perturbed is not None
        assert perturbed.gross_tonnage is not None
        ratios.append(perturbed.gross_tonnage / 1000.0)
    assert all(0.80 <= ratio <= 0.90 or 1.10 <= ratio <= 1.20 for ratio in ratios)
    assert any(ratio < 1.0 for ratio in ratios)
    assert any(ratio > 1.0 for ratio in ratios)


def test_tonnage_noise_needs_a_tonnage() -> None:
    assert perturb_record(_record("S0", "한진부산", gross=None), "kor_tonnage_noise_10_20") is None


def test_heavy_typo_is_deterministic_and_seed_dependent() -> None:
    records = [_record(f"S{index}", "BLUE OCEAN STAR") for index in range(20)]
    first = [perturb_record(record, "lat_heavy_typo", 0) for record in records]
    second = [perturb_record(record, "lat_heavy_typo", 0) for record in records]
    other = [perturb_record(record, "lat_heavy_typo", 1) for record in records]
    assert first == second
    assert first != other


def test_existing_scenarios_are_unchanged() -> None:
    golden = {
        ("kor_tonnage_x1.08", "S0"): ("한진부산", 1080.0, 972.0000000000001),
        ("kor_tonnage_x1.12", "S0"): ("한진부산", 1120.0, 1008.0000000000001),
        ("kor_tonnage_x0.90", "S1"): ("글로비스 스피릿", 900.0, None),
        ("kor_tonnage_missing", "S0"): ("한진부산", None, None),
        ("kor_syllable_deleted", "S0"): ("한진부", 1000.0, 900.0),
        ("kor_syllable_inserted", "S0"): ("한진진부산", 1000.0, 900.0),
        ("kor_vowel_swapped", "S0"): ("한쥔부산", 1000.0, 900.0),
        ("kor_spaces_removed", "S1"): ("글로비스스피릿", 1000.0, None),
        ("kor_flag_type_missing", "S0"): ("한진부산", 1000.0, 900.0),
        ("lat_extra_typo", "S2"): ("CALID", 1000.0, None),
        ("lat_tonnage_missing", "S2"): ("CALIDA", None, None),
    }
    records = {
        "S0": _record("S0", "한진부산", gross=1000.0, international=900.0),
        "S1": _record("S1", "글로비스 스피릿"),
        "S2": _record("S2", "CALIDA"),
    }
    for (scenario, sample_id), expected in golden.items():
        perturbed = perturb_record(records[sample_id], scenario)
        assert perturbed is not None, scenario
        assert (
            perturbed.vessel_name,
            perturbed.gross_tonnage,
            perturbed.international_gross_tonnage,
        ) == expected, scenario


def test_unseen_names_are_known_scenarios() -> None:
    assert set(UNSEEN_PERTURBATIONS) <= set(PERTURBATIONS)
    assert UNSEEN_PERTURBATIONS == (
        "kor_tokens_rotated",
        "lat_tokens_rotated",
        "lat_heavy_typo",
        "kor_vowel_swapped",
        "kor_syllable_inserted",
        "kor_spaces_removed",
        "kor_flag_type_missing",
    )
