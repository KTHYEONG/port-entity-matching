"""Contract tests for the training-only damage families and catalog-synthetic Latin rows."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig, recipe_config
from port_entity_matching.families import FamilyContext, family_copies
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.records import records_from_frame
from port_entity_matching.synthetic import SYNTHETIC_PREFIX, synthetic_latin_rows
from port_entity_matching.train import build_training_data

_NAMES = ["GRAND ACE 11", "GLOVIS SPIRIT", "SHINING ROAD", "SEA STAR", "MORNING COURIER", "KASHIMA MARU 3"]


def _catalog() -> pd.DataFrame:
    rows = []
    for i in range(40):
        name = f"{_NAMES[i % len(_NAMES)]} {i}"
        rows.append(
            {
                "vessel_id": f"V{i:02d}",
                "vessel_type": "Tanker" if i % 2 else "Bulk",
                "vessel_name": name.title(),
                "size": 1000 + 100 * i,
                "unit": "DWT",
                "deadweight_tonnage": 3000 + 100 * i,
                "gross_tonnage": 1000 + 100 * i,
                "flag_name": "Panama",
            }
        )
    return pd.DataFrame(rows)


def _train() -> pd.DataFrame:
    rows = []
    for i in range(12):
        hangul = i % 3 == 0
        rows.append(
            {
                "sample_id": f"T{i:03d}",
                "vessel_name": "그랜드 에이스 일레븐" if hangul else _NAMES[i % len(_NAMES)],
                "call_sign": f"CS{i}",
                "gross_tonnage": 1000.0 + 100 * i,
                "international_gross_tonnage": 1000.0 + 100 * i,
                "flag_name": "파나마",
                "vessel_type": "유조선" if i % 2 else "산물선",
                "arrival_year": 2020,
                "target_vessel_id": f"V{i:02d}",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def context(tmp_path: Path) -> FamilyContext:
    lexicon = tmp_path / "rev.jsonl"
    lexicon.write_text(
        "".join(
            json.dumps({"name": n, "hangul": h}, ensure_ascii=False) + "\n"
            for n, h in [("A B", "현대 상선"), ("C D", "대한 해운")]
        ),
        encoding="utf-8",
    )
    return FamilyContext.build(_train(), _catalog(), lexicon)


def test_copies_are_deterministic_and_per_record(context: FamilyContext) -> None:
    config = recipe_config("rev2syn")
    records = records_from_frame(_train())
    batch, sources = family_copies(records, config, context)
    assert family_copies(records, config, context) == (batch, sources)
    for i, record in enumerate(records):
        alone, _ = family_copies([record], config, context)
        assert alone == [copy for copy, source in zip(batch, sources, strict=True) if source == i]


def test_rates_control_attribute_families(context: FamilyContext) -> None:
    records = records_from_frame(_train())
    off = replace(
        recipe_config("rev2syn"),
        augment_wrong_tonnage_rate=0.0,
        augment_type_noise_rate=0.0,
        augment_hangul_noise_rate=0.0,
    )
    tags = {copy.sample_id.split("#")[1] for copy in family_copies(records, off, context)[0]}
    assert not tags & {"aug20", "aug21", "aug22"}
    always = replace(off, augment_type_noise_rate=1.0)
    typed = [copy for copy in family_copies(records, always, context)[0] if copy.sample_id.endswith("#aug20")]
    assert len(typed) == len(records)
    assert all(copy.vessel_type != record.vessel_type for copy, record in zip(typed, records, strict=True))


def test_invalid_rate_rejected(context: FamilyContext) -> None:
    config = replace(recipe_config("rev2syn"), augment_type_noise_rate=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        family_copies(records_from_frame(_train()), config, context)


def test_families_need_a_context() -> None:
    catalog = _catalog()
    index = CatalogIndex.build(catalog)
    maps = AttributeMaps.fit(_train(), index)
    config = replace(recipe_config("rev2syn"), llm_enabled=False)
    with pytest.raises(ValueError, match="FamilyContext"):
        build_training_data(_train(), index, maps, config)


def test_recipe_config_rejects_unknown_name() -> None:
    assert recipe_config("default") == MatchConfig()
    with pytest.raises(ValueError, match="Unknown recipe"):
        recipe_config("nope")


def _varied_records() -> list:
    hangul = [
        "그랜드 에이스 일레븐",
        "글로비스 캡틴 스피릿",
        "현대 프라이드 호",
        "모닝 쿠리어 에이스",
        "칼리다",
        "카시마 마루 제3호",
    ]
    latin = ["GRAND ACE 11", "GLOVIS SPIRIT", "MORNING COURIER EXPRESS", "SEA STAR 3", "KASHIMA MARU 10", "AB"]
    rows = []
    for i in range(240):
        name = hangul[i % len(hangul)] if i % 2 else latin[i % len(latin)]
        rows.append(
            {
                "sample_id": f"S{i:04d}",
                "vessel_name": name,
                "call_sign": None,
                "gross_tonnage": [5.0, 1234.0, 98765.0][i % 3],
                "international_gross_tonnage": None if i % 5 == 0 else [5.0, 1234.0, 98765.0][i % 3],
                "flag_name": None,
                "vessel_type": "유조선",
                "arrival_year": 2020,
                "target_vessel_id": f"V{i % 40:02d}",
            }
        )
    return rows


def test_every_family_fires_and_changes_the_record(tmp_path: Path) -> None:
    frame = pd.DataFrame(_varied_records())
    lexicon = tmp_path / "rev.jsonl"
    lexicon.write_text(json.dumps({"name": "A B", "hangul": "현대 상선"}, ensure_ascii=False) + "\n", encoding="utf-8")
    context = FamilyContext.build(frame.assign(vessel_type=["유조선", "산물선"] * 120), _catalog(), lexicon)
    config = replace(
        recipe_config("rev2syn"),
        augment_wrong_tonnage_rate=1.0,
        augment_type_noise_rate=1.0,
        augment_hangul_noise_rate=1.0,
    )
    records = records_from_frame(frame)
    copies, sources = family_copies(records, config, context)
    tags = {copy.sample_id.split("#")[1] for copy in copies}
    assert tags >= {f"aug{n}" for n in (7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22)}
    for copy, source in zip(copies, sources, strict=True):
        original = records[source]
        assert copy.sample_id.startswith(original.sample_id + "#")
        assert copy != replace(original, sample_id=copy.sample_id)
        if copy.sample_id.endswith(("#aug7", "#aug8")):
            assert original.vessel_name.startswith(copy.vessel_name)
        if copy.sample_id.endswith("#aug21"):
            assert copy.gross_tonnage == copy.international_gross_tonnage
            assert copy.gross_tonnage != (original.international_gross_tonnage or original.gross_tonnage)


def test_type_noise_blanks_when_no_other_type_exists(context: FamilyContext) -> None:
    single = replace(context, vessel_types=("유조선",), vessel_type_probs=(1.0,))
    config = replace(recipe_config("rev2syn"), augment_type_noise_rate=1.0)
    records = [replace(r, vessel_type="유조선") for r in records_from_frame(_train())]
    typed = [c for c in family_copies(records, config, single)[0] if c.sample_id.endswith("#aug20")]
    assert len(typed) == len(records)
    assert all(c.vessel_type is None for c in typed)


def test_short_names_and_missing_tonnage_get_no_damage_copies(context: FamilyContext) -> None:
    config = replace(
        recipe_config("rev2syn"),
        augment_wrong_tonnage_rate=1.0,
        augment_hangul_noise_rate=1.0,
        augment_type_noise_rate=0.0,
    )
    rows = [
        _varied_records()[1] | {"sample_id": f"K{i}", "vessel_name": "흥아", "gross_tonnage": None,
                                "international_gross_tonnage": None}
        for i in range(60)
    ]  # fmt: skip
    copies, _ = family_copies(records_from_frame(pd.DataFrame(rows)), config, context)
    assert not [c for c in copies if c.sample_id.endswith(("#aug8", "#aug21", "#aug22"))]


def test_synthetic_and_derived_rows_get_no_copies(context: FamilyContext) -> None:
    config = replace(recipe_config("rev2syn"), augment_type_noise_rate=1.0)
    records = records_from_frame(_train())
    skipped = [replace(records[0], sample_id=SYNTHETIC_PREFIX + "000001"), replace(records[1], sample_id="T001#aug6")]
    assert family_copies(skipped, config, context) == ([], [])
