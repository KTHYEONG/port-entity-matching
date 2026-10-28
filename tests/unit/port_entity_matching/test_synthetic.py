"""Contract tests for catalog-synthetic Latin training rows."""

from __future__ import annotations

import re

import pandas as pd

from port_entity_matching.synthetic import SYNTHETIC_PREFIX, synthetic_latin_rows

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


def test_synthetic_rows_respect_exclusions_and_seed() -> None:
    train, catalog = _train(), _catalog()
    excluded = ["V20", "V21", "V22"]
    rows = synthetic_latin_rows(train, catalog, 10, excluded, seed=0)
    assert len(rows) == 10
    assert rows["sample_id"].str.startswith(SYNTHETIC_PREFIX).all()
    drawn = set(rows["target_vessel_id"])
    assert not drawn & set(excluded)
    assert not drawn & set(train["target_vessel_id"])
    assert rows.equals(synthetic_latin_rows(train, catalog, 10, excluded, seed=0))
    assert set(rows.columns) >= {"vessel_name", "gross_tonnage", "target_vessel_id"}


def test_synthetic_rows_cover_the_damage_mixture() -> None:
    names = ["GRAND ACE 3", "GLOVIS SPIRIT EXPRESS", "MORNING COURIER", "SEA STAR 10", "SHINING ROAD"]
    catalog = pd.DataFrame(
        [
            {
                "vessel_id": f"C{i:03d}",
                "vessel_type": "Tanker",
                "vessel_name": f"{names[i % len(names)]} {i}",
                "size": 1000 + i,
                "unit": "DWT",
                "deadweight_tonnage": 3000 + i,
                "gross_tonnage": 1000 + i,
                "flag_name": "Panama",
            }
            for i in range(300)
        ]
    )
    train = _train().assign(target_vessel_id=[f"C{i:03d}" for i in range(12)], vessel_type="유조선")
    rows = synthetic_latin_rows(train, catalog, 250, [], seed=1)
    clean = {re.sub(r"\s+", " ", n.upper()) for n in catalog["vessel_name"]}
    damaged = rows[~rows["vessel_name"].isin(clean)]
    assert 0.4 < len(damaged) / len(rows) < 0.8
    assert rows["vessel_name"].str.startswith(("MV ", "M/V ")).any()
    assert rows["vessel_type"].eq("유조선").all()


def test_synthetic_rows_keep_columns_when_pool_is_empty() -> None:
    train = _train()
    catalog = _catalog()
    rows = synthetic_latin_rows(train, catalog, 5, list(catalog["vessel_id"]), seed=0)
    assert rows.empty
    assert "target_vessel_id" in rows.columns


def test_short_token_names_still_yield_valid_rows() -> None:
    catalog = pd.DataFrame(
        [
            {
                "vessel_id": f"S{i:03d}",
                "vessel_type": "Tanker",
                "vessel_name": f"OX {i % 10}" if i % 2 else f"SEA OX {i}",
                "size": 500 + i,
                "unit": "DWT",
                "deadweight_tonnage": 900 + i,
                "gross_tonnage": 500 + i,
                "flag_name": "Panama",
            }
            for i in range(400)
        ]
    )
    train = _train().assign(target_vessel_id=[f"S{i:03d}" for i in range(12)], vessel_type="유조선")
    rows = synthetic_latin_rows(train, catalog, 380, [], seed=3)
    assert len(rows) == 380
    assert rows["vessel_name"].str.len().gt(0).all()
    assert rows["target_vessel_id"].isin(set(catalog["vessel_id"])).all()
