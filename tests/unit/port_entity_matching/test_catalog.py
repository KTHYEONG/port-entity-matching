"""Contract tests for the catalog index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.normalize import latin_phonetic_key_v2


def _catalog(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ("vessel_type", "flag_name"):
        if column not in frame.columns:
            frame[column] = ""
    return frame


def test_duplicate_vessel_id_rejected() -> None:
    frame = _catalog(
        [
            {"vessel_id": "A1", "vessel_name": "Alpha", "gross_tonnage": 100},
            {"vessel_id": "A1", "vessel_name": "Alpha Two", "gross_tonnage": 200},
        ]
    )
    with pytest.raises(ValueError, match="duplicate vessel_id"):
        CatalogIndex.build(frame)


def test_missing_column_rejected() -> None:
    frame = _catalog([{"vessel_id": "A1", "vessel_name": "Alpha"}])
    with pytest.raises(ValueError, match="gross_tonnage"):
        CatalogIndex.build(frame)


def test_empty_catalog_rejected() -> None:
    frame = pd.DataFrame(columns=["vessel_id", "vessel_name", "gross_tonnage", "vessel_type", "flag_name"])
    with pytest.raises(ValueError, match="empty"):
        CatalogIndex.build(frame)


def test_gt_range_skips_missing_and_non_positive() -> None:
    frame = _catalog(
        [
            {"vessel_id": "A0", "vessel_name": "Zero", "gross_tonnage": 0},
            {"vessel_id": "A1", "vessel_name": "Hundred", "gross_tonnage": 100},
            {"vessel_id": "A2", "vessel_name": "Two Hundred", "gross_tonnage": 200},
            {"vessel_id": "A3", "vessel_name": "Missing", "gross_tonnage": None},
            {"vessel_id": "A4", "vessel_name": "Two Five", "gross_tonnage": 205},
        ]
    )
    index = CatalogIndex.build(frame)
    assert index.gt_range(195, 210).tolist() == [2, 4]
    assert index.gt_range(0, 1000).tolist() == [1, 2, 4]
    assert index.gt_range(300, 400).tolist() == []
    assert len(index) == 5


def test_gt_exact_and_same_gt_count() -> None:
    frame = _catalog(
        [
            {"vessel_id": "A0", "vessel_name": "Hundred", "gross_tonnage": 100},
            {"vessel_id": "A1", "vessel_name": "Two Hundred", "gross_tonnage": 200},
            {"vessel_id": "A2", "vessel_name": "Other Two Hundred", "gross_tonnage": 200},
        ]
    )
    index = CatalogIndex.build(frame)
    assert index.gt_exact(200).tolist() == [1, 2]
    assert index.same_gt_count[1] == 2
    assert index.same_gt_count[0] == 1


def test_token_postings_and_document_frequency() -> None:
    frame = _catalog(
        [
            {"vessel_id": "A0", "vessel_name": "Glovis Sun", "gross_tonnage": 100},
            {"vessel_id": "A1", "vessel_name": "Glovis Sky", "gross_tonnage": 200},
            {"vessel_id": "A2", "vessel_name": "Other Vessel", "gross_tonnage": 300},
        ]
    )
    index = CatalogIndex.build(frame)
    assert index.token_postings("GLOVIS").tolist() == [0, 1]
    assert index.token_df["GLOVIS"] == 2
    assert index.token_postings("UNKNOWN").tolist() == []
    assert index.token_postings("UNKNOWN").dtype == np.int64


def test_non_finite_bounds_are_empty() -> None:
    index = CatalogIndex.build(_catalog([{"vessel_id": "A0", "vessel_name": "Alpha", "gross_tonnage": 100}]))
    assert index.gt_range(300, 100).tolist() == []
    assert index.gt_range(float("nan"), 100).tolist() == []
    assert index.gt_exact(float("nan")).tolist() == []


def test_missing_names_and_attributes_become_empty_strings() -> None:
    empty_frame = pd.DataFrame(
        [{"vessel_id": "A0", "vessel_name": None, "gross_tonnage": 100, "vessel_type": None, "flag_name": None}]
    )
    nan_frame = pd.DataFrame(
        [{"vessel_id": "A1", "vessel_name": np.nan, "gross_tonnage": 200, "vessel_type": np.nan, "flag_name": np.nan}]
    )
    for frame in (empty_frame, nan_frame):
        index = CatalogIndex.build(frame)
        assert index.names == ("",)
        assert index.canonical == [""]
        assert index.vessel_type == ("",)
        assert index.flag == ("",)


def test_names_and_keys_are_derived_from_vessel_name() -> None:
    frame = _catalog([{"vessel_id": "A0", "vessel_name": "M/V Stena Promise 16", "gross_tonnage": 100}])
    index = CatalogIndex.build(frame)
    assert index.canonical[0] == "STENAPROMISE16"
    assert index.tokens[0] == frozenset({"STENA", "PROMISE", "16"})
    assert index.digits[0] == ("16",)
    assert index.key_cons[0] == "MPSTNPRMS16"
    assert index.id_to_pos == {"A0": 0}


def test_raw_attributes_and_nullable_tonnage() -> None:
    frame = pd.DataFrame(
        {
            "vessel_id": ["A0", "A1"],
            "vessel_name": ["Alpha", "nan"],
            "gross_tonnage": pd.Series([100, pd.NA], dtype="Float64"),
            "vessel_type": [" Cargo ", "nan"],
            "flag_name": [" Panama ", "nan"],
        },
        index=[0, 1],
    )
    index = CatalogIndex.build(frame)
    assert index.vessel_type == (" Cargo ", "nan")
    assert index.flag == (" Panama ", "nan")
    assert index.names == ("Alpha", "nan")
    assert np.isnan(index.gt[1])
    assert index.same_gt_count.tolist() == [1, 0]
    assert index.gt_range(-np.inf, np.inf).tolist() == [0]


def test_duplicate_ids_checked_after_text_conversion() -> None:
    frame = pd.DataFrame({"vessel_id": [None, np.nan], "vessel_name": ["A", "B"], "gross_tonnage": [100, 200]})
    with pytest.raises(ValueError, match="duplicate vessel_id"):
        CatalogIndex.build(frame)


def test_key_v2_columns_align_with_catalog_rows() -> None:
    frame = _catalog(
        [
            {"vessel_id": "A0", "vessel_name": "M/V Stena Promise 16", "gross_tonnage": 100},
            {"vessel_id": "A1", "vessel_name": "Glovis Spirit", "gross_tonnage": 200},
            {"vessel_id": "A2", "vessel_name": "BW Wren", "gross_tonnage": 300},
        ]
    )
    index = CatalogIndex.build(frame)

    assert len(index.key2_cons) == len(index.key2_vow) == 3
    for position, name in enumerate(index.names):
        assert index.key2_cons[position] == latin_phonetic_key_v2(name, False)
        assert index.key2_vow[position] == latin_phonetic_key_v2(name, True)


def test_sorted_and_word_key_fields() -> None:
    frame = _catalog(
        [
            {"vessel_id": "A0", "vessel_name": "BLUE STAR", "gross_tonnage": 100},
            {"vessel_id": "A1", "vessel_name": "STAR BLUE", "gross_tonnage": 200},
            {"vessel_id": "A2", "vessel_name": "RED SUN", "gross_tonnage": 300},
        ]
    )
    index = CatalogIndex.build(frame)
    assert index.sorted_canonical[0] == index.sorted_canonical[1]
    assert index.sorted_key2_cons[0] == index.sorted_key2_cons[1]
    assert index.token_idf("STAR") < index.token_idf("RED")
    from port_entity_matching.normalize import latin_phonetic_key_v2

    star_key = latin_phonetic_key_v2("STAR", False)
    assert index.word_key_postings(star_key).tolist() == [0, 1]
    assert index.word_key_postings("UNKNOWN").tolist() == []
