"""Contract tests for record loading and typed record conversion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from port_entity_matching.records import (
    RECORD_COLUMNS,
    QueryRecord,
    read_csv_checked,
    records_from_frame,
)


def _write_csv(path: Path, rows: list[dict[str, object]], encoding: str = "utf-8") -> Path:
    pd.DataFrame(rows, columns=list(RECORD_COLUMNS)).to_csv(path, index=False, encoding=encoding)
    return path


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "s1",
            "vessel_name": "HANG HAE",
            "call_sign": "abc",
            "gross_tonnage": 1000.0,
            "international_gross_tonnage": 2000.0,
            "flag_name": "Korea",
            "vessel_type": "bulk",
            "arrival_year": 2019,
        },
        {
            "sample_id": "s2",
            "vessel_name": "SECOND",
            "call_sign": "def",
            "gross_tonnage": 10.0,
            "international_gross_tonnage": 20.0,
            "flag_name": "Panama",
            "vessel_type": "tanker",
            "arrival_year": 2020,
        },
    ]


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"

    with pytest.raises(FileNotFoundError, match=r"absent\.csv"):
        read_csv_checked(missing, RECORD_COLUMNS)


def test_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    frame = pd.DataFrame(_sample_rows()).drop(columns=["call_sign"])
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="call_sign"):
        read_csv_checked(path, RECORD_COLUMNS)


def test_bom_header_parsed(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "bom.csv", _sample_rows(), encoding="utf-8-sig")

    frame = read_csv_checked(path, RECORD_COLUMNS)

    assert next(iter(frame.columns)) == "sample_id"
    assert frame["sample_id"].tolist() == ["s1", "s2"]


def test_missing_values_become_none(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "vessel_name": "NAME",
                "call_sign": None,
                "gross_tonnage": float("nan"),
                "international_gross_tonnage": 0,
                "flag_name": None,
                "vessel_type": "  ",
                "arrival_year": None,
            }
        ],
        columns=list(RECORD_COLUMNS),
    ).to_csv(path, index=False)

    record = records_from_frame(read_csv_checked(path, RECORD_COLUMNS))[0]

    assert record.gross_tonnage is None
    assert record.international_gross_tonnage is None
    assert record.flag_name is None
    assert record.call_sign is None
    assert record.vessel_type is None
    assert record.arrival_year is None
    assert record.vessel_name == "NAME"


def test_empty_csv_reports_path_and_columns(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no header") as error:
        read_csv_checked(path, RECORD_COLUMNS)
    assert str(path) in str(error.value)
    assert "sample_id" in str(error.value)


def test_identifier_columns_preserve_leading_zeros(tmp_path: Path) -> None:
    path = tmp_path / "identifiers.csv"
    path.write_text("sample_id,vessel_id,call_sign\n001,002,003\n", encoding="utf-8")
    frame = read_csv_checked(path, ("sample_id", "vessel_id", "call_sign"))
    assert frame.iloc[0].tolist() == ["001", "002", "003"]


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, pd.NaT, "", "  "])
def test_scalar_missing_values(missing: object) -> None:
    row = dict.fromkeys(RECORD_COLUMNS, missing)
    row["sample_id"] = "s1"
    record = records_from_frame(pd.DataFrame([row]))[0]
    assert record.vessel_name == ""
    assert record.call_sign is None
    assert record.flag_name is None
    assert record.vessel_type is None
    assert record.gross_tonnage is None
    assert record.international_gross_tonnage is None
    assert record.arrival_year is None


def test_conversion_is_row_independent_and_handles_empty_frame() -> None:
    frame = pd.DataFrame(_sample_rows())
    records = records_from_frame(frame)
    assert records_from_frame(frame.iloc[[1, 0]]) == records[::-1]
    assert records_from_frame(frame.iloc[[0]]) == records[:1]
    assert records_from_frame(frame.iloc[:0]) == []


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), 10**400])
def test_nonfinite_or_overflowing_numbers_are_unknown(value: object) -> None:
    row = _sample_rows()[0]
    row.update(gross_tonnage=value, international_gross_tonnage=value, arrival_year=value)
    record = records_from_frame(pd.DataFrame([row], dtype=object))[0]
    assert record.gross_tonnage is None
    assert record.international_gross_tonnage is None
    assert record.arrival_year is None


def test_order_preserved(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "records.csv", _sample_rows())

    records = records_from_frame(read_csv_checked(path, RECORD_COLUMNS))

    assert [record.sample_id for record in records] == ["s1", "s2"]


def test_extra_columns_ignored() -> None:
    frame = pd.DataFrame(_sample_rows()).assign(target_vessel_id=["v1", "v2"])

    records = records_from_frame(frame)

    assert len(records) == 2
    assert all(isinstance(record, QueryRecord) for record in records)


def test_frame_missing_columns_raises() -> None:
    frame = pd.DataFrame(_sample_rows()).drop(columns=["arrival_year"])

    with pytest.raises(ValueError, match="arrival_year"):
        records_from_frame(frame)


def test_unparsable_numbers_become_none() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": 7,
                "vessel_name": None,
                "call_sign": "abc",
                "gross_tonnage": "n/a",
                "international_gross_tonnage": -5.0,
                "flag_name": "Korea",
                "vessel_type": "bulk",
                "arrival_year": "2019.0",
            },
            {
                "sample_id": "s2",
                "vessel_name": "OTHER",
                "call_sign": "abc",
                "gross_tonnage": None,
                "international_gross_tonnage": None,
                "flag_name": "Korea",
                "vessel_type": "bulk",
                "arrival_year": "unknown",
            },
        ]
    )

    records = records_from_frame(frame)

    assert records[0].sample_id == "7"
    assert records[0].vessel_name == ""
    assert records[0].gross_tonnage is None
    assert records[0].international_gross_tonnage is None
    assert records[0].arrival_year == 2019
    assert records[1].gross_tonnage is None
    assert records[1].arrival_year is None


def test_missing_identifier_and_infinite_year() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": None,
                "vessel_name": "NAME",
                "call_sign": "abc",
                "gross_tonnage": 1.0,
                "international_gross_tonnage": 1.0,
                "flag_name": "Korea",
                "vessel_type": "bulk",
                "arrival_year": float("inf"),
            }
        ]
    )

    record = records_from_frame(frame)[0]

    assert record.sample_id == "None"
    assert record.arrival_year is None
