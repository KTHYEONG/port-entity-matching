"""Typed loading of port-call records and the vessel catalog.

The record schema is fixed by the competition input files; missing values are
normalized to ``None`` so that downstream code never has to reason about NaN.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd

RECORD_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "vessel_name",
    "call_sign",
    "gross_tonnage",
    "international_gross_tonnage",
    "flag_name",
    "vessel_type",
    "arrival_year",
)
CATALOG_COLUMNS: tuple[str, ...] = (
    "vessel_id",
    "vessel_type",
    "vessel_name",
    "size",
    "unit",
    "deadweight_tonnage",
    "gross_tonnage",
    "flag_name",
)

_STRING_DTYPES: dict[str, Any] = {"sample_id": str, "vessel_id": str, "call_sign": str}


@dataclass(frozen=True)
class QueryRecord:
    """One port-call record with missing values normalized to None."""

    sample_id: str
    vessel_name: str
    call_sign: str | None
    gross_tonnage: float | None
    international_gross_tonnage: float | None
    flag_name: str | None
    vessel_type: str | None
    arrival_year: int | None
    alt_names: tuple[str, ...] = ()


def read_csv_checked(path: str | Path, required: Sequence[str]) -> pd.DataFrame:
    """Read a UTF-8(-sig) CSV; raise FileNotFoundError if absent, ValueError listing missing required columns."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"CSV file not found: {resolved}")
    try:
        frame = pd.read_csv(resolved, encoding="utf-8-sig", dtype=_STRING_DTYPES)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"CSV {resolved} has no header; required columns: {', '.join(required)}") from error
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV {resolved} is missing required columns: {', '.join(missing)}")
    return frame


def records_from_frame(frame: pd.DataFrame) -> list[QueryRecord]:
    """Convert record rows to QueryRecords in frame order; raise ValueError on missing RECORD_COLUMNS."""
    missing = [column for column in RECORD_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Record frame is missing required columns: {', '.join(missing)}")
    records = [
        QueryRecord(
            sample_id=str(row.get("sample_id")),
            vessel_name=_as_name(row.get("vessel_name")),
            call_sign=_as_optional_text(row.get("call_sign")),
            gross_tonnage=_as_optional_tonnage(row.get("gross_tonnage")),
            international_gross_tonnage=_as_optional_tonnage(row.get("international_gross_tonnage")),
            flag_name=_as_optional_text(row.get("flag_name")),
            vessel_type=_as_optional_text(row.get("vessel_type")),
            arrival_year=_as_optional_year(row.get("arrival_year")),
        )
        for row in frame.to_dict(orient="records")
    ]
    return records


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, Real) and value != value


def _as_name(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value).strip()


def _as_optional_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _as_optional_tonnage(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        tonnage = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(tonnage) or tonnage <= 0:
        return None
    return tonnage


def _as_optional_year(value: object) -> int | None:
    if _is_missing(value):
        return None
    try:
        year = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(year):
        return None
    return int(year)
