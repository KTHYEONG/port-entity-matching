"""Smoothed P(catalog attribute | record attribute) tables fitted on training rows."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from port_entity_matching.catalog import CatalogIndex

__all__ = ["AttributeMaps"]

_REQUIRED_COLUMNS: tuple[str, ...] = ("vessel_type", "flag_name", "target_vessel_id")


class AttributeMaps:
    """Smoothing-estimated attribute compatibility tables; fitted on training rows only."""

    __slots__ = (
        "alpha",
        "flag_counts",
        "flag_prior_counts",
        "flag_totals",
        "flag_value_count",
        "prior_total",
        "type_counts",
        "type_prior_counts",
        "type_totals",
        "type_value_count",
    )

    def __init__(
        self,
        alpha: float,
        type_counts: dict[str, dict[str, int]],
        type_totals: dict[str, int],
        type_prior_counts: dict[str, int],
        flag_counts: dict[str, dict[str, int]],
        flag_totals: dict[str, int],
        flag_prior_counts: dict[str, int],
        type_value_count: int,
        flag_value_count: int,
        prior_total: int,
    ) -> None:
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = float(alpha)
        self.type_counts = type_counts
        self.type_totals = type_totals
        self.type_prior_counts = type_prior_counts
        self.flag_counts = flag_counts
        self.flag_totals = flag_totals
        self.flag_prior_counts = flag_prior_counts
        self.type_value_count = type_value_count
        self.flag_value_count = flag_value_count
        self.prior_total = prior_total

    @classmethod
    def fit(cls, train: pd.DataFrame, index: CatalogIndex, alpha: float = 1.0) -> AttributeMaps:
        """Count record-type→catalog-type and record-flag→catalog-flag pairs; ValueError on missing columns."""
        missing = [column for column in _REQUIRED_COLUMNS if column not in train.columns]
        if missing:
            raise ValueError(f"Train frame is missing required columns: {', '.join(missing)}")
        type_counts: dict[str, dict[str, int]] = {}
        type_totals: dict[str, int] = {}
        type_prior_counts: dict[str, int] = {}
        flag_counts: dict[str, dict[str, int]] = {}
        flag_totals: dict[str, int] = {}
        flag_prior_counts: dict[str, int] = {}
        prior_total = 0

        for record_type, record_flag, raw_target in zip(
            train["vessel_type"].tolist(),
            train["flag_name"].tolist(),
            train["target_vessel_id"].tolist(),
            strict=True,
        ):
            target = _as_text(raw_target)
            position = index.id_to_pos.get(target)
            if position is None:
                continue
            prior_total += 1
            catalog_type = index.vessel_type[position]
            catalog_flag = index.flag[position]
            type_prior_counts[catalog_type] = type_prior_counts.get(catalog_type, 0) + 1
            flag_prior_counts[catalog_flag] = flag_prior_counts.get(catalog_flag, 0) + 1

            key_type = _as_optional_text(record_type)
            if key_type is not None:
                bucket = type_counts.setdefault(key_type, {})
                bucket[catalog_type] = bucket.get(catalog_type, 0) + 1
                type_totals[key_type] = type_totals.get(key_type, 0) + 1
            key_flag = _as_optional_text(record_flag)
            if key_flag is not None:
                bucket = flag_counts.setdefault(key_flag, {})
                bucket[catalog_flag] = bucket.get(catalog_flag, 0) + 1
                flag_totals[key_flag] = flag_totals.get(key_flag, 0) + 1

        return cls(
            alpha=alpha,
            type_counts=type_counts,
            type_totals=type_totals,
            type_prior_counts=type_prior_counts,
            flag_counts=flag_counts,
            flag_totals=flag_totals,
            flag_prior_counts=flag_prior_counts,
            type_value_count=len(set(index.vessel_type)),
            flag_value_count=len(set(index.flag)),
            prior_total=prior_total,
        )

    def type_logprob(self, record_type: str | None, catalog_type: str) -> float:
        """log P(catalog_type | record_type); NaN when the record type is missing."""
        if record_type is None:
            return math.nan
        return _smoothed_logprob(
            self.type_counts.get(record_type, {}).get(catalog_type, 0),
            self.type_totals.get(record_type, 0),
            self.type_value_count,
            self.alpha,
        )

    def flag_logprob(self, record_flag: str | None, catalog_flag: str) -> float:
        """log P(catalog_flag | record_flag); NaN when the record flag is missing."""
        if record_flag is None:
            return math.nan
        return _smoothed_logprob(
            self.flag_counts.get(record_flag, {}).get(catalog_flag, 0),
            self.flag_totals.get(record_flag, 0),
            self.flag_value_count,
            self.alpha,
        )

    def type_logprior(self, catalog_type: str) -> float:
        """log of the smoothed share of catalog_type among the training targets."""
        return _smoothed_logprob(
            self.type_prior_counts.get(catalog_type, 0),
            self.prior_total,
            self.type_value_count,
            self.alpha,
        )

    def flag_logprior(self, catalog_flag: str) -> float:
        """log of the smoothed share of catalog_flag among the training targets."""
        return _smoothed_logprob(
            self.flag_prior_counts.get(catalog_flag, 0),
            self.prior_total,
            self.flag_value_count,
            self.alpha,
        )

    def to_json(self, path: str | Path) -> None:
        """Write every count table and the smoothing strength."""
        payload = {
            "alpha": self.alpha,
            "type_counts": self.type_counts,
            "type_totals": self.type_totals,
            "type_prior_counts": self.type_prior_counts,
            "flag_counts": self.flag_counts,
            "flag_totals": self.flag_totals,
            "flag_prior_counts": self.flag_prior_counts,
            "type_value_count": self.type_value_count,
            "flag_value_count": self.flag_value_count,
            "prior_total": self.prior_total,
        }
        destination = Path(path)
        if destination.parent != Path(""):
            destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> AttributeMaps:
        """Load the count tables written by to_json."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Attribute map file {path} must contain a JSON object")
        try:
            return cls(
                alpha=float(raw["alpha"]),
                type_counts=_nested_counts(raw["type_counts"]),
                type_totals=_flat_counts(raw["type_totals"]),
                type_prior_counts=_flat_counts(raw["type_prior_counts"]),
                flag_counts=_nested_counts(raw["flag_counts"]),
                flag_totals=_flat_counts(raw["flag_totals"]),
                flag_prior_counts=_flat_counts(raw["flag_prior_counts"]),
                type_value_count=int(raw["type_value_count"]),
                flag_value_count=int(raw["flag_value_count"]),
                prior_total=int(raw["prior_total"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Attribute map file {path} is malformed: {error}") from error


def _smoothed_logprob(count: int, total: int, value_count: int, alpha: float) -> float:
    return math.log((count + alpha) / (total + alpha * value_count))


def _as_text(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def _as_optional_text(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    return text or None


def _flat_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object of counts")
    return {str(key): int(count) for key, count in value.items()}


def _nested_counts(value: object) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object of counts")
    nested: dict[str, dict[str, int]] = {}
    for outer, inner in value.items():
        nested[str(outer)] = _flat_counts(inner)
    return nested
