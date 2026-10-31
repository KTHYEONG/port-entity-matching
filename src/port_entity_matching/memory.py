"""Unique training call-sign mappings persisted as catalog vessel ids."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from port_entity_matching.catalog import CatalogIndex

__all__ = ["CallSignMemory"]

_REQUIRED_COLUMNS: tuple[str, ...] = ("call_sign", "target_vessel_id")


@dataclass(frozen=True, eq=False)
class CallSignMemory:
    """Train-fitted call_sign to vessel_id lookup restricted to ids present in the catalog."""

    _vessel_ids: dict[str, str]
    _positions: dict[str, int]

    @classmethod
    def fit(cls, train: pd.DataFrame, index: CatalogIndex) -> CallSignMemory:
        """Build from train rows with call_sign and target_vessel_id; raise ValueError if either column is absent."""
        missing = [column for column in _REQUIRED_COLUMNS if column not in train.columns]
        if missing:
            raise ValueError(f"Train frame is missing required columns: {', '.join(missing)}")

        grouped: dict[str, set[str]] = {}
        for raw_sign, raw_target in zip(train["call_sign"], train["target_vessel_id"], strict=True):
            sign = _as_text(raw_sign).strip()
            target = _as_text(raw_target)
            if not sign or not target:
                continue
            grouped.setdefault(sign, set()).add(target)

        vessel_ids = {sign: next(iter(targets)) for sign, targets in grouped.items() if len(targets) == 1}
        known = {sign: target for sign, target in vessel_ids.items() if target in index.id_to_pos}
        positions = {sign: index.id_to_pos[target] for sign, target in known.items()}
        return cls(_vessel_ids=known, _positions=positions)

    def lookup(self, call_sign: str | None) -> int | None:
        """Catalog position of the memorized vessel, or None."""
        if not isinstance(call_sign, str):
            return None
        return self._positions.get(call_sign.strip())

    def known_positions(self) -> frozenset[int]:
        """Catalog positions of every memorized vessel."""
        return frozenset(self._positions.values())

    def __len__(self) -> int:
        """Number of memorized call signs."""
        return len(self._positions)

    def to_json(self, path: str | Path) -> None:
        """Write {call_sign: vessel_id} sorted by call_sign."""
        payload = {sign: self._vessel_ids[sign] for sign in sorted(self._vessel_ids)}
        destination = Path(path)
        if destination.parent != Path(""):
            destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path, index: CatalogIndex) -> CallSignMemory:
        """Load a saved mapping, dropping ids absent from the catalog."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Call-sign memory file {path} must contain a JSON object")
        vessel_ids: dict[str, str] = {}
        positions: dict[str, int] = {}
        for raw_sign, raw_target in raw.items():
            sign = _as_text(raw_sign).strip()
            target = _as_text(raw_target)
            if not sign or not target or target not in index.id_to_pos:
                continue
            vessel_ids[sign] = target
            positions[sign] = index.id_to_pos[target]
        return cls(_vessel_ids=vessel_ids, _positions=positions)


def _as_text(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)
