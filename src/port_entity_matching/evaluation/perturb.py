"""Deterministic record perturbations used for robustness stress tests.

Every perturbation is derived from the record itself, so the same record, scenario
and seed always yield the same degraded copy, independent of batch composition.
"""

from __future__ import annotations

import dataclasses
import zlib
from collections.abc import Callable

import numpy as np

from port_entity_matching.augment import drop_hangul_syllable
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.records import QueryRecord

__all__ = ["PERTURBATIONS", "perturb_record"]

PERTURBATIONS: tuple[str, ...] = (
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
)

_SYLLABLE_BASE = 0xAC00
_JUNGSEONG: tuple[str, ...] = (
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
)
_MEDIAL_SWAP: dict[str, str] = {
    "ㅐ": "ㅔ",
    "ㅔ": "ㅐ",
    "ㅓ": "ㅗ",
    "ㅗ": "ㅓ",
    "ㅜ": "ㅡ",
    "ㅡ": "ㅜ",
    "ㅣ": "ㅟ",
    "ㅟ": "ㅣ",
}

Handler = Callable[[QueryRecord, np.random.Generator], QueryRecord | None]


def perturb_record(record: QueryRecord, scenario: str, seed: int = 0) -> QueryRecord | None:
    """Return the degraded copy of record for a named scenario, or None when the scenario does not apply."""
    handler = _SCENARIOS.get(scenario)
    if handler is None:
        raise ValueError(f"Unknown scenario '{scenario}'. Valid scenarios: {', '.join(PERTURBATIONS)}")
    rng = np.random.default_rng(zlib.crc32(f"{seed}|{scenario}|{record.sample_id}".encode()))
    perturbed = handler(record, rng)
    if perturbed is None or perturbed == record:
        return None
    return perturbed


def _syllable_positions(name: str) -> list[int]:
    return [position for position, char in enumerate(name) if _SYLLABLE_BASE <= ord(char) <= 0xD7A3]


def _is_hangul(record: QueryRecord) -> bool:
    return bool(is_hangul_name(record.vessel_name))


def _scaled_tonnage(record: QueryRecord, factor: float) -> QueryRecord:
    gross = None if record.gross_tonnage is None else record.gross_tonnage * factor
    international = (
        None if record.international_gross_tonnage is None else record.international_gross_tonnage * factor
    )
    return dataclasses.replace(
        record,
        gross_tonnage=gross,
        international_gross_tonnage=international,
    )


def _tonnage_factor(factor: float) -> Handler:
    def handler(record: QueryRecord, rng: np.random.Generator) -> QueryRecord | None:
        if not _is_hangul(record):
            return None
        return _scaled_tonnage(record, factor)

    return handler


def _missing_tonnage(record: QueryRecord, rng: np.random.Generator) -> QueryRecord | None:
    if not _is_hangul(record):
        return None
    return dataclasses.replace(record, gross_tonnage=None, international_gross_tonnage=None)


def _deleted_syllable(record: QueryRecord, rng: np.random.Generator) -> QueryRecord | None:
    if not _is_hangul(record):
        return None
    name = drop_hangul_syllable(record.vessel_name, rng, 3)
    return dataclasses.replace(record, vessel_name=name)


def _inserted_syllable(record: QueryRecord, rng: np.random.Generator) -> QueryRecord | None:
    if not _is_hangul(record):
        return None
    positions = _syllable_positions(record.vessel_name)
    position = positions[int(rng.integers(len(positions)))]
    name = record.vessel_name
    return dataclasses.replace(record, vessel_name=f"{name[: position + 1]}{name[position]}{name[position + 1 :]}")


def _swapped_vowel(record: QueryRecord, rng: np.random.Generator) -> QueryRecord | None:
    if not _is_hangul(record):
        return None
    name = record.vessel_name
    candidates = [position for position in _syllable_positions(name) if _medial(name[position]) in _MEDIAL_SWAP]
    if not candidates:
        return None
    position = candidates[int(rng.integers(len(candidates)))]
    return dataclasses.replace(record, vessel_name=_swap_medial(name, position))


def _removed_spaces(record: QueryRecord, rng: np.random.Generator) -> QueryRecord | None:
    if not _is_hangul(record):
        return None
    return dataclasses.replace(record, vessel_name=record.vessel_name.replace(" ", ""))


def _missing_flag_and_type(record: QueryRecord, rng: np.random.Generator) -> QueryRecord | None:
    if not _is_hangul(record):
        return None
    return dataclasses.replace(record, flag_name=None, vessel_type="기타선")


def _latin_typo(record: QueryRecord, rng: np.random.Generator) -> QueryRecord | None:
    if _is_hangul(record):
        return None
    name = record.vessel_name
    positions = [position for position, char in enumerate(name) if char.isalpha()]
    if len(positions) < 4:
        return None
    position = positions[int(rng.integers(len(positions)))]
    if rng.random() < 0.5:
        return dataclasses.replace(record, vessel_name=f"{name[:position]}{name[position + 1 :]}")
    return dataclasses.replace(record, vessel_name=f"{name[: position + 1]}{name[position]}{name[position + 1 :]}")


def _latin_missing_tonnage(record: QueryRecord, rng: np.random.Generator) -> QueryRecord | None:
    if _is_hangul(record):
        return None
    return dataclasses.replace(record, gross_tonnage=None, international_gross_tonnage=None)


def _medial(char: str) -> str:
    code = ord(char) - _SYLLABLE_BASE
    return _JUNGSEONG[(code % 588) // 28]


def _swap_medial(name: str, position: int) -> str:
    code = ord(name[position]) - _SYLLABLE_BASE
    initial = code // 588
    medial = (code % 588) // 28
    final = code % 28
    swapped = _JUNGSEONG.index(_MEDIAL_SWAP[_JUNGSEONG[medial]])
    return f"{name[:position]}{chr(_SYLLABLE_BASE + initial * 588 + swapped * 28 + final)}{name[position + 1 :]}"


_SCENARIOS: dict[str, Handler] = {
    "kor_tonnage_x1.08": _tonnage_factor(1.08),
    "kor_tonnage_x1.12": _tonnage_factor(1.12),
    "kor_tonnage_x0.90": _tonnage_factor(0.90),
    "kor_tonnage_missing": _missing_tonnage,
    "kor_syllable_deleted": _deleted_syllable,
    "kor_syllable_inserted": _inserted_syllable,
    "kor_vowel_swapped": _swapped_vowel,
    "kor_spaces_removed": _removed_spaces,
    "kor_flag_type_missing": _missing_flag_and_type,
    "lat_extra_typo": _latin_typo,
    "lat_tonnage_missing": _latin_missing_tonnage,
}
