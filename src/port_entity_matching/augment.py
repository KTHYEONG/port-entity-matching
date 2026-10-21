"""Training-only degraded copies of Hangul records.

The copies mimic the distortions observed in the port-call records (inflated
tonnage, missing tonnage, dropped Hangul syllable) so the ranker learns to
cope with them. They are used for training only, never at inference time.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

from port_entity_matching.config import MatchConfig
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.records import QueryRecord

__all__ = ["degraded_copies", "drop_hangul_syllable"]

_SYLLABLE_FIRST = 0xAC00
_SYLLABLE_LAST = 0xD7A3


def degraded_copies(
    records: Sequence[QueryRecord],
    config: MatchConfig,
) -> tuple[list[QueryRecord], list[int]]:
    """Return training-only degraded copies of Hangul records and, per copy, the index of its source record."""
    if not math.isfinite(config.augment_tonnage_factor) or config.augment_tonnage_factor <= 0:
        raise ValueError("augment_tonnage_factor must be positive")
    if config.augment_min_syllables < 1:
        raise ValueError("augment_min_syllables must be at least 1")
    rng = np.random.default_rng(config.augment_seed)
    copies: list[QueryRecord] = []
    sources: list[int] = []
    for source_index, record in enumerate(records):
        if not is_hangul_name(record.vessel_name):
            continue
        copies.append(_with_inflated_tonnage(record, config.augment_tonnage_factor))
        copies.append(_without_tonnage(record))
        copies.append(_with_dropped_syllable(record, rng, config.augment_min_syllables))
        sources.extend([source_index, source_index, source_index])
    return copies, sources


def _with_inflated_tonnage(record: QueryRecord, factor: float) -> QueryRecord:
    gross = None if record.gross_tonnage is None else record.gross_tonnage * factor
    international = (
        None if record.international_gross_tonnage is None else record.international_gross_tonnage * factor
    )
    return dataclasses.replace(
        record,
        sample_id=f"{record.sample_id}#aug0",
        gross_tonnage=gross,
        international_gross_tonnage=international,
    )


def _without_tonnage(record: QueryRecord) -> QueryRecord:
    return dataclasses.replace(
        record,
        sample_id=f"{record.sample_id}#aug1",
        gross_tonnage=None,
        international_gross_tonnage=None,
    )


def drop_hangul_syllable(name: str, rng: np.random.Generator, min_syllables: int) -> str:
    """Remove one randomly chosen Hangul syllable when the name has at least min_syllables; else return name unchanged."""
    positions = [
        position
        for position, char in enumerate(name)
        if _SYLLABLE_FIRST <= ord(char) <= _SYLLABLE_LAST
    ]
    if positions and len(positions) >= min_syllables:
        target = positions[int(rng.integers(len(positions)))]
        return name[:target] + name[target + 1 :]
    return name


def _with_dropped_syllable(record: QueryRecord, rng: np.random.Generator, min_syllables: int) -> QueryRecord:
    name = drop_hangul_syllable(record.vessel_name, rng, min_syllables)
    return dataclasses.replace(record, sample_id=f"{record.sample_id}#aug2", vessel_name=name)
