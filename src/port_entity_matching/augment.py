"""Training-only degraded copies of Hangul records.

The copies mimic the distortions observed in the port-call records (inflated
tonnage, missing tonnage, dropped Hangul syllable) so the ranker learns to
cope with them. They are used for training only, never at inference time.

The module also exposes the notation-distortion helpers (token reversal, token
dropping and character edits). They are pure apart from the caller-supplied
RNG and are shared by the training and evaluation paths.
"""

from __future__ import annotations

import dataclasses
import math
import string
from collections.abc import Sequence

import numpy as np

from port_entity_matching.config import MatchConfig
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.records import QueryRecord

__all__ = [
    "char_edits",
    "degraded_copies",
    "drop_hangul_syllable",
    "drop_token",
    "reverse_tokens",
]

_SYLLABLE_FIRST = 0xAC00
_SYLLABLE_LAST = 0xD7A3

_ASCII_UPPERCASE: str = string.ascii_uppercase
_MIN_EDIT_LETTERS = 4


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


def reverse_tokens(name: str) -> str:
    """Whitespace tokens of name in reverse order joined by single spaces; names with < 2 tokens are returned unchanged."""
    tokens = name.split()
    if len(tokens) < 2:
        return name
    return " ".join(reversed(tokens))


def drop_token(name: str, rng: np.random.Generator) -> str:
    """Remove one uniformly chosen whitespace token; names with < 2 tokens are returned unchanged."""
    tokens = name.split()
    if len(tokens) < 2:
        return name
    dropped = int(rng.integers(len(tokens)))
    return " ".join(token for position, token in enumerate(tokens) if position != dropped)


def char_edits(name: str, rng: np.random.Generator, count: int) -> str:
    """Apply count random edits (delete, duplicate, transpose, substitute) at ASCII-letter positions; names with < 4 letters are returned unchanged."""
    if count < 0:
        raise ValueError("count must not be negative")
    text = name
    for _ in range(count):
        positions = [position for position, char in enumerate(text) if _is_ascii_letter(char)]
        if len(positions) < _MIN_EDIT_LETTERS:
            break
        position = positions[int(rng.integers(len(positions)))]
        operation = int(rng.integers(4))
        if operation == 0:
            text = text[:position] + text[position + 1 :]
        elif operation == 1:
            text = f"{text[: position + 1]}{text[position]}{text[position + 1 :]}"
        elif operation == 2:
            text = _transpose(text, position)
        else:
            text = _substitute(text, position, rng)
    return text


def _is_ascii_letter(char: str) -> bool:
    """Report whether char is an ASCII letter (A-Z or a-z)."""
    return ("A" <= char <= "Z") or ("a" <= char <= "z")


def _transpose(text: str, position: int) -> str:
    """Swap the character at position with the next one, or with the previous one at the end of the string."""
    if position + 1 < len(text):
        return f"{text[:position]}{text[position + 1]}{text[position]}{text[position + 2 :]}"
    return f"{text[: position - 1]}{text[position]}{text[position - 1]}{text[position + 1 :]}"


def _substitute(text: str, position: int, rng: np.random.Generator) -> str:
    """Replace the character at position with a uniformly drawn uppercase letter different from the original one."""
    original = text[position].upper()
    alphabet = [letter for letter in _ASCII_UPPERCASE if letter != original]
    replacement = alphabet[int(rng.integers(len(alphabet)))]
    return f"{text[:position]}{replacement}{text[position + 1 :]}"
