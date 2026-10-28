"""Training-only damage families appended after the notation copies (never used at inference).

Families: truncation (#aug7 Latin, #aug8 Hangul), operator-prefix rename (#aug17 Latin, #aug19 Hangul), notation
variants (#aug10-14 Latin: initialism, token abbreviation, vowel drop, number words, roman numerals; #aug15-16 Hangul:
token cut, vowel swap), wrong tonnage (#aug21), Hangul syllable/jamo noise (#aug22) and vessel-type noise (#aug20).
Every draw uses a generator seeded by crc32(salt|sample_id), so a record's copies never depend on other records.
"""

from __future__ import annotations

import dataclasses
import json
import re
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from port_entity_matching.config import MatchConfig
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.records import QueryRecord

__all__ = ["FamilyContext", "family_copies"]

_SYLLABLE_BASE = 0xAC00
_LEXICON_SIZE = 300
_NUMBER_WORDS = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"]
_ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
_VOWELS = set("AEIOU")
_SMALL_NUMBER_RE = re.compile(r"\b([0-9]|10)\b")
_HANGUL_RE = re.compile("[가-힣]")
# Vowel pairs commonly confused in Hangul loanword spelling: ㅐ<->ㅔ, ㅓ<->ㅗ, ㅜ<->ㅡ (jungseong indices).
_VOWEL_SWAP = {1: 5, 5: 1, 4: 8, 8: 4, 13: 18, 18: 13}


@dataclass(frozen=True)
class FamilyContext:
    """Lexicons and label-side lookups the families draw from (fitted on train + catalog only)."""

    latin_first_tokens: tuple[str, ...]
    hangul_first_tokens: tuple[str, ...]
    vessel_types: tuple[str, ...]
    vessel_type_probs: tuple[float, ...]
    deadweight: dict[str, float]
    target_of: dict[str, str]

    @classmethod
    def build(cls, train: pd.DataFrame, catalog: pd.DataFrame, rename_lexicon: str | Path) -> FamilyContext:
        """Lexicons from catalog names, Hangul first tokens of a catalog rendering JSONL (file order), train types."""
        latin = Counter(
            _upper_spaced(name).split()[0] for name in catalog["vessel_name"] if len(str(name).split()) >= 2
        )
        with Path(rename_lexicon).open(encoding="utf-8") as handle:
            renderings = [json.loads(line)["hangul"] for line in handle]
        hangul = Counter(h.split()[0] for h in renderings if len(h.split()) >= 2 and _HANGUL_RE.search(h.split()[0]))
        counts = train["vessel_type"].value_counts()
        deadweight = pd.to_numeric(catalog["deadweight_tonnage"], errors="coerce")
        return cls(
            latin_first_tokens=tuple(word for word, _ in latin.most_common(_LEXICON_SIZE)),
            hangul_first_tokens=tuple(word for word, _ in hangul.most_common(_LEXICON_SIZE)),
            vessel_types=tuple(str(value) for value in counts.index),
            vessel_type_probs=tuple(float(value) for value in (counts / counts.sum()).to_numpy()),
            deadweight=dict(zip(catalog["vessel_id"].astype(str), deadweight.astype(float), strict=True)),
            target_of=dict(zip(train["sample_id"].astype(str), train["target_vessel_id"].astype(str), strict=True)),
        )


def family_copies(
    records: Sequence[QueryRecord], config: MatchConfig, context: FamilyContext
) -> tuple[list[QueryRecord], list[int]]:
    """Training-only family copies and, per copy, its source index; rows with `#` or a `SYN` prefix are skipped."""
    for rate in (config.augment_wrong_tonnage_rate, config.augment_type_noise_rate, config.augment_hangul_noise_rate):
        if not 0.0 <= rate <= 1.0:
            raise ValueError("family rates must be between 0 and 1")
    copies: list[QueryRecord] = []
    sources: list[int] = []

    def add(index: int, record: QueryRecord) -> None:
        copies.append(record)
        sources.append(index)

    for index, record in enumerate(records):
        sid, name = record.sample_id, record.vessel_name
        if "#" in sid or sid.startswith("SYN"):
            continue
        hangul = is_hangul_name(name)
        if hangul:
            named = [("#aug8", _hangul_truncation(sid, name))]
            named += _hangul_rename(sid, name, context) + _hangul_variants(sid, name)
        else:
            named = [("#aug7", _latin_truncation(sid, name))]
            named += _latin_rename(sid, name, context) + _latin_variants(sid, name)
        for tag, new_name in named:
            if new_name:
                add(index, dataclasses.replace(record, sample_id=sid + tag, vessel_name=new_name))
        rng = _rng("ton", sid)
        if rng.random() < config.augment_wrong_tonnage_rate:
            tonnage = _wrong_tonnage(record, rng, context)
            if tonnage:
                add(
                    index,
                    dataclasses.replace(
                        record, sample_id=sid + "#aug21", gross_tonnage=tonnage, international_gross_tonnage=tonnage
                    ),
                )
        if hangul:
            rng = _rng("kn", sid)
            if rng.random() < config.augment_hangul_noise_rate:
                noisy = _hangul_noise(name, rng)
                if noisy:
                    add(index, dataclasses.replace(record, sample_id=sid + "#aug22", vessel_name=noisy))
        rng = _rng("type", sid)
        if rng.random() < config.augment_type_noise_rate:
            vessel_type: str | None
            alternatives = any(other != record.vessel_type for other in context.vessel_types)
            if rng.random() < 0.4 or not alternatives:
                vessel_type = None
            else:
                vessel_type = record.vessel_type
                while vessel_type == record.vessel_type:
                    pick = rng.choice(len(context.vessel_types), p=np.asarray(context.vessel_type_probs))
                    vessel_type = context.vessel_types[int(pick)]
            add(index, dataclasses.replace(record, sample_id=sid + "#aug20", vessel_type=vessel_type))
    return copies, sources


def _rng(salt: str, sample_id: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{salt}|{sample_id}".encode()))


def _upper_spaced(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).upper()).strip()


def _cut_hangul(name: str, fraction: float) -> str | None:
    chars = [i for i, c in enumerate(name) if not c.isspace()]
    if len(chars) < 4:
        return None
    keep = max(2, int(len(chars) * fraction))
    out = name[: chars[keep - 1] + 1].rstrip()
    return out if out != name else None


def _hangul_truncation(sid: str, name: str, rate: float = 0.3) -> str | None:
    rng = _rng("ktrunc", sid)
    if rng.random() >= rate:
        return None
    return _cut_hangul(name, float(rng.uniform(0.5, 0.9)))


def _latin_truncation(sid: str, name: str, rate: float = 0.2) -> str | None:
    rng = _rng("trunc", sid)
    if rng.random() >= rate or len(name) < 8:
        return None
    if rng.random() < 0.5:
        cut = int(len(name) * rng.uniform(0.5, 0.9))
    else:
        field_length = int(rng.integers(10, 16))
        if len(name) <= field_length:
            return None
        cut = field_length
    out = name[: max(3, cut)].rstrip()
    return out if out != name else None


def _latin_rename(sid: str, name: str, context: FamilyContext, p_prefix: float = 0.15) -> list[tuple[str, str]]:
    rng = _rng("lren", sid)
    tokens = name.split()
    if len(tokens) >= 2 and rng.random() < p_prefix:
        word = context.latin_first_tokens[int(rng.integers(len(context.latin_first_tokens)))]
        if word != tokens[0]:
            return [("#aug17", " ".join([word, *tokens[1:]]))]
    return []


def _hangul_rename(sid: str, name: str, context: FamilyContext, p_prefix: float = 0.15) -> list[tuple[str, str]]:
    rng = _rng("kren", sid)
    tokens = name.split()
    if len(tokens) >= 2 and rng.random() < p_prefix:
        word = context.hangul_first_tokens[int(rng.integers(len(context.hangul_first_tokens)))]
        if word != tokens[0]:
            return [("#aug19", " ".join([word, *tokens[1:]]))]
    return []


def _latin_variants(sid: str, name: str) -> list[tuple[str, str]]:
    tokens = name.split()
    out: list[tuple[str, str]] = []
    rng = _rng("lfam", sid)
    if len(tokens) >= 2 and rng.random() < 0.08:
        k = 2 if len(tokens) == 2 or rng.random() < 0.6 else 3
        rest = (" " + " ".join(tokens[k:])) if len(tokens) > k else ""
        out.append(("#aug10", "".join(w[0] for w in tokens[:k]) + rest))
    candidates = [j for j in range(len(tokens)) if len(tokens[j]) >= 5 and tokens[j].isalpha()]
    if candidates and rng.random() < 0.08:
        j = candidates[int(rng.integers(len(candidates)))]
        length = int(rng.integers(3, 5))
        out.append(("#aug11", " ".join([*tokens[:j], tokens[j][:length], *tokens[j + 1 :]])))
    candidates = [j for j in range(len(tokens)) if len(tokens[j]) >= 4 and tokens[j].isalpha()]
    if candidates and rng.random() < 0.05:
        j = candidates[int(rng.integers(len(candidates)))]
        stripped = tokens[j][0] + "".join(c for c in tokens[j][1:] if c not in _VOWELS)
        out.append(("#aug12", " ".join([*tokens[:j], stripped, *tokens[j + 1 :]])))
    if _SMALL_NUMBER_RE.search(name) and rng.random() < 0.5:
        if rng.random() < 0.5:
            out.append(("#aug13", _SMALL_NUMBER_RE.sub(lambda m: _NUMBER_WORDS[int(m.group(1))], name)))
        else:
            out.append(("#aug14", re.sub(r"\b([1-9]|10)\b", lambda m: _ROMAN[int(m.group(1))], name)))
    return [(tag, new) for tag, new in out if new and new != name]


def _hangul_variants(sid: str, name: str) -> list[tuple[str, str]]:
    tokens = name.split()
    out: list[tuple[str, str]] = []
    rng = _rng("kfam", sid)
    candidates = [j for j in range(len(tokens)) if len(tokens[j]) >= 4]
    if candidates and rng.random() < 0.15:
        j = candidates[int(rng.integers(len(candidates)))]
        out.append(("#aug15", " ".join([*tokens[:j], tokens[j][:2], *tokens[j + 1 :]])))
    syllables = [i for i, c in enumerate(name) if "가" <= c <= "힣"]
    if syllables and rng.random() < 0.1:
        swappable = [i for i in syllables if ((ord(name[i]) - _SYLLABLE_BASE) % 588) // 28 in _VOWEL_SWAP]
        if swappable:
            p = swappable[int(rng.integers(len(swappable)))]
            code = ord(name[p]) - _SYLLABLE_BASE
            cho, jung, jong = code // 588, (code % 588) // 28, code % 28
            swapped = chr(_SYLLABLE_BASE + cho * 588 + _VOWEL_SWAP[jung] * 28 + jong)
            out.append(("#aug16", name[:p] + swapped + name[p + 1 :]))
    return [(tag, new) for tag, new in out if new and new != name]


def _digit_typo(value: float, rng: np.random.Generator) -> float | None:
    digits = list(str(round(value)))
    if len(digits) < 2:
        return None
    op = rng.integers(3)
    p = int(rng.integers(len(digits)))
    if op == 0 and len(digits) > 2:
        del digits[p]
    elif op == 1:
        digits[p] = str(rng.integers(10))
    else:
        p = min(p, len(digits) - 2)
        digits[p], digits[p + 1] = digits[p + 1], digits[p]
    text = "".join(digits).lstrip("0")
    return float(text) if text and float(text) != value else None


def _wrong_tonnage(record: QueryRecord, rng: np.random.Generator, context: FamilyContext) -> float | None:
    """One damaged tonnage: +-10-50% deviation, a digit typo, or the target's catalog DWT."""
    base = record.international_gross_tonnage or record.gross_tonnage
    if not base:
        return None
    family = rng.integers(3)
    if family == 0:
        return float(base * (1 + (1 if rng.random() < 0.5 else -1) * rng.uniform(0.10, 0.50)))
    if family == 1:
        return _digit_typo(base, rng)
    target = context.target_of.get(record.sample_id)
    dwt = context.deadweight.get(target, np.nan) if target is not None else np.nan
    return float(dwt) if dwt and dwt > 0 and abs(np.log(dwt / base)) > 0.1 else None


def _hangul_noise(name: str, rng: np.random.Generator) -> str | None:
    """1-2 Latin-train-style edits on Hangul: syllable delete / duplicate / adjacent swap, or a jamo substitution."""
    text = name
    for _ in range(1 if rng.random() < 0.7 else 2):
        positions = [i for i, c in enumerate(text) if "가" <= c <= "힣"]
        if len(positions) < 3:
            break
        op = rng.integers(4)
        i = positions[int(rng.integers(1, len(positions)))]
        if op == 0:
            text = text[:i] + text[i + 1 :]
        elif op == 1:
            text = text[:i] + text[i] + text[i:]
        elif op == 2:
            j = i - 1
            if "가" <= text[j] <= "힣":
                text = text[:j] + text[i] + text[j] + text[i + 1 :]
        else:
            code = ord(text[i]) - _SYLLABLE_BASE
            cho, jung, jong = code // 588, (code % 588) // 28, code % 28
            part = rng.integers(3)
            if part == 0:
                cho = int(rng.integers(19))
            elif part == 1:
                jung = int(rng.integers(21))
            else:
                jong = int(rng.integers(28))
            text = text[:i] + chr(_SYLLABLE_BASE + cho * 588 + jung * 28 + jong) + text[i + 1 :]
    return text if text != name else None
