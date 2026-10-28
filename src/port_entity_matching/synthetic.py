"""Catalog-synthetic Latin training rows (training-only data, never used at inference).

Uncurated catalog vessels of port-call types get a broad name-damage mixture (40% clean, 22% one edit, 13% 2-4 edits
with substitutions, 18% token operations, 7% both); attributes are borrowed from a random train Latin row (tonnage
ratios, flag/type through train co-occurrence maps). sample_ids start with ``SYN``.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd

__all__ = ["SYNTHETIC_PREFIX", "synthetic_latin_rows"]

SYNTHETIC_PREFIX = "SYN"

_AZ = string.ascii_uppercase
_NUMBER_WORDS = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"]
_ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
_VOWELS = set("AEIOU")
_COMMON_TOKENS = 300
_RETYPED_TONNAGE_RATE = 0.08
_FLAG_MISSING_RATE = 0.17
_COLUMNS = (
    "sample_id",
    "vessel_name",
    "call_sign",
    "gross_tonnage",
    "international_gross_tonnage",
    "flag_name",
    "vessel_type",
    "arrival_year",
    "target_vessel_id",
)


def synthetic_latin_rows(
    train: pd.DataFrame, catalog: pd.DataFrame, n: int, exclude_ids: Iterable[str], seed: int = 0
) -> pd.DataFrame:
    """Up to n synthetic rows (record columns + target_vessel_id); train targets and exclude_ids are never drawn."""
    rng = np.random.default_rng(seed)
    merged = train.merge(catalog, left_on="target_vessel_id", right_on="vessel_id", suffixes=("", "_c"))
    latin = merged[~merged.vessel_name.str.contains(r"[가-힣]", na=False)].reset_index(drop=True)
    flag_map = (
        latin.dropna(subset=["flag_name"]).groupby("flag_name_c").flag_name.agg(lambda s: s.value_counts().index[0])
    ).to_dict()
    type_map = merged.groupby("vessel_type_c").vessel_type.agg(lambda s: s.value_counts().index[0]).to_dict()
    port_types = set(merged.vessel_type_c)
    common = [
        word
        for word, _ in Counter(w for x in catalog.vessel_name.str.upper() for w in str(x).split()).most_common(
            _COMMON_TOKENS
        )
    ]
    gt = pd.to_numeric(catalog.gross_tonnage, errors="coerce").to_numpy()
    excluded = set(exclude_ids) | set(train.target_vessel_id)
    pool = [
        i
        for i in range(len(catalog))
        if catalog.vessel_id[i] not in excluded
        and gt[i] > 0
        and catalog.vessel_type[i] in port_types
        and len(re.sub(r"[^A-Z0-9]", "", str(catalog.vessel_name[i]).upper())) >= 3
    ]
    pick = rng.choice(pool, size=min(n, len(pool)), replace=False)
    rows = []
    for k, i in enumerate(pick):
        donor = latin.iloc[int(rng.integers(len(latin)))]
        gt_ratio = donor.gross_tonnage / donor.gross_tonnage_c if donor.gross_tonnage_c else np.nan
        igt_ratio = donor.international_gross_tonnage / donor.gross_tonnage_c if donor.gross_tonnage_c else np.nan
        gross, international = gt[i] * gt_ratio, gt[i] * igt_ratio
        if rng.random() < _RETYPED_TONNAGE_RATE:
            factor = 1 + (1 if rng.random() < 0.5 else -1) * rng.uniform(0.1, 0.5)
            gross, international = gross * factor, international * factor
        name = re.sub(r"\s+", " ", str(catalog.vessel_name[i]).upper()).strip()
        rows.append(
            {
                "sample_id": f"{SYNTHETIC_PREFIX}{k:06d}",
                "vessel_name": _damage(name, rng, common),
                "call_sign": None,
                "gross_tonnage": gross,
                "international_gross_tonnage": international,
                "flag_name": None if rng.random() < _FLAG_MISSING_RATE else flag_map.get(catalog.flag_name[i]),
                "vessel_type": type_map.get(catalog.vessel_type[i]),
                "arrival_year": int(rng.integers(2015, 2027)),
                "target_vessel_id": catalog.vessel_id[i],
            }
        )
    return pd.DataFrame(rows, columns=list(_COLUMNS))


def _char_edits(text: str, count: int, rng: np.random.Generator, substitute: bool) -> str:
    chars = list(text)
    for _ in range(count):
        if len(chars) < 3:
            break
        op = rng.integers(4 if substitute else 3)
        p = int(rng.integers(1, len(chars) - 1))
        if op == 0:
            del chars[p]
        elif op == 1:
            chars.insert(p, chars[p])
        elif op == 2:
            chars[p], chars[p + 1] = chars[p + 1], chars[p]
        else:
            chars[p] = _AZ[rng.integers(26)] if chars[p].isalpha() else str(rng.integers(10))
    return "".join(chars)


def _token_op(name: str, rng: np.random.Generator, common: list[str]) -> str:  # noqa: C901, PLR0911
    tokens = name.split()
    ops = ["vowel", "extra", "mv", "nospace"]
    if len(tokens) >= 2:
        ops += ["drop", "swap", "init", "abbrev"]
    if re.search(r"\b([0-9]|10)\b", name):
        ops += ["numw", "roman"]
    op = ops[int(rng.integers(len(ops)))]
    if op == "drop":
        i = int(rng.integers(len(tokens)))
        return " ".join(tokens[:i] + tokens[i + 1 :])
    if op == "swap":
        i = int(rng.integers(len(tokens) - 1))
        tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
        return " ".join(tokens)
    if op == "init":
        return "".join(w[0] for w in tokens[:2]) + (" " + " ".join(tokens[2:]) if len(tokens) > 2 else "")
    if op == "abbrev":
        candidates = [j for j in range(len(tokens)) if len(tokens[j]) >= 5 and tokens[j].isalpha()]
        if not candidates:
            return name
        j = candidates[int(rng.integers(len(candidates)))]
        tokens[j] = tokens[j][: int(rng.integers(3, 5))]
        return " ".join(tokens)
    if op == "vowel":
        candidates = [j for j in range(len(tokens)) if len(tokens[j]) >= 4 and tokens[j].isalpha()]
        if not candidates:
            return name
        j = candidates[int(rng.integers(len(candidates)))]
        tokens[j] = tokens[j][0] + "".join(ch for ch in tokens[j][1:] if ch not in _VOWELS)
        return " ".join(tokens)
    if op == "extra":
        return name + " " + common[int(rng.integers(len(common)))]
    if op == "mv":
        return ("M/V " if rng.random() < 0.5 else "MV ") + name
    if op == "nospace":
        return name.replace(" ", "")
    if op == "numw":
        return re.sub(r"\b([0-9]|10)\b", lambda m: _NUMBER_WORDS[int(m.group(1))], name)
    return re.sub(r"\b([1-9]|10)\b", lambda m: _ROMAN[int(m.group(1))], name)


def _damage(name: str, rng: np.random.Generator, common: list[str]) -> str:
    u = rng.random()
    if u < 0.40:
        return name
    if u < 0.62:
        return _char_edits(name, 1, rng, rng.random() < 0.3)
    if u < 0.75:
        return _char_edits(name, int(rng.integers(2, 5)), rng, True)
    if u < 0.93:
        return _token_op(name, rng, common)
    return _char_edits(_token_op(name, rng, common), int(rng.integers(1, 3)), rng, True)
