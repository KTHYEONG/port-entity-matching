"""Catalog Hangul renderings from a train-fitted token dictionary and letter-by-letter acronym spelling.

Train Hangul records aligned token by token with their catalog names give clerk spellings of recurring tokens
(Keoyoung -> 거영, KMTC -> 케이엠티씨); upper-case acronyms without a dictionary entry are spelled letter by letter.
The result is an extra reverse-channel rendering file.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

__all__ = ["build_token_renderings", "catalog_tokens", "read_renderings", "record_tokens", "write_renderings"]

LETTER_NAMES: dict[str, str] = dict(
    zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        [
            "에이",
            "비",
            "씨",
            "디",
            "이",
            "에프",
            "지",
            "에이치",
            "아이",
            "제이",
            "케이",
            "엘",
            "엠",
            "엔",
            "오",
            "피",
            "큐",
            "알",
            "에스",
            "티",
            "유",
            "브이",
            "더블유",
            "엑스",
            "와이",
            "제트",
        ],
        strict=True,
    )
)
_HANGUL_TOKEN_RE = re.compile(r"[가-힣]+")
_FILL_TOKEN_RE = re.compile(r"[가-힣0-9]+")
_ROMAN_RE = re.compile(r"[IVXL]+")


def catalog_tokens(name: str) -> list[str]:
    """Catalog name tokens split on space, hyphen and period, with 'No. 7' reduced to '7'."""
    text = re.sub(r"\bNo\.?\s*(\d+)", r"\1", str(name), flags=re.IGNORECASE)
    return [token for token in re.split(r"[\s\-\.]+", text) if token]


def record_tokens(name: str) -> list[str]:
    """Hangul record tokens with 제N / N호 / trailing 호 removed and digit runs split off."""
    text = re.sub(r"제\s*(\d+)", r" \1 ", str(name))
    text = re.sub(r"(\d+)\s*호", r" \1 ", text)
    text = re.sub(r"(?<=[가-힣])호$", "", text.strip())
    text = re.sub(r"(?<=[가-힣])(?=\d)|(?<=\d)(?=[가-힣])", " ", text)
    return [token for token in text.split() if token]


def _dictionary(train: pd.DataFrame, catalog_names: Mapping[str, str]) -> dict[str, str]:
    hangul = train[train["vessel_name"].astype(str).str.contains("[가-힣]")]
    pairs = hangul.drop_duplicates(["target_vessel_id", "vessel_name"])
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for vessel_id, name in zip(pairs["target_vessel_id"], pairs["vessel_name"], strict=True):
        if vessel_id not in catalog_names:
            continue
        source, target = catalog_tokens(catalog_names[vessel_id]), record_tokens(name)
        if len(source) != len(target):
            continue
        for token, spelled in zip(source, target, strict=True):
            if not token.isdigit() and _HANGUL_TOKEN_RE.fullmatch(spelled):
                table[token.upper()][spelled] += 1
    return {token: counts.most_common(1)[0][0] for token, counts in table.items()}


def _render(name: str, dictionary: Mapping[str, str], existing: list[str]) -> str | None:
    tokens = catalog_tokens(name)
    if not tokens:
        return None
    fixed: list[str | None] = []
    for token in tokens:
        if token.isdigit():
            fixed.append(token)
        elif token.upper() in dictionary:
            fixed.append(dictionary[token.upper()])
        elif (
            token.isupper()
            and token.isalpha()
            and 2 <= len(token) <= 5
            and len(tokens) > 1
            and not _ROMAN_RE.fullmatch(token)
        ):
            fixed.append("".join(LETTER_NAMES[letter] for letter in token))
        else:
            fixed.append(None)
    if all(token is None or token.isdigit() for token in fixed):
        return None
    if None in fixed:
        # Tokens without a dictionary entry are taken from an existing Hangul rendering with the same token count.
        fill = next(
            (
                split
                for split in (record_tokens(rendering) for rendering in existing)
                if len(split) == len(tokens) and all(_FILL_TOKEN_RE.fullmatch(token) for token in split)
            ),
            None,
        )
        if fill is None:
            return None
        fixed = [token if token is not None else fill[slot] for slot, token in enumerate(fixed)]
    rendering = " ".join(token for token in fixed if token is not None)
    return None if rendering in existing else rendering


def build_token_renderings(
    train: pd.DataFrame, catalog: pd.DataFrame, existing: Mapping[str, list[str]]
) -> dict[str, str]:
    """{catalog name: new Hangul rendering} for names the dictionary or acronym rule changes, in catalog order."""
    catalog_names = dict(zip(catalog["vessel_id"], catalog["vessel_name"].astype(str), strict=True))
    dictionary = _dictionary(train, catalog_names)
    out: dict[str, str] = {}
    for name in dict.fromkeys(catalog["vessel_name"].astype(str)):
        rendering = _render(name, dictionary, existing.get(name, []))
        if rendering is not None:
            out[name] = rendering
    return out


def read_renderings(paths: Iterable[str | Path]) -> dict[str, list[str]]:
    """Merge rendering JSONL files into {catalog name: renderings in file order}."""
    merged: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                hangul = entry["hangul"]
                merged[entry["name"]].extend(hangul if isinstance(hangul, list) else [hangul])
    return dict(merged)


def write_renderings(renderings: Mapping[str, str], path: str | Path) -> None:
    """Write {name, hangul} JSONL lines in mapping order."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for name, hangul in renderings.items():
            handle.write(json.dumps({"name": name, "hangul": hangul}, ensure_ascii=False) + "\n")
