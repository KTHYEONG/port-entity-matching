"""Deterministic text canonicalization for Latin and Hangul vessel names.

The module exposes two key spaces that are meant to be comparable:

* a Latin key space built from consonant classes of romanized names, and
* a Hangul key space built from the jamo decomposition of Korean names.

Both spaces share the same class alphabet, so a Korean rendering of a vessel name
can be compared against its Latin catalog entry with a single edit distance.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

__all__ = [
    "canonical_latin",
    "decode_letter_acronym",
    "digit_tokens",
    "hangul_phonetic_key",
    "hangul_variants",
    "is_hangul_name",
    "latin_phonetic_key",
    "latin_tokens",
    "normalize_text",
]

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9A-Z]+")
_DIGIT_RUN_RE = re.compile(r"\d+")
_HANGUL_SYLLABLE_RE = re.compile(r"[\uac00-\ud7a3]")
_HANGUL_TOKEN_RE = re.compile(r"^[\uac00-\ud7a3]+$")
_VESSEL_PREFIX_RE = re.compile(r"^(?:M\s*/\s*V|M\.?\s*V\.?)\s+")
_COLLAPSE_RE = re.compile(r"(.)\1+")

_CHOSEONG = (
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)
_JUNGSEONG = (
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
_JONGSEONG = (
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)

_INITIAL_CLASS = {
    "ㄱ": "K",
    "ㄲ": "K",
    "ㅋ": "K",
    "ㄴ": "N",
    "ㄷ": "T",
    "ㄸ": "T",
    "ㅌ": "T",
    "ㄹ": "R",
    "ㅁ": "M",
    "ㅂ": "P",
    "ㅃ": "P",
    "ㅍ": "P",
    "ㅅ": "S",
    "ㅆ": "S",
    "ㅈ": "J",
    "ㅉ": "J",
    "ㅊ": "J",
    "ㅇ": "",
    "ㅎ": "",
}
_FINAL_CLASS = {
    "": "",
    "ㄱ": "K",
    "ㄲ": "K",
    "ㅋ": "K",
    "ㄴ": "N",
    "ㄷ": "T",
    "ㅌ": "T",
    "ㄹ": "R",
    "ㅁ": "M",
    "ㅂ": "P",
    "ㅍ": "P",
    "ㅅ": "S",
    "ㅆ": "S",
    "ㅇ": "N",
    "ㅈ": "J",
    "ㅊ": "J",
    "ㅎ": "",
}
_COMPOUND_FINAL_HEAD = {
    "ㄳ": "ㄱ",
    "ㄵ": "ㄴ",
    "ㄶ": "ㄴ",
    "ㄺ": "ㄹ",
    "ㄻ": "ㄹ",
    "ㄼ": "ㄹ",
    "ㄽ": "ㄹ",
    "ㄾ": "ㄹ",
    "ㄿ": "ㄹ",
    "ㅀ": "ㄹ",
    "ㅄ": "ㅂ",
}
_VOWEL_CLASS = {
    "ㅏ": "A",
    "ㅐ": "E",
    "ㅑ": "YA",
    "ㅒ": "YE",
    "ㅓ": "O",
    "ㅔ": "E",
    "ㅕ": "YO",
    "ㅖ": "YE",
    "ㅗ": "O",
    "ㅘ": "WA",
    "ㅙ": "WE",
    "ㅚ": "WE",
    "ㅛ": "YO",
    "ㅜ": "U",
    "ㅝ": "WO",
    "ㅞ": "WE",
    "ㅟ": "WI",
    "ㅠ": "YU",
    "ㅡ": "",
    "ㅢ": "I",
    "ㅣ": "I",
}
_LATIN_CLASS = {
    "B": "P",
    "P": "P",
    "F": "P",
    "V": "P",
    "C": "K",
    "G": "K",
    "K": "K",
    "Q": "K",
    "D": "T",
    "T": "T",
    "L": "R",
    "R": "R",
    "M": "M",
    "N": "N",
    "S": "S",
    "Z": "J",
    "J": "J",
    "H": "",
    "W": "",
    "Y": "",
}
_VOWELS = "AEIOU"
_LATIN_REWRITES: tuple[tuple[str, str], ...] = (
    ("TION", "SON"),
    ("SION", "SON"),
    ("PH", "P"),
    ("CH", "J"),
    ("SH", "S"),
    ("TH", "S"),
    ("CK", "K"),
    ("QU", "K"),
    ("X", "KS"),
    ("C(?=[EIY])", "S"),
    ("G(?=[EIY])", "J"),
    ("GH", ""),
    ("NG", "N"),
)

_LETTER_UNITS: dict[str, str] = {
    "에이": "A",
    "비": "B",
    "씨": "C",
    "시": "C",
    "디": "D",
    "이": "E",
    "에프": "F",
    "지": "G",
    "쥐": "G",
    "에이치": "H",
    "에치": "H",
    "아이": "I",
    "제이": "J",
    "케이": "K",
    "엘": "L",
    "엠": "M",
    "엔": "N",
    "앤": "N",
    "오": "O",
    "피": "P",
    "큐": "Q",
    "알": "R",
    "에스": "S",
    "에쓰": "S",
    "티": "T",
    "유": "U",
    "브이": "V",
    "뷔": "V",
    "더블유": "W",
    "더블류": "W",
    "엑스": "X",
    "와이": "Y",
    "제트": "Z",
}
_UNITS_BY_LENGTH_DESC: tuple[str, ...] = tuple(
    sorted(_LETTER_UNITS, key=lambda unit: (-len(unit), unit)),
)

_JE_NUMBER_RE = re.compile(r"제\s*(\d+)\s*호?")
_NUMBER_HO_RE = re.compile(r"(\d+)\s*호")
_ALPHANUMERIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def normalize_text(value: object) -> str:
    """NFKC + uppercase + whitespace-collapsed string; non-str/NaN → ''."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).upper()
    return _WHITESPACE_RE.sub(" ", text).strip()


def is_hangul_name(value: object) -> bool:
    """True iff the value contains at least one precomposed Hangul syllable (U+AC00-U+D7A3)."""
    if not isinstance(value, str):
        return False
    return _HANGUL_SYLLABLE_RE.search(value) is not None


def canonical_latin(value: object) -> str:
    """Compact [0-9A-Z] key of a Latin name with vessel prefixes removed."""
    text = _strip_vessel_prefix(normalize_text(value))
    return _NON_ALNUM_RE.sub("", text)


def latin_tokens(value: object) -> tuple[str, ...]:
    """Ordered [0-9A-Z]+ tokens of a Latin name with vessel prefixes removed."""
    text = _strip_vessel_prefix(normalize_text(value))
    return tuple(token for token in _NON_ALNUM_RE.split(text) if token)


def digit_tokens(value: object) -> tuple[str, ...]:
    """Ordered digit runs with leading zeros stripped ('0' stays '0')."""
    text = normalize_text(value)
    return tuple(run.lstrip("0") or "0" for run in _DIGIT_RUN_RE.findall(text))


def decode_letter_acronym(token: str) -> str | None:
    """Latin letters if the whole Hangul token spells English letter names (>=2 letters), else None."""
    if not isinstance(token, str) or not _HANGUL_TOKEN_RE.fullmatch(token):
        return None
    lengths = _acronym_segment_lengths(token)
    if lengths is None:
        return None
    letters = []
    position = 0
    for length in lengths:
        unit = token[position : position + length]
        letters.append(_LETTER_UNITS[unit])
        position += length
    if len(letters) < 2:
        return None
    return "".join(letters)


def hangul_variants(value: object) -> tuple[str, ...]:
    """Deduplicated literal variants of a Hangul name after numbering, suffix and acronym rules."""
    text = normalize_text(value)
    if not text:
        return ()
    text = _JE_NUMBER_RE.sub(r" NO \1 ", text)
    text = _NUMBER_HO_RE.sub(r" \1 ", text)
    tokens = text.split()
    literals = []
    for index, word in enumerate(tokens):
        if index > 0 and word == "투":
            literals.append("II")
            continue
        decoded = decode_letter_acronym(word)
        literals.append(decoded if decoded else word)
    base = " ".join(literals)
    variants = [base]
    without_suffix = _drop_trailing_ho(base)
    if without_suffix is not None:
        variants.append(without_suffix)
    return tuple(dict.fromkeys(variants))


def latin_phonetic_key(value: str, with_vowels: bool) -> str:
    """Consonant-class (optionally vowel) key of Latin text shared with the Hangul key space."""
    text = normalize_text(value)
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    compact = "".join(char for char in stripped.upper() if char in _ALPHANUMERIC)
    for pattern, replacement in _LATIN_REWRITES:
        compact = re.sub(pattern, replacement, compact)
    key = []
    for char in compact:
        if char in _VOWELS:
            if with_vowels:
                key.append(char)
        elif char.isdigit():
            key.append(char)
        else:
            key.append(_LATIN_CLASS.get(char, char))
    return _collapse("".join(key))


def hangul_phonetic_key(literal: str, with_vowels: bool) -> str:
    """Phonetic key of a Hangul literal (may contain Latin runs and digits) in the shared key space."""
    text = normalize_text(literal)
    key: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if _is_hangul_syllable(char):
            initial, medial, final = _decompose_syllable(char)
            key.append(_INITIAL_CLASS[initial])
            if with_vowels:
                key.append(_VOWEL_CLASS[medial])
            key.append(_FINAL_CLASS[_COMPOUND_FINAL_HEAD.get(final, final)])
            index += 1
        elif "A" <= char <= "Z":
            end = index
            while end < len(text) and "A" <= text[end] <= "Z":
                end += 1
            key.append(latin_phonetic_key(text[index:end], with_vowels))
            index = end
        elif char.isdigit():
            key.append(char)
            index += 1
        else:
            index += 1
    return _collapse("".join(key))


def _strip_vessel_prefix(text: str) -> str:
    return _VESSEL_PREFIX_RE.sub("", text, count=1)


def _collapse(text: str) -> str:
    return _COLLAPSE_RE.sub(r"\1", text)


def _is_hangul_syllable(char: str) -> bool:
    return "가" <= char <= "힣"


def _decompose_syllable(char: str) -> tuple[str, str, str]:
    code = ord(char) - 0xAC00
    return _CHOSEONG[code // 588], _JUNGSEONG[(code % 588) // 28], _JONGSEONG[code % 28]


def _drop_trailing_ho(base: str) -> str | None:
    """Return the name without its trailing 호 suffix, or None when the suffix is part of the name."""
    if not base.endswith("호"):
        return None
    candidate = base[:-1].strip()
    if len(_HANGUL_SYLLABLE_RE.findall(candidate)) < 2:
        return None
    return candidate


@lru_cache(maxsize=4096)
def _acronym_segment_lengths(token: str) -> tuple[int, ...] | None:
    """Return unit lengths of the preferred full segmentation of a Hangul token."""
    best: list[tuple[int, tuple[int, ...]] | None] = [None] * (len(token) + 1)
    best[0] = (0, ())
    for position in range(len(token)):
        current = best[position]
        if current is None:
            continue
        for unit in _UNITS_BY_LENGTH_DESC:
            if not token.startswith(unit, position):
                continue
            end = position + len(unit)
            candidate = (current[0] + 1, current[1] + (len(unit),))
            existing = best[end]
            candidate_rank = _segment_rank(candidate)
            existing_rank = _segment_rank(existing) if existing is not None else None
            if existing_rank is None or candidate_rank < existing_rank:
                best[end] = candidate
    final = best[len(token)]
    if final is None:
        return None
    return final[1]


def _segment_rank(segment: tuple[int, tuple[int, ...]]) -> tuple[int, tuple[int, ...]]:
    """Rank segmentations by fewest letters, then by longest unit at the earliest position."""
    return (segment[0], tuple(-length for length in segment[1]))
