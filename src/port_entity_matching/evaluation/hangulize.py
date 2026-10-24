"""Rule-based English to Hangul transliteration with style randomization.

Port of the evidence script used for the twin evaluation. Every draw is derived
from the caller-provided generator, so identical generator states yield
identical renderings.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np

__all__ = ["random_style", "transliterate_latin"]

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def _syllable(initial: str, medial: str, final: str = "") -> str:
    return chr(0xAC00 + (CHO.index(initial) * 21 + JUNG.index(medial)) * 28 + JONG.index(final))


CONS: dict[str, str] = {
    "B": "ㅂ", "P": "ㅍ", "F": "ㅍ", "V": "ㅂ", "D": "ㄷ", "T": "ㅌ", "G": "ㄱ",
    "K": "ㅋ", "C": "ㅋ", "Q": "ㅋ", "M": "ㅁ", "N": "ㄴ", "L": "ㄹ", "R": "ㄹ",
    "S": "ㅅ", "Z": "ㅈ", "J": "ㅈ", "H": "ㅎ", "X": "ㅋ", "SH": "ㅅ", "CH": "ㅊ",
    "TH": "ㅅ", "DH": "ㄷ", "NG": "ㅇ", "TS": "ㅊ", "DZ": "ㅈ",
}
VOW: dict[str, str] = {
    "A": "ㅏ", "AE": "ㅐ", "E": "ㅔ", "I": "ㅣ", "O": "ㅗ", "U": "ㅜ", "EO": "ㅓ",
    "EU": "ㅡ", "AI": "ㅏ+ㅣ", "EI": "ㅔ+ㅣ", "AU": "ㅏ+ㅜ", "OI": "ㅗ+ㅣ",
    "OU": "ㅗ+ㅜ", "IA": "ㅣ+ㅏ", "IO": "ㅣ+ㅗ", "YA": "ㅑ", "YE": "ㅖ",
    "YO": "ㅛ", "YU": "ㅠ", "WA": "ㅘ", "WE": "ㅞ", "WI": "ㅟ", "WO": "ㅝ",
}
FINAL_OK: dict[str, str] = {
    "ㄱ": "ㄱ", "ㅋ": "ㄱ", "ㄴ": "ㄴ", "ㄹ": "ㄹ", "ㅁ": "ㅁ",
    "ㅂ": "ㅂ", "ㅍ": "ㅂ", "ㅇ": "ㅇ", "ㅅ": "ㅅ", "ㅌ": "ㅅ",
}

DEFAULT_STYLE: dict[str, float] = {
    "final_r_silent": 1.0,
    "er_as_eo": 1.0,
    "magic_e": 1.0,
    "a_as_ae": 0.5,
    "o_as_eo": 0.0,
    "final_stop_keep": 0.7,
    "v_as_b": 1.0,
    "double_s": 0.0,
    "ie_as_i": 1.0,
}

_STYLE_RANGES: tuple[tuple[str, float, float], ...] = (
    ("a_as_ae", 0.0, 1.0),
    ("o_as_eo", 0.0, 0.4),
    ("final_stop_keep", 0.3, 1.0),
    ("ie_as_i", 0.5, 1.0),
    ("double_s", 0.0, 0.5),
)

LETTER_NAMES: dict[str, str] = dict(
    zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ["에이", "비", "씨", "디", "이", "에프", "지", "에이치", "아이", "제이", "케이", "엘", "엠", "엔", "오", "피", "큐", "알", "에스", "티", "유", "브이", "더블유", "엑스", "와이", "제트"],
        strict=True,
    )
)
_VOWEL_SET = frozenset("AEIOUY")


def random_style(rng: np.random.Generator) -> dict[str, float]:
    """Draw one clerk style; ranges as in the evidence script (a_as_ae 0–1, o_as_eo 0–0.4, final_stop_keep 0.3–1, ie_as_i 0.5–1, double_s 0–0.5)."""  # noqa: RUF002
    return {key: float(rng.uniform(low, high)) for key, low, high in _STYLE_RANGES}


def transliterate_latin(name: str, rng: np.random.Generator, style: Mapping[str, float]) -> str:
    """Rule-based Korean loanword rendering of a Latin vessel name; digits and unknown symbols pass through."""
    merged = dict(DEFAULT_STYLE)
    merged.update(style)
    out: list[str] = []
    for token in re.split(r"[\s\-/]+", str(name).upper().strip()):
        cleaned = re.sub(r"[^A-Z0-9]", "", token)
        if not cleaned:
            continue
        if cleaned.isdigit():
            out.append(cleaned)
            continue
        match = re.match(r"([A-Z]+)(\d*)$", cleaned)
        if not match:
            out.append(cleaned)
            continue
        word, digits = match.groups()
        if (len(word) <= 4 and not (set(word) & _VOWEL_SET)) or (
            len(word) <= 3 and len(set(word) & _VOWEL_SET) == 0
        ):
            out.append("".join(LETTER_NAMES[c] for c in word) + digits)
            continue
        out.append(_render(_units(word, rng, merged), rng, merged) + digits)
    return " ".join(out)


def _units(word: str, rng: np.random.Generator, st: Mapping[str, float]) -> list[tuple[str, str]]:
    w = word
    rules = [
        (r"PH", "F"), (r"CK", "K"), (r"GH", ""), (r"QU", "KW"), (r"X", "KS"),
        (r"WR", "R"), (r"KN", "N"), (r"C(?=[EIY])", "S"), (r"G(?=[EIY])", "J"),
        (r"TION", "SHUN"), (r"SION", "SHUN"),
    ]
    for pat, rep in rules:
        w = re.sub(pat, rep, w)
    w = re.sub(r"([BCDFGKLMNPRSTVZ])\1", r"\1", w)
    out: list[tuple[str, str]] = []
    i = 0
    n = len(w)

    def is_vowel(ch: str) -> bool:
        return ch in "AEIOUY"

    while i < n:
        ch = w[i]
        two = w[i : i + 2]
        three = w[i : i + 3]
        if two in ("SH", "CH", "TH", "NG", "TS") and not (two == "NG" and i + 2 < n and is_vowel(w[i + 2])):
            out.append(("C", two))
            i += 2
            continue
        if ch == "Y":
            if i + 1 < n and w[i + 1] in "AEOU":
                out.append(("V", "Y" + w[i + 1] if w[i + 1] != "E" else "YE"))
                i += 2
                continue
            out.append(("V", "I"))
            i += 1
            continue
        if ch == "W":
            if i + 1 < n and w[i + 1] in "AEIO":
                out.append(("V", "W" + w[i + 1]))
                i += 2
                continue
            if i > 0 and is_vowel(w[i - 1]):
                if out and out[-1][0] == "V":
                    out[-1] = ("V", out[-1][1] + "U") if out[-1][1] in ("A", "O", "E") else out[-1]
                i += 1
                continue
            out.append(("V", "U"))
            i += 1
            continue
        if is_vowel(ch):
            if three in ("EAU",):
                out.append(("V", "O"))
                i += 3
                continue
            if two in ("EE", "EA", "IE") and st["ie_as_i"] >= rng.random():
                out.append(("V", "I"))
                i += 2
                continue
            if two in ("OO",):
                out.append(("V", "U"))
                i += 2
                continue
            if two in ("AI", "AY", "EI", "EY"):
                out.append(("V", "EI"))
                i += 2
                continue
            if two in ("OU", "OW"):
                out.append(("V", "AU"))
                i += 2
                continue
            if two in ("AU", "AW"):
                out.append(("V", "O"))
                i += 2
                continue
            if two in ("OI", "OY"):
                out.append(("V", "OI"))
                i += 2
                continue
            if two in ("OA",):
                out.append(("V", "O"))
                i += 2
                continue
            if two in ("IA", "IO") and i + 2 < n:
                out.append(("V", two))
                i += 2
                continue
            if ch == "E" and i == n - 1 and i >= 2 and st["magic_e"] >= rng.random():
                i += 1
                continue
            if ch == "E" and two == "ER" and (i + 2 >= n or not is_vowel(w[i + 2])) and st["er_as_eo"] >= rng.random():
                out.append(("V", "EO"))
                i += 2 if st["final_r_silent"] >= rng.random() else 1
                continue
            if ch == "A" and st["a_as_ae"] > rng.random() and i + 1 < n and not is_vowel(w[i + 1]):
                out.append(("V", "AE"))
                i += 1
                continue
            if ch == "O" and st["o_as_eo"] > rng.random():
                out.append(("V", "EO"))
                i += 1
                continue
            if ch == "U" and i + 1 < n and not is_vowel(w[i + 1]) and rng.random() < 0.4:
                out.append(("V", "EO"))
                i += 1
                continue
            out.append(("V", ch))
            i += 1
            continue
        if ch == "R" and i > 0 and is_vowel(w[i - 1]) and (i + 1 >= n or not is_vowel(w[i + 1])) and st["final_r_silent"] >= rng.random():
            i += 1
            continue
        if ch == "V" and st["v_as_b"] < rng.random():
            out.append(("C", "B"))
            i += 1
            continue
        if ch.isalpha():
            out.append(("C", ch))
        i += 1
    return out


def _render(units: list[tuple[str, str]], rng: np.random.Generator, st: Mapping[str, float]) -> str:
    res: list[str] = []
    i = 0
    n = len(units)
    while i < n:
        kind, unit = units[i]
        if kind == "V":
            parts = VOW.get(unit, "ㅓ").split("+")
            res.append(_syllable("ㅇ", parts[0]))
            res.extend(_syllable("ㅇ", extra) for extra in parts[1:])
            i += 1
            continue
        jamo = CONS.get(unit, "ㅅ")
        nxt = units[i + 1] if i + 1 < n else None
        if nxt is not None and nxt[0] == "V":
            parts = VOW.get(nxt[1], "ㅓ").split("+")
            initial = jamo if jamo != "ㅇ" else "ㅇ"
            if unit == "NG" and res and _can_take_final(res[-1]):
                res[-1] = _add_final(res[-1], "ㅇ")
                initial = "ㄱ"
            if unit == "L" and res and _can_take_final(res[-1]) and rng.random() < 0.9:
                res[-1] = _add_final(res[-1], "ㄹ")
            res.append(_syllable(initial, parts[0]))
            res.extend(_syllable("ㅇ", extra) for extra in parts[1:])
            i += 2
            continue
        prev_open = bool(res) and _can_take_final(res[-1])
        nxt_is_cons = nxt is not None and nxt[0] == "C"
        if prev_open and unit in ("N", "M", "NG", "L", "R") and unit != "R":
            res[-1] = _add_final(res[-1], FINAL_OK.get(jamo, "ㄴ") if unit != "NG" else "ㅇ")
        elif (
            prev_open
            and unit in ("K", "C", "P", "T", "B", "G", "D")
            and (nxt is None or nxt_is_cons)
            and st["final_stop_keep"] > rng.random()
            and jamo in FINAL_OK
        ):
            res[-1] = _add_final(res[-1], FINAL_OK[jamo])
        elif unit in ("SH", "CH", "J", "DZ"):
            res.append(_syllable(jamo, "ㅣ"))
        elif unit == "S" and st["double_s"] > rng.random():
            res.append(_syllable("ㅆ", "ㅡ"))
        else:
            res.append(_syllable(jamo, "ㅡ"))
        i += 1
    return "".join(res)


def _can_take_final(char: str) -> bool:
    code = ord(char) - 0xAC00
    return 0 <= code < 11172 and code % 28 == 0


def _add_final(char: str, final: str) -> str:
    return chr(ord(char) + JONG.index(final))
