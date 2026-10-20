"""Contract tests for deterministic name canonicalization."""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein
import pytest

from port_entity_matching.normalize import (
    canonical_latin,
    decode_letter_acronym,
    digit_tokens,
    hangul_phonetic_key,
    hangul_variants,
    is_hangul_name,
    latin_phonetic_key,
    latin_tokens,
    normalize_text,
)


def test_nfkc_and_roman_numerals() -> None:
    assert canonical_latin("AL JABBRIYAH Ⅱ") == "ALJABBRIYAHII"


def test_nbsp_catalog_name() -> None:
    assert latin_tokens("Kai\xa0Sheng\xa0You 16") == ("KAI", "SHENG", "YOU", "16")


def test_vessel_prefix_stripped() -> None:
    assert canonical_latin("M/V DENVER") == "DENVER"
    assert canonical_latin("M.V. DENVER") == "DENVER"
    assert canonical_latin("MV. DENVER") == "DENVER"
    assert canonical_latin("MVELO") == "MVELO"


def test_hyphen_equals_space() -> None:
    assert canonical_latin("STOLT-CEDAR") == canonical_latin("Stolt Cedar")


def test_non_string_input_is_safe() -> None:
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    assert canonical_latin(None) == ""
    assert latin_tokens(None) == ()
    assert digit_tokens(float("nan")) == ()
    assert is_hangul_name(None) is False
    assert hangul_variants(None) == ()
    assert decode_letter_acronym(None) is None


def test_hangul_detection() -> None:
    assert is_hangul_name("흥아 사라") is True
    assert is_hangul_name("SEA V") is False


def test_digit_tokens_across_scripts() -> None:
    assert digit_tokens("제7하나") == ("7",)
    assert digit_tokens("No. 7 Hana") == ("7",)
    assert digit_tokens("WAN XIANG 0988") == ("988",)
    assert digit_tokens("000") == ("0",)


def test_acronym_decoding() -> None:
    assert decode_letter_acronym("케이엠티씨") == "KMTC"
    assert decode_letter_acronym("에스티아이") == "STI"
    assert decode_letter_acronym("에스아이티씨") == "SITC"
    assert decode_letter_acronym("에이스") is None
    assert decode_letter_acronym("오션") is None
    assert decode_letter_acronym("엑스") is None


def test_numbering_rules() -> None:
    variants = hangul_variants("제2흥아 파이오니어호")

    assert "NO 2" in variants[0]
    assert any(not variant.endswith("호") for variant in variants)
    assert len(variants) == len(set(variants))


def test_trailing_ho_dropped_when_suffix() -> None:
    assert hangul_variants("성호 에이스호") == ("성호 에이스호", "성호 에이스")


def test_suffix_kept_when_part_of_name() -> None:
    assert hangul_variants("성호") == ("성호",)


def test_roman_numeral_token_replaced() -> None:
    assert hangul_variants("제3호 투")[0] == "NO 3 II"


def test_loanword_keys_align() -> None:
    hangul_key = hangul_phonetic_key("리플렉션", with_vowels=False)
    latin_key = latin_phonetic_key("Reflection", with_vowels=False)

    assert Levenshtein.distance(hangul_key, latin_key) <= 1


def test_acronym_shares_key_space() -> None:
    variant = hangul_variants("케이엠티씨 서울")[0]

    hangul_key = hangul_phonetic_key(variant, with_vowels=False)
    latin_key = latin_phonetic_key("KMTC Seoul", with_vowels=False)

    assert hangul_key.startswith(latin_phonetic_key("KMTC", with_vowels=False))
    assert hangul_key == latin_key


def test_vowel_keys_differ_from_consonant_keys() -> None:
    assert latin_phonetic_key("SEOUL", with_vowels=True) != latin_phonetic_key("SEOUL", with_vowels=False)
    assert hangul_phonetic_key("서울", with_vowels=True) != hangul_phonetic_key("서울", with_vowels=False)


def test_determinism() -> None:
    name = "제2 흥아 파이오니어 호"

    assert canonical_latin(name) == canonical_latin(name)
    assert latin_tokens(name) == latin_tokens(name)
    assert digit_tokens(name) == digit_tokens(name)
    assert hangul_variants(name) == hangul_variants(name)
    assert hangul_phonetic_key(name, with_vowels=False) == hangul_phonetic_key(name, with_vowels=False)
    assert latin_phonetic_key(name, with_vowels=False) == latin_phonetic_key(name, with_vowels=False)


def test_acronym_requires_entire_hangul_token() -> None:
    assert decode_letter_acronym("케이엠\n") is None
    assert decode_letter_acronym("케이엠A") is None
    assert decode_letter_acronym("에이치비") == "HB"
    assert decode_letter_acronym("더블유비") == "WB"


@pytest.mark.parametrize("value", [None, float("nan"), 7, "", "  "])
def test_empty_and_nonstring_normalization(value: object) -> None:
    assert normalize_text(value) == ""
    assert canonical_latin(value) == ""
    assert latin_tokens(value) == ()
    assert digit_tokens(value) == ()
    assert hangul_variants(value) == ()
    assert is_hangul_name(value) is False
    assert latin_phonetic_key(value, False) == ""
    assert hangul_phonetic_key(value, True) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [("TION", "SN"), ("SION", "SN"), ("PH", "P"), ("CH", "J"),
     ("SH", "S"), ("TH", "S"), ("CK", "K"), ("QU", "K"),
     ("X", "KS"), ("CE", "S"), ("GE", "J"), ("GH", ""), ("NG", "N")],
)
def test_ordered_latin_rewrites(value: str, expected: str) -> None:
    assert latin_phonetic_key(value, False) == expected


def test_variant_token_position_and_unicode_boundaries() -> None:
    assert hangul_variants("투 서울 투") == ("투 서울 II",)
    assert hangul_variants("가호") == ("가호",)
    assert hangul_phonetic_key("값 닭 12!", False) == "KPTR12"
    assert latin_phonetic_key("Réflection", False) == latin_phonetic_key("Reflection", False)
    assert canonical_latin("MVELO 한글") == "MVELO"
    assert latin_tokens("M/V DENVER") == ("DENVER",)
