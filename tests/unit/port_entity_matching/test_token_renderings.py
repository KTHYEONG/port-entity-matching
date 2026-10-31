from __future__ import annotations

import pandas as pd

from port_entity_matching.token_renderings import (
    build_token_renderings,
    catalog_tokens,
    read_renderings,
    record_tokens,
    write_renderings,
)


def _catalog(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"vessel_id": [f"V{i}" for i in range(len(names))], "vessel_name": names})


def _train(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame({"vessel_name": [p[0] for p in pairs], "target_vessel_id": [p[1] for p in pairs]})


def test_tokenizers_normalize_numbers_and_suffixes() -> None:
    assert catalog_tokens("Ocean Pioneer No. 2") == ["Ocean", "Pioneer", "2"]
    assert catalog_tokens("Heung-A Xiamen") == ["Heung", "A", "Xiamen"]
    assert record_tokens("제2오션 파이오니어호") == ["2", "오션", "파이오니어"]


def test_dictionary_tokens_render_unseen_vessels_of_the_same_fleet() -> None:
    catalog = _catalog(["Keoyoung Star", "Keoyoung Blue 5"])
    train = _train([("거영스타", "V0")])
    existing = {"Keoyoung Blue 5": ["케이유영 블루 5"]}
    out = build_token_renderings(_train([("거영 스타", "V0")]), catalog, existing)
    assert out["Keoyoung Blue 5"] == "거영 블루 5"
    assert "Keoyoung Star" in out
    assert build_token_renderings(train, catalog, {}) == {}


def test_acronyms_are_spelled_and_roman_numerals_are_not() -> None:
    catalog = _catalog(["SITC Weihai", "Spiekeroog III"])
    existing = {"SITC Weihai": ["시트씨 웨이하이"], "Spiekeroog III": ["스피케로그 3"]}
    out = build_token_renderings(_train([]), catalog, existing)
    assert out == {"SITC Weihai": "에스아이티씨 웨이하이"}


def test_existing_rendering_is_not_duplicated() -> None:
    catalog = _catalog(["YN Busan"])
    assert build_token_renderings(_train([]), catalog, {"YN Busan": ["와이엔 부산"]}) == {}


def test_names_without_an_aligned_fill_are_skipped() -> None:
    catalog = _catalog(["DAT Venus Star"])
    assert build_token_renderings(_train([]), catalog, {"DAT Venus Star": ["닷 비너스"]}) == {}


def test_round_trip(tmp_path) -> None:
    path = tmp_path / "rev.jsonl"
    write_renderings({"YN Busan": "와이엔 부산"}, path)
    assert read_renderings([path]) == {"YN Busan": ["와이엔 부산"]}


def test_targets_outside_the_catalog_and_empty_names_are_ignored() -> None:
    catalog = _catalog(["", "Keoyoung Blue 5"])
    out = build_token_renderings(_train([("거영 스타", "V9")]), catalog, {"Keoyoung Blue 5": ["케이유영 블루 5"]})
    assert out == {}
