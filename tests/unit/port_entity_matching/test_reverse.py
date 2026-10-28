"""Contract tests for the reverse (catalog Hangul rendering) channel."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from port_entity_matching.artifacts import ArtifactBundle
from port_entity_matching.candidates import generate_candidates
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.features import FEATURE_NAMES, build_features, feature_names
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.pipeline import Matcher
from port_entity_matching.records import QueryRecord, records_from_frame
from port_entity_matching.reverse import REVERSE_FEATURE_NAMES, ReverseChannel, jamo_string
from port_entity_matching.train import fit_bundle

_CATALOG = [
    ("V0", "Glovis Spirit", 5000.0),
    ("V1", "Grand Ace", 1000.0),
    ("V2", "Grand Ace", 9000.0),
    ("V3", "Calida", 61332.0),
    ("V4", "Kashima Maru", 8000.0),
]
_RENDERINGS = {
    "Glovis Spirit": "글로비스 스피릿",
    "Grand Ace": "그랜드 에이스",
    "Calida": "칼리다",
    "Kashima Maru": "카시마 마루",
}


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "vessel_id": vid,
                "vessel_type": "Tanker",
                "vessel_name": name,
                "size": gt,
                "unit": "DWT",
                "deadweight_tonnage": gt,
                "gross_tonnage": gt,
                "flag_name": "Panama",
            }
            for vid, name, gt in _CATALOG
        ]
    )


def _write(path: Path, mapping: dict[str, object]) -> Path:
    path.write_text(
        "".join(json.dumps({"name": n, "hangul": h}, ensure_ascii=False) + "\n" for n, h in mapping.items()),
        encoding="utf-8",
    )
    return path


def _record(sample_id: str, name: str, tonnage: float | None) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "vessel_name": name,
        "call_sign": None,
        "gross_tonnage": tonnage,
        "international_gross_tonnage": tonnage,
        "flag_name": "파나마",
        "vessel_type": "Tanker",
        "arrival_year": 2020,
    }


def _records(rows: list[dict[str, object]]) -> list[QueryRecord]:
    return records_from_frame(pd.DataFrame(rows))


@pytest.fixture
def channel(tmp_path: Path) -> ReverseChannel:
    return ReverseChannel.load(CatalogIndex.build(_catalog()), [_write(tmp_path / "rev_a.jsonl", _RENDERINGS)])


def test_jamo_string_decomposes_and_drops_spaces() -> None:
    assert jamo_string("가 각") == "ㄱㅏㄱㅏㄱ"
    assert jamo_string("ace 7") == "ACE7"


def test_load_requires_existing_files(tmp_path: Path) -> None:
    index = CatalogIndex.build(_catalog())
    with pytest.raises(ValueError, match="at least one"):
        ReverseChannel.load(index, [])
    with pytest.raises(FileNotFoundError):
        ReverseChannel.load(index, [tmp_path / "missing.jsonl"])


def test_hangul_row_gains_rendering_candidates_latin_row_unchanged(channel: ReverseChannel) -> None:
    config = MatchConfig()
    index = CatalogIndex.build(_catalog())
    records = _records([_record("H", "칼리다", 61332.0), _record("L", "CALIDA", 61332.0)])
    base = generate_candidates(records, index, None, config)
    extended = channel.extend(records, base, config)
    assert 3 in extended[0].positions.tolist()
    assert list(extended[0].positions) == sorted(extended[0].positions)
    assert extended[1] is base[1]


def test_columns_rank_exact_rendering_first_and_gate_tonnage(channel: ReverseChannel) -> None:
    config = MatchConfig()
    record = _records([_record("H", "그랜드 에이스", 1000.0)])[0]
    positions = np.array([0, 1, 2], dtype=np.int64)
    columns = channel.columns(record, positions, config, hangul=True)
    assert columns["rev_jamo_sim"][1] == pytest.approx(1.0)
    assert columns["rev_jamo_rank"][1] == 0
    assert columns["rev_ton_rank"][1] == 0
    assert math.isnan(columns["rev_ton_rank"][2])
    latin = channel.columns(record, positions, config, hangul=False)
    assert all(np.isnan(latin[name]).all() for name in REVERSE_FEATURE_NAMES)


def test_similarity_is_max_over_files_and_variants(tmp_path: Path) -> None:
    index = CatalogIndex.build(_catalog())
    first = _write(tmp_path / "rev_a.jsonl", _RENDERINGS)
    second = _write(tmp_path / "rev_b.jsonl", {"Calida": ["깔리다", "카리다"]})
    channel = ReverseChannel.load(index, [first, second])
    record = _records([_record("H", "카리다", None)])[0]
    sims = channel.columns(record, np.array([3], dtype=np.int64), MatchConfig(), hangul=True)["rev_jamo_sim"]
    assert sims[0] == pytest.approx(1.0)


def test_feature_matrix_appends_reverse_columns(channel: ReverseChannel) -> None:
    config = MatchConfig()
    index = CatalogIndex.build(_catalog())
    records = _records([_record("H", "칼리다", 61332.0)])
    sets = generate_candidates(records, index, None, config, channel)
    maps = AttributeMaps.fit(pd.DataFrame(columns=["vessel_type", "flag_name", "target_vessel_id"]), index)
    matrix = build_features(records, sets, index, maps, config, channel)
    assert feature_names(True) == FEATURE_NAMES + REVERSE_FEATURE_NAMES
    assert matrix.shape[1] == len(feature_names(True))


def _train() -> pd.DataFrame:
    names = [("칼리다", "V3", 61332.0), ("글로비스 스피릿", "V0", 5000.0), ("CALIDA", "V3", 61332.0)]
    names += [("그랜드 에이스", "V1", 1000.0), ("GRAND ACE", "V2", 9000.0), ("카시마 마루", "V4", 8000.0)]
    rows = [
        _record(f"T{i}", n, gt) | {"target_vessel_id": t, "call_sign": f"CS{i}"} for i, (n, t, gt) in enumerate(names)
    ]
    return pd.DataFrame(rows)


def test_bundle_round_trip_and_row_independence(tmp_path: Path, channel: ReverseChannel) -> None:
    config = MatchConfig(lgbm_num_rounds=10, lgbm_min_data_in_leaf=1, num_threads=1, ensemble_size=1)
    bundle = fit_bundle(_train(), _catalog(), config, reverse=channel)
    out = tmp_path / "model"
    bundle.save(out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reverse"] == {"files": ["rev_a.jsonl"]}
    assert manifest["feature_names"] == list(feature_names(True))
    assert (out / "rev_a.jsonl").is_file()

    matcher = Matcher.from_dir(out, _catalog())
    assert matcher.bundle.reverse is not None
    rows = [_record("R0", "칼리다", 61332.0), _record("R1", "그랜드 에이스", 1000.0), _record("R2", "CALIDA", None)]
    batch = matcher.predict(_records(rows))
    alone = [matcher.predict(_records([row]))[0] for row in rows]
    shuffled = matcher.predict(_records([rows[2], rows[0], rows[1]]))
    assert batch == alone
    assert shuffled == [batch[2], batch[0], batch[1]]


def test_reverse_section_without_reverse_features_is_rejected(tmp_path: Path, channel: ReverseChannel) -> None:
    config = MatchConfig(lgbm_num_rounds=5, lgbm_min_data_in_leaf=1, num_threads=1, ensemble_size=1)
    fit_bundle(_train(), _catalog(), config).save(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reverse"] = {"files": ["rev_a.jsonl"]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="feature names"):
        ArtifactBundle.load(tmp_path, CatalogIndex.build(_catalog()))


def test_duplicate_rendering_file_names_rejected(tmp_path: Path) -> None:
    index = CatalogIndex.build(_catalog())
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _write(tmp_path / "a" / "rev.jsonl", _RENDERINGS)
    second = _write(tmp_path / "b" / "rev.jsonl", _RENDERINGS)
    channel = ReverseChannel.load(index, [first, second])
    config = MatchConfig(lgbm_num_rounds=5, lgbm_min_data_in_leaf=1, num_threads=1, ensemble_size=1)
    bundle = fit_bundle(_train(), _catalog(), config, reverse=channel)
    with pytest.raises(ValueError, match="unique"):
        bundle.save(tmp_path / "model")


@pytest.mark.parametrize("section", [{"files": []}, {"files": [1]}, ["rev_a.jsonl"]])
def test_malformed_reverse_section_rejected(tmp_path: Path, channel: ReverseChannel, section: object) -> None:
    config = MatchConfig(lgbm_num_rounds=5, lgbm_min_data_in_leaf=1, num_threads=1, ensemble_size=1)
    fit_bundle(_train(), _catalog(), config, reverse=channel).save(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reverse"] = section
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="reverse section"):
        ArtifactBundle.load(tmp_path, CatalogIndex.build(_catalog()))


def _cli_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from port_entity_matching import config as config_module
    from port_entity_matching.transliterate import CachedTransliterator

    monkeypatch.setattr(CachedTransliterator, "translate", lambda self, names: dict.fromkeys(names, ("CALIDA",)))
    small = {
        **config_module.RECIPES["rev2syn"],
        "synthetic_latin_rows": 2,
        "lgbm_num_rounds": 5,
        "lgbm_min_data_in_leaf": 1,
        "num_threads": 1,
        "ensemble_size": 1,
    }
    monkeypatch.setitem(config_module.RECIPES, "rev2syn", small)
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    (llm_dir / "config.json").write_text("{}", encoding="utf-8")
    _train().to_csv(tmp_path / "train.csv", index=False)
    _catalog().to_csv(tmp_path / "catalog.csv", index=False)
    rendering = _write(tmp_path / "rev_a.jsonl", _RENDERINGS)
    (tmp_path / "exclude.txt").write_text("V9\n\n", encoding="utf-8")
    return [
        "--train", str(tmp_path / "train.csv"),
        "--catalog", str(tmp_path / "catalog.csv"),
        "--llm-model", str(llm_dir),
        "--llm-cache", str(tmp_path / "cache"),
        "--recipe", "rev2syn",
        "--reverse", str(rendering),
        "--synthetic-exclude", str(tmp_path / "exclude.txt"),
    ]  # fmt: skip


def test_cli_fits_recipe_with_reverse_channel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from port_entity_matching.train import main

    args = _cli_inputs(tmp_path, monkeypatch)
    out = tmp_path / "model"
    assert main([*args, "--rename-lexicon", str(tmp_path / "rev_a.jsonl"), "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reverse"] == {"files": ["rev_a.jsonl"]}
    assert manifest["config"]["augment_families"] is True


def test_cli_rejects_incomplete_recipe_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from port_entity_matching.train import main

    args = _cli_inputs(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        main([*args, "--out", str(tmp_path / "model")])
    with pytest.raises(SystemExit):
        main([*args, "--rename-lexicon", str(tmp_path / "rev_a.jsonl"), "--cv"])
