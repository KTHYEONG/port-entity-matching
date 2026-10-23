"""Contract tests for the serializable artifact bundle."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from port_entity_matching.artifacts import ArtifactBundle
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.features import FEATURE_NAMES
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.memory import CallSignMemory
from port_entity_matching.ranker import score_ensemble, train_ensemble


def _index() -> CatalogIndex:
    frame = pd.DataFrame(
        [
            {
                "vessel_id": "V0",
                "vessel_name": "Alpha",
                "gross_tonnage": 100,
                "vessel_type": "Container",
                "flag_name": "Panama",
            },
            {
                "vessel_id": "V1",
                "vessel_name": "Beta",
                "gross_tonnage": 200,
                "vessel_type": "Bulk",
                "flag_name": "Liberia",
            },
        ]
    )
    return CatalogIndex.build(frame)


def _bundle(config: MatchConfig | None = None) -> ArtifactBundle:
    settings = config or MatchConfig(lgbm_num_rounds=10, lgbm_min_data_in_leaf=1, num_threads=1)
    index = _index()
    train = pd.DataFrame(
        [
            {"call_sign": "CS_A", "vessel_type": "컨테이너선", "flag_name": "파나마", "target_vessel_id": "V0"},
            {"call_sign": "CS_B", "vessel_type": "벌크선", "flag_name": "라이베리아", "target_vessel_id": "V1"},
        ]
    )
    maps = AttributeMaps.fit(train, index)
    memory = CallSignMemory.fit(train, index)
    features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
    features[0, 0] = 1.0
    features[1, 0] = 0.0
    features[2, 0] = 1.0
    features[3, 0] = 0.0
    labels = np.array([1, 0, 1, 0], dtype=np.int64)
    ranker = train_ensemble(features, labels, [2, 2], settings)
    return ArtifactBundle(ranker=ranker, maps=maps, memory=memory, config=settings)


def _features() -> np.ndarray:
    features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
    features[0, 0] = 1.0
    features[2, 0] = 1.0
    return features


def test_round_trip(tmp_path) -> None:
    bundle = _bundle(MatchConfig(lgbm_num_rounds=10, lgbm_min_data_in_leaf=1, num_threads=1, ensemble_size=2))
    index = _index()
    bundle.save(tmp_path)
    for name in ("model_0.txt", "model_1.txt", "maps.json", "memory.json", "manifest.json"):
        assert (tmp_path / name).is_file()

    restored = ArtifactBundle.load(tmp_path, index)
    assert restored.config == bundle.config
    features = _features()
    assert np.array_equal(
        score_ensemble(restored.ranker, features, restored.config),
        score_ensemble(bundle.ranker, features, bundle.config),
    )
    assert restored.memory.lookup("CS_A") == bundle.memory.lookup("CS_A")
    assert restored.memory.lookup("CS_MISSING") is None
    assert restored.maps.type_logprob("컨테이너선", "Container") == bundle.maps.type_logprob("컨테이너선", "Container")


def test_old_manifest_keeps_stored_threads(tmp_path) -> None:
    bundle = _bundle(MatchConfig(lgbm_num_rounds=2, num_threads=4, ensemble_size=1))
    bundle.save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["config"]["latin_prefilter_k"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    restored = ArtifactBundle.load(tmp_path, _index())
    assert restored.config.num_threads == 4
    assert restored.config.latin_prefilter_k == 200
    assert MatchConfig().num_threads == 6


def test_manifest_records_every_member(tmp_path) -> None:
    bundle = _bundle(MatchConfig(lgbm_num_rounds=10, lgbm_min_data_in_leaf=1, num_threads=1, ensemble_size=3))
    bundle.save(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    section = manifest["ensemble"]
    assert section["files"] == ["model_0.txt", "model_1.txt", "model_2.txt"]
    assert section["means"] == list(bundle.ranker.means)
    assert section["stds"] == list(bundle.ranker.stds)


@pytest.mark.parametrize("feature_count", [39, len(FEATURE_NAMES) - 1])
def test_feature_drift_rejected(tmp_path, feature_count: int) -> None:
    bundle = _bundle()
    bundle.save(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_names"] = list(FEATURE_NAMES[:feature_count])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="feature names"):
        ArtifactBundle.load(tmp_path, _index())


def test_missing_file_rejected(tmp_path) -> None:
    bundle = _bundle()
    bundle.save(tmp_path)
    (tmp_path / "maps.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"maps\.json"):
        ArtifactBundle.load(tmp_path, _index())


def test_missing_member_file_rejected(tmp_path) -> None:
    bundle = _bundle(MatchConfig(lgbm_num_rounds=10, lgbm_min_data_in_leaf=1, num_threads=1, ensemble_size=2))
    bundle.save(tmp_path)
    (tmp_path / "model_1.txt").unlink()
    with pytest.raises(FileNotFoundError, match=r"model_1\.txt"):
        ArtifactBundle.load(tmp_path, _index())


def test_manifest_without_ensemble_rejected(tmp_path) -> None:
    bundle = _bundle()
    bundle.save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["ensemble"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ensemble section"):
        ArtifactBundle.load(tmp_path, _index())


@pytest.mark.parametrize(
    ("files", "means", "stds"),
    [
        ([], [], []),
        (["model_0.txt"], [], []),
        (["model_0.txt", "model_1.txt"], [1.0], [1.0]),
        ("model_0.txt", [1.0], [1.0]),
    ],
)
def test_malformed_ensemble_section_rejected(tmp_path, files, means, stds) -> None:
    bundle = _bundle()
    bundle.save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["ensemble"] = {"files": files, "means": means, "stds": stds}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ensemble section"):
        ArtifactBundle.load(tmp_path, _index())


def test_manifest_must_be_an_object(tmp_path) -> None:
    bundle = _bundle()
    bundle.save(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(["model"]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        ArtifactBundle.load(tmp_path, _index())


def test_manifest_config_must_be_an_object(tmp_path) -> None:
    bundle = _bundle()
    bundle.save(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"] = ["seed"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed config"):
        ArtifactBundle.load(tmp_path, _index())


@pytest.mark.parametrize("feature_names", [42, "is_hangul", None, {}])
def test_malformed_feature_names_rejected(tmp_path, feature_names) -> None:
    _bundle().save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["feature_names"] = feature_names
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="feature names"):
        ArtifactBundle.load(tmp_path, _index())


def test_booster_feature_drift_rejected_even_with_valid_manifest(tmp_path) -> None:
    _bundle().save(tmp_path)
    path = tmp_path / "model_0.txt"
    model = path.read_text(encoding="utf-8")
    path.write_text(model.replace("feature_names=is_hangul ", "feature_names=renamed "), encoding="utf-8")
    with pytest.raises(ValueError, match="Booster feature names"):
        ArtifactBundle.load(tmp_path, _index())


@pytest.mark.parametrize(
    ("key", "value"),
    [("means", None), ("means", "invalid"), ("means", float("nan")),
     ("stds", 0.0), ("stds", -1.0), ("stds", float("inf"))],
)
def test_invalid_manifest_normalization_rejected(tmp_path, key, value) -> None:
    _bundle().save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["ensemble"][key][0] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=r"numeric|finite|positive"):
        ArtifactBundle.load(tmp_path, _index())


def test_fewshot_round_trip(tmp_path) -> None:
    bundle = _bundle()
    fewshot = (("글로비스 스피릿", "GLOVIS SPIRIT"), ("흥아", "HEUNG-A"))
    bundle = ArtifactBundle(
        ranker=bundle.ranker, maps=bundle.maps, memory=bundle.memory, config=bundle.config, fewshot=fewshot
    )
    bundle.save(tmp_path)
    assert (tmp_path / "llm_fewshot.json").is_file()
    pairs = json.loads((tmp_path / "llm_fewshot.json").read_text(encoding="utf-8"))
    assert pairs == [["글로비스 스피릿", "GLOVIS SPIRIT"], ["흥아", "HEUNG-A"]]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm"] == {"fewshot_file": "llm_fewshot.json", "weights_dir": "llm"}
    restored = ArtifactBundle.load(tmp_path, _index())
    assert restored.fewshot == fewshot


def test_legacy_manifest_without_llm_loads_empty_fewshot(tmp_path) -> None:
    bundle = _bundle()
    bundle.save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["llm"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert ArtifactBundle.load(tmp_path, _index()).fewshot == ()


def test_missing_fewshot_file_rejected(tmp_path) -> None:
    bundle = _bundle()
    bundle.save(tmp_path)
    (tmp_path / "llm_fewshot.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"llm_fewshot\.json"):
        ArtifactBundle.load(tmp_path, _index())


def test_malformed_llm_section_rejected(tmp_path) -> None:
    _bundle().save(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["llm"] = []
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed llm section"):
        ArtifactBundle.load(tmp_path, _index())


@pytest.mark.parametrize("pairs", [{}, ["pair"], [["only-one"]], [["가나", 1]]])
def test_malformed_fewshot_rejected(tmp_path, pairs) -> None:
    _bundle().save(tmp_path)
    (tmp_path / "llm_fewshot.json").write_text(json.dumps(pairs), encoding="utf-8")
    with pytest.raises(ValueError, match=r"contain a list|malformed pair"):
        ArtifactBundle.load(tmp_path, _index())
