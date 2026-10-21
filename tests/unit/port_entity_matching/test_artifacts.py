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
from port_entity_matching.ranker import score_pairs, train_ranker


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
    booster = train_ranker(features, labels, [2, 2], settings)
    return ArtifactBundle(booster=booster, maps=maps, memory=memory, config=settings)


def test_round_trip(tmp_path) -> None:
    bundle = _bundle()
    index = _index()
    bundle.save(tmp_path)
    for name in ("model.txt", "maps.json", "memory.json", "manifest.json"):
        assert (tmp_path / name).is_file()

    restored = ArtifactBundle.load(tmp_path, index)
    assert restored.config == bundle.config
    features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
    assert np.array_equal(score_pairs(restored.booster, features, restored.config), score_pairs(bundle.booster, features, bundle.config))
    assert restored.memory.lookup("CS_A") == bundle.memory.lookup("CS_A")
    assert restored.memory.lookup("CS_MISSING") is None
    assert restored.maps.type_logprob("컨테이너선", "Container") == bundle.maps.type_logprob("컨테이너선", "Container")


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
    (tmp_path / "model.txt").unlink()
    with pytest.raises(FileNotFoundError, match=r"model\.txt"):
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
    path = tmp_path / "model.txt"
    model = path.read_text(encoding="utf-8")
    path.write_text(model.replace("feature_names=is_hangul ", "feature_names=renamed "), encoding="utf-8")
    with pytest.raises(ValueError, match="Booster feature names"):
        ArtifactBundle.load(tmp_path, _index())
