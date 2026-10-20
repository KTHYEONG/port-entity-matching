"""Serializable bundle of every artifact inference needs besides the catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb

from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig, config_from_dict, config_to_dict
from port_entity_matching.features import FEATURE_NAMES
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.memory import CallSignMemory
from port_entity_matching.version import __version__

__all__ = ["ArtifactBundle"]

MODEL_FILE = "model.txt"
MAPS_FILE = "maps.json"
MEMORY_FILE = "memory.json"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class ArtifactBundle:
    """Everything inference needs besides the catalog."""

    booster: lgb.Booster
    maps: AttributeMaps
    memory: CallSignMemory
    config: MatchConfig

    def save(self, model_dir: str | Path) -> None:
        """Write the booster, attribute maps, call-sign memory and manifest."""
        destination = Path(model_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(destination / MODEL_FILE))
        self.maps.to_json(destination / MAPS_FILE)
        self.memory.to_json(destination / MEMORY_FILE)
        manifest = {
            "feature_names": list(FEATURE_NAMES),
            "config": config_to_dict(self.config),
            "package_version": __version__,
        }
        (destination / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, model_dir: str | Path, index: CatalogIndex) -> ArtifactBundle:
        """Load a saved bundle; ValueError on feature drift, FileNotFoundError on missing files."""
        source = Path(model_dir)
        for name in (MODEL_FILE, MAPS_FILE, MEMORY_FILE, MANIFEST_FILE):
            if not (source / name).is_file():
                raise FileNotFoundError(f"Artifact file not found: {source / name}")
        manifest = json.loads((source / MANIFEST_FILE).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Artifact manifest {source / MANIFEST_FILE} must contain a JSON object")
        feature_names = manifest.get("feature_names")
        if feature_names != list(FEATURE_NAMES):
            raise ValueError("Artifact manifest feature names do not match the current feature set")
        raw_config = manifest.get("config", {})
        if not isinstance(raw_config, dict):
            raise ValueError(f"Artifact manifest {source / MANIFEST_FILE} has a malformed config section")
        booster = lgb.Booster(model_file=str(source / MODEL_FILE))
        if booster.feature_name() != list(FEATURE_NAMES):
            raise ValueError("Booster feature names do not match the current feature set")
        maps = AttributeMaps.from_json(source / MAPS_FILE)
        memory = CallSignMemory.from_json(source / MEMORY_FILE, index)
        return cls(booster=booster, maps=maps, memory=memory, config=config_from_dict(raw_config))
