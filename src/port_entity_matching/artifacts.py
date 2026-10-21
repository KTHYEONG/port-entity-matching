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
from port_entity_matching.ranker import RankerEnsemble
from port_entity_matching.version import __version__

__all__ = ["ArtifactBundle"]

MEMBER_FILE_TEMPLATE = "model_{index}.txt"
MAPS_FILE = "maps.json"
MEMORY_FILE = "memory.json"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class ArtifactBundle:
    """Everything inference needs besides the catalog."""

    ranker: RankerEnsemble
    maps: AttributeMaps
    memory: CallSignMemory
    config: MatchConfig

    def save(self, model_dir: str | Path) -> None:
        """Write the ensemble members, attribute maps, call-sign memory and manifest."""
        destination = Path(model_dir)
        destination.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        for index, booster in enumerate(self.ranker.boosters):
            name = MEMBER_FILE_TEMPLATE.format(index=index)
            booster.save_model(str(destination / name))
            files.append(name)
        self.maps.to_json(destination / MAPS_FILE)
        self.memory.to_json(destination / MEMORY_FILE)
        manifest = {
            "feature_names": list(FEATURE_NAMES),
            "config": config_to_dict(self.config),
            "package_version": __version__,
            "ensemble": {
                "files": files,
                "means": list(self.ranker.means),
                "stds": list(self.ranker.stds),
            },
        }
        (destination / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, model_dir: str | Path, index: CatalogIndex) -> ArtifactBundle:
        """Load a saved bundle; ValueError on feature drift, FileNotFoundError on missing files."""
        source = Path(model_dir)
        for name in (MAPS_FILE, MEMORY_FILE, MANIFEST_FILE):
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
        section = manifest.get("ensemble")
        if not isinstance(section, dict):
            raise ValueError(f"Artifact manifest {source / MANIFEST_FILE} has no ensemble section")
        files = section.get("files")
        means = section.get("means")
        stds = section.get("stds")
        if not isinstance(files, list) or not isinstance(means, list) or not isinstance(stds, list):
            raise ValueError("Artifact manifest ensemble section is malformed")
        if not files or len(files) != len(means) or len(files) != len(stds):
            raise ValueError("Artifact manifest ensemble section must be non-empty and of equal length")
        try:
            member_means = tuple(float(value) for value in means)
            member_stds = tuple(float(value) for value in stds)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Artifact manifest ensemble constants must be numeric") from error
        boosters = []
        for name in files:
            path = source / str(name)
            if not path.is_file():
                raise FileNotFoundError(f"Artifact file not found: {path}")
            booster = lgb.Booster(model_file=str(path))
            if booster.feature_name() != list(FEATURE_NAMES):
                raise ValueError("Booster feature names do not match the current feature set")
            boosters.append(booster)
        ranker = RankerEnsemble(
            boosters=tuple(boosters),
            means=member_means,
            stds=member_stds,
        )
        maps = AttributeMaps.from_json(source / MAPS_FILE)
        memory = CallSignMemory.from_json(source / MEMORY_FILE, index)
        return cls(ranker=ranker, maps=maps, memory=memory, config=config_from_dict(raw_config))
