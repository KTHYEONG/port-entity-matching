"""Serializable bundle of every artifact inference needs besides the catalog."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb

from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig, config_from_dict, config_to_dict
from port_entity_matching.features import feature_names
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.memory import CallSignMemory
from port_entity_matching.ranker import RankerEnsemble
from port_entity_matching.reverse import ReverseChannel
from port_entity_matching.version import __version__

__all__ = ["ArtifactBundle"]

MEMBER_FILE_TEMPLATE = "model_{index}.txt"
MAPS_FILE = "maps.json"
MEMORY_FILE = "memory.json"
MANIFEST_FILE = "manifest.json"
FEWSHOT_FILE = "llm_fewshot.json"
LLM_WEIGHTS_DIR = "llm"


@dataclass(frozen=True)
class ArtifactBundle:
    """Everything inference needs besides the catalog."""

    ranker: RankerEnsemble
    maps: AttributeMaps
    memory: CallSignMemory
    config: MatchConfig
    fewshot: tuple[tuple[str, str], ...] = ()
    reverse: ReverseChannel | None = None

    def save(self, model_dir: str | Path) -> None:
        """Write the ensemble members, attribute maps, call-sign memory, rendering files and manifest."""
        destination = Path(model_dir)
        destination.mkdir(parents=True, exist_ok=True)
        reverse_files: list[str] = []
        if self.reverse is not None:
            for source in self.reverse.files:
                if source.resolve() != (destination / source.name).resolve():
                    shutil.copyfile(source, destination / source.name)
                reverse_files.append(source.name)
            if len(set(reverse_files)) != len(reverse_files):
                raise ValueError("Rendering file names must be unique")
        files: list[str] = []
        for index, booster in enumerate(self.ranker.boosters):
            name = MEMBER_FILE_TEMPLATE.format(index=index)
            booster.save_model(str(destination / name))
            files.append(name)
        self.maps.to_json(destination / MAPS_FILE)
        self.memory.to_json(destination / MEMORY_FILE)
        (destination / FEWSHOT_FILE).write_text(
            json.dumps([list(pair) for pair in self.fewshot], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest = {
            "feature_names": list(feature_names(self.reverse is not None)),
            "config": config_to_dict(self.config),
            "package_version": __version__,
            "ensemble": {
                "files": files,
                "means": list(self.ranker.means),
                "stds": list(self.ranker.stds),
            },
            "llm": {"fewshot_file": FEWSHOT_FILE, "weights_dir": LLM_WEIGHTS_DIR},
        }
        if reverse_files:
            manifest["reverse"] = {"files": reverse_files}
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
        reverse_section = manifest.get("reverse")
        reverse_names: list[str] = []
        if reverse_section is not None:
            raw_files = reverse_section.get("files") if isinstance(reverse_section, dict) else None
            if not isinstance(raw_files, list) or not raw_files or not all(isinstance(f, str) for f in raw_files):
                raise ValueError(f"Artifact manifest {source / MANIFEST_FILE} has a malformed reverse section")
            reverse_names = raw_files
        expected_names = list(feature_names(bool(reverse_names)))
        if manifest.get("feature_names") != expected_names:
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
            if booster.feature_name() != expected_names:
                raise ValueError("Booster feature names do not match the current feature set")
            boosters.append(booster)
        ranker = RankerEnsemble(
            boosters=tuple(boosters),
            means=member_means,
            stds=member_stds,
        )
        maps = AttributeMaps.from_json(source / MAPS_FILE)
        memory = CallSignMemory.from_json(source / MEMORY_FILE, index)
        llm_section = manifest.get("llm")
        if llm_section is None:
            fewshot: tuple[tuple[str, str], ...] = ()
        else:
            if not isinstance(llm_section, dict):
                raise ValueError(f"Artifact manifest {source / MANIFEST_FILE} has a malformed llm section")
            fewshot_name = llm_section.get("fewshot_file", FEWSHOT_FILE)
            fewshot_path = source / str(fewshot_name)
            if not fewshot_path.is_file():
                raise FileNotFoundError(f"Artifact file not found: {fewshot_path}")
            raw_pairs = json.loads(fewshot_path.read_text(encoding="utf-8"))
            if not isinstance(raw_pairs, list):
                raise ValueError(f"Artifact fewshot file {fewshot_path} must contain a list")
            parsed: list[tuple[str, str]] = []
            for entry in raw_pairs:
                if (
                    not isinstance(entry, list)
                    or len(entry) != 2
                    or not isinstance(entry[0], str)
                    or not isinstance(entry[1], str)
                ):
                    raise ValueError(f"Artifact fewshot file {fewshot_path} has a malformed pair")
                parsed.append((entry[0], entry[1]))
            fewshot = tuple(parsed)
        reverse = ReverseChannel.load(index, [source / name for name in reverse_names]) if reverse_names else None
        return cls(
            ranker=ranker,
            maps=maps,
            memory=memory,
            config=config_from_dict(raw_config),
            fewshot=fewshot,
            reverse=reverse,
        )
