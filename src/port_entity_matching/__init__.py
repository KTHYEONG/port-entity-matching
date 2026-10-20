"""Port entity matching package."""

from port_entity_matching.artifacts import ArtifactBundle
from port_entity_matching.candidates import (
    SRC_FALLBACK,
    SRC_MEMORY,
    SRC_NAME,
    SRC_TOKEN,
    SRC_TONNAGE,
    CandidateSet,
    generate_candidates,
)
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig, config_from_dict, config_to_dict
from port_entity_matching.features import FEATURE_NAMES, build_features
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.memory import CallSignMemory
from port_entity_matching.pipeline import Matcher, Prediction, run_submission
from port_entity_matching.ranker import score_pairs, select_top1, train_ranker
from port_entity_matching.version import __version__

__all__ = [
    "FEATURE_NAMES",
    "SRC_FALLBACK",
    "SRC_MEMORY",
    "SRC_NAME",
    "SRC_TOKEN",
    "SRC_TONNAGE",
    "ArtifactBundle",
    "AttributeMaps",
    "CallSignMemory",
    "CandidateSet",
    "CatalogIndex",
    "MatchConfig",
    "Matcher",
    "Prediction",
    "__version__",
    "build_features",
    "config_from_dict",
    "config_to_dict",
    "generate_candidates",
    "run_submission",
    "score_pairs",
    "select_top1",
    "train_ranker",
]
