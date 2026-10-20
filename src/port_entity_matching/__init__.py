"""Port entity matching package."""

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
from port_entity_matching.memory import CallSignMemory

__version__ = "0.1.0"

__all__ = [
    "SRC_FALLBACK",
    "SRC_MEMORY",
    "SRC_NAME",
    "SRC_TOKEN",
    "SRC_TONNAGE",
    "CallSignMemory",
    "CandidateSet",
    "CatalogIndex",
    "MatchConfig",
    "config_from_dict",
    "config_to_dict",
    "generate_candidates",
]
