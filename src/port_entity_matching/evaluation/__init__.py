"""Evaluation utilities: record perturbations and out-of-fold statistics.

This subpackage is development-only; it is never part of the shipped submission.
"""

from port_entity_matching.evaluation.cli import main, parse_overrides
from port_entity_matching.evaluation.hangulize import random_style, transliterate_latin
from port_entity_matching.evaluation.perturb import PERTURBATIONS, UNSEEN_PERTURBATIONS, perturb_record
from port_entity_matching.evaluation.stats import (
    OOF_COLUMNS,
    PairedDifference,
    bucket_scores,
    load_oof,
    paired_difference,
    suite_scores,
    vessel_bootstrap_ci,
)
from port_entity_matching.evaluation.twin import build_twin, reverse_llm, twin_scenarios

__all__ = [
    "OOF_COLUMNS",
    "PERTURBATIONS",
    "UNSEEN_PERTURBATIONS",
    "PairedDifference",
    "bucket_scores",
    "build_twin",
    "load_oof",
    "main",
    "paired_difference",
    "parse_overrides",
    "perturb_record",
    "random_style",
    "reverse_llm",
    "suite_scores",
    "transliterate_latin",
    "twin_scenarios",
    "vessel_bootstrap_ci",
]
