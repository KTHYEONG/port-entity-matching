"""Evaluation utilities: record perturbations and out-of-fold statistics.

This subpackage is development-only; it is never part of the shipped submission.
"""

from port_entity_matching.evaluation.perturb import PERTURBATIONS, perturb_record
from port_entity_matching.evaluation.stats import (
    OOF_COLUMNS,
    PairedDifference,
    bucket_scores,
    load_oof,
    paired_difference,
    vessel_bootstrap_ci,
)

__all__ = [
    "OOF_COLUMNS",
    "PERTURBATIONS",
    "PairedDifference",
    "bucket_scores",
    "load_oof",
    "paired_difference",
    "perturb_record",
    "vessel_bootstrap_ci",
]
