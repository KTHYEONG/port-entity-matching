"""LightGBM lambdarank model over pair features and deterministic top-1 selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from numbers import Integral

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from port_entity_matching.candidates import CandidateSet
from port_entity_matching.config import MatchConfig
from port_entity_matching.features import FEATURE_NAMES

__all__ = [
    "RankerEnsemble",
    "score_ensemble",
    "score_pairs",
    "select_top1",
    "train_ensemble",
    "train_ranker",
]


def train_ranker(
    features: NDArray[np.float32],
    labels: NDArray[np.int64],
    group_sizes: Sequence[int],
    config: MatchConfig,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> lgb.Booster:
    """Fit a lambdarank model; ValueError on size mismatch or a group without exactly one positive."""
    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels).reshape(-1)
    if len(group_sizes) == 0 or any(not isinstance(size, Integral) or size <= 0 for size in group_sizes):
        raise ValueError("group sizes must be positive integers and non-empty")
    sizes = [int(size) for size in group_sizes]
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError(f"features must have {len(feature_names)} columns")
    if matrix.shape[0] != targets.shape[0]:
        raise ValueError("features and labels must have the same number of rows")
    if sum(sizes) != matrix.shape[0]:
        raise ValueError("group sizes must sum to the number of feature rows")
    if np.any((targets != 0) & (targets != 1)):
        raise ValueError("labels must be binary")

    offset = 0
    for size in sizes:
        group = targets[offset : offset + size]
        if int(np.count_nonzero(group == 1)) != 1:
            raise ValueError("every group must contain exactly one positive label")
        offset += size

    dataset = lgb.Dataset(matrix, label=targets, group=sizes, feature_name=list(feature_names))
    params: dict[str, object] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [1],
        "learning_rate": config.lgbm_learning_rate,
        "num_leaves": config.lgbm_num_leaves,
        "min_data_in_leaf": config.lgbm_min_data_in_leaf,
        "feature_fraction": config.lgbm_feature_fraction,
        "seed": config.seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": config.num_threads,
        "verbosity": -1,
    }
    return lgb.train(params, dataset, num_boost_round=config.lgbm_num_rounds)


def score_pairs(booster: lgb.Booster, features: NDArray[np.float32], config: MatchConfig) -> NDArray[np.float64]:
    """Return one float64 relevance score per feature row."""
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    predictions = booster.predict(matrix, num_threads=config.num_threads)
    return np.asarray(predictions, dtype=np.float64).reshape(-1)


def select_top1(scores: NDArray[np.float64], candidate_sets: Sequence[CandidateSet]) -> list[int]:
    """Return the best catalog position per candidate set; equal scores fall back to the lowest position."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != sum(len(candidate_set.positions) for candidate_set in candidate_sets):
        raise ValueError("scores must match the total candidate count")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    selected: list[int] = []
    offset = 0
    for candidate_set in candidate_sets:
        size = len(candidate_set.positions)
        chunk = values[offset : offset + size]
        offset += size
        if size == 0:
            raise ValueError("candidate sets must be non-empty")
        best = float(np.max(chunk))
        tied = np.flatnonzero(chunk == best)
        positions = candidate_set.positions[tied]
        selected.append(int(positions[int(np.argmin(positions))]))
    return selected


@dataclass(frozen=True)
class RankerEnsemble:
    """Seed-bagged lambdarank boosters with per-member score normalization constants."""

    boosters: tuple[lgb.Booster, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject incomplete members or invalid affine normalization constants."""
        if not self.boosters:
            raise ValueError("ensemble must contain at least one member")
        if len(self.boosters) != len(self.means) or len(self.boosters) != len(self.stds):
            raise ValueError("ensemble members and constants must have equal lengths")
        if not all(math.isfinite(mean) for mean in self.means):
            raise ValueError("ensemble means must be finite")
        if not all(math.isfinite(std) and std > 0 for std in self.stds):
            raise ValueError("ensemble stds must be finite and positive")


def train_ensemble(
    features: NDArray[np.float32],
    labels: NDArray[np.int64],
    group_sizes: Sequence[int],
    config: MatchConfig,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> RankerEnsemble:
    """Fit config.ensemble_size lambdarank members; same validation errors as train_ranker."""
    if config.ensemble_size < 1:
        raise ValueError("ensemble_size must be at least 1")
    boosters: list[lgb.Booster] = []
    means: list[float] = []
    stds: list[float] = []
    for member in range(config.ensemble_size):
        member_config = replace(config, seed=config.seed + member)
        booster = train_ranker(features, labels, group_sizes, member_config, feature_names)
        scores = score_pairs(booster, features, member_config)
        means.append(float(np.mean(scores)))
        stds.append(_finite_std(scores))
        boosters.append(booster)
    return RankerEnsemble(boosters=tuple(boosters), means=tuple(means), stds=tuple(stds))


def score_ensemble(
    ensemble: RankerEnsemble,
    features: NDArray[np.float32],
    config: MatchConfig,
) -> NDArray[np.float64]:
    """Mean of per-member normalized scores, one float64 per feature row."""
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    total = np.zeros(matrix.shape[0], dtype=np.float64)
    for booster, mean, std in zip(ensemble.boosters, ensemble.means, ensemble.stds, strict=True):
        total += (score_pairs(booster, matrix, config) - mean) / std
    return total / len(ensemble.boosters)


def _finite_std(scores: NDArray[np.float64]) -> float:
    value = float(np.std(scores))
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    return value
