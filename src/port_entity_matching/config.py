"""Immutable configuration for the vessel matching pipeline.

Every tunable consumed by retrieval, feature building, ranking and cross-validation
lives here so that downstream modules never embed magic numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, cast


@dataclass(frozen=True)
class MatchConfig:
    """Immutable tunables for retrieval, features, ranking and CV."""

    name_top_k: int = 30
    hangul_key_top_k: int = 30
    tonnage_window: float = 0.06
    hangul_tonnage_window: float = 0.10
    hangul_window_keep: int = 200
    latin_exact_gt_cap: int = 50
    rare_token_max_df: int = 50
    rare_token_min_len: int = 3
    max_candidates: int = 400
    band_low: float = 0.025
    band_high: float = 0.055
    near_tol: float = 0.01
    query_chunk: int = 256
    batch_size: int = 512
    num_threads: int = 4
    seed: int = 42
    cv_splits: int = 5
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 31
    lgbm_num_rounds: int = 400
    lgbm_min_data_in_leaf: int = 20
    lgbm_feature_fraction: float = 0.9


def config_to_dict(config: MatchConfig) -> dict[str, int | float]:
    """Return a JSON-serializable mapping of all fields."""
    return cast("dict[str, int | float]", asdict(config))


def config_from_dict(data: Mapping[str, object]) -> MatchConfig:
    """Build a config from a mapping; unknown keys raise ValueError, missing keys use defaults."""
    known = {field.name for field in fields(MatchConfig)}
    unknown = sorted(str(key) for key in data if key not in known)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    values = {key: value for key, value in data.items() if key in known}
    return MatchConfig(**cast("dict[str, Any]", values))
