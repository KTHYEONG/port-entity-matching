"""Immutable configuration for the vessel matching pipeline.

Every tunable consumed by retrieval, feature building, ranking and cross-validation
lives here so that downstream modules never embed magic numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, cast


@dataclass(frozen=True)
class MatchConfig:
    """Immutable tunables for retrieval, features, ranking and CV."""

    name_top_k: int = 30
    latin_prefilter_k: int = 200
    hangul_key_top_k: int = 30
    tonnage_window: float = 0.06
    hangul_tonnage_window: float = 0.10
    hangul_window_keep: int = 200
    latin_exact_gt_cap: int = 50
    rare_token_max_df: int = 5000
    rare_token_min_len: int = 3
    token_top_k: int = 30
    token_df_cap: int = 2000
    token_match_threshold: float = 0.8
    max_candidates: int = 400
    band_low: float = 0.025
    band_high: float = 0.055
    near_tol: float = 0.01
    query_chunk: int = 256
    batch_size: int = 512
    num_threads: int = 6
    seed: int = 42
    cv_splits: int = 5
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 31
    lgbm_num_rounds: int = 400
    lgbm_min_data_in_leaf: int = 20
    lgbm_feature_fraction: float = 0.9
    augment: bool = True
    augment_tonnage_factor: float = 1.08
    augment_min_syllables: int = 3
    augment_seed: int = 0
    augment_notation: bool = True
    augment_hangul_notation: bool = False
    augment_hangul_tonnage_noise: bool = False
    augment_latin_rate: float = 0.3
    augment_tonnage_low: float = 0.05
    augment_tonnage_high: float = 0.25
    ensemble_size: int = 5
    llm_enabled: bool = True
    llm_fewshot: int = 12
    llm_fewshot_seed: int = 0
    llm_num_beams: int = 5
    llm_max_new_tokens: int = 24
    llm_retrieval_beams: int = 2
    llm_max_name_chars: int = 64
    llm_rerank_enabled: bool = True
    llm_rerank_top_k: int = 10
    llm_rerank_weight: float = 0.2
    reverse_top_k: int = 20
    reverse_pool: int = 200
    reverse_tonnage_top_k: int = 5
    reverse_tonnage_window: float = 0.1
    augment_families: bool = False
    augment_wrong_tonnage_rate: float = 0.0
    augment_type_noise_rate: float = 0.0
    augment_hangul_noise_rate: float = 0.0
    synthetic_latin_rows: int = 0
    synthetic_seed: int = 0
    # Inference-only knobs for Latin records (0 = off / use the shared value); fitting never reads them.
    latin_wide_name_top_k: int = 0
    latin_wide_prefilter_k: int = 0
    latin_wide_token_top_k: int = 0
    latin_wratio_top_k: int = 0
    latin_wratio_tonnage_k: int = 0
    latin_wratio_pool: int = 200
    latin_wratio_tonnage_window: float = 0.1
    # Test vessels are disjoint from train (LB-confirmed), so a ranked train vessel without a call-sign hit is
    # re-picked as the best non-train candidate within the top `exclude_train_top_k`.
    exclude_train_vessels: bool = False
    exclude_train_top_k: int = 50


# Named training recipes: overrides applied on top of MatchConfig defaults. "rev2syn" is the LB-best recipe
# (reverse channel + damage families + 15k synthetic Latin rows); it also needs the rendering files at fit time.
RECIPES: dict[str, dict[str, object]] = {
    "default": {},
    "rev2syn": {
        "augment_families": True,
        "augment_wrong_tonnage_rate": 0.3,
        "augment_type_noise_rate": 0.3,
        "augment_hangul_noise_rate": 0.6,
        "synthetic_latin_rows": 15000,
        "synthetic_seed": 0,
        "latin_wide_name_top_k": 100,
        "latin_wide_prefilter_k": 1000,
        "latin_wide_token_top_k": 60,
        "latin_wratio_top_k": 10,
        "latin_wratio_tonnage_k": 5,
    },
}


def latin_inference_config(config: MatchConfig) -> MatchConfig | None:
    """Config for ranking Latin records at inference, or None when no Latin-only knob is set."""
    if not (
        config.latin_wide_name_top_k
        or config.latin_wide_prefilter_k
        or config.latin_wide_token_top_k
        or config.latin_wratio_top_k
        or config.latin_wratio_tonnage_k
    ):
        return None
    return replace(
        config,
        name_top_k=config.latin_wide_name_top_k or config.name_top_k,
        latin_prefilter_k=config.latin_wide_prefilter_k or config.latin_prefilter_k,
        token_top_k=config.latin_wide_token_top_k or config.token_top_k,
    )


def recipe_config(name: str) -> MatchConfig:
    """MatchConfig for a named recipe; ValueError for an unknown name."""
    if name not in RECIPES:
        raise ValueError(f"Unknown recipe: {name}")
    return config_from_dict(RECIPES[name])


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
