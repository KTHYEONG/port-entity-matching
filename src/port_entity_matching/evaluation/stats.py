"""Out-of-fold statistics: bucket scores, vessel-cluster bootstrap CIs and paired comparisons.

Frames are always joined on ``sample_id``; row positions are never used to align
two runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from port_entity_matching.train import DIAGNOSTIC_COLUMNS

__all__ = [
    "OOF_COLUMNS",
    "PairedDifference",
    "bucket_scores",
    "load_oof",
    "paired_difference",
    "vessel_bootstrap_ci",
]

OOF_COLUMNS: tuple[str, ...] = DIAGNOSTIC_COLUMNS

_STRING_COLUMNS: tuple[str, ...] = ("sample_id", "target_vessel_id", "pred_vessel_id", "top2_vessel_id")
_BOOLEAN_COLUMNS: tuple[str, ...] = ("correct", "is_hangul")


@dataclass(frozen=True)
class PairedDifference:
    """Score difference b - a with a vessel-cluster bootstrap CI and per-vessel flip counts."""

    delta: float
    ci_low: float
    ci_high: float
    vessels_fixed: int
    vessels_broken: int
    rows_fixed: int
    rows_broken: int


def load_oof(path: str | Path) -> pd.DataFrame:
    """Read an OOF CSV (clean rows); ValueError on missing columns or duplicate sample_id."""
    frame = pd.read_csv(path, converters=dict.fromkeys(_STRING_COLUMNS, str))
    if "scenario" in frame.columns:
        frame = frame[frame["scenario"].astype(str) == "clean"]
    missing = [column for column in OOF_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"OOF CSV is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    for column in _STRING_COLUMNS:
        frame[column] = frame[column].astype(str)
    for column in _BOOLEAN_COLUMNS:
        frame[column] = frame[column].map(_to_bool)
    if frame["sample_id"].duplicated().any():
        raise ValueError("OOF CSV contains duplicate sample_id values")
    return frame.reset_index(drop=True)


def bucket_scores(oof: pd.DataFrame) -> dict[str, float]:
    """Return {'kor', 'lat', 'score'} accuracies with score = 0.7·kor + 0.3·lat."""
    hangul = oof["is_hangul"].astype(bool).to_numpy()
    correct = oof["correct"].astype(bool).to_numpy()
    kor = float(correct[hangul].mean()) if hangul.any() else math.nan
    lat = float(correct[~hangul].mean()) if (~hangul).any() else math.nan
    score = 0.7 * kor + 0.3 * lat if not (math.isnan(kor) or math.isnan(lat)) else math.nan
    return {"kor": kor, "lat": lat, "score": score}


def vessel_bootstrap_ci(oof: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% percentile CI per bucket and score, resampling vessels (target_vessel_id) with replacement."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    frame = _sorted_by_id(oof)
    codes, vessels = pd.factorize(frame["target_vessel_id"].astype(str))
    n_vessels = len(vessels)
    if n_vessels == 0:
        raise ValueError("OOF frame contains no rows")
    hangul = frame["is_hangul"].astype(bool).to_numpy()
    correct = frame["correct"].astype(bool).to_numpy()
    kor_correct = _counts(codes, (correct & hangul).astype(float), n_vessels)
    kor_total = _counts(codes, hangul.astype(float), n_vessels)
    lat_correct = _counts(codes, (correct & ~hangul).astype(float), n_vessels)
    lat_total = _counts(codes, (~hangul).astype(float), n_vessels)

    rng = np.random.default_rng(seed)
    kor_draws: list[float] = []
    lat_draws: list[float] = []
    score_draws: list[float] = []
    for _ in range(n_boot):
        weights = _counts(rng.integers(0, n_vessels, n_vessels), None, n_vessels)
        kor = _ratio(weights, kor_correct, kor_total)
        lat = _ratio(weights, lat_correct, lat_total)
        kor_draws.append(kor)
        lat_draws.append(lat)
        score_draws.append(0.7 * kor + 0.3 * lat)

    return {
        "kor": _percentile_ci(kor_draws),
        "lat": _percentile_ci(lat_draws),
        "score": _percentile_ci(score_draws),
    }


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> PairedDifference:
    """Compare two OOF frames joined on sample_id; ValueError when their sample_id sets or targets differ."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    left = _aligned(a, b)
    right = _aligned(b, a)
    codes, vessels = pd.factorize(left["target_vessel_id"].astype(str))
    n_vessels = len(vessels)
    if n_vessels == 0:
        raise ValueError("OOF frame contains no rows")
    hangul = left["is_hangul"].astype(bool).to_numpy()
    correct_a = left["correct"].astype(bool).to_numpy()
    correct_b = right["correct"].astype(bool).to_numpy()

    kor_a = _counts(codes, (correct_a & hangul).astype(float), n_vessels)
    kor_b = _counts(codes, (correct_b & hangul).astype(float), n_vessels)
    lat_a = _counts(codes, (correct_a & ~hangul).astype(float), n_vessels)
    lat_b = _counts(codes, (correct_b & ~hangul).astype(float), n_vessels)
    kor_total = _counts(codes, hangul.astype(float), n_vessels)
    lat_total = _counts(codes, (~hangul).astype(float), n_vessels)

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        weights = _counts(rng.integers(0, n_vessels, n_vessels), None, n_vessels)
        score_a = 0.7 * _ratio(weights, kor_a, kor_total) + 0.3 * _ratio(weights, lat_a, lat_total)
        score_b = 0.7 * _ratio(weights, kor_b, kor_total) + 0.3 * _ratio(weights, lat_b, lat_total)
        draws.append(score_b - score_a)

    score_a = bucket_scores(left)["score"]
    score_b = bucket_scores(right)["score"]
    delta = score_b - score_a
    rows_fixed = int(np.count_nonzero(~correct_a & correct_b))
    rows_broken = int(np.count_nonzero(correct_a & ~correct_b))
    per_vessel_a = _counts(codes, correct_a.astype(float), n_vessels)
    per_vessel_b = _counts(codes, correct_b.astype(float), n_vessels)
    vessels_fixed = int(np.count_nonzero(per_vessel_b > per_vessel_a))
    vessels_broken = int(np.count_nonzero(per_vessel_b < per_vessel_a))
    ci_low, ci_high = _percentile_ci(draws)
    return PairedDifference(
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        vessels_fixed=vessels_fixed,
        vessels_broken=vessels_broken,
        rows_fixed=rows_fixed,
        rows_broken=rows_broken,
    )


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text not in {"0", "1", "false", "true"}:
        raise ValueError(f"Invalid OOF boolean: {value!r}")
    return text in {"1", "true"}


def _sorted_by_id(oof: pd.DataFrame) -> pd.DataFrame:
    return oof.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def _aligned(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a["sample_id"].astype(str).duplicated().any() or b["sample_id"].astype(str).duplicated().any():
        raise ValueError("OOF frames contain duplicate sample_id values")
    ids_a = set(a["sample_id"].astype(str))
    ids_b = set(b["sample_id"].astype(str))
    if ids_a != ids_b:
        raise ValueError("OOF frames have different sample_id sets")
    left = a.copy()
    left["sample_id"] = left["sample_id"].astype(str)
    left = left.set_index("sample_id", drop=False)
    right = b.copy()
    right["sample_id"] = right["sample_id"].astype(str)
    right = right.set_index("sample_id", drop=False)
    order = sorted(ids_a)
    left = left.loc[order]
    right = right.loc[order]
    if not (left["target_vessel_id"].astype(str).to_numpy() == right["target_vessel_id"].astype(str).to_numpy()).all():
        raise ValueError("OOF frames disagree on target_vessel_id for the same sample_id")
    if not (left["is_hangul"].astype(bool).to_numpy() == right["is_hangul"].astype(bool).to_numpy()).all():
        raise ValueError("OOF frames disagree on is_hangul for the same sample_id")
    return left.reset_index(drop=True)


def _counts(codes: NDArray[np.intp], weights: NDArray[np.float64] | None, size: int) -> NDArray[np.float64]:
    if weights is None:
        return np.asarray(np.bincount(codes, minlength=size), dtype=np.float64)
    return np.asarray(np.bincount(codes, weights=weights, minlength=size), dtype=np.float64)


def _ratio(weights: NDArray[np.float64], correct: NDArray[np.float64], total: NDArray[np.float64]) -> float:
    denominator = float((weights * total).sum())
    if denominator == 0.0:
        return math.nan
    return float((weights * correct).sum()) / denominator


def _percentile_ci(draws: list[float]) -> tuple[float, float]:
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
