"""Out-of-fold statistics: bucket scores, vessel-cluster bootstrap CIs and paired comparisons.

Frames are always joined on ``sample_id``; row positions are never used to align
two runs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from port_entity_matching.evaluation.perturb import UNSEEN_PERTURBATIONS
from port_entity_matching.train import DIAGNOSTIC_COLUMNS

__all__ = [
    "OOF_COLUMNS",
    "PairedDifference",
    "bucket_scores",
    "load_oof",
    "paired_difference",
    "suite_scores",
    "vessel_bootstrap_ci",
]

OOF_COLUMNS: tuple[str, ...] = DIAGNOSTIC_COLUMNS

_STRING_COLUMNS: tuple[str, ...] = ("sample_id", "target_vessel_id", "pred_vessel_id", "top2_vessel_id")
_BOOLEAN_COLUMNS: tuple[str, ...] = ("correct", "is_hangul")

_SUITE_GROUPS: tuple[str, ...] = ("all", "seen", "unseen")


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


def load_oof(path: str | Path, scenario: str = "clean") -> pd.DataFrame:
    """Read an OOF CSV, keeping the rows of one scenario when a scenario column exists; ValueError on missing columns, duplicate sample_id, or no rows for the scenario."""
    frame = pd.read_csv(path, converters=dict.fromkeys(_STRING_COLUMNS, str))
    missing = [column for column in OOF_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"OOF CSV is missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    if "scenario" in frame.columns:
        frame = frame[frame["scenario"].astype(str) == scenario]
    elif scenario != "clean":
        raise ValueError(f"OOF CSV has no scenario column, so scenario '{scenario}' cannot be selected")
    for column in _STRING_COLUMNS:
        frame[column] = frame[column].astype(str)
    for column in _BOOLEAN_COLUMNS:
        frame[column] = frame[column].map(_to_bool)
    if frame["sample_id"].duplicated().any():
        raise ValueError("OOF CSV contains duplicate sample_id values")
    if len(frame) == 0:
        raise ValueError(f"OOF CSV contains no rows for scenario '{scenario}'")
    return frame.reset_index(drop=True)


def bucket_scores(oof: pd.DataFrame) -> dict[str, float]:
    """Return {'kor', 'lat', 'score'} accuracies with score = 0.7·kor + 0.3·lat."""
    hangul = oof["is_hangul"].astype(bool).to_numpy()
    correct = oof["correct"].astype(bool).to_numpy()
    kor = float(correct[hangul].mean()) if hangul.any() else math.nan
    lat = float(correct[~hangul].mean()) if (~hangul).any() else math.nan
    score = 0.7 * kor + 0.3 * lat if not (math.isnan(kor) or math.isnan(lat)) else math.nan
    return {"kor": kor, "lat": lat, "score": score}


def suite_scores(oof_long: pd.DataFrame, scenarios: Sequence[str]) -> dict[str, dict[str, float | int]]:
    """Mean perturbed accuracy per bucket over clean plus the given scenarios, split into 'seen', 'unseen' and 'all' groups."""
    if "scenario" in oof_long.columns:
        scenario_values = oof_long["scenario"].astype(str).to_numpy()
    else:
        scenario_values = np.full(len(oof_long), "clean", dtype=object)
    correct = oof_long["correct"].astype(bool).to_numpy()
    hangul = oof_long["is_hangul"].astype(bool).to_numpy()

    kor_items: dict[str, float] = {}
    lat_items: dict[str, float] = {}
    clean_rows = scenario_values == "clean"
    if bool(clean_rows.any()):
        clean_hangul = clean_rows & hangul
        clean_latin = clean_rows & ~hangul
        if bool(clean_hangul.any()):
            kor_items["clean"] = float(correct[clean_hangul].mean())
        if bool(clean_latin.any()):
            lat_items["clean"] = float(correct[clean_latin].mean())

    selected: list[str] = []
    for name in dict.fromkeys(scenarios):
        bucket = _scenario_bucket(name)
        rows = scenario_values == name
        if not bool(rows.any()):
            continue
        selected.append(name)
        if bucket == "kor":
            kor_items[name] = float(correct[rows].mean())
        else:
            lat_items[name] = float(correct[rows].mean())

    groups: dict[str, list[str]] = {
        "all": selected,
        "seen": [name for name in selected if name not in UNSEEN_PERTURBATIONS],
        "unseen": [name for name in selected if name in UNSEEN_PERTURBATIONS],
    }
    return {group: _suite_group_scores(groups[group], kor_items, lat_items) for group in _SUITE_GROUPS}


def _scenario_bucket(name: str) -> str:
    """Return the bucket of a scenario name from its prefix; ValueError on an unknown prefix."""
    if name.startswith("kor_"):
        return "kor"
    if name.startswith("lat_"):
        return "lat"
    raise ValueError(f"Unknown scenario bucket for '{name}': expected a 'kor_' or 'lat_' prefix")


def _suite_group_scores(
    names: Sequence[str],
    kor_items: dict[str, float],
    lat_items: dict[str, float],
) -> dict[str, float | int]:
    """Average the per-scenario items of one suite group, including the clean item."""
    kor_values = [kor_items[key] for key in ("clean", *names) if key in kor_items]
    lat_values = [lat_items[key] for key in ("clean", *names) if key in lat_items]
    kor = float(np.mean(kor_values)) if kor_values else math.nan
    lat = float(np.mean(lat_values)) if lat_values else math.nan
    score = 0.7 * kor + 0.3 * lat if not (math.isnan(kor) or math.isnan(lat)) else math.nan
    return {
        "kor": kor,
        "lat": lat,
        "score": score,
        "n_kor": len(kor_values),
        "n_lat": len(lat_values),
    }


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
    """Compare two OOF frames joined on sample_id; ValueError when their sample_id sets or targets differ.

    The score is the bucket-weighted competition metric when both scripts are present.
    A single-script selection (one perturbation scenario) falls back to the plain
    mean accuracy, because the weighted score is undefined without the other bucket.
    """
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
    weighted = bool(hangul.any()) and bool((~hangul).any())

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        weights = _counts(rng.integers(0, n_vessels, n_vessels), None, n_vessels)
        score_a = _draw_score(weights, kor_a, kor_total, lat_a, lat_total, weighted, hangul)
        score_b = _draw_score(weights, kor_b, kor_total, lat_b, lat_total, weighted, hangul)
        draws.append(score_b - score_a)

    score_a = _point_score(left, correct_a)
    score_b = _point_score(right, correct_b)
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


def _draw_score(
    weights: NDArray[np.float64],
    kor_correct: NDArray[np.float64],
    kor_total: NDArray[np.float64],
    lat_correct: NDArray[np.float64],
    lat_total: NDArray[np.float64],
    weighted: bool,
    hangul: NDArray[np.bool_],
) -> float:
    """Return the bootstrap score of one draw: the weighted metric, or the single present bucket."""
    if weighted:
        return 0.7 * _ratio(weights, kor_correct, kor_total) + 0.3 * _ratio(weights, lat_correct, lat_total)
    if bool(hangul.any()):
        return _ratio(weights, kor_correct, kor_total)
    return _ratio(weights, lat_correct, lat_total)


def _point_score(frame: pd.DataFrame, correct: NDArray[np.bool_]) -> float:
    """Return the point score of a frame: the weighted metric, or the plain mean accuracy for one script."""
    score = bucket_scores(frame)["score"]
    if not math.isnan(score):
        return score
    return float(correct.mean())


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
