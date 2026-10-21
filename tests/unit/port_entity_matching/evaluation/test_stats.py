"""Contract tests for the out-of-fold statistics helpers."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from port_entity_matching.evaluation.stats import (
    OOF_COLUMNS,
    bucket_scores,
    load_oof,
    paired_difference,
    vessel_bootstrap_ci,
)
from port_entity_matching.train import DIAGNOSTIC_COLUMNS


def _frame(rows: list[tuple[str, bool, str, str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "is_hangul": int(hangul),
                "target_vessel_id": target,
                "pred_vessel_id": predicted,
                "correct": int(correct),
                "top1_score": 1.0,
                "top2_vessel_id": "",
                "top2_score": 0.0,
                "margin": 1.0,
                "target_retrieved": 1,
                "n_candidates": 3,
            }
            for sample_id, hangul, target, predicted, correct in rows
        ],
        columns=list(DIAGNOSTIC_COLUMNS),
    )


def _mixed_frame() -> pd.DataFrame:
    kor_rows = [(f"K{index:03d}", True, f"V{index:03d}", f"V{index:03d}", index != 0) for index in range(10)]
    latin_rows = [(f"L{index:03d}", False, f"W{index:03d}", f"W{index:03d}", index != 0) for index in range(100)]
    return _frame([*kor_rows, *latin_rows])


def test_oof_columns_match_diagnostic_columns() -> None:
    assert OOF_COLUMNS == DIAGNOSTIC_COLUMNS


def test_weighted_score() -> None:
    scores = bucket_scores(_mixed_frame())
    assert scores["kor"] == pytest.approx(0.9, rel=1e-12)
    assert scores["lat"] == pytest.approx(0.99, rel=1e-12)
    assert scores["score"] == pytest.approx(0.7 * 0.9 + 0.3 * 0.99, rel=1e-12)


def test_missing_bucket_is_nan() -> None:
    frame = _frame([("K000", True, "V000", "V000", True)])
    scores = bucket_scores(frame)
    assert math.isnan(scores["lat"])
    assert math.isnan(scores["score"])


def test_row_order_does_not_matter() -> None:
    frame = _mixed_frame()
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert bucket_scores(frame) == bucket_scores(shuffled)
    assert vessel_bootstrap_ci(frame) == vessel_bootstrap_ci(shuffled)
    difference = paired_difference(frame, shuffled)
    assert difference.delta == pytest.approx(0.0, abs=1e-12)
    assert (difference.ci_low, difference.ci_high) == (0.0, 0.0)
    assert difference.rows_fixed == 0
    assert difference.rows_broken == 0
    assert difference.vessels_fixed == 0
    assert difference.vessels_broken == 0


def test_ci_brackets_the_point_estimate() -> None:
    frame = _mixed_frame()
    scores = bucket_scores(frame)
    intervals = vessel_bootstrap_ci(frame, n_boot=200, seed=7)
    for key, (low, high) in intervals.items():
        assert low <= scores[key] <= high
    assert vessel_bootstrap_ci(frame, n_boot=200, seed=7) == intervals


def test_flip_counting() -> None:
    base = [
        ("S0", True, "V0", "V0", True),
        ("S1", True, "V0", "V0", True),
        ("S2", False, "V1", "V1", True),
        ("S3", False, "V1", "V1", True),
    ]
    improved = [
        ("S0", True, "V0", "V0", True),
        ("S1", True, "V0", "V0", True),
        ("S2", False, "V1", "V1", False),
        ("S3", False, "V1", "V1", True),
    ]
    difference = paired_difference(_frame(base), _frame(improved))
    assert difference.rows_fixed == 0
    assert difference.rows_broken == 1
    assert difference.vessels_broken == 1
    assert difference.vessels_fixed == 0


def test_vessel_fixed_count() -> None:
    base = [
        ("S0", True, "V0", "V0", False),
        ("S1", False, "V1", "V1", True),
    ]
    improved = [
        ("S0", True, "V0", "V0", True),
        ("S1", False, "V1", "V1", True),
    ]
    difference = paired_difference(_frame(base), _frame(improved))
    assert difference.rows_fixed == 1
    assert difference.rows_broken == 0
    assert difference.vessels_fixed == 1
    assert difference.vessels_broken == 0
    assert difference.delta > 0.0


def test_mismatched_frames_rejected() -> None:
    base = _mixed_frame()
    other = base[base["sample_id"] != "K000"].reset_index(drop=True)
    with pytest.raises(ValueError, match="sample_id"):
        paired_difference(base, other)
    changed = base.copy()
    changed.loc[0, "target_vessel_id"] = "OTHER"
    with pytest.raises(ValueError, match="target_vessel_id"):
        paired_difference(base, changed)


def test_load_oof_keeps_only_clean_rows(tmp_path: Path) -> None:
    frame = _mixed_frame()
    long_frame = pd.concat([frame.assign(scenario="clean"), frame.assign(scenario="kor_tonnage_missing")])
    path = tmp_path / "oof_long.csv"
    long_frame.to_csv(path, index=False)
    loaded = load_oof(path)
    assert len(loaded) == len(frame)
    assert set(loaded["sample_id"]) == set(frame["sample_id"])
    assert loaded["correct"].dtype == bool
    assert loaded["is_hangul"].dtype == bool
    assert loaded["sample_id"].dtype == object


def test_load_oof_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "oof.csv"
    _mixed_frame().drop(columns=["correct"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="correct"):
        load_oof(path)


def test_load_oof_rejects_duplicate_ids(tmp_path: Path) -> None:
    frame = _mixed_frame()
    path = tmp_path / "oof.csv"
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_oof(path)


def test_load_oof_reads_boolean_columns(tmp_path: Path) -> None:
    frame = _mixed_frame()
    path = tmp_path / "oof.csv"
    frame.assign(correct=frame["correct"].astype(bool)).to_csv(path, index=False)
    loaded = load_oof(path)
    assert loaded["correct"].dtype == bool
    assert loaded["correct"].tolist() == frame["correct"].astype(bool).tolist()


def test_empty_frame_rejected() -> None:
    empty = _frame([])
    with pytest.raises(ValueError, match="no rows"):
        vessel_bootstrap_ci(empty)
    with pytest.raises(ValueError, match="no rows"):
        paired_difference(empty, empty)


def test_single_bucket_ci_is_nan() -> None:
    frame = _frame([("K000", True, "V000", "V000", True), ("K001", True, "V001", "V001", False)])
    intervals = vessel_bootstrap_ci(frame, n_boot=50, seed=1)
    assert math.isnan(intervals["lat"][0])
    assert math.isnan(intervals["lat"][1])


def test_is_hangul_mismatch_rejected() -> None:
    base = _mixed_frame()
    changed = base.copy()
    changed.loc[0, "is_hangul"] = 0
    with pytest.raises(ValueError, match="is_hangul"):
        paired_difference(base, changed)


def test_string_ids_survive_csv_inference(tmp_path: Path) -> None:
    frame = _frame([("0001", True, "001", "NA", True)])
    frame["top2_vessel_id"] = "002"
    path = tmp_path / "ids.csv"
    frame.to_csv(path, index=False)
    loaded = load_oof(path)
    assert loaded.loc[0, ["sample_id", "target_vessel_id", "pred_vessel_id", "top2_vessel_id"]].tolist() == [
        "0001", "001", "NA", "002",
    ]


@pytest.mark.parametrize("column", OOF_COLUMNS)
def test_all_diagnostic_columns_required(tmp_path: Path, column: str) -> None:
    path = tmp_path / "missing.csv"
    _mixed_frame().drop(columns=column).to_csv(path, index=False)
    with pytest.raises(ValueError, match=column):
        load_oof(path)


def test_invalid_boolean_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    _mixed_frame().assign(correct="unknown").to_csv(path, index=False)
    with pytest.raises(ValueError, match="boolean"):
        load_oof(path)


def test_duplicate_frames_rejected() -> None:
    frame = _mixed_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        paired_difference(duplicated, frame)
    with pytest.raises(ValueError, match="duplicate"):
        paired_difference(frame, duplicated)


@pytest.mark.parametrize("n_boot", [0, -1])
def test_invalid_bootstrap_count_rejected(n_boot: int) -> None:
    frame = _mixed_frame()
    with pytest.raises(ValueError, match="n_boot"):
        vessel_bootstrap_ci(frame, n_boot=n_boot)
    with pytest.raises(ValueError, match="n_boot"):
        paired_difference(frame, frame, n_boot=n_boot)


def test_percentile_is_not_clamped_to_point() -> None:
    import numpy as np

    frame = _frame([
        ("0", True, "A", "A", True), ("1", False, "A", "A", True),
        ("2", True, "B", "X", False), ("3", False, "B", "X", False),
    ])
    rng = np.random.default_rng(0)
    accuracy = float((rng.integers(0, 2, 2) == 0).mean())
    intervals = vessel_bootstrap_ci(frame, n_boot=1)
    assert intervals["kor"] == (accuracy, accuracy)
    difference = paired_difference(frame.assign(correct=0), frame, n_boot=1)
    assert difference.ci_low == pytest.approx(accuracy)
    assert difference.ci_high == pytest.approx(accuracy)
