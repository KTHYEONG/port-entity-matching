"""Slow integration checks: inference budget rehearsal and CV regression floors."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
import pytest

from port_entity_matching.config import MatchConfig
from port_entity_matching.pipeline import run_submission
from port_entity_matching.records import CATALOG_COLUMNS, RECORD_COLUMNS, read_csv_checked
from port_entity_matching.train import cross_validate, cross_validate_run

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "raw" / "open"
MODEL_DIR = REPO_ROOT / "artifacts" / "model"
TRAIN_PATH = DATA_DIR / "train.csv"
CATALOG_PATH = DATA_DIR / "vessel_catalog.csv"

REHEARSAL_ROWS = 3000
TIME_BUDGET_SECONDS = 300.0
PEAK_RSS_LIMIT_MB = 8 * 1024
HANGUL_FLOOR = 0.95
NON_HANGUL_FLOOR = 0.995

pytestmark = pytest.mark.slow


def _require_inputs() -> None:
    if not DATA_DIR.is_dir():
        pytest.skip(f"competition data not available at {DATA_DIR}")
    if not MODEL_DIR.is_dir():
        pytest.skip(f"fitted artifacts not available at {MODEL_DIR}")


def test_full_catalog_rehearsal(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _require_inputs()
    train = read_csv_checked(TRAIN_PATH, (*RECORD_COLUMNS, "target_vessel_id"))
    catalog = read_csv_checked(CATALOG_PATH, CATALOG_COLUMNS)
    pseudo_test = train.head(REHEARSAL_ROWS).drop(columns=["target_vessel_id"])
    test_path = tmp_path / "test.csv"
    pseudo_test.to_csv(test_path, index=False)
    output_path = tmp_path / "output" / "submission.csv"

    with caplog.at_level(logging.INFO):
        started = time.perf_counter()
        written = run_submission(test_path, CATALOG_PATH, MODEL_DIR, output_path)
        elapsed = time.perf_counter() - started

    assert written == output_path
    assert elapsed < TIME_BUDGET_SECONDS, f"rehearsal took {elapsed:.1f}s"
    frame = pd.read_csv(written)
    assert len(frame) == REHEARSAL_ROWS
    assert frame["sample_id"].tolist() == pseudo_test["sample_id"].tolist()
    assert set(frame["target_vessel_id"]) <= set(catalog["vessel_id"])

    peaks = [float(value) for value in _logged_peak_rss(caplog)]
    assert peaks, "no peak RSS was logged"
    assert max(peaks) < PEAK_RSS_LIMIT_MB, f"peak RSS {max(peaks):.0f} MB exceeds the budget"


def test_cv_regression_guard() -> None:
    _require_inputs()
    train = read_csv_checked(TRAIN_PATH, (*RECORD_COLUMNS, "target_vessel_id"))
    catalog = read_csv_checked(CATALOG_PATH, CATALOG_COLUMNS)
    report = cross_validate(train, catalog, MatchConfig())
    assert report.hangul_accuracy >= HANGUL_FLOOR, f"hangul accuracy {report.hangul_accuracy:.4f}"
    assert report.non_hangul_accuracy >= NON_HANGUL_FLOOR, f"non-hangul accuracy {report.non_hangul_accuracy:.4f}"


def test_full_data_cv_reproduces_the_recorded_run() -> None:
    _require_inputs()
    recorded_path = REPO_ROOT / "artifacts" / "cv_oof.csv"
    if not recorded_path.is_file():
        pytest.skip(f"recorded OOF run not available at {recorded_path}")
    train = read_csv_checked(TRAIN_PATH, (*RECORD_COLUMNS, "target_vessel_id"))
    catalog = read_csv_checked(CATALOG_PATH, CATALOG_COLUMNS)
    run = cross_validate_run(train, catalog, MatchConfig())
    clean = run.oof[run.oof["scenario"] == "clean"].set_index("sample_id")
    recorded = pd.read_csv(recorded_path).set_index("sample_id")
    assert set(clean.index) == set(recorded.index)
    assert (clean["pred_vessel_id"] == recorded.loc[clean.index, "pred_vessel_id"]).all()
    assert run.report.weighted_score == pytest.approx(0.9875, abs=1e-4)


def _logged_peak_rss(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Peak RSS values in MB from the ``[SYS]`` submission log line."""
    values: list[str] = []
    for record in caplog.records:
        message = record.getMessage()
        if "[SYS]" not in message:
            continue
        values.extend(re.findall(r"peak_rss_mb=([0-9.]+)", message))
    return values
