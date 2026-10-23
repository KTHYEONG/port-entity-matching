"""Slow integration checks: inference budget rehearsal and CV regression floors."""

from __future__ import annotations

import logging
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import pytest

from port_entity_matching.config import MatchConfig
from port_entity_matching.evaluation.stats import bucket_scores
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.pipeline import Matcher, run_submission
from port_entity_matching.records import CATALOG_COLUMNS, RECORD_COLUMNS, read_csv_checked, records_from_frame
from port_entity_matching.train import cross_validate, cross_validate_run

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "raw" / "open"
MODEL_DIR = REPO_ROOT / "artifacts" / "model"
LLM_MODEL_DIR = Path(os.environ.get("LLM_BUDGET_MODEL_DIR", str(MODEL_DIR)))
TRAIN_PATH = DATA_DIR / "train.csv"
CATALOG_PATH = DATA_DIR / "vessel_catalog.csv"

REHEARSAL_ROWS = 3000
TIME_BUDGET_SECONDS = 900.0
PEAK_RSS_LIMIT_MB = 8 * 1024
HANGUL_FLOOR = 0.95
NON_HANGUL_FLOOR = 0.995

pytestmark = pytest.mark.slow


def _require_inputs(model_dir: Path = MODEL_DIR) -> None:
    if not DATA_DIR.is_dir():
        pytest.skip(f"competition data not available at {DATA_DIR}")
    if not model_dir.is_dir():
        pytest.skip(f"fitted artifacts not available at {model_dir}")


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
    recorded_score = bucket_scores(recorded.reset_index())["score"]
    assert run.report.weighted_score == pytest.approx(recorded_score, abs=1e-4)


def _logged_peak_rss(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Peak RSS values in MB from the ``[SYS]`` submission log line."""
    values: list[str] = []
    for record in caplog.records:
        message = record.getMessage()
        if "[SYS]" not in message:
            continue
        values.extend(re.findall(r"peak_rss_mb=([0-9.]+)", message))
    return values


def _require_llm() -> None:
    _require_inputs(LLM_MODEL_DIR)
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not (LLM_MODEL_DIR / "llm" / "config.json").is_file():
        pytest.skip(f"transliteration weights not available at {LLM_MODEL_DIR / 'llm'}")
    manifest = json.loads((LLM_MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["llm_enabled"], "LLM rehearsal requires an enabled bundle"
    shots = LLM_MODEL_DIR / manifest["llm"]["fewshot_file"]
    assert json.loads(shots.read_text(encoding="utf-8")), "LLM rehearsal requires few-shot pairs"


def _pseudo_llm_frame(train: pd.DataFrame, total: int = 4632) -> pd.DataFrame:
    hangul = train[train["vessel_name"].map(is_hangul_name)]
    latin = train[~train["vessel_name"].map(is_hangul_name)]
    need = max(0, total - len(hangul))
    frame = pd.concat([hangul, latin.head(need)], ignore_index=True)
    frame = frame.drop(columns=["target_vessel_id"])
    # Training-row memory would hide changes in the LLM/ranker path.
    frame["call_sign"] = None
    return frame


def _logged_llm_seconds(caplog: pytest.LogCaptureFixture) -> float:
    values: list[float] = []
    for record in caplog.records:
        values.extend(float(value) for value in re.findall(r"llm_seconds=([0-9.]+)", record.getMessage()))
    assert values and max(values) > 0, "no active LLM stage was logged"
    return max(values)


def _logged_elapsed(caplog: pytest.LogCaptureFixture) -> float:
    values: list[float] = []
    for record in caplog.records:
        values.extend(float(value) for value in re.findall(r"elapsed_seconds=([0-9.]+)", record.getMessage()))
    assert values, "no elapsed_seconds was logged"
    return max(values)


def test_llm_server_budget_projection(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _require_llm()
    train = read_csv_checked(TRAIN_PATH, (*RECORD_COLUMNS, "target_vessel_id"))
    pseudo = _pseudo_llm_frame(train)
    assert len(pseudo) == 4632
    test_path = tmp_path / "test.csv"
    pseudo.to_csv(test_path, index=False)
    output_path = tmp_path / "output" / "submission.csv"
    with caplog.at_level(logging.INFO):
        run_submission(test_path, CATALOG_PATH, LLM_MODEL_DIR, output_path)
    total = _logged_elapsed(caplog)
    llm_seconds = _logged_llm_seconds(caplog)
    projected = llm_seconds * 2.0 + (total - llm_seconds) + 120
    logging.getLogger(__name__).info(
        "[EVAL] llm_budget rows=%d llm_seconds=%.2f total_seconds=%.2f projected_server_seconds=%.2f",
        len(pseudo), llm_seconds, total, projected,
    )
    assert "llm unavailable" not in caplog.text
    assert projected < 25 * 60


def test_llm_row_independence(tmp_path: Path) -> None:
    _require_llm()
    train = read_csv_checked(TRAIN_PATH, (*RECORD_COLUMNS, "target_vessel_id"))
    pseudo = _pseudo_llm_frame(train)
    catalog = read_csv_checked(CATALOG_PATH, CATALOG_COLUMNS)
    matcher = Matcher.from_dir(LLM_MODEL_DIR, catalog)
    assert matcher.transliterator is not None
    full_records = records_from_frame(pseudo)
    full = matcher.predict(full_records)
    selected = [position for position, record in enumerate(full_records) if is_hangul_name(record.vessel_name)][:5]
    assert len(selected) == 5
    alone = [matcher.predict([full_records[position]])[0] for position in selected]
    assert [full[position] for position in selected] == alone
    assert matcher.predict([full_records[position] for position in selected]) == alone
    shuffled_frame = pseudo.sample(frac=1, random_state=42)
    shuffled = matcher.predict(records_from_frame(shuffled_frame))
    assert [full[position] for position in shuffled_frame.index] == shuffled
