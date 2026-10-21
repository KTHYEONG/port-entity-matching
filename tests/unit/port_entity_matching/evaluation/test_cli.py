"""Contract tests for the evaluation CLI: overrides, artifacts, stress table and comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from port_entity_matching.config import MatchConfig
from port_entity_matching.evaluation.cli import _cast_value, main, parse_overrides
from port_entity_matching.evaluation.perturb import PERTURBATIONS
from port_entity_matching.evaluation.stats import OOF_COLUMNS

_VESSELS: list[tuple[str, str, str, int, str, str]] = [
    ("V00", "KAROS", "카로스", 1000, "Tanker", "Panama"),
    ("V01", "TOMOS", "토모스", 2000, "Container", "Liberia"),
    ("V02", "SAMOS", "사모스", 3000, "Tanker", "Malta"),
    ("V03", "PAKOS", "파코스", 4000, "Bulk", "Panama"),
    ("V04", "NOMOS", "노모스", 5000, "Tanker", "Greece"),
    ("V05", "KOMOS", "코모스", 6000, "Container", "Panama"),
    ("V06", "ROMOS", "로모스", 7000, "Tanker", "Cyprus"),
    ("V07", "SONIS", "소니스", 8000, "Container", "Panama"),
    ("V08", "PONOS", "포노스", 9000, "Tanker", "Marshall Islands"),
    ("V09", "TONIS", "토니스", 10000, "Bulk", "Panama"),
]


def _write_fixture(root: Path) -> tuple[Path, Path]:
    """Write a tiny catalog and training CSV with Korean and Latin rows."""
    catalog = pd.DataFrame(
        [
            {
                "vessel_id": vessel_id,
                "vessel_type": vessel_type,
                "vessel_name": name,
                "size": tonnage,
                "unit": "DWT",
                "deadweight_tonnage": tonnage,
                "gross_tonnage": tonnage,
                "flag_name": flag,
            }
            for vessel_id, name, _korean, tonnage, vessel_type, flag in _VESSELS
        ]
    )
    catalog_path = root / "vessel_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    train = pd.DataFrame(
        [
            {
                "sample_id": f"T{index:05d}",
                "vessel_name": korean if index % 2 == 0 else name,
                "call_sign": f"CS_{vessel_id}",
                "gross_tonnage": tonnage,
                "international_gross_tonnage": tonnage,
                "flag_name": "파나마" if flag == "Panama" else "라이베리아",
                "vessel_type": "유조선" if vessel_type == "Tanker" else "컨테이너선",
                "arrival_year": 2020,
                "target_vessel_id": vessel_id,
            }
            for index, (vessel_id, name, korean, tonnage, vessel_type, flag) in enumerate(_VESSELS)
        ]
    )
    train_path = root / "train.csv"
    train.to_csv(train_path, index=False)
    return train_path, catalog_path


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A tiny catalog/training pair with ten vessels, five Korean and five Latin rows."""
    root = tmp_path_factory.mktemp("tiny")
    return _write_fixture(root)


@pytest.fixture(scope="module")
def cv_run(tiny_dataset: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the cv subcommand once and return its output directory."""
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path_factory.mktemp("cv")
    assert main([
        "cv",
        "--train", str(train_path),
        "--catalog", str(catalog_path),
        "--out-dir", str(out_dir),
        "--set", "ensemble_size=1",
    ]) == 0
    return out_dir


def test_typed_overrides_are_cast_by_field_type() -> None:
    config = parse_overrides(["hangul_tonnage_window=0.08", "augment=false", "ensemble_size=2"])
    assert config.hangul_tonnage_window == pytest.approx(0.08)
    assert isinstance(config.hangul_tonnage_window, float)
    assert config.augment is False
    assert config.ensemble_size == 2
    assert isinstance(config.ensemble_size, int)
    defaults = MatchConfig()
    assert config.cv_splits == defaults.cv_splits
    assert config.seed == defaults.seed


def test_bool_overrides_accept_only_documented_literals() -> None:
    assert parse_overrides(["augment=TRUE"]).augment is True
    assert parse_overrides(["augment=0"]).augment is False
    with pytest.raises(ValueError, match="augment"):
        parse_overrides(["augment=yes"])


@pytest.mark.parametrize(
    ("pair", "key"),
    [("nope=1", "nope"), ("ensemble_size=two", "ensemble_size"), ("augment", "augment")],
)
def test_bad_overrides_are_rejected(pair: str, key: str) -> None:
    with pytest.raises(ValueError, match=key):
        parse_overrides([pair])


def test_float_override_with_a_non_numeric_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="tonnage_window"):
        parse_overrides(["tonnage_window=wide"])


def test_unsupported_field_types_are_rejected() -> None:
    with pytest.raises(ValueError, match="vessel_name"):
        _cast_value("vessel_name", "value", "text")


def test_cv_writes_artifacts(cv_run: Path, tiny_dataset: tuple[Path, Path]) -> None:
    oof_path = cv_run / "oof.csv"
    summary_path = cv_run / "summary.json"
    assert oof_path.is_file()
    assert summary_path.is_file()

    oof = pd.read_csv(oof_path)
    train_rows = len(pd.read_csv(tiny_dataset[0]))
    assert list(oof.columns) == list(OOF_COLUMNS)
    assert len(oof) == train_rows
    assert "scenario" not in oof.columns

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("config", "kor", "lat", "score", "ci", "elapsed_seconds"):
        assert key in summary
    assert summary["config"]["ensemble_size"] == 1
    assert summary["elapsed_seconds"] >= 0.0


def test_stress_writes_per_scenario_table(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path / "stress"
    assert main([
        "stress",
        "--train", str(train_path),
        "--catalog", str(catalog_path),
        "--out-dir", str(out_dir),
        "--scenarios", "kor_tonnage_missing,lat_tonnage_missing",
        "--set", "ensemble_size=1",
    ]) == 0
    assert (out_dir / "oof_long.csv").is_file()

    report = json.loads((out_dir / "stress.json").read_text(encoding="utf-8"))
    assert set(report) == {"kor_tonnage_missing", "lat_tonnage_missing"}
    for entry in report.values():
        assert entry["affected_rows"] > 0
        for field in ("acc_clean", "acc_perturbed", "delta"):
            assert field in entry


def test_compare_is_order_insensitive(cv_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    oof = pd.read_csv(cv_run / "oof.csv")
    shuffled_path = tmp_path / "oof_shuffled.csv"
    oof.sample(frac=1.0, random_state=11).to_csv(shuffled_path, index=False)

    report_path = tmp_path / "compare.json"
    exit_code = main([
        "compare", str(cv_run / "oof.csv"), str(shuffled_path), "--json", str(report_path)
    ])
    assert exit_code == 0

    printed = capsys.readouterr().out
    assert "+0.0000" in printed
    assert "vessels_fixed=0" in printed
    assert "vessels_broken=0" in printed
    assert "rows_fixed=0" in printed
    assert "rows_broken=0" in printed

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["delta"] == pytest.approx(0.0, abs=1e-12)
    assert report["rows_fixed"] == 0
    assert report["rows_broken"] == 0


def test_unknown_scenario_is_reported(tiny_dataset: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train_path, catalog_path = tiny_dataset
    exit_code = main([
        "stress",
        "--train", str(train_path),
        "--catalog", str(catalog_path),
        "--out-dir", str(tmp_path / "stress"),
        "--scenarios", "bogus",
    ])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "bogus" in captured.err
    assert "kor_tonnage_missing" in captured.err


def test_all_scenarios_are_run_by_default(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path / "stress_all"
    assert main([
        "stress",
        "--train", str(train_path),
        "--catalog", str(catalog_path),
        "--out-dir", str(out_dir),
        "--set", "ensemble_size=1",
    ]) == 0
    report = json.loads((out_dir / "stress.json").read_text(encoding="utf-8"))
    assert set(report) == set(PERTURBATIONS)


def test_a_scenario_without_affected_rows_is_reported(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path / "stress_empty"
    assert main([
        "stress",
        "--train", str(train_path),
        "--catalog", str(catalog_path),
        "--out-dir", str(out_dir),
        "--scenarios", "kor_spaces_removed",
        "--set", "ensemble_size=1",
    ]) == 0
    report = json.loads((out_dir / "stress.json").read_text(encoding="utf-8"))
    entry = report["kor_spaces_removed"]
    assert entry["affected_rows"] == 0
    assert entry["acc_clean"] is None
    assert entry["delta"] is None


def test_output_directory_is_created(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path / "nested" / "run"
    assert main([
        "cv",
        "--train", str(train_path),
        "--catalog", str(catalog_path),
        "--out-dir", str(out_dir),
        "--set", "ensemble_size=1",
    ]) == 0
    assert (out_dir / "oof.csv").is_file()


def test_writing_inside_a_data_directory_is_refused(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    exit_code = main([
        "cv",
        "--train", str(train_path),
        "--catalog", str(catalog_path),
        "--out-dir", str(tmp_path / "data" / "eval"),
    ])
    assert exit_code == 1
