"""End-to-end contract tests for the packaged submission entry point."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import port_entity_matching
from port_entity_matching.config import MatchConfig
from port_entity_matching.pipeline import run_submission
from port_entity_matching.train import fit_bundle
from tools.build_submit import build_submit, smoke_test

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_SOURCE = REPO_ROOT / "submission" / "script.py"
PACKAGE_SOURCE = Path(port_entity_matching.__file__).parent

_CATALOG: list[tuple[str, str, float, str, str]] = [
    ("V00", "Glovis Spirit", 5000, "Tanker", "Panama"),
    ("V01", "Grand Ace 11", 1000, "Container", "Liberia"),
    ("V02", "Calida", 61332, "Tanker", "Malta"),
    ("V03", "Shining Road", 2000, "Bulk", "Panama"),
]

_TRAIN: list[tuple[str, str, str, float | None]] = [
    ("글로비스 스피릿", "V00", "Tanker", 5000),
    ("GRAND ACE 11", "V01", "Container", 1000),
    ("CALIDA", "V02", "Tanker", 61332),
    ("SHINING ROAD", "V03", "Bulk", 2000),
]

_TEST_ROWS: list[dict[str, object]] = [
    {
        "sample_id": "S0",
        "vessel_name": "CALIDA",
        "call_sign": "CS_UNKNOWN_0",
        "gross_tonnage": 61332,
        "international_gross_tonnage": 61332,
        "flag_name": "파나마",
        "vessel_type": "Tanker",
        "arrival_year": 2020,
    },
    {
        "sample_id": "S1",
        "vessel_name": "글로비스 스피릿",
        "call_sign": "CS_UNKNOWN_1",
        "gross_tonnage": 5000,
        "international_gross_tonnage": 5000,
        "flag_name": "파나마",
        "vessel_type": "Tanker",
        "arrival_year": 2021,
    },
]


def _catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "vessel_id": vessel_id,
                "vessel_type": vessel_type,
                "vessel_name": vessel_name,
                "size": tonnage,
                "unit": "DWT",
                "deadweight_tonnage": tonnage,
                "gross_tonnage": tonnage,
                "flag_name": flag,
            }
            for vessel_id, vessel_name, tonnage, vessel_type, flag in _CATALOG
        ]
    )


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"T{number:05d}",
                "vessel_name": name,
                "call_sign": f"CS_{target}_{number}",
                "gross_tonnage": tonnage,
                "international_gross_tonnage": tonnage,
                "flag_name": "파나마",
                "vessel_type": vessel_type,
                "arrival_year": 2020,
                "target_vessel_id": target,
            }
            for number, (name, target, vessel_type, tonnage) in enumerate(_TRAIN)
        ]
    )


def _package(tmp_path: Path) -> Path:
    """Lay out a runnable submission tree: script.py, model/package, model/artifacts, data/."""
    root = tmp_path / "package"
    (root / "data").mkdir(parents=True)
    shutil.copy2(SCRIPT_SOURCE, root / "script.py")
    shutil.copytree(
        PACKAGE_SOURCE,
        root / "model" / "port_entity_matching",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    bundle = fit_bundle(
        _train_frame(),
        _catalog_frame(),
        MatchConfig(lgbm_num_rounds=20, lgbm_min_data_in_leaf=1, cv_splits=2, num_threads=1),
    )
    bundle.save(root / "model" / "artifacts")
    pd.DataFrame(_TEST_ROWS).to_csv(root / "data" / "test.csv", index=False)
    _catalog_frame().to_csv(root / "data" / "vessel_catalog.csv", index=False)
    return root


def test_script_produces_submission(tmp_path: Path) -> None:
    root = _package(tmp_path)
    expected_path = tmp_path / "expected.csv"
    run_submission(
        root / "data" / "test.csv",
        root / "data" / "vessel_catalog.csv",
        root / "model" / "artifacts",
        expected_path,
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "script.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    produced = root / "output" / "submission.csv"
    assert produced.is_file()
    assert produced.read_bytes() == expected_path.read_bytes()


def test_script_without_test_file_fails_clearly(tmp_path: Path) -> None:
    root = _package(tmp_path)
    (root / "data" / "test.csv").unlink()
    result = subprocess.run(  # noqa: S603
        [sys.executable, "script.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "data/test.csv" in result.stderr


@pytest.mark.parametrize("name", ["model_0.txt", "maps.json", "memory.json", "manifest.json"])
def test_build_submit_requires_every_artifact(tmp_path: Path, name: str) -> None:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from build_submit import REQUIRED_ARTIFACTS, build_submit  # noqa: PLC0415

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for artifact in REQUIRED_ARTIFACTS:
        (model_dir / artifact).write_text("{}", encoding="utf-8")
    (model_dir / name).unlink()
    with pytest.raises(SystemExit) as error:
        build_submit(model_dir, tmp_path / "submit.zip")
    assert error.value.code == 1


def test_build_submit_zip_layout(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from build_submit import REQUIRED_ARTIFACTS, build_submit  # noqa: PLC0415

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for artifact in REQUIRED_ARTIFACTS:
        (model_dir / artifact).write_text("{}", encoding="utf-8")
    (model_dir / "__pycache__").mkdir()
    (model_dir / "__pycache__" / "junk.pyc").write_bytes(b"")
    (model_dir / "extra.pyc").write_bytes(b"")

    zip_path = build_submit(model_dir, tmp_path / "submit.zip")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "script.py" in names
        assert "requirements.txt" in names
        assert "model/port_entity_matching/pipeline.py" in names
        assert "model/artifacts/model_0.txt" in names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
        assert all(not name.startswith("model/port_entity_matching/../") for name in names)


def test_zip_build_is_byte_identical_and_smoke_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tools.build_submit import main

    root = _package(tmp_path)
    model_dir = root / "model" / "artifacts"
    first = build_submit(model_dir, tmp_path / "first.zip")
    before = first.read_bytes()
    for source in model_dir.iterdir():
        source.touch()
    second = build_submit(model_dir, tmp_path / "second.zip")
    assert before == second.read_bytes()
    assert main([
        "--model-dir", str(model_dir), "--out", str(first), "--smoke-data", str(root / "data")
    ]) == 0
    assert "smoke_ok" in capsys.readouterr().out


def test_zip_cannot_overwrite_or_embed_model_artifacts(tmp_path: Path) -> None:
    root = _package(tmp_path)
    model_dir = root / "model" / "artifacts"
    model_file = model_dir / "model_0.txt"
    original = model_file.read_bytes()
    for output in (model_file, model_dir / "submit.zip"):
        with pytest.raises(ValueError, match="outside"):
            build_submit(model_dir, output)
    assert model_file.read_bytes() == original


@pytest.mark.parametrize("script", ["raise RuntimeError('smoke failure')", "x = 1"])
def test_smoke_rejects_failed_or_missing_output(tmp_path: Path, script: str) -> None:
    root = _package(tmp_path)
    archive_path = tmp_path / "broken.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("script.py", script)
    with pytest.raises(RuntimeError, match="smoke run"):
        smoke_test(archive_path, root / "data")
