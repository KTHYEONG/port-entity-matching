#!/usr/bin/env python3
"""Assemble the competition submission ZIP from the fitted artifacts.

The archive has no top-level folder: ``script.py``, ``requirements.txt`` and
``model/`` (package plus artifacts) sit at the zip root.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "src" / "port_entity_matching"
SCRIPT_DIR = REPO_ROOT / "submission"

REQUIRED_ARTIFACTS: tuple[str, ...] = ("model_0.txt", "maps.json", "memory.json", "manifest.json")
EXCLUDED_SUFFIXES: tuple[str, ...] = (".pyc",)
EXCLUDED_DIRS: tuple[str, ...] = ("__pycache__",)
FIXED_TIMESTAMP: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)


def build_submit(
    model_dir: Path,
    out_zip: Path,
    package_dir: Path = PACKAGE_DIR,
    script_dir: Path = SCRIPT_DIR,
) -> Path:
    """Assemble script.py, requirements.txt, model/port_entity_matching/*.py, model/artifacts/* into a root-level zip."""
    model_dir = Path(model_dir)
    out_zip = Path(out_zip)
    if out_zip.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("Output ZIP must be outside the model artifact directory")
    missing = [name for name in REQUIRED_ARTIFACTS if not (model_dir / name).is_file()]
    if missing:
        print(
            f"error: model artifacts missing in {model_dir}: {', '.join(missing)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    entries: dict[str, Path] = {
        "script.py": script_dir / "script.py",
        "requirements.txt": script_dir / "requirements.txt",
    }
    for source in sorted(package_dir.glob("*.py")):
        entries[f"model/port_entity_matching/{source.name}"] = source
    for source in sorted(model_dir.iterdir()):
        if source.is_dir() or source.suffix in EXCLUDED_SUFFIXES or source.name in EXCLUDED_DIRS:
            continue
        entries[f"model/artifacts/{source.name}"] = source

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname in sorted(entries):
            info = zipfile.ZipInfo(arcname, date_time=FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, entries[arcname].read_bytes())
    return out_zip


def smoke_test(zip_path: Path, smoke_data: Path) -> None:
    """Run the extracted submission against smoke data; raise RuntimeError on failure."""
    smoke_data = Path(smoke_data)
    with tempfile.TemporaryDirectory(prefix="submit_smoke_") as temporary:
        root = Path(temporary) / "package"
        root.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(root)
        data_dir = root / "data"
        data_dir.mkdir()
        for name in ("test.csv", "vessel_catalog.csv"):
            shutil.copy2(smoke_data / name, data_dir / name)
        result = subprocess.run(  # noqa: S603
            [sys.executable, "script.py"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        output = root / "output" / "submission.csv"
        if result.returncode != 0:
            raise RuntimeError(f"smoke run failed with exit code {result.returncode}: {result.stderr.strip()}")
        if not output.is_file():
            raise RuntimeError("smoke run did not produce output/submission.csv")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: build the submission zip and optionally smoke-test it."""
    parser = argparse.ArgumentParser(prog="build_submit", description="Build the competition submission zip.")
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "artifacts" / "model")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "submit.zip")
    parser.add_argument("--smoke-data", type=Path, default=None, help="directory with test.csv and vessel_catalog.csv")
    args = parser.parse_args(argv)

    zip_path = build_submit(args.model_dir, args.out)
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.namelist()
    print(f"zip={zip_path} size_bytes={zip_path.stat().st_size} entries={len(entries)}")

    if args.smoke_data is not None:
        smoke_test(zip_path, args.smoke_data)
        print("smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
