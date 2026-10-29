#!/usr/bin/env python3
"""Assemble the competition submission ZIP from the fitted artifacts.

The archive has no top-level folder: ``script.py``, ``requirements.txt`` and
``model/`` (package plus artifacts) sit at the zip root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
DEFAULT_LLM_DIR = REPO_ROOT / "artifacts" / "hf" / "kanana-1.5-2.1b-instruct-2505"

REQUIRED_ARTIFACTS: tuple[str, ...] = ("model_0.txt", "maps.json", "memory.json", "manifest.json")
FEWSHOT_ARTIFACT = "llm_fewshot.json"
EXCLUDED_SUFFIXES: tuple[str, ...] = (".pyc",)
EXCLUDED_DIRS: tuple[str, ...] = ("__pycache__",)
FIXED_TIMESTAMP: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)
LLM_SUFFIXES: tuple[str, ...] = (".json", ".safetensors", ".txt")
MAX_COMPRESSED_BYTES = 10 * 1024**3
MAX_UNCOMPRESSED_BYTES = 32 * 1024**3
_CHUNK_BYTES = 1024 * 1024


def _llm_enabled(model_dir: Path) -> bool:
    manifest = model_dir / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("Invalid artifact manifest") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("config", {}), dict):
        raise ValueError("Invalid artifact manifest config")
    return payload.get("config", {}).get("llm_enabled") is True


def _llm_weight_sources(llm_dir: Path) -> list[Path]:
    sources: list[Path] = []
    for source in sorted(llm_dir.iterdir()):
        if not source.is_file():
            continue
        name = source.name
        if name.startswith(("tokenizer", "LICENSE")) or source.suffix in LLM_SUFFIXES:
            sources.append(source)
    return sources


def _write_streamed(archive: zipfile.ZipFile, arcname: str, source: Path, compress_type: int) -> None:
    info = zipfile.ZipInfo(arcname, date_time=FIXED_TIMESTAMP)
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    with open(source, "rb") as handle, archive.open(info, "w", force_zip64=True) as dest:
        shutil.copyfileobj(handle, dest, length=_CHUNK_BYTES)


def build_submit(
    model_dir: Path,
    out_zip: Path,
    package_dir: Path = PACKAGE_DIR,
    script_dir: Path = SCRIPT_DIR,
    llm_dir: Path | None = None,
) -> Path:
    """Assemble script.py, requirements.txt, model/port_entity_matching/*.py, model/artifacts/* into a root-level zip."""
    model_dir = Path(model_dir)
    out_zip = Path(out_zip)
    if out_zip.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("Output ZIP must be outside the model artifact directory")
    required = list(REQUIRED_ARTIFACTS)
    llm_required = _llm_enabled(model_dir)
    if llm_required:
        required.append(FEWSHOT_ARTIFACT)
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        print(
            f"error: model artifacts missing in {model_dir}: {', '.join(missing)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    resolved_llm = Path(llm_dir) if llm_dir is not None else DEFAULT_LLM_DIR
    llm_sources: list[Path] = []
    if llm_required:
        if out_zip.resolve().is_relative_to(resolved_llm.resolve()):
            raise ValueError("Output ZIP must be outside the LLM weights directory")
        has_config = (resolved_llm / "config.json").is_file()
        has_weights = any(source.is_file() for source in resolved_llm.glob("*.safetensors"))
        if not (has_config and has_weights):
            print(f"error: LLM weights missing in {resolved_llm}", file=sys.stderr)
            raise SystemExit(1)
        llm_sources = _llm_weight_sources(resolved_llm)

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
    for source in llm_sources:
        entries[f"model/artifacts/llm/{source.name}"] = source

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname in sorted(entries):
            source = entries[arcname]
            compress = zipfile.ZIP_STORED if source.suffix == ".safetensors" else zipfile.ZIP_DEFLATED
            _write_streamed(archive, arcname, source, compress)
    compressed = out_zip.stat().st_size
    with zipfile.ZipFile(out_zip) as archive:
        uncompressed = sum(item.file_size for item in archive.infolist())
    if compressed > MAX_COMPRESSED_BYTES or uncompressed > MAX_UNCOMPRESSED_BYTES:
        out_zip.unlink(missing_ok=True)
        raise ValueError("Submission ZIP exceeds the size budget")
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
    parser.add_argument("--llm-dir", type=Path, default=DEFAULT_LLM_DIR, help="transliteration weights directory")
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / "artifacts" / "eval" / "lb" / "ledger.jsonl")
    parser.add_argument("--ledger-id", type=str, default=None)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args(argv)

    if args.ledger_id is None and not args.local_only:
        print(
            "refusing to package: register a ledger entry (tools/lb.py register) or pass --local-only",
            file=sys.stderr,
        )
        return 2
    if args.ledger_id is not None and args.local_only:
        print("refusing to package: pass exactly one of --ledger-id or --local-only", file=sys.stderr)
        return 2

    ledger = None
    if args.ledger_id is not None:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from port_entity_matching.evaluation.lb.ledger import Ledger  # noqa: PLC0415

        ledger = Ledger.load(Path(args.ledger))
        current = {entry.entry_id: entry for entry in ledger.entries()}
        entry = current.get(args.ledger_id)
        if entry is None or entry.status != "registered":
            print(f"refusing to package: ledger entry '{args.ledger_id}' is not registered", file=sys.stderr)
            return 2

    zip_path = build_submit(args.model_dir, args.out, llm_dir=args.llm_dir)
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.namelist()
    print(f"zip={zip_path} size_bytes={zip_path.stat().st_size} entries={len(entries)}")

    if args.smoke_data is not None:
        smoke_test(zip_path, args.smoke_data)
        print("smoke_ok")

    if ledger is not None and args.ledger_id is not None:
        digest = hashlib.sha256()
        with open(zip_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        try:
            ledger.attach_zip(args.ledger_id, digest.hexdigest())
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        print(f"ledger_id={args.ledger_id} sha256={digest.hexdigest()}")
    elif args.local_only:
        print("local_only=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
