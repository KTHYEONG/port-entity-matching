#!/usr/bin/env python3
"""Package an LB diagnostic probe from a base submission zip, gated by the LB ledger.

The probe zip is the base zip with ``script.py`` replaced by a probe entry script,
``probe.json`` holding the per-channel plan, optional extra files and alternative
ranker bundles under ``model/artifacts_<suffix>/``. A ledger id is required unless
``--local-only`` is passed, and the plan's K/L policies must equal the policies
registered for that id, so the packaged probe is exactly the registered one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "artifacts" / "eval" / "lb" / "ledger.jsonl"
_CHUNK_BYTES = 1 << 24


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def build_probe(
    base_zip: Path,
    out: Path,
    script: Path,
    plan: dict[str, object],
    extras: Sequence[tuple[str, Path]] = (),
    alt_models: Sequence[Path] = (),
    requirements: Path | None = None,
) -> Path:
    """Write the probe zip; ValueError on a plan without K/L or an extra/alt path that is missing."""
    if "K" not in plan or "L" not in plan:
        raise ValueError("plan must hold 'K' and 'L' policies")
    for _, path in extras:
        if not path.is_file():
            raise ValueError(f"extra file missing: {path}")
    for alt in alt_models:
        if not alt.is_dir():
            raise ValueError(f"alt model dir missing: {alt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base_zip) as zin, zipfile.ZipFile(out, "w", allowZip64=True) as zout:
        for info in zin.infolist():
            if info.filename == "script.py":
                zout.writestr(info, script.read_text(encoding="utf-8"))
            elif info.filename == "requirements.txt" and requirements is not None:
                zout.writestr(info, requirements.read_text(encoding="utf-8"))
            else:
                with zin.open(info) as source, zout.open(info, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source, target, _CHUNK_BYTES)
        zout.writestr("probe.json", json.dumps(plan, ensure_ascii=False))
        for arcname, path in extras:
            zout.write(path, arcname, zipfile.ZIP_DEFLATED)
        for alt in alt_models:
            suffix = alt.name.removeprefix("model_kanana_").removeprefix("model_")
            for item in sorted(alt.iterdir()):
                if item.is_file():
                    zout.write(item, f"model/artifacts_{suffix}/{item.name}", zipfile.ZIP_DEFLATED)
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_extra(raw: str) -> tuple[str, Path]:
    arcname, sep, path = raw.partition("=")
    if not sep or not arcname or not path:
        raise ValueError(f"invalid --extra {raw!r}: expected ARCNAME=PATH")
    return arcname, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: build a ledger-registered probe zip."""
    parser = argparse.ArgumentParser(prog="build_probe", description="Package a ledger-registered LB probe.")
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True, help="probe entry script copied to script.py")
    parser.add_argument("--plan", type=Path, required=True, help="JSON file with K and L policies (+ optional keys)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra", action="append", default=[], help="ARCNAME=PATH added to the zip root")
    parser.add_argument("--alt-model", action="append", type=Path, default=[], help="ranker bundle directory")
    parser.add_argument("--requirements", type=Path, default=None, help="replacement requirements.txt")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ledger-id", type=str, default=None)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args(argv)

    if (args.ledger_id is None) == (not args.local_only):
        print(
            "refusing to package: pass exactly one of --ledger-id (registered via tools/lb.py) or --local-only",
            file=sys.stderr,
        )
        return 2
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        extras = [_parse_extra(raw) for raw in args.extra]
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not isinstance(plan, dict):
        print("error: plan must be a JSON object", file=sys.stderr)
        return 2

    ledger = None
    if args.ledger_id is not None:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from port_entity_matching.evaluation.lb.ledger import Ledger  # noqa: PLC0415

        ledger = Ledger.load(args.ledger)
        entry = {item.entry_id: item for item in ledger.entries()}.get(args.ledger_id)
        if entry is None or entry.status != "registered":
            print(f"refusing to package: ledger entry '{args.ledger_id}' is not registered", file=sys.stderr)
            return 2
        for channel, spec in (("K", entry.k), ("L", entry.l)):
            if _canonical(plan.get(channel)) != _canonical(dict(spec.policy)):
                print(
                    f"refusing to package: plan {channel} policy differs from ledger entry '{args.ledger_id}'",
                    file=sys.stderr,
                )
                return 2

    try:
        zip_path = build_probe(
            args.base_zip, args.out, args.script, plan, extras, args.alt_model, args.requirements
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"zip={zip_path} size_bytes={zip_path.stat().st_size}")
    if ledger is not None:
        digest = _sha256(zip_path)
        ledger.attach_zip(args.ledger_id, digest)
        print(f"ledger_id={args.ledger_id} sha256={digest}")
    else:
        print("local_only=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
