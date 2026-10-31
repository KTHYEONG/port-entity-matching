#!/usr/bin/env python3
"""Derive a model directory from a fitted one without refitting.

Adds token-dictionary renderings to the reverse channel and overrides inference-only config fields; the ensemble,
maps and memory are copied unchanged.

usage: derive_model.py --from artifacts/model_rev2syn_combo --to artifacts/model_final \
    [--train data/raw/open/train.csv --catalog data/raw/open/vessel_catalog.csv \
     --token-renderings rev_dict_catalog.jsonl] [--add-reverse rev_k8b_catalog.jsonl ...] \
    [--set exclude_train_vessels=true ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

from port_entity_matching.artifacts import MANIFEST_FILE  # noqa: E402
from port_entity_matching.config import config_from_dict, config_to_dict  # noqa: E402
from port_entity_matching.token_renderings import (  # noqa: E402
    build_token_renderings,
    read_renderings,
    write_renderings,
)

INFERENCE_ONLY = {"exclude_train_vessels", "exclude_train_top_k"}


def _parse_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        return float(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", type=Path, required=True)
    parser.add_argument("--to", dest="target", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=None, help="required with --token-renderings")
    parser.add_argument("--catalog", type=Path, default=None, help="required with --token-renderings")
    parser.add_argument("--token-renderings", type=str, default=None, help="file name of the added rendering file")
    parser.add_argument("--add-reverse", type=Path, action="append", default=[], help="existing rendering JSONL to add")
    parser.add_argument("--set", action="append", default=[], help="KEY=VALUE inference-only config override")
    args = parser.parse_args(argv)

    if args.target.exists():
        parser.error(f"{args.target} already exists")
    if args.token_renderings and (args.train is None or args.catalog is None):
        parser.error("--token-renderings needs --train and --catalog")
    manifest = json.loads((args.source / MANIFEST_FILE).read_text(encoding="utf-8"))
    config = dict(manifest["config"])
    for item in args.set:
        key, sep, raw = item.partition("=")
        if not sep or key not in INFERENCE_ONLY:
            parser.error(f"--set accepts only {sorted(INFERENCE_ONLY)}, got {item!r}")
        config[key] = _parse_value(raw)
    manifest["config"] = config_to_dict(config_from_dict(config))

    shutil.copytree(args.source, args.target)
    if args.token_renderings:
        reverse = manifest.get("reverse", {}).get("files", [])
        if not reverse:
            parser.error("source model has no reverse channel")
        existing = read_renderings(args.source / name for name in reverse)
        renderings = build_token_renderings(pd.read_csv(args.train), pd.read_csv(args.catalog), existing)
        write_renderings(renderings, args.target / args.token_renderings)
        manifest["reverse"] = {"files": [*reverse, args.token_renderings]}
        print(f"token renderings: {len(renderings)} names -> {args.token_renderings}")
    for path in args.add_reverse:
        reverse = manifest.get("reverse", {}).get("files", [])
        if not reverse or path.name in reverse:
            parser.error(f"cannot add {path.name}: no reverse channel or duplicate file name")
        shutil.copyfile(path, args.target / path.name)
        manifest["reverse"] = {"files": [*reverse, path.name]}
        print(f"added rendering file {path.name}")
    (args.target / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"derived {args.target} from {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
