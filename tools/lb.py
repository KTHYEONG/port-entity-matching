#!/usr/bin/env python3
"""Thin entry point for the LB ledger CLI."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from port_entity_matching.evaluation.lb.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
