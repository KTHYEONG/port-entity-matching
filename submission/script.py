"""Evaluation entry point: data/test.csv + data/vessel_catalog.csv to output/submission.csv."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "model"))

from port_entity_matching.pipeline import run_submission  # noqa: E402

TEST_PATH = "data/test.csv"
CATALOG_PATH = "data/vessel_catalog.csv"
OUTPUT_PATH = "output/submission.csv"


def main() -> None:
    """Evaluation entry: data/test.csv + data/vessel_catalog.csv -> output/submission.csv."""
    logging.basicConfig(level=logging.INFO)
    run_submission(
        test_path=TEST_PATH,
        catalog_path=CATALOG_PATH,
        model_dir=SCRIPT_DIR / "model" / "artifacts",
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
