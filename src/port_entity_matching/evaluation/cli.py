"""Evaluation CLI: cross-validation, stress tests and paired run comparisons.

The CLI is development-only: it lives in the non-shipped ``evaluation``
subpackage and is never imported by the submission path. Every command is
row-independent, deterministic and writes only below the requested output
directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig, config_from_dict, config_to_dict
from port_entity_matching.evaluation.cache import CandidateCache
from port_entity_matching.evaluation.paths import guard_data_path
from port_entity_matching.evaluation.perturb import PERTURBATIONS, perturb_record
from port_entity_matching.evaluation.stats import (
    OOF_COLUMNS,
    bucket_scores,
    load_oof,
    paired_difference,
    suite_scores,
    vessel_bootstrap_ci,
)
from port_entity_matching.records import CATALOG_COLUMNS, RECORD_COLUMNS, read_csv_checked
from port_entity_matching.train import cross_validate_run

__all__ = ["main", "parse_overrides"]

logger = logging.getLogger(__name__)

_BOOL_TRUE: frozenset[str] = frozenset({"true", "1"})
_BOOL_FALSE: frozenset[str] = frozenset({"false", "0"})


def parse_overrides(pairs: Sequence[str]) -> MatchConfig:
    """Build a MatchConfig from 'key=value' strings typed by MatchConfig field types; ValueError on unknown key or bad value."""
    defaults = cast("dict[str, object]", dict(config_to_dict(MatchConfig())))
    values: dict[str, object] = dict(defaults)
    for pair in pairs:
        key, separator, raw = str(pair).partition("=")
        key = key.strip()
        if not separator:
            raise ValueError(f"Invalid override '{pair}': expected KEY=VALUE (key '{key}')")
        if key not in defaults:
            raise ValueError(f"Unknown config key '{key}'")
        values[key] = _cast_value(key, raw.strip(), defaults[key])
    return config_from_dict(values)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluation CLI with subcommands cv, stress, compare; returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        if args.command == "cv":
            return _run_cv(args)
        if args.command == "stress":
            return _run_stress(args)
        return _run_compare(args)
    except (ValueError, FileNotFoundError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree with the cv, stress and compare subcommands."""
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluation toolkit: vessel-disjoint CV, perturbation stress tests and paired comparisons.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one MatchConfig field; repeatable",
    )
    common.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache/candidates"))
    common.add_argument("--no-cache", action="store_true")

    cv = subparsers.add_parser("cv", parents=[common], help="run vessel-disjoint cross-validation")
    cv.add_argument("--train", required=True, type=Path, help="training CSV with target_vessel_id")
    cv.add_argument("--catalog", required=True, type=Path, help="vessel catalog CSV")
    cv.add_argument("--out-dir", required=True, type=Path, help="directory for oof.csv and summary.json")

    stress = subparsers.add_parser("stress", parents=[common], help="run perturbation stress tests")
    stress.add_argument("--train", required=True, type=Path, help="training CSV with target_vessel_id")
    stress.add_argument("--catalog", required=True, type=Path, help="vessel catalog CSV")
    stress.add_argument("--out-dir", required=True, type=Path, help="directory for oof_long.csv and stress.json")
    stress.add_argument(
        "--scenarios",
        default="all",
        help="comma-separated perturbation names, or 'all' for every known scenario",
    )
    stress.add_argument("--perturb-seed", type=int, default=0, help="seed of the perturbation draws")

    compare = subparsers.add_parser("compare", help="compare two out-of-fold CSV runs")
    compare.add_argument("a", type=Path, help="baseline OOF CSV")
    compare.add_argument("b", type=Path, help="candidate OOF CSV")
    compare.add_argument("--scenario", default="clean", help="scenario name to compare (default: clean)")
    compare.add_argument("--json", dest="json_path", type=Path, default=None, help="optional JSON report path")
    return parser


def _run_cv(args: argparse.Namespace) -> int:
    """Run vessel-disjoint cross-validation and write oof.csv plus summary.json."""
    started = time.perf_counter()
    config = parse_overrides(_overrides(args))
    train, catalog = _load_data(args)
    out_dir = _prepare_out_dir(args.out_dir)
    index = CatalogIndex.build(catalog)
    cache = _open_cache(args, catalog, config)
    try:
        if cache is None:
            run = cross_validate_run(train, catalog, config, scenarios=None, index=index)
        else:
            run = cross_validate_run(
                train,
                catalog,
                config,
                scenarios=None,
                candidate_fn=lambda records: cache.fetch(records, index, config),
                index=index,
            )
    finally:
        _flush_cache(cache)
    clean = _clean_rows(run.oof)
    _write_oof(out_dir / "oof.csv", clean)
    summary = _cv_summary(clean, config, time.perf_counter() - started)
    _write_json(out_dir / "summary.json", summary)
    logger.info(
        "[EVAL] cv score=%s kor=%s lat=%s error_vessels=%s",
        summary["score"],
        summary["kor"],
        summary["lat"],
        summary["error_vessels"],
    )
    logger.info("[SYS] cv artifacts written out_dir=%s elapsed_seconds=%.1f", out_dir, summary["elapsed_seconds"])
    sys.stdout.write(f"out_dir={out_dir} files=oof.csv,summary.json\n")
    return 0


def _run_stress(args: argparse.Namespace) -> int:
    """Run the perturbation stress scenarios and write oof_long.csv plus stress.json."""
    started = time.perf_counter()
    scenarios = _parse_scenarios(args.scenarios)
    config = parse_overrides(_overrides(args))
    train, catalog = _load_data(args)
    out_dir = _prepare_out_dir(args.out_dir)
    seed = int(args.perturb_seed)
    scenario_map = {name: (lambda record, s=name: perturb_record(record, s, seed)) for name in scenarios}
    index = CatalogIndex.build(catalog)
    cache = _open_cache(args, catalog, config)
    try:
        if cache is None:
            run = cross_validate_run(train, catalog, config, scenarios=scenario_map, index=index)
        else:
            run = cross_validate_run(
                train,
                catalog,
                config,
                scenarios=scenario_map,
                candidate_fn=lambda records: cache.fetch(records, index, config),
                index=index,
            )
    finally:
        _flush_cache(cache)
    _write_oof(out_dir / "oof_long.csv", run.oof)
    report = {name: _stress_entry(run.oof, name) for name in scenarios}
    _write_json(out_dir / "stress.json", report)
    suite = suite_scores(run.oof, scenarios)
    _write_json(
        out_dir / "suite.json",
        {
            group: {key: _number(value) if isinstance(value, float) else value for key, value in entry.items()}
            for group, entry in suite.items()
        },
    )
    logger.info(
        "[EVAL] suite all_score=%s seen_score=%s unseen_score=%s",
        _format(suite["all"]["score"]),
        _format(suite["seen"]["score"]),
        _format(suite["unseen"]["score"]),
    )
    logger.info("[EVAL] stress scenarios=%d elapsed_seconds=%.1f", len(scenarios), time.perf_counter() - started)
    logger.info("[SYS] stress artifacts written out_dir=%s", out_dir)
    sys.stdout.write(f"out_dir={out_dir} files=oof_long.csv,stress.json,suite.json\n")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    """Compare two out-of-fold runs and report the paired score difference."""
    left = load_oof(args.a, args.scenario)
    right = load_oof(args.b, args.scenario)
    difference = paired_difference(left, right)
    payload: dict[str, Any] = {
        "delta": _number(difference.delta),
        "ci_low": _number(difference.ci_low),
        "ci_high": _number(difference.ci_high),
        "vessels_fixed": difference.vessels_fixed,
        "vessels_broken": difference.vessels_broken,
        "rows_fixed": difference.rows_fixed,
        "rows_broken": difference.rows_broken,
    }
    sys.stdout.write(
        f"delta={_format(difference.delta)} "
        f"ci95=[{_format(difference.ci_low)}, {_format(difference.ci_high)}] "
        f"vessels_fixed={difference.vessels_fixed} vessels_broken={difference.vessels_broken} "
        f"rows_fixed={difference.rows_fixed} rows_broken={difference.rows_broken}\n"
    )
    if args.json_path is not None:
        _write_json(_guard_data_path(Path(args.json_path)), payload)
    return 0


def _load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the training rows and the catalog, logging only row counts."""
    train = read_csv_checked(args.train, (*RECORD_COLUMNS, "target_vessel_id"))
    if train["sample_id"].isna().any() or train["sample_id"].duplicated().any():
        raise ValueError("Train sample_id values must be non-missing and unique")
    catalog = read_csv_checked(args.catalog, CATALOG_COLUMNS)
    logger.info("[DATA] loaded train_rows=%d catalog_rows=%d", len(train), len(catalog))
    return train, catalog


def _cast_value(key: str, raw: str, default: object) -> object:
    """Cast one override string to the type of the matching config field default."""
    if isinstance(default, bool):
        text = raw.strip().lower()
        if text in _BOOL_TRUE:
            return True
        if text in _BOOL_FALSE:
            return False
        raise ValueError(f"Invalid boolean value for '{key}': '{raw}' (expected true|false|1|0)")
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError as error:
            raise ValueError(f"Invalid integer value for '{key}': '{raw}'") from error
    if isinstance(default, float):
        try:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("value must be finite")
            return value
        except ValueError as error:
            raise ValueError(f"Invalid float value for '{key}': '{raw}'") from error
    raise ValueError(f"Unsupported config field type for '{key}'")


def _overrides(args: argparse.Namespace) -> list[str]:
    """Return the repeated --set values as a list."""
    return [str(value) for value in (getattr(args, "overrides", None) or [])]


def _parse_scenarios(text: str) -> list[str]:
    """Split a --scenarios value into scenario names; ValueError on an unknown name."""
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    if names == ["all"]:
        return list(PERTURBATIONS)
    if not names or "all" in names:
        raise ValueError("Scenarios must be a non-empty comma list or 'all' alone")
    unknown = [name for name in names if name not in PERTURBATIONS]
    if unknown:
        raise ValueError(f"Unknown scenario '{unknown[0]}'. Valid scenarios: {', '.join(PERTURBATIONS)}")
    return list(dict.fromkeys(names))


def _clean_rows(oof: pd.DataFrame) -> pd.DataFrame:
    """Return the clean out-of-fold rows in OOF_COLUMNS order without the scenario column."""
    clean = oof[oof["scenario"].astype(str) == "clean"]
    return clean.drop(columns=["scenario"]).loc[:, list(OOF_COLUMNS)].reset_index(drop=True)


def _cv_summary(clean: pd.DataFrame, config: MatchConfig, elapsed: float) -> dict[str, Any]:
    """Summarize the clean out-of-fold rows into the summary.json payload."""
    scores = bucket_scores(clean)
    hangul = clean["is_hangul"].astype(bool).to_numpy()
    correct = clean["correct"].astype(bool).to_numpy()
    retrieved = clean["target_retrieved"].astype(bool).to_numpy()
    errors = {
        "kor": int(np.count_nonzero(~correct & hangul)),
        "lat": int(np.count_nonzero(~correct & ~hangul)),
    }
    error_vessels = int(clean.loc[~correct, "target_vessel_id"].astype(str).nunique())
    recall = {
        "kor": float(retrieved[hangul].mean()) if bool(hangul.any()) else math.nan,
        "lat": float(retrieved[~hangul].mean()) if bool((~hangul).any()) else math.nan,
    }
    intervals = {key: [_number(low), _number(high)] for key, (low, high) in vessel_bootstrap_ci(clean).items()}
    return {
        "config": config_to_dict(config),
        "kor": _number(scores["kor"]),
        "lat": _number(scores["lat"]),
        "score": _number(scores["score"]),
        "errors": errors,
        "error_vessels": error_vessels,
        "recall": {key: _number(value) for key, value in recall.items()},
        "ci": intervals,
        "vessel_bootstrap_ci": intervals,
        "elapsed_seconds": round(float(elapsed), 3),
    }


def _stress_entry(oof: pd.DataFrame, scenario: str) -> dict[str, Any]:
    """Summarize one perturbation scenario against the clean rows of the same records."""
    perturbed = oof[oof["scenario"].astype(str) == scenario]
    affected = len(perturbed)
    if affected == 0:
        return {
            "affected_rows": 0,
            "recall_perturbed": None,
            "acc_clean": None,
            "acc_perturbed": None,
            "delta": None,
        }
    clean = oof[oof["scenario"].astype(str) == "clean"][["sample_id", "correct"]]
    merged = perturbed.merge(clean, on="sample_id", how="left", suffixes=("", "_clean"), validate="one_to_one")
    if merged["correct_clean"].isna().any():
        raise ValueError("Perturbed sample_id values must have matching clean rows")
    acc_perturbed = float(perturbed["correct"].astype(bool).to_numpy().mean())
    acc_clean = float(merged["correct_clean"].astype(bool).to_numpy().mean())
    recall = float(perturbed["target_retrieved"].astype(bool).to_numpy().mean())
    return {
        "affected_rows": affected,
        "recall_perturbed": recall,
        "acc_clean": _number(acc_clean),
        "acc_perturbed": _number(acc_perturbed),
        "delta": _number(acc_perturbed - acc_clean) if not math.isnan(acc_clean) else None,
    }


def _number(value: float) -> float | None:
    """Return a JSON-safe float; NaN becomes None."""
    number = float(value)
    return None if math.isnan(number) else number


def _format(value: float) -> str:
    """Format a signed fixed-point number; NaN becomes 'nan'."""
    number = float(value)
    return "nan" if math.isnan(number) else f"{number:+.4f}"


def _prepare_out_dir(path: Path) -> Path:
    """Create the output directory when missing; refuse to write inside a data directory."""
    resolved = _guard_data_path(Path(path))
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _guard_data_path(path: Path) -> Path:
    resolved: Path = guard_data_path(Path(path))
    return resolved


def _open_cache(args: argparse.Namespace, catalog: pd.DataFrame, config: MatchConfig) -> CandidateCache | None:
    """Open the candidate cache unless --no-cache is given."""
    if bool(getattr(args, "no_cache", False)):
        return None
    return CandidateCache.open(args.cache_dir, catalog, config)


def _flush_cache(cache: CandidateCache | None) -> None:
    """Flush the cache and log hit/miss statistics."""
    if cache is None:
        return
    cache.flush()
    hits, misses = cache.stats
    logger.info("[SYS] candidate_cache hits=%d misses=%d fingerprint=%s", hits, misses, cache.fingerprint)


def _write_oof(path: Path, frame: pd.DataFrame) -> None:
    """Write an out-of-fold frame as CSV."""
    frame.to_csv(_guard_data_path(path), index=False, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    """Write a JSON payload with sorted keys; NaN values are rejected."""
    destination = _guard_data_path(Path(path))
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    destination.write_text(text, encoding="utf-8")
