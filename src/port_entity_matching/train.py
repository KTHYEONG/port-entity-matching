"""Training, vessel-disjoint cross-validation and the CLI entry point."""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from port_entity_matching.artifacts import ArtifactBundle
from port_entity_matching.candidates import generate_candidates
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.features import build_features
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.memory import CallSignMemory
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.ranker import score_pairs, train_ranker
from port_entity_matching.records import (
    CATALOG_COLUMNS,
    RECORD_COLUMNS,
    read_csv_checked,
    records_from_frame,
)

__all__ = [
    "CVReport",
    "TrainingData",
    "build_training_data",
    "cross_validate",
    "fit_bundle",
    "fold_indices",
    "main",
]

logger = logging.getLogger(__name__)

DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "is_hangul",
    "target_vessel_id",
    "pred_vessel_id",
    "correct",
    "top1_score",
    "top2_vessel_id",
    "top2_score",
    "margin",
    "target_retrieved",
    "n_candidates",
)


@dataclass(frozen=True)
class TrainingData:
    """Ranker training matrix plus candidate recall bookkeeping."""

    features: NDArray[np.float32]
    labels: NDArray[np.int64]
    group_sizes: list[int]
    recall_hangul: float
    recall_latin: float


@dataclass(frozen=True)
class CVReport:
    """Vessel-disjoint cross-validation metrics."""

    hangul_accuracy: float
    non_hangul_accuracy: float
    weighted_score: float
    recall_hangul: float
    recall_latin: float
    n_hangul: int
    n_non_hangul: int


def fold_indices(train: pd.DataFrame, n_splits: int) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Split train rows into vessel-disjoint folds grouped by target_vessel_id."""
    if "target_vessel_id" not in train.columns:
        raise ValueError("Train frame is missing required columns: target_vessel_id")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    groups = train["target_vessel_id"].astype(str).to_numpy()
    n_groups = int(pd.Series(groups).nunique())
    if n_groups < n_splits:
        raise ValueError(f"Cannot build {n_splits} vessel-disjoint folds from {n_groups} vessels")
    splitter = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros((len(train), 1), dtype=np.float64)
    return [
        (np.asarray(train_index, dtype=np.int64), np.asarray(valid_index, dtype=np.int64))
        for train_index, valid_index in splitter.split(placeholder, groups=groups)
    ]


def build_training_data(
    train: pd.DataFrame,
    index: CatalogIndex,
    maps: AttributeMaps,
    config: MatchConfig,
) -> TrainingData:
    """Build the candidate matrix, features and labels; rows whose target is not retrieved are excluded."""
    records = records_from_frame(train)
    targets = [str(value) for value in train["target_vessel_id"].tolist()]
    candidate_sets = generate_candidates(records, index, None, config)
    features = build_features(records, candidate_sets, index, maps, config)

    sizes = [len(candidate_set.positions) for candidate_set in candidate_sets]
    offsets = np.cumsum([0, *sizes])
    keep = np.zeros(features.shape[0], dtype=bool)
    labels = np.zeros(features.shape[0], dtype=np.int64)
    group_sizes: list[int] = []
    retrieved_hangul = retrieved_latin = 0
    total_hangul = total_latin = 0

    for position, (record, target) in enumerate(zip(records, targets, strict=True)):
        hangul = is_hangul_name(record.vessel_name)
        if hangul:
            total_hangul += 1
        else:
            total_latin += 1
        catalog_position = index.id_to_pos.get(target)
        if catalog_position is None:
            continue
        local = np.flatnonzero(candidate_sets[position].positions == catalog_position)
        if local.size == 0:
            continue
        start = int(offsets[position])
        end = int(offsets[position + 1])
        labels[start + int(local[0])] = 1
        keep[start:end] = True
        group_sizes.append(end - start)
        if hangul:
            retrieved_hangul += 1
        else:
            retrieved_latin += 1

    return TrainingData(
        features=features[keep],
        labels=labels[keep],
        group_sizes=group_sizes,
        recall_hangul=retrieved_hangul / total_hangul if total_hangul else 0.0,
        recall_latin=retrieved_latin / total_latin if total_latin else 0.0,
    )


def fit_bundle(train: pd.DataFrame, catalog: pd.DataFrame, config: MatchConfig) -> ArtifactBundle:
    """Fit the attribute maps, the call-sign memory and the ranker on all given train rows."""
    index = CatalogIndex.build(catalog)
    maps = AttributeMaps.fit(train, index)
    memory = CallSignMemory.fit(train, index)
    training = build_training_data(train, index, maps, config)
    booster = train_ranker(training.features, training.labels, training.group_sizes, config)
    logger.info(
        "[ALGO] fitted rows=%d groups=%d recall_hangul=%.4f recall_latin=%.4f",
        len(train),
        len(training.group_sizes),
        training.recall_hangul,
        training.recall_latin,
    )
    return ArtifactBundle(booster=booster, maps=maps, memory=memory, config=config)


def cross_validate(
    train: pd.DataFrame,
    catalog: pd.DataFrame,
    config: MatchConfig,
    diagnostics_path: str | Path | None = None,
) -> CVReport:
    """Run vessel-disjoint cross-validation without call-sign memory and report the bucket metrics."""
    index = CatalogIndex.build(catalog)
    folds = fold_indices(train, config.cv_splits)
    logger.info(
        "[DATA] cv start train_rows=%d catalog_rows=%d folds=%d",
        len(train),
        len(index),
        len(folds),
    )

    diagnostic_rows: list[dict[str, object]] = []
    hangul_correct = latin_correct = 0
    hangul_total = latin_total = 0
    hangul_retrieved = latin_retrieved = 0

    for fold_number, (train_index, valid_index) in enumerate(folds, start=1):
        fold_train = train.iloc[train_index]
        fold_valid = train.iloc[valid_index]
        maps = AttributeMaps.fit(fold_train, index)
        training = build_training_data(fold_train, index, maps, config)
        booster = train_ranker(training.features, training.labels, training.group_sizes, config)
        logger.info(
            "[ALGO] fold=%d train_rows=%d groups=%d recall_hangul=%.4f recall_latin=%.4f",
            fold_number,
            len(fold_train),
            len(training.group_sizes),
            training.recall_hangul,
            training.recall_latin,
        )

        records = records_from_frame(fold_valid)
        fold_targets = [str(value) for value in fold_valid["target_vessel_id"].tolist()]
        candidate_sets = generate_candidates(records, index, None, config)
        features = build_features(records, candidate_sets, index, maps, config)
        scores = score_pairs(booster, features, config)

        sizes = [len(candidate_set.positions) for candidate_set in candidate_sets]
        offsets = np.cumsum([0, *sizes])
        fold_hangul = fold_hangul_correct = 0
        fold_latin = fold_latin_correct = 0
        for position, record in enumerate(records):
            start = int(offsets[position])
            end = int(offsets[position + 1])
            chunk = scores[start:end]
            catalog_positions = candidate_sets[position].positions
            top_offset, top_score, second_id, second_score = _best_two(chunk, catalog_positions, index)
            predicted = index.vessel_ids[catalog_positions[top_offset]]
            target = fold_targets[position]
            hangul = is_hangul_name(record.vessel_name)
            retrieved = target in {index.vessel_ids[catalog_position] for catalog_position in catalog_positions}
            correct = predicted == target
            margin = float(top_score - second_score) if second_score is not None else math.nan
            diagnostic_rows.append(
                {
                    "sample_id": record.sample_id,
                    "is_hangul": int(hangul),
                    "target_vessel_id": target,
                    "pred_vessel_id": predicted,
                    "correct": int(correct),
                    "top1_score": float(top_score),
                    "top2_vessel_id": second_id,
                    "top2_score": float(second_score) if second_score is not None else math.nan,
                    "margin": margin,
                    "target_retrieved": int(retrieved),
                    "n_candidates": len(catalog_positions),
                }
            )
            if hangul:
                fold_hangul += 1
                fold_hangul_correct += int(correct)
                hangul_retrieved += int(retrieved)
            else:
                fold_latin += 1
                fold_latin_correct += int(correct)
                latin_retrieved += int(retrieved)

        hangul_total += fold_hangul
        hangul_correct += fold_hangul_correct
        latin_total += fold_latin
        latin_correct += fold_latin_correct
        logger.info(
            "[EVAL] fold=%d hangul_accuracy=%.4f (%d rows) non_hangul_accuracy=%.4f (%d rows)",
            fold_number,
            fold_hangul_correct / fold_hangul if fold_hangul else math.nan,
            fold_hangul,
            fold_latin_correct / fold_latin if fold_latin else math.nan,
            fold_latin,
        )

    hangul_accuracy = hangul_correct / hangul_total if hangul_total else math.nan
    latin_accuracy = latin_correct / latin_total if latin_total else math.nan
    weighted = 0.7 * hangul_accuracy + 0.3 * latin_accuracy
    logger.info(
        "[EVAL] overall hangul_accuracy=%.4f non_hangul_accuracy=%.4f weighted_score=%.4f",
        hangul_accuracy,
        latin_accuracy,
        weighted,
    )

    if diagnostics_path is not None:
        _write_diagnostics(diagnostics_path, diagnostic_rows)
        logger.info("[DATA] diagnostics written rows=%d path=%s", len(diagnostic_rows), diagnostics_path)

    return CVReport(
        hangul_accuracy=hangul_accuracy,
        non_hangul_accuracy=latin_accuracy,
        weighted_score=weighted,
        recall_hangul=hangul_retrieved / hangul_total if hangul_total else 0.0,
        recall_latin=latin_retrieved / latin_total if latin_total else 0.0,
        n_hangul=hangul_total,
        n_non_hangul=latin_total,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: fit and save artifacts and/or run cross-validation."""
    parser = argparse.ArgumentParser(prog="port_entity_matching.train", description="Fit and validate the vessel matcher.")
    parser.add_argument("--train", required=True, type=Path, help="training CSV with target_vessel_id")
    parser.add_argument("--catalog", required=True, type=Path, help="vessel catalog CSV")
    parser.add_argument("--out", type=Path, default=None, help="directory the fitted artifacts are written to")
    parser.add_argument("--cv", action="store_true", help="run vessel-disjoint cross-validation")
    parser.add_argument("--diagnostics", type=Path, default=None, help="CSV path for the out-of-fold diagnostics")
    args = parser.parse_args(argv)
    if args.cv is False and args.out is None:
        parser.error("nothing to do: pass --cv and/or --out")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    train = read_csv_checked(args.train, (*RECORD_COLUMNS, "target_vessel_id"))
    catalog = read_csv_checked(args.catalog, CATALOG_COLUMNS)
    config = MatchConfig()
    logger.info("[DATA] loaded train_rows=%d catalog_rows=%d", len(train), len(catalog))

    if args.cv:
        report = cross_validate(train, catalog, config, args.diagnostics)
        logger.info(
            "[EVAL] cv weighted_score=%.4f hangul_accuracy=%.4f non_hangul_accuracy=%.4f",
            report.weighted_score,
            report.hangul_accuracy,
            report.non_hangul_accuracy,
        )
    if args.out is not None:
        bundle = fit_bundle(train, catalog, config)
        bundle.save(args.out)
        logger.info("[SYS] artifacts written path=%s", args.out)
    return 0


def _best_two(
    scores: NDArray[np.float64],
    catalog_positions: NDArray[np.int64],
    index: CatalogIndex,
) -> tuple[int, float, str, float | None]:
    """Return the best offset and score plus the runner-up vessel id and score."""
    order = sorted(range(len(catalog_positions)), key=lambda offset: (-float(scores[offset]), int(catalog_positions[offset])))
    best = order[0]
    second_id = ""
    second_score: float | None = None
    if len(order) > 1:
        runner_up = order[1]
        second_id = index.vessel_ids[int(catalog_positions[runner_up])]
        second_score = float(scores[runner_up])
    return best, float(scores[best]), second_id, second_score


def _write_diagnostics(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    destination = Path(path)
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(DIAGNOSTIC_COLUMNS))
    frame.to_csv(destination, index=False, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())  # pragma: no cover
