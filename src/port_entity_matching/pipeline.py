"""Inference pipeline: one deterministic vessel_id per record, independently.

The pipeline binds a catalog index to fitted artifacts and evaluates records in
fixed-size chunks. Every record's prediction depends only on its own fields,
the catalog and the fitted artifacts.
"""

from __future__ import annotations

import logging
import resource
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from port_entity_matching.artifacts import ArtifactBundle
from port_entity_matching.candidates import generate_candidates
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.features import build_features
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.ranker import score_ensemble, select_top1
from port_entity_matching.records import (
    CATALOG_COLUMNS,
    RECORD_COLUMNS,
    QueryRecord,
    read_csv_checked,
    records_from_frame,
)
from port_entity_matching.transliterate import (
    Transliterator,
    TransliteratorUnavailable,
    attach_alt_names,
)

__all__ = ["Matcher", "Prediction", "run_submission"]

logger = logging.getLogger(__name__)

SUBMISSION_COLUMNS: tuple[str, ...] = ("sample_id", "target_vessel_id")


@dataclass(frozen=True)
class Prediction:
    """Top-k ranked output for one record (dev/diagnostics path)."""

    sample_id: str
    vessel_ids: tuple[str, ...]
    scores: tuple[float, ...]
    memory_hit: bool


@dataclass(frozen=True)
class _Ranked:
    """Ranked candidate positions of one record with its call-sign memory hit."""

    positions: tuple[int, ...]
    scores: tuple[float, ...]
    memory_position: int | None
    n_candidates: int


class Matcher:
    """Catalog index + fitted bundle; predicts one vessel_id per record independently."""

    def __init__(self, index: CatalogIndex, bundle: ArtifactBundle, transliterator: Transliterator | None = None) -> None:
        """Bind a built catalog index to a loaded bundle."""
        if bundle.config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.index = index
        self.bundle = bundle
        self.transliterator = transliterator
        self.llm_load_seconds = 0.0
        self.llm_seconds = 0.0

    @classmethod
    def from_dir(cls, model_dir: str | Path, catalog: pd.DataFrame) -> Matcher:
        """Build index, load bundle, and load the transliterator from <model_dir>/llm when enabled and available."""
        index = CatalogIndex.build(catalog)
        bundle = ArtifactBundle.load(model_dir, index)
        transliterator: Transliterator | None = None
        load_seconds = 0.0
        if bundle.config.llm_enabled and bundle.fewshot:
            started = time.perf_counter()
            try:
                transliterator = Transliterator.load(Path(model_dir) / "llm", bundle.fewshot, bundle.config)
            except TransliteratorUnavailable as error:
                logger.warning("[SYS] llm unavailable reason=%s", type(error).__name__)
                transliterator = None
            finally:
                load_seconds = time.perf_counter() - started
            logger.info("[SYS] llm_load_seconds=%.2f", load_seconds)
        matcher = cls(index, bundle, transliterator)
        matcher.llm_load_seconds = load_seconds
        return matcher

    def predict(self, records: Sequence[QueryRecord]) -> list[str]:
        """One catalog vessel_id per record, in input order."""
        records = self._apply_llm(records)
        ranked = self._rank(records)
        self._log_summary(records, ranked)
        predictions: list[str] = []
        for item in ranked:
            position = item.memory_position if item.memory_position is not None else item.positions[0]
            predictions.append(self.index.vessel_ids[position])
        return predictions

    def predict_topk(self, records: Sequence[QueryRecord], k: int = 3) -> list[Prediction]:
        """Ranked top-k ids and scores per record for error analysis."""
        if k < 1:
            raise ValueError("k must be at least 1")
        records = self._apply_llm(records)
        ranked = self._rank(records)
        self._log_summary(records, ranked)
        predictions: list[Prediction] = []
        for record, item in zip(records, ranked, strict=True):
            positions = item.positions
            scores = item.scores
            if item.memory_position is not None:
                head = item.memory_position
                head_score = scores[positions.index(head)]
                rest = tuple(
                    (position, score)
                    for position, score in zip(positions, scores, strict=True)
                    if position != head
                )
                ordered = ((head, head_score), *rest)
            else:
                ordered = tuple(zip(positions, scores, strict=True))
            selected = ordered[:k]
            predictions.append(
                Prediction(
                    sample_id=record.sample_id,
                    vessel_ids=tuple(self.index.vessel_ids[position] for position, _ in selected),
                    scores=tuple(float(score) for _, score in selected),
                    memory_hit=item.memory_position is not None,
                )
            )
        return predictions

    def _apply_llm(self, records: Sequence[QueryRecord]) -> list[QueryRecord]:
        """Attach transliterated alt names for Hangul records lacking them."""
        records = list(records)
        if not self.bundle.config.llm_enabled or self.transliterator is None or not records:
            return records
        names = sorted(
            {record.vessel_name for record in records if not record.alt_names and is_hangul_name(record.vessel_name)}
        )
        if not names:
            return records
        started = time.perf_counter()
        table = self.transliterator.translate(names)
        elapsed = time.perf_counter() - started
        self.llm_seconds += elapsed
        logger.info(
            "[SYS] llm_seconds=%.2f names=%d generated=%d",
            elapsed,
            len(names),
            len(table),
        )
        pending = [record for record in records if not record.alt_names]
        attached_pending: list[QueryRecord] = attach_alt_names(pending, table)
        replacements = iter(attached_pending)
        attached = [record if record.alt_names else next(replacements) for record in records]
        return attached

    def _rank(self, records: Sequence[QueryRecord]) -> list[_Ranked]:
        """Rank candidates for every record, processing fixed-size chunks."""
        config = self.bundle.config
        step = config.batch_size
        ranked: list[_Ranked] = []
        for start in range(0, len(records), step):
            ranked.extend(self._rank_chunk(records[start : start + step]))
        return ranked

    def _rank_chunk(self, records: Sequence[QueryRecord]) -> list[_Ranked]:
        """Retrieve, featurize and score one chunk of records."""
        config = self.bundle.config
        candidate_sets = generate_candidates(records, self.index, self.bundle.memory, config)
        features = build_features(records, candidate_sets, self.index, self.bundle.maps, config)
        scores = score_ensemble(self.bundle.ranker, features, config)
        tops = select_top1(scores, candidate_sets)
        ranked: list[_Ranked] = []
        offset = 0
        for record, candidate_set, top in zip(records, candidate_sets, tops, strict=True):
            size = len(candidate_set.positions)
            chunk = scores[offset : offset + size]
            offset += size
            score_by_position = {
                int(position): float(value) for position, value in zip(candidate_set.positions, chunk, strict=True)
            }
            order = sorted(score_by_position, key=lambda position: (-score_by_position[position], position))
            head = int(top)
            positions = (head, *(position for position in order if position != head))
            values = tuple(score_by_position[position] for position in positions)
            ranked.append(
                _Ranked(
                    positions=positions,
                    scores=values,
                    memory_position=self.bundle.memory.lookup(record.call_sign),
                    n_candidates=size,
                )
            )
        return ranked

    def _log_summary(self, records: Sequence[QueryRecord], ranked: Sequence[_Ranked]) -> None:
        """Log counts and shapes only; never names, call signs, ids or predictions."""
        if not records:
            return
        sizes = [item.n_candidates for item in ranked]
        hangul_rows = sum(1 for record in records if is_hangul_name(record.vessel_name))
        memory_hits = sum(1 for item in ranked if item.memory_position is not None)
        logger.info(
            "[ALGO] inference rows=%d hangul_rows=%d candidates_mean=%.1f candidates_max=%d memory_hits=%d",
            len(records),
            hangul_rows,
            sum(sizes) / len(sizes),
            max(sizes),
            memory_hits,
        )


def run_submission(
    test_path: str | Path,
    catalog_path: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Read test + catalog, predict, write sample_id,target_vessel_id CSV; return output path."""
    started = time.perf_counter()
    test = read_csv_checked(test_path, RECORD_COLUMNS)
    catalog = read_csv_checked(catalog_path, CATALOG_COLUMNS)
    _validate_sample_ids(test)
    records = records_from_frame(test)
    logger.info(
        "[DATA] submission loaded test_rows=%d catalog_rows=%d hangul_rows=%d",
        len(test),
        len(catalog),
        sum(1 for record in records if is_hangul_name(record.vessel_name)),
    )

    matcher = Matcher.from_dir(model_dir, catalog)
    llm_load_seconds = matcher.llm_load_seconds
    predictions = matcher.predict(records)
    logger.info(
        "[SYS] llm_load_seconds=%.2f llm_seconds=%.2f",
        llm_load_seconds,
        matcher.llm_seconds,
    )

    known_ids = set(matcher.index.vessel_ids)
    invalid = sum(1 for prediction in predictions if prediction not in known_ids)
    if invalid:
        raise RuntimeError(f"{invalid} predicted vessel ids are absent from the catalog")

    frame = pd.DataFrame(
        {
            "sample_id": test["sample_id"].astype(str).to_numpy(),
            "target_vessel_id": predictions,
        },
        columns=list(SUBMISSION_COLUMNS),
    )
    destination = Path(output_path)
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")
    logger.info(
        "[EVAL] submission written rows=%d elapsed_seconds=%.2f",
        len(frame),
        time.perf_counter() - started,
    )
    logger.info(
        "[SYS] peak_rss_mb=%.1f",
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    )
    return destination


def _validate_sample_ids(test: pd.DataFrame) -> None:
    """Raise ValueError when sample_id contains nulls or duplicates."""
    if test["sample_id"].isna().any() or not test["sample_id"].is_unique:
        raise ValueError("test sample_id must be non-null and unique")
