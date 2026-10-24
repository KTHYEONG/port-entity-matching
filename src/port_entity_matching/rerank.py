"""LLM likelihood rerank of top-k Hangul candidates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from port_entity_matching.candidates import CandidateSet
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.normalize import is_hangul_name
from port_entity_matching.records import QueryRecord
from port_entity_matching.transliterate import CandidateScoreFn

__all__ = ["rerank_scores"]


def rerank_scores(
    records: Sequence[QueryRecord],
    candidate_sets: Sequence[CandidateSet],
    scores: NDArray[np.float64],
    index: CatalogIndex,
    scorer: CandidateScoreFn,
    config: MatchConfig,
) -> NDArray[np.float64]:
    """Return scores where each Hangul record's top-k candidates are re-scored by LLM likelihood; other rows unchanged."""
    shape = scores.shape
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    total = sum(len(candidate_set.positions) for candidate_set in candidate_sets)
    if values.shape[0] != total:
        raise ValueError("scores must match the total candidate count")
    out = values.copy()
    if not config.llm_enabled or not config.llm_rerank_enabled:
        return out.reshape(shape)
    weight = float(config.llm_rerank_weight)
    top_k = int(config.llm_rerank_top_k)
    offset = 0
    for record, candidate_set in zip(records, candidate_sets, strict=True):
        size = len(candidate_set.positions)
        original = values[offset : offset + size]
        if is_hangul_name(record.vessel_name) and size >= 2 and top_k >= 1:
            k = min(top_k, size)
            order = sorted(
                range(size),
                key=lambda local: (-float(original[local]), int(candidate_set.positions[local])),
            )
            top = order[:k]
            names = [index.names[int(candidate_set.positions[local])] for local in top]
            likelihoods = tuple(float(value) for value in scorer(record.vessel_name, names))
            if len(likelihoods) != k:
                raise ValueError("scorer must return one likelihood per candidate")
            if not all(np.isfinite(likelihoods)):
                raise ValueError("scorer returned a non-finite likelihood")
            peak = max(likelihoods)
            adjusted = {
                local: float(original[local]) + weight * (lp - peak) for local, lp in zip(top, likelihoods, strict=True)
            }
            top_min_original = min(float(original[local]) for local in top)
            top_min_adjusted = min(adjusted.values())
            for local in top:
                out[offset + local] = adjusted[local]
            if k < size:
                rest = [local for local in range(size) if local not in adjusted]
                shift = top_min_adjusted - top_min_original - 1.0
                for local in rest:
                    out[offset + local] = float(original[local]) + shift
        offset += size
    if not bool(np.isfinite(out).all()):
        raise ValueError("reranked scores must be finite")
    return out.reshape(shape)
