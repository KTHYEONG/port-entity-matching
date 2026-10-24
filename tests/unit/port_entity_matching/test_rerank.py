"""Invariant tests for the LLM likelihood rerank stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from port_entity_matching.candidates import CandidateSet
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.records import QueryRecord
from port_entity_matching.rerank import rerank_scores


def _config(**overrides: object) -> MatchConfig:
    defaults: dict[str, object] = {"llm_enabled": True, "llm_rerank_enabled": True}
    defaults.update(overrides)
    return MatchConfig(**defaults)  # type: ignore[arg-type]


def _index(size: int = 15) -> CatalogIndex:
    frame = pd.DataFrame(
        [
            {
                "vessel_id": f"V{i:02d}",
                "vessel_type": "Tanker",
                "vessel_name": f"Vessel {i:02d}",
                "size": 1000 + i,
                "unit": "DWT",
                "deadweight_tonnage": 1000 + i,
                "gross_tonnage": 1000 + i,
                "flag_name": "Panama",
            }
            for i in range(size)
        ]
    )
    return CatalogIndex.build(frame)


def _record(name: str = "흥아") -> QueryRecord:
    return QueryRecord(
        sample_id="R0",
        vessel_name=name,
        call_sign=None,
        gross_tonnage=1000.0,
        international_gross_tonnage=1000.0,
        flag_name=None,
        vessel_type=None,
        arrival_year=None,
    )


def _sets(positions: list[int]) -> list[CandidateSet]:
    return [
        CandidateSet(
            positions=np.asarray(positions, dtype=np.int64),
            sources=np.zeros(len(positions), dtype=np.uint8),
        )
    ]


def test_latin_rows_unchanged() -> None:
    index = _index()
    records = [_record("CALIDA")]
    candidate_sets = _sets([0, 1, 2])
    scores = np.array([1.0, 0.5, 0.0], dtype=np.float64)

    def scorer(name: str, names: object) -> tuple[float, ...]:
        raise AssertionError("scorer must not be called")

    out = rerank_scores(records, candidate_sets, scores, index, scorer, _config())
    np.testing.assert_array_equal(out, scores)


@pytest.mark.parametrize("flag", ["llm_enabled", "llm_rerank_enabled"])
def test_disabled_config_is_noop(flag: str) -> None:
    index = _index()
    records = [_record()]
    candidate_sets = _sets([0, 1, 2])
    scores = np.array([1.0, 0.5, 0.0], dtype=np.float64)

    def scorer(name: str, names: object) -> tuple[float, ...]:
        raise AssertionError("scorer must not be called")

    out = rerank_scores(records, candidate_sets, scores, index, scorer, _config(**{flag: False}))
    np.testing.assert_array_equal(out, scores)


def test_likelihood_flips_close_pair() -> None:
    index = _index()
    records = [_record()]
    candidate_sets = _sets([3, 7])
    scores = np.array([1.00, 0.98], dtype=np.float64)
    out = rerank_scores(records, candidate_sets, scores, index, lambda name, names: (-2.0, -0.1), _config())
    assert int(np.argmax(out)) == 1


def test_large_ranker_margin_survives() -> None:
    index = _index()
    records = [_record()]
    candidate_sets = _sets([3, 7])
    scores = np.array([3.0, 0.5], dtype=np.float64)
    out = rerank_scores(records, candidate_sets, scores, index, lambda name, names: (-2.0, 0.0), _config())
    assert int(np.argmax(out)) == 0


def test_only_top_k_scored() -> None:
    index = _index(20)
    records = [_record()]
    positions = list(range(15))
    candidate_sets = _sets(positions)
    scores = np.array([15.0 - i for i in range(15)], dtype=np.float64)
    seen: list[list[str]] = []

    def scorer(name: str, names: list[str]) -> tuple[float, ...]:
        seen.append(list(names))
        return tuple(-100.0 if i == 0 else 0.0 for i in range(len(names)))

    out = rerank_scores(records, candidate_sets, scores, index, scorer, _config(llm_rerank_top_k=10))
    assert len(seen) == 1
    assert seen[0] == [f"Vessel {i:02d}" for i in range(10)]
    top_min = float(np.min(out[:10]))
    assert bool((out[10:] < top_min).all())
    assert list(out[10:]) == sorted(out[10:], reverse=True)
    np.testing.assert_allclose(out[10:] - scores[10:], np.full(5, -12.0))


def test_ties_use_lowest_position() -> None:
    index = _index(20)
    records = [_record()]
    positions = [5, 2, 9, 1, 7]
    candidate_sets = _sets(positions)
    scores = np.zeros(5, dtype=np.float64)
    seen: list[list[str]] = []

    def scorer(name: str, names: list[str]) -> tuple[float, ...]:
        seen.append(list(names))
        return tuple(0.0 for _ in names)

    out = rerank_scores(records, candidate_sets, scores, index, scorer, _config(llm_rerank_top_k=2))
    assert seen[0] == ["Vessel 01", "Vessel 02"]
    assert float(out[3]) == float(out[1])


def test_input_not_mutated_and_finite() -> None:
    index = _index()
    records = [_record()]
    candidate_sets = _sets([0, 1, 2])
    scores = np.array([1.0, 0.9, 0.1], dtype=np.float64)
    snapshot = scores.copy()
    out = rerank_scores(records, candidate_sets, scores, index, lambda name, names: (-1.0, -0.5, -0.2), _config())
    np.testing.assert_array_equal(scores, snapshot)
    assert out is not scores
    assert bool(np.isfinite(out).all())


@pytest.mark.parametrize("lp", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_likelihood_raises(lp: float) -> None:
    index = _index()
    records = [_record()]
    candidate_sets = _sets([0, 1])
    scores = np.array([1.0, 0.9], dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        rerank_scores(records, candidate_sets, scores, index, lambda name, names: (lp, -0.1), _config())


@pytest.mark.parametrize("enabled", [False, True])
def test_output_preserves_shape_and_storage_independence(enabled: bool) -> None:
    scores = np.array([[1.0], [0.98]], dtype=np.float64)
    out = rerank_scores(
        [_record()], _sets([0, 1]), scores, _index(), lambda name, names: (-2.0, -0.1),
        _config(llm_rerank_enabled=enabled),
    )
    assert out.shape == scores.shape
    assert not np.shares_memory(out, scores)
    np.testing.assert_array_equal(scores, [[1.0], [0.98]])


def test_single_candidate_untouched() -> None:
    index = _index()
    records = [_record()]
    candidate_sets = _sets([4])
    scores = np.array([2.5], dtype=np.float64)

    def scorer(name: str, names: object) -> tuple[float, ...]:
        raise AssertionError("scorer must not be called")

    out = rerank_scores(records, candidate_sets, scores, index, scorer, _config())
    np.testing.assert_array_equal(out, scores)


def test_score_count_mismatch_raises() -> None:
    index = _index()
    records = [_record()]
    candidate_sets = _sets([0, 1, 2])
    scores = np.array([1.0, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="total candidate count"):
        rerank_scores(records, candidate_sets, scores, index, lambda name, names: (-1.0,), _config())


def test_scorer_length_mismatch_raises() -> None:
    index = _index()
    records = [_record()]
    candidate_sets = _sets([0, 1])
    scores = np.array([1.0, 0.9], dtype=np.float64)
    with pytest.raises(ValueError, match="one likelihood per candidate"):
        rerank_scores(records, candidate_sets, scores, index, lambda name, names: (-1.0,), _config())


def test_non_finite_output_raises() -> None:
    index = _index()
    records = [_record()]
    candidate_sets = _sets([0, 1])
    scores = np.array([float("inf"), 0.0], dtype=np.float64)
    with pytest.raises(ValueError, match="must be finite"):
        rerank_scores(records, candidate_sets, scores, index, lambda name, names: (-1.0, -1.0), _config())
