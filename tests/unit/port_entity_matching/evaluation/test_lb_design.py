"""Invariant tests for probe ranking."""

from __future__ import annotations

import numpy as np
import pytest

from port_entity_matching.evaluation.lb.design import rank_probes
from port_entity_matching.evaluation.lb.ledger import Fact
from port_entity_matching.evaluation.lb.response import ResponseMatrix, parse_cell


def _matrix() -> ResponseMatrix:
    cells = ("K|g1|clean|exact", "K|g2|clean|exact")
    values = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    return ResponseMatrix(
        script="K",
        cells=cells,
        families=tuple(parse_cell(c) for c in cells),
        keys=("t", "info", "const"),
        kinds={"t": "abs", "info": "abs", "const": "abs"},
        values=values,
        rows=np.array([30, 30], dtype=np.int64),
    )


def test_informative_probe_ranks_first() -> None:
    ranked = rank_probes(_matrix(), [], ["info", "const"], ["t"])
    assert [score.candidate for score in ranked] == ["info", "const"]
    assert ranked[0].expected_width < ranked[1].expected_width
    assert ranked[0].feasible_points == 5


def test_unobservable_candidate_omitted() -> None:
    matrix = _matrix()
    fact = Fact(
        entry_id="e1", channel="K", policy_key="info", kind="abs",
        base=None, value=532 * 0.6, prediction_hit=None,
    )
    assert rank_probes(matrix, [fact], ["info"], ["t"]) == []


def test_deterministic_order() -> None:
    matrix = _matrix()
    first = rank_probes(matrix, [], ["info", "const"], ["t"])
    second = rank_probes(matrix, [], ["info", "const"], ["t"])
    assert first == second


def test_self_target_respects_previous_candidate_fact() -> None:
    fact = Fact("e1", "K", "info", "abs", None, 266.0, None)
    score = rank_probes(_matrix(), [fact], ["info"], ["info"])[0]
    assert score.feasible_points < 5
    assert score.expected_width < 100


def test_invalid_grid_rejected() -> None:
    with pytest.raises(ValueError, match="grid"):
        rank_probes(_matrix(), [], ["info"], ["t"], grid=1)


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValueError, match="ghost"):
        rank_probes(_matrix(), [], ["ghost"], ["t"])
    with pytest.raises(ValueError, match="ghost"):
        rank_probes(_matrix(), [], ["info"], ["ghost"])
