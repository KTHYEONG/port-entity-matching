"""Invariant tests for set-membership inference and the LOO gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from port_entity_matching.evaluation.lb.decode import N_K
from port_entity_matching.evaluation.lb.feasible import (
    FeasibilityError,
    Interval,
    classify,
    feasible_interval,
    loo_gate,
)
from port_entity_matching.evaluation.lb.ledger import Fact
from port_entity_matching.evaluation.lb.response import ResponseMatrix, parse_cell

CELLS = ("K|g1|clean|exact", "K|g1|jamo|exact", "K|g2|clean|exact", "K|g2|jamo|exact")


def _matrix() -> ResponseMatrix:
    values = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return ResponseMatrix(
        script="K",
        cells=CELLS,
        families=tuple(parse_cell(c) for c in CELLS),
        keys=("a", "b", "h"),
        kinds={"a": "abs", "b": "abs", "h": "abs"},
        values=values,
        rows=np.array([25, 25, 25, 25], dtype=np.int64),
    )


def _fact(key: str, value: float, entry: str = "e1") -> Fact:
    return Fact(
        entry_id=entry, channel="K", policy_key=key, kind="abs",
        base=None, value=value, prediction_hit=None,
    )


def test_true_mixture_is_inside_every_interval() -> None:
    matrix = _matrix()
    interval = feasible_interval(matrix, [_fact("a", N_K * 0.3), _fact("b", N_K * 0.7)], "h")
    assert interval.lo - 1e-6 <= N_K * 0.3 <= interval.hi + 1e-6


def test_inconsistent_facts_raise() -> None:
    matrix = _matrix()
    with pytest.raises(FeasibilityError, match="family library inconsistent with LB facts") as excinfo:
        feasible_interval(matrix, [_fact("a", 10.0, "e1"), _fact("a", 100.0, "e2")], "b")
    assert "a" in str(excinfo.value)


def test_more_facts_never_widen() -> None:
    matrix = _matrix()
    narrow_facts = [_fact("a", N_K * 0.3)]
    wide = feasible_interval(matrix, narrow_facts, "h")
    narrow = feasible_interval(matrix, [*narrow_facts, _fact("b", N_K * 0.7)], "h")
    assert narrow.lo >= wide.lo - 1e-9
    assert narrow.hi <= wide.hi + 1e-9


def test_unknown_cells_only_widen() -> None:
    matrix = _matrix()
    facts = [_fact("a", N_K * 0.3), _fact("b", N_K * 0.7)]
    plain = feasible_interval(matrix, facts, "h")
    widened = feasible_interval(matrix, facts, "h", unknown_cells={CELLS[0]})
    assert widened.lo <= plain.lo + 1e-6
    assert widened.hi >= plain.hi - 1e-6


def test_self_facts_are_excluded() -> None:
    matrix = _matrix()
    facts = [_fact("a", N_K * 0.3), _fact("b", N_K * 0.7)]
    with_self = feasible_interval(matrix, [*facts, _fact("h", 10000.0)], "h")
    without_self = feasible_interval(matrix, facts, "h")
    assert with_self == without_self


def test_unknown_fact_key_fails_closed() -> None:
    matrix = _matrix()
    facts = [_fact("a", N_K * 0.3), _fact("zzz", 5.0)]
    with pytest.raises(ValueError, match="zzz"):
        feasible_interval(matrix, facts, "h")
    interval = feasible_interval(matrix, facts, "h", ignore=["zzz"])
    assert interval.lo <= interval.hi


def test_classify_is_strict() -> None:
    assert classify(Interval(lo=0.0, hi=5.0)) == "lb_required"
    assert classify(Interval(lo=0.1, hi=5.0)) == "adopt"
    assert classify(Interval(lo=-3.0, hi=-0.1)) == "reject"


def test_empty_held_out_rejected() -> None:
    with pytest.raises(ValueError, match="held_out"):
        loo_gate(_matrix(), [_fact("a", 1.0)], [])


def test_conflicting_held_out_observations_fail_closed() -> None:
    with pytest.raises(ValueError, match="Conflicting LB facts"):
        loo_gate(_two_cell_matrix(), [_fact("p", 0), _fact("p", 1064)], ["p"])


def test_target_kind_mismatch_rejected() -> None:
    fact = Fact("bad", "K", "h", "net", "base", 0, None)
    with pytest.raises(ValueError, match="Kind mismatch"):
        feasible_interval(_matrix(), [fact], "h")


def _two_cell_matrix(p_second: float = 1.0) -> ResponseMatrix:
    cells = ("K|g1|clean|exact", "K|g2|clean|exact")
    values = np.array([[1.0, 0.0], [1.0, p_second]], dtype=np.float64)
    return ResponseMatrix(
        script="K",
        cells=cells,
        families=tuple(parse_cell(c) for c in cells),
        keys=("a", "p"),
        kinds={"a": "abs", "p": "abs"},
        values=values,
        rows=np.array([20, 20], dtype=np.int64),
    )


def test_gate_reports_coverage() -> None:
    matrix = _two_cell_matrix()
    report = loo_gate(matrix, [_fact("a", N_K * 0.5), _fact("p", 0.0)], ["p"])
    assert report.coverage < 1
    assert report.passes is False
    assert report.intervals["p"].lo > 0


def test_gate_passes_when_covered() -> None:
    matrix = _two_cell_matrix(p_second=0.0)
    report = loo_gate(matrix, [_fact("a", N_K * 0.5), _fact("p", N_K * 0.5)], ["p"])
    assert report.coverage == 1.0
    assert report.passes is True


@pytest.mark.heavy
@pytest.mark.slow
def test_acceptance_loo_gate_reproduces_probe() -> None:
    from port_entity_matching.evaluation.lb.ledger import DEFAULT_LEDGER_PATH, Ledger
    from port_entity_matching.evaluation.lb.response import response_matrix

    pol = Path("scratch/twin/pol.csv")
    models = Path("scratch/twin/models.csv")
    synk = Path("scratch/synk/models_k.csv")
    if not pol.is_file() or not models.is_file() or not synk.is_file():
        pytest.skip("scratch inputs absent")
    if not DEFAULT_LEDGER_PATH.is_file():
        pytest.skip("ledger absent")
    import pandas as pd

    frame = pd.read_csv(pol, dtype=str)
    model_cols = pd.read_csv(models, dtype=str)
    synk_cols = pd.read_csv(synk, dtype=str).rename(columns={"model_rev2synkf": "rev2synkf"})
    frame = frame.merge(model_cols, on="sample_id", how="inner").merge(synk_cols, on="sample_id", how="inner")
    frame["rev2syn"] = frame["base"]
    held_out = ["knn", "rev", "rev2", "rev3syn", "rev3synk", "rev2synkf"]
    kinds = {fact.policy_key: fact.kind for fact in Ledger.load(DEFAULT_LEDGER_PATH).facts() if fact.channel == "K"}
    kinds = {key: kind for key, kind in kinds.items() if key in frame.columns}
    matrix = response_matrix(frame, "K", kinds, "base")
    facts = [fact for fact in Ledger.load(DEFAULT_LEDGER_PATH).facts() if fact.policy_key in matrix.keys]
    assert len([fact for fact in facts if fact.channel == "K" and fact.policy_key not in held_out]) == 18
    report = loo_gate(matrix, facts, held_out)
    assert report.coverage == 1.0
    assert report.decisive == 0
