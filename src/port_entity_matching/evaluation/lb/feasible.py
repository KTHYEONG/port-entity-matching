"""Set-membership inference over cell mixtures consistent with LB facts.

Development-only; never imported by ``submission/`` or ``pipeline.py``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from port_entity_matching.evaluation.lb.decode import N_K, N_L
from port_entity_matching.evaluation.lb.ledger import Fact, FactKind
from port_entity_matching.evaluation.lb.response import ResponseMatrix

__all__ = [
    "FeasibilityError",
    "GateReport",
    "Interval",
    "Verdict",
    "classify",
    "fact_sigma",
    "feasible_interval",
    "loo_gate",
]

Verdict = Literal["adopt", "reject", "lb_required"]


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float


class FeasibilityError(ValueError):
    """No cell mixture is consistent with the LB facts."""


@dataclass(frozen=True)
class GateReport:
    intervals: Mapping[str, Interval]
    truth: Mapping[str, float]
    coverage: float
    decisive: int
    passes: bool


def fact_sigma(kind: FactKind, value: float, n: int) -> float:
    """Tolerance scale of one LB fact, in rows."""
    if kind == "abs":
        q = min(1.0, max(0.0, value / n)) if n > 0 else 0.0
        return max(3.0, math.sqrt(n * q * (1.0 - q)))
    if kind in ("comp", "net"):
        return math.sqrt(1.0 + abs(value))
    raise ValueError(f"Invalid kind {kind!r}")


def _extreme(kind: FactKind, upper: bool) -> float:
    if kind == "net":
        return 1.0 if upper else -1.0
    if kind in ("abs", "comp"):
        return 1.0 if upper else 0.0
    raise ValueError(f"Invalid kind {kind!r}")


def feasible_interval(
    matrix: ResponseMatrix, facts: Sequence[Fact], target: str, z: float = 2.0,
    unknown_cells: Collection[str] = (), ignore: Collection[str] = (),
) -> Interval:
    """Range of the target's predicted rows over mixtures consistent with the facts."""
    if not math.isfinite(z) or z < 0:
        raise ValueError(f"Invalid tolerance z={z}: expected z >= 0")
    index = {key: i for i, key in enumerate(matrix.keys)}
    if target not in index:
        raise ValueError(f"Unknown target key {target!r}")
    n = N_K if matrix.script == "K" else N_L
    cells = list(matrix.cells)
    m = len(cells)
    if m == 0:
        raise FeasibilityError("family library inconsistent with LB facts: matrix has no cells")
    ignored = set(ignore)
    cons: list[tuple[int, float, float, str]] = []
    for fact in facts:
        if fact.channel != matrix.script:
            continue
        if fact.policy_key not in index:
            if fact.policy_key in ignored:
                continue
            raise ValueError(f"Unknown policy key {fact.policy_key!r} (use ignore to skip)")
        if fact.kind != matrix.kinds[fact.policy_key]:
            raise ValueError(
                f"Kind mismatch for {fact.policy_key!r}: "
                f"fact {fact.kind!r} vs matrix {matrix.kinds[fact.policy_key]!r}"
            )
        if fact.policy_key == target:
            continue
        label = f"{fact.policy_key}({fact.entry_id})={fact.value}"
        cons.append((index[fact.policy_key], fact.value, fact_sigma(fact.kind, fact.value, n), label))
    kind = matrix.kinds[target]
    unknown = set(unknown_cells)
    base_row = np.array(matrix.values[index[target]], dtype=np.float64)
    row_lo = base_row.copy()
    row_hi = base_row.copy()
    for j, cell in enumerate(cells):
        if cell in unknown:
            row_lo[j] = _extreme(kind, upper=False)
            row_hi[j] = _extreme(kind, upper=True)
    a_eq = np.ones((1, m), dtype=np.float64)
    b_eq = np.ones(1, dtype=np.float64)
    bounds = [(0.0, None)] * m
    a_ub: NDArray[np.float64] | None = None
    b_ub: NDArray[np.float64] | None = None
    if cons:
        a_ub = np.empty((2 * len(cons), m), dtype=np.float64)
        b_ub = np.empty(2 * len(cons), dtype=np.float64)
        for r, (ki, value, sigma, _) in enumerate(cons):
            resp = n * np.array(matrix.values[ki], dtype=np.float64)
            tol = z * sigma
            a_ub[2 * r] = resp
            b_ub[2 * r] = value + tol
            a_ub[2 * r + 1] = -resp
            b_ub[2 * r + 1] = tol - value
    lo_hit = linprog(c=n * row_lo, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    hi_hit = linprog(c=-n * row_hi, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if lo_hit.status == 2 or hi_hit.status == 2:
        raise _infeasible(cons, m, matrix, n, z)
    if lo_hit.status != 0 or hi_hit.status != 0:
        raise RuntimeError(f"LP solver failed: {lo_hit.message}; {hi_hit.message}")
    return Interval(lo=float(lo_hit.fun), hi=float(-hi_hit.fun))


def _infeasible(
    cons: list[tuple[int, float, float, str]],
    m: int,
    matrix: ResponseMatrix,
    n: int,
    z: float,
) -> FeasibilityError:
    """Name the facts with the largest residual from a least-slack solve."""
    f = len(cons)
    c = np.zeros(m + f, dtype=np.float64)
    c[m:] = 1.0
    a_eq = np.zeros((1, m + f), dtype=np.float64)
    a_eq[0, :m] = 1.0
    b_eq = np.ones(1, dtype=np.float64)
    bounds = [(0.0, None)] * (m + f)
    a_ub = np.zeros((2 * f, m + f), dtype=np.float64)
    b_ub = np.zeros(2 * f, dtype=np.float64)
    for r, (ki, value, sigma, _) in enumerate(cons):
        resp = n * np.array(matrix.values[ki], dtype=np.float64)
        tol = z * sigma
        a_ub[2 * r, :m] = resp
        a_ub[2 * r, m + r] = -1.0
        b_ub[2 * r] = value + tol
        a_ub[2 * r + 1, :m] = -resp
        a_ub[2 * r + 1, m + r] = -1.0
        b_ub[2 * r + 1] = tol - value
    res = linprog(c=c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    slacks = np.array(res.x[m:], dtype=np.float64) if res.status == 0 and res.x is not None else np.zeros(f)
    ranked = sorted(range(f), key=lambda r: (float(slacks[r]), cons[r][3]), reverse=True)
    worst = [r for r in ranked if float(slacks[r]) > 1e-9][:3] or ranked[:3]
    names = ", ".join(cons[r][3] for r in worst)
    return FeasibilityError(f"family library inconsistent with LB facts: {names}")


def classify(interval: Interval, threshold: float = 0.0) -> Verdict:
    """Adopt above the threshold, reject below it, else require the LB."""
    if interval.lo > threshold:
        return "adopt"
    if interval.hi < threshold:
        return "reject"
    return "lb_required"


def loo_gate(matrix: ResponseMatrix, facts: Sequence[Fact], held_out: Sequence[str], z: float = 2.0) -> GateReport:
    """Leave-one-key-out intervals for each held-out fact key against its LB value."""
    if len(held_out) == 0:
        raise ValueError("held_out must not be empty")
    if len(set(held_out)) != len(held_out):
        raise ValueError("Duplicate held-out keys")
    script_facts = [fact for fact in facts if fact.channel == matrix.script]
    by_key: dict[str, list[float]] = {}
    for fact in script_facts:
        by_key.setdefault(fact.policy_key, []).append(float(fact.value))
    intervals: dict[str, Interval] = {}
    truth: dict[str, float] = {}
    for key in held_out:
        if key not in by_key:
            raise ValueError(f"No LB fact for held-out key {key!r}")
        rest = [fact for fact in facts if not (fact.channel == matrix.script and fact.policy_key == key)]
        intervals[key] = feasible_interval(matrix, rest, key, z=z)
        values = by_key[key]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Conflicting LB facts for held-out key {key!r}")
        truth[key] = values[0]
    hits = sum(1 for key in held_out if intervals[key].lo <= truth[key] <= intervals[key].hi)
    decisive = sum(1 for key in held_out if intervals[key].lo > 0 or intervals[key].hi < 0)
    return GateReport(
        intervals=intervals, truth=truth, coverage=hits / len(held_out),
        decisive=decisive, passes=hits == len(held_out),
    )
