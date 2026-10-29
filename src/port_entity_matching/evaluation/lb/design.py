"""Rank candidate measurement policies by expected target-interval width.

Development-only; never imported by ``submission/`` or ``pipeline.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from port_entity_matching.evaluation.lb.feasible import (
    FeasibilityError,
    Interval,
    feasible_interval,
)
from port_entity_matching.evaluation.lb.ledger import Fact
from port_entity_matching.evaluation.lb.response import ResponseMatrix

__all__ = ["ProbeScore", "rank_probes"]


@dataclass(frozen=True)
class ProbeScore:
    candidate: str
    expected_width: float
    worst_width: float
    feasible_points: int


def rank_probes(
    matrix: ResponseMatrix, facts: Sequence[Fact], candidates: Sequence[str], targets: Sequence[str],
    grid: int = 5, z: float = 2.0,
) -> list[ProbeScore]:
    """Rank candidates by mean target-interval width after observing them."""
    if grid < 2:
        raise ValueError(f"Invalid grid {grid}: expected grid >= 2")
    for key in (*candidates, *targets):
        if key not in matrix.kinds:
            raise ValueError(f"Unknown key {key!r}")
    if len(targets) == 0:
        raise ValueError("targets must not be empty")
    script_facts = [fact for fact in facts if fact.channel == matrix.script]
    scored: list[ProbeScore] = []
    for candidate in candidates:
        kind = matrix.kinds[candidate]
        current = feasible_interval(matrix, script_facts, candidate, z=z)
        step = (current.hi - current.lo) / (grid - 1)
        widths: list[float] = []
        for i in range(grid):
            value = current.lo + i * step
            hype = Fact(
                entry_id="hypothetical", channel=matrix.script, policy_key=candidate,
                kind=kind, base=None, value=value, prediction_hit=None,
            )
            augmented = [*script_facts, hype]
            try:
                point_widths = [_target_width(matrix, augmented, target, z) for target in targets]
            except FeasibilityError:
                continue
            widths.append(sum(point_widths) / len(point_widths))
        if not widths:
            continue
        scored.append(
            ProbeScore(
                candidate=candidate, expected_width=sum(widths) / len(widths),
                worst_width=max(widths), feasible_points=len(widths),
            )
        )
    scored.sort(key=lambda score: (score.expected_width, score.worst_width, score.candidate))
    return scored


def _target_width(
    matrix: ResponseMatrix, augmented: list[Fact], target: str, z: float,
) -> float:
    """Target interval width given a hypothetical observation, exact for self-targets."""
    if target == augmented[-1].policy_key:
        alias = target + "#objective"
        while alias in matrix.keys:
            alias += "#"
        index = matrix.keys.index(target)
        extended = replace(
            matrix,
            keys=(*matrix.keys, alias),
            kinds={**matrix.kinds, alias: matrix.kinds[target]},
            values=np.vstack([matrix.values, matrix.values[index]]),
        )
        interval: Interval = feasible_interval(extended, augmented, alias, z=z)
        return float(interval.hi - interval.lo)
    other: Interval = feasible_interval(matrix, augmented, target, z=z)
    return float(other.hi - other.lo)
