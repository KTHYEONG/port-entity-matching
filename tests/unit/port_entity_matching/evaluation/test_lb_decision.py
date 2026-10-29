"""Invariant tests for the LB verdict."""

from __future__ import annotations

import pytest

from port_entity_matching.evaluation.lb.decision import (
    churn_from_pair,
    lb_verdict,
    private_gain_probability,
)


def test_reference_probabilities() -> None:
    assert private_gain_probability(3, 12) == pytest.approx(0.890, abs=0.005)
    assert private_gain_probability(1, 26) == pytest.approx(0.61, abs=0.005)
    assert private_gain_probability(5, 26) == pytest.approx(0.92, abs=0.005)
    assert private_gain_probability(-5, 26) == pytest.approx(0.08, abs=0.005)


def test_symmetry() -> None:
    assert private_gain_probability(4, 20) + private_gain_probability(-4, 20) == pytest.approx(1.0, abs=1e-12)


def test_zero_churn_is_fair_coin() -> None:
    assert private_gain_probability(0, 0) == 0.5


def test_impossible_churn_rejected() -> None:
    with pytest.raises(ValueError, match="churn"):
        private_gain_probability(5, 3)
    with pytest.raises(ValueError, match="churn"):
        private_gain_probability(0, -1)
    with pytest.raises(ValueError, match="churn"):
        private_gain_probability(float("nan"), 12)


def test_pair_derivation() -> None:
    assert churn_from_pair(9, 4) == 14
    with pytest.raises(ValueError, match="breaks"):
        churn_from_pair(2, 4)


def test_verdict_bands() -> None:
    assert lb_verdict(3, 12, 0.9) == "inconclusive"
    assert lb_verdict(8, 26) == "adopt"
    assert lb_verdict(-8, 26) == "reject"
    with pytest.raises(ValueError, match="threshold"):
        lb_verdict(8, 26, 0.5)
