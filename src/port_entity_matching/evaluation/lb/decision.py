"""LB verdict from public net rows and churned rows.

Development-only; never imported by ``submission/`` or ``pipeline.py``.
"""

from __future__ import annotations

import math
from typing import Literal

from scipy.stats import norm

__all__ = ["LbVerdict", "churn_from_pair", "lb_verdict", "private_gain_probability"]

LbVerdict = Literal["adopt", "reject", "inconclusive"]


def private_gain_probability(net: float, churn: float) -> float:
    """Probability the full-test gain is positive given public net and churn."""
    if not math.isfinite(net) or not math.isfinite(churn) or churn < 0 or churn < abs(net):
        raise ValueError(f"Impossible churn={churn} for net={net}: expected churn >= |net| >= 0")
    if churn == 0:
        return 0.5
    return float(norm.cdf(2.0 * net / math.sqrt(2.0 * churn)))


def churn_from_pair(rescued: float, net: float) -> float:
    """Churn (fixes plus breaks) from a complement count and a net count."""
    breaks = rescued - net
    if not math.isfinite(rescued) or not math.isfinite(net) or rescued < 0 or breaks < -1e-9:
        raise ValueError(f"Impossible pair rescued={rescued} net={net}: breaks must be >= 0")
    return rescued + max(breaks, 0.0)


def lb_verdict(net: float, churn: float, threshold: float = 0.9) -> LbVerdict:
    """Adopt, reject, or stay inconclusive from the private gain probability."""
    if not 0.5 < threshold < 1:
        raise ValueError(f"Invalid threshold {threshold}: expected 0.5 < threshold < 1")
    prob = private_gain_probability(net, churn)
    if prob >= threshold:
        return "adopt"
    if prob <= 1 - threshold:
        return "reject"
    return "inconclusive"
