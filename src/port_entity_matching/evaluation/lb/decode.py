"""Score decoding for leaderboard (LB) values.

Development-only; never imported by ``submission/`` or ``pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = ["N_K", "N_L", "W_K", "W_L", "DecodeError", "Decoded", "decode_in_windows", "decode_score"]

N_K: Final[int] = 532
N_L: Final[int] = 1784
W_K: Final[float] = 0.7
W_L: Final[float] = 0.3

_TOL: Final[float] = 5e-10


@dataclass(frozen=True)
class Decoded:
    k: int
    l: float


class DecodeError(ValueError):
    """Raised when a score cannot be decoded uniquely in the given windows."""


def _reconstructed(k: int, l: float) -> float:
    return W_K * k / N_K + W_L * l / N_L


def decode_score(score: float) -> list[Decoded]:
    """All (K, L) with W_K*K/N_K + W_L*L/N_L == score (K integer, L in half rows); may be empty."""
    found: list[Decoded] = []
    for k in range(N_K + 1):
        l_star = (score - W_K * k / N_K) * N_L / W_L
        nearest_half = round(l_star * 2.0) / 2.0
        if nearest_half < 0.0 or nearest_half > float(N_L):
            continue
        if abs(_reconstructed(k, nearest_half) - score) < _TOL:
            found.append(Decoded(k=k, l=nearest_half))
    found.sort(key=lambda d: d.k)
    return found


def decode_in_windows(
    score: float, k_window: tuple[float, float], l_window: tuple[float, float]
) -> Decoded:
    """The unique decoded pair inside both windows; DecodeError if none or several."""
    k_lo, k_hi = k_window
    l_lo, l_hi = l_window
    if k_lo > k_hi:
        raise ValueError(f"Invalid K window {(k_lo, k_hi)}: lo > hi")
    if l_lo > l_hi:
        raise ValueError(f"Invalid L window {(l_lo, l_hi)}: lo > hi")
    pairs = decode_score(score)
    inside = [d for d in pairs if k_lo <= d.k <= k_hi and l_lo <= d.l <= l_hi]
    if len(inside) == 1:
        return inside[0]
    raise DecodeError(f"Expected one decoded pair in K{k_window}/L{l_window}, got {len(inside)}; all pairs={pairs}")
