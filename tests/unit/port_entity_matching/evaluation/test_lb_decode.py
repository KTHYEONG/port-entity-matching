"""Invariant tests for LB score decoding."""

from __future__ import annotations

from itertools import pairwise

import pytest

from port_entity_matching.evaluation.lb.decode import (
    Decoded,
    DecodeError,
    _reconstructed,
    decode_in_windows,
    decode_score,
)


def test_known_submissions_decode_exactly() -> None:
    assert decode_in_windows(0.9525342223, (440, 532), (1680, 1784)) == Decoded(k=505, l=1713)
    assert decode_in_windows(0.9550020651, (440, 532), (1680, 1784)) == Decoded(k=504, l=1735.5)
    assert decode_in_windows(0.9497389072, (440, 532), (1680, 1784)) == Decoded(k=500, l=1735.5)
    assert decode_in_windows(0.0067220321, (0, 27), (0, 71)) == Decoded(k=3, l=16.5)


def test_ambiguity_is_reported_not_resolved() -> None:
    # Mid-range scores admit several (K, L) pairs spaced by the exact
    # period Dk=57 / Dl=-446; the decoder must list them all.
    score = round(_reconstructed(400, 1000.0), 10)
    pairs = decode_score(score)
    assert len(pairs) > 1
    ordered = sorted(pairs, key=lambda d: d.k)
    assert [d.k for d in ordered] == sorted(d.k for d in pairs)
    for first, second in pairwise(ordered):
        assert second.k - first.k == 57


def test_single_pair_scores_decode_unambiguously() -> None:
    # The recorded LB scores sit at the edge of the count range, so only one
    # pair falls inside [0, N]; the decoder returns it rather than failing.
    assert decode_score(0.9525342223) == [Decoded(k=505, l=1713)]


def test_window_without_match_raises() -> None:
    with pytest.raises(DecodeError, match=r"505"):
        decode_in_windows(0.9525342223, (0, 10), (1680, 1784))


def test_inverted_window_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        decode_in_windows(0.9525342223, (5, 1), (1680, 1784))
    with pytest.raises(ValueError, match="window"):
        decode_in_windows(0.9525342223, (440, 532), (9, 1))
