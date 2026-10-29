"""Invariant tests for per-cell policy response matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from port_entity_matching.evaluation.lb.response import (
    Family,
    circular_cells,
    parse_cell,
    response_matrix,
)

C1 = "K|g1|clean|exact"
C2 = "K|g1|jamo|exact"


def _rows(cell: str, n: int, pred: str, base: str, start: int) -> list[dict[str, str]]:
    return [
        {
            "sample_id": f"s{start + i}",
            "script": "K",
            "cell": cell,
            "target_vessel_id": "V1",
            "base": base,
            "p": pred,
        }
        for i in range(n)
    ]


def _comp_frame() -> pd.DataFrame:
    rows = _rows(C1, 20, "V1", "VX", 0) + _rows(C1, 10, "VX", "V1", 20)
    rows += _rows(C2, 20, "V1", "VX", 30) + _rows(C2, 10, "VX", "V1", 50)
    return pd.DataFrame(rows)


def test_fact_units_per_kind() -> None:
    matrix = response_matrix(_comp_frame(), "K", {"p": "comp"}, "base")
    assert matrix.cells == (C1, C2)
    assert matrix.keys == ("p",)
    assert matrix.rows.tolist() == [30, 30]
    np.testing.assert_allclose(matrix.values[0], [20 / 30, 20 / 30])


def test_net_equals_accuracy_difference() -> None:
    rows = _rows(C1, 20, "V1", "V1", 0) + _rows(C2, 20, "VX", "VX", 20)
    matrix = response_matrix(pd.DataFrame(rows), "K", {"p": "net"}, "base")
    np.testing.assert_allclose(matrix.values[0], [0.0, 0.0])


def test_tiny_cell_fails_closed() -> None:
    rows = _rows("K|tiny|clean|exact", 5, "V1", "VX", 0) + _rows(C1, 20, "V1", "VX", 5)
    with pytest.raises(ValueError, match="tiny"):
        response_matrix(pd.DataFrame(rows), "K", {"p": "abs"}, "base", min_rows=20)


def test_duplicate_sample_ids_rejected() -> None:
    rows = _rows(C1, 20, "V1", "VX", 0) + _rows(C2, 20, "V1", "VX", 0)
    with pytest.raises(ValueError, match="Duplicate"):
        response_matrix(pd.DataFrame(rows), "K", {"p": "abs"}, "base")


def test_missing_columns_listed() -> None:
    frame = pd.DataFrame([{"sample_id": "s0"}])
    with pytest.raises(ValueError, match="base"):
        response_matrix(frame, "K", {"p": "abs"}, "base")


def test_missing_cell_fails_closed() -> None:
    frame = _comp_frame()
    frame.loc[0, "cell"] = None
    with pytest.raises(ValueError, match="Missing cell"):
        response_matrix(frame, "K", {"p": "abs"}, "base")


def test_invalid_kind_rejected() -> None:
    with pytest.raises(ValueError, match="bogus"):
        response_matrix(_comp_frame(), "K", {"p": "bogus"}, "base")  # type: ignore[dict-item]


def test_circular_cells_by_provenance() -> None:
    rows = _rows("K|qwen|clean|exact", 20, "V1", "VX", 0)
    rows += _rows("K|midm|jamo|none", 20, "V1", "VX", 20)
    matrix = response_matrix(pd.DataFrame(rows), "K", {"p": "abs"}, "base")
    assert circular_cells(matrix, {"qwen"}) == frozenset({"K|qwen|clean|exact"})
    assert circular_cells(matrix, set()) == frozenset()


def test_parse_cell_round_trip() -> None:
    assert parse_cell(C1) == Family(script="K", generator="g1", noise="clean", tonnage="exact")
    with pytest.raises(ValueError, match="Invalid cell"):
        parse_cell("K|only-two-parts")
    with pytest.raises(ValueError, match="Invalid cell"):
        parse_cell("X|g|n|t")
