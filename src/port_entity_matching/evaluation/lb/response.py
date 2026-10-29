"""Per-cell policy response matrix in LB fact units.

Development-only; never imported by ``submission/`` or ``pipeline.py``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from port_entity_matching.evaluation.lb.ledger import FactKind

__all__ = ["Family", "ResponseMatrix", "circular_cells", "parse_cell", "response_matrix"]


@dataclass(frozen=True)
class Family:
    script: Literal["K", "L"]
    generator: str
    noise: str
    tonnage: str


@dataclass(frozen=True)
class ResponseMatrix:
    script: Literal["K", "L"]
    cells: tuple[str, ...]
    families: tuple[Family, ...]
    keys: tuple[str, ...]
    kinds: Mapping[str, FactKind]
    values: NDArray[np.float64]
    rows: NDArray[np.int64]


def parse_cell(cell: str) -> Family:
    """Split a ``SCRIPT|generator|noise|tonnage`` cell name into its family."""
    parts = cell.split("|")
    if len(parts) != 4 or parts[0] not in ("K", "L"):
        raise ValueError(f"Invalid cell {cell!r}: expected 'SCRIPT|generator|noise|tonnage'")
    script = cast("Literal['K', 'L']", parts[0])
    return Family(script=script, generator=parts[1], noise=parts[2], tonnage=parts[3])


def response_matrix(
    frame: pd.DataFrame, script: Literal["K", "L"], kinds: Mapping[str, FactKind], base: str, min_rows: int = 20
) -> ResponseMatrix:
    """Mean per-row fact-unit response of each policy column within each cell."""
    if script not in ("K", "L"):
        raise ValueError(f"Invalid script {script!r}: expected 'K' or 'L'")
    required = ["sample_id", "script", "cell", "target_vessel_id", base, *kinds.keys()]
    missing = [c for c in dict.fromkeys(required) if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if bool(frame["sample_id"].duplicated().any()):
        dups = frame.loc[frame["sample_id"].duplicated(keep=False), "sample_id"].unique().tolist()
        raise ValueError(f"Duplicate sample_id values: {dups}")
    for key, kind in kinds.items():
        if kind not in ("abs", "comp", "net"):
            raise ValueError(f"Invalid kind {kind!r} for key {key!r}")
    sub = frame.loc[frame["script"] == script]
    if sub["cell"].isna().any():
        raise ValueError("Missing cell values")
    counts = sub["cell"].value_counts()
    tiny = sorted(counts[counts < min_rows].index.tolist())
    if tiny:
        raise ValueError(f"Cells with fewer than {min_rows} rows: {tiny}")
    cells = tuple(sorted(counts.index.tolist()))
    keys = tuple(kinds.keys())
    target = sub["target_vessel_id"].to_numpy()
    base_hit = (sub[base].to_numpy() == target)
    cell_arr = sub["cell"].to_numpy()
    vals = np.empty((len(keys), len(cells)), dtype=np.float64)
    for i, key in enumerate(keys):
        pred_hit = (sub[key].to_numpy() == target)
        kind = kinds[key]
        if kind == "abs":
            unit = pred_hit.astype(np.float64)
        elif kind == "comp":
            unit = (pred_hit & ~base_hit).astype(np.float64)
        else:
            unit = pred_hit.astype(np.float64) - base_hit.astype(np.float64)
        means = pd.Series(unit, index=cell_arr).groupby(level=0).mean()
        vals[i, :] = np.array([float(means[c]) for c in cells], dtype=np.float64)
    rows = np.array([int(counts[c]) for c in cells], dtype=np.int64)
    return ResponseMatrix(
        script=script, cells=cells, families=tuple(parse_cell(c) for c in cells),
        keys=keys, kinds=dict(kinds), values=vals, rows=rows,
    )


def circular_cells(matrix: ResponseMatrix, processes: Collection[str]) -> frozenset[str]:
    """Cells whose generator or noise token matches a training process name."""
    wanted = set(processes)
    if not wanted:
        return frozenset()
    return frozenset(
        cell for cell, family in zip(matrix.cells, matrix.families, strict=True)
        if family.generator in wanted or family.noise in wanted
    )
