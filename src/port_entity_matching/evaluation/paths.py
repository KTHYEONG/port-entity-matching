"""Resolve output paths while refusing to write inside data directories."""

from __future__ import annotations

from pathlib import Path

__all__ = ["guard_data_path"]


def guard_data_path(path: Path) -> Path:
    """Resolve a path and reject any location inside a directory named 'data'."""
    resolved = path.expanduser().resolve()
    if "data" in resolved.parts:
        raise ValueError(f"Refusing to write evaluation outputs inside a data directory: {resolved}")
    return resolved
