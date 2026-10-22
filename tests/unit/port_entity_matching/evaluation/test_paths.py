"""Output path safety contracts."""

from pathlib import Path

import pytest

from port_entity_matching.evaluation.paths import guard_data_path


def test_output_path_is_resolved(tmp_path: Path) -> None:
    assert guard_data_path(tmp_path / "nested" / ".." / "output") == tmp_path / "output"


def test_data_directory_and_symlink_are_refused(tmp_path: Path) -> None:
    protected = tmp_path / "data"
    protected.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(protected, target_is_directory=True)
    for path in (protected / "output", alias / "output"):
        with pytest.raises(ValueError, match="data directory"):
            guard_data_path(path)
