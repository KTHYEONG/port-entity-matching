"""Contract test for the recorded package version."""

from __future__ import annotations

import re

from port_entity_matching import __version__
from port_entity_matching.version import __version__ as module_version


def test_version_is_consistent_and_release_like() -> None:
    assert __version__ == module_version
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__) is not None
