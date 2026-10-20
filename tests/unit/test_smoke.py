"""Basic package smoke test."""

import port_entity_matching


def test_package_version() -> None:
    assert port_entity_matching.__version__ == "0.1.0"
