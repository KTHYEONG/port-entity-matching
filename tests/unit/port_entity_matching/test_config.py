"""Contract tests for the immutable matching configuration."""

from __future__ import annotations

import pytest

from port_entity_matching.config import MatchConfig, config_from_dict, config_to_dict


def test_round_trip_preserves_values() -> None:
    config = MatchConfig(name_top_k=7, tonnage_window=0.12, seed=1234)

    restored = config_from_dict(config_to_dict(config))

    assert restored == config


def test_round_trip_covers_every_field() -> None:
    payload = config_to_dict(MatchConfig())

    assert set(payload) == set(MatchConfig.__dataclass_fields__)
    assert all(isinstance(value, (int, float)) for value in payload.values())


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValueError, match="bogus"):
        config_from_dict({"bogus": 1})


def test_missing_keys_use_defaults() -> None:
    config = config_from_dict({"name_top_k": 5})

    assert config.name_top_k == 5
    assert config == MatchConfig(name_top_k=5)
