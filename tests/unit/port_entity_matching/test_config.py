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


def test_legacy_manifest_uses_llm_defaults() -> None:
    config = config_from_dict({"name_top_k": 5})
    assert config.augment_hangul_notation is False
    assert config.augment_hangul_tonnage_noise is False
    assert config.llm_enabled is True
    assert (config.llm_fewshot, config.llm_fewshot_seed) == (12, 0)
    assert (config.llm_num_beams, config.llm_max_new_tokens, config.llm_retrieval_beams) == (5, 24, 2)
    assert config.llm_max_name_chars == 64
    assert config.llm_rerank_enabled is True
    assert config.llm_rerank_top_k == 10
    assert config.llm_rerank_weight == 0.2


def test_old_manifest_uses_token_retrieval_defaults() -> None:
    config = config_from_dict({"name_top_k": 5})
    assert config.token_top_k == 30
    assert config.token_df_cap == 2000
    assert config.token_match_threshold == 0.8


def test_hangul_tonnage_window_default_and_round_trip() -> None:
    assert MatchConfig().hangul_tonnage_window == 0.10
    assert config_from_dict({"name_top_k": 5}).hangul_tonnage_window == 0.10
    assert config_from_dict({"hangul_tonnage_window": 0.08}).hangul_tonnage_window == 0.08
    assert config_from_dict(config_to_dict(MatchConfig(hangul_tonnage_window=0.08))).hangul_tonnage_window == 0.08


def test_latin_tonnage_window_is_untouched_by_hangul_window() -> None:
    assert MatchConfig().tonnage_window == 0.06


def test_training_ensemble_defaults() -> None:
    config = MatchConfig()
    assert config.augment is True
    assert config.augment_tonnage_factor == 1.08
    assert config.augment_min_syllables == 3
    assert config.augment_seed == 0
    assert config.ensemble_size == 5


def test_augmentation_and_ensemble_fields_round_trip() -> None:
    restored = config_from_dict(config_to_dict(MatchConfig()))
    assert restored == MatchConfig()
    assert isinstance(restored.augment, bool)


def test_json_boolean_stays_a_boolean() -> None:
    assert config_from_dict({"augment": False}).augment is False
    assert config_from_dict({"augment": True}).augment is True


def test_notation_augmentation_defaults_and_round_trip() -> None:
    config = MatchConfig()
    assert config.augment_notation is True
    assert config.augment_latin_rate == 0.3
    assert config.augment_tonnage_low == 0.05
    assert config.augment_tonnage_high == 0.25
    assert config_from_dict(config_to_dict(config)) == config


def test_notation_copies_off_by_default() -> None:
    assert MatchConfig().augment_hangul_notation is False


def test_rerank_defaults() -> None:
    config = MatchConfig()
    assert config.llm_rerank_enabled is True
    assert config.llm_rerank_top_k == 10
    assert config.llm_rerank_weight == 0.2
    assert config_from_dict({}) == config


def test_old_manifest_without_notation_fields_uses_defaults() -> None:
    payload = {
        key: value
        for key, value in config_to_dict(MatchConfig()).items()
        if key not in ("augment_notation", "augment_latin_rate", "augment_tonnage_low", "augment_tonnage_high")
    }
    config = config_from_dict(payload)
    assert config.augment_notation is True
    assert config.augment_latin_rate == 0.3
    assert config.augment_tonnage_low == 0.05
    assert config.augment_tonnage_high == 0.25
