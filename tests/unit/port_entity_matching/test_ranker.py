"""Contract tests for the LightGBM ranker and the top-1 selection."""

from __future__ import annotations

import numpy as np
import pytest

from port_entity_matching.candidates import SRC_NAME, CandidateSet
from port_entity_matching.config import MatchConfig
from port_entity_matching.features import FEATURE_NAMES
from port_entity_matching.ranker import (
    score_ensemble,
    score_pairs,
    select_top1,
    train_ensemble,
    train_ranker,
)


def _config(**overrides: int | float) -> MatchConfig:
    defaults: dict[str, int | float] = {
        "lgbm_num_rounds": 20,
        "lgbm_min_data_in_leaf": 1,
        "lgbm_learning_rate": 0.2,
        "num_threads": 1,
    }
    defaults.update(overrides)
    return MatchConfig(**defaults)


def _separable_data(groups: int = 12) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = np.random.default_rng(20261020)
    features = rng.random((groups * 2, len(FEATURE_NAMES))).astype(np.float32)
    labels = np.zeros(groups * 2, dtype=np.int64)
    for group in range(groups):
        features[group * 2, 0] = 1.0
        features[group * 2 + 1, 0] = 0.0
        labels[group * 2] = 1
    return features, labels, [2] * groups


def _candidate_set(positions: list[int]) -> CandidateSet:
    return CandidateSet(
        np.asarray(positions, dtype=np.int64),
        np.full(len(positions), SRC_NAME, dtype=np.uint8),
    )


def test_learns_separable_signal() -> None:
    features, labels, groups = _separable_data()
    booster = train_ranker(features, labels, groups, _config())
    scores = score_pairs(booster, features, _config())
    sets = [_candidate_set([2 * group, 2 * group + 1]) for group in range(len(groups))]
    selected = select_top1(scores, sets)
    assert selected == [2 * group for group in range(len(groups))]


def test_group_without_positive_rejected() -> None:
    features, labels, groups = _separable_data(groups=2)
    labels[0] = 0
    with pytest.raises(ValueError, match="exactly one positive"):
        train_ranker(features, labels, groups, _config())


def test_group_size_mismatch_rejected() -> None:
    features, labels, _ = _separable_data(groups=2)
    with pytest.raises(ValueError, match="group sizes"):
        train_ranker(features, labels, [5], _config())


def test_label_row_mismatch_rejected() -> None:
    features, labels, groups = _separable_data(groups=2)
    with pytest.raises(ValueError, match="same number of rows"):
        train_ranker(features, labels[:-1], groups, _config())


def test_tie_break_by_position() -> None:
    scores = np.zeros(2)
    sets = [_candidate_set([9, 3])]
    assert select_top1(scores, sets) == [3]


def test_select_top1_handles_multiple_sets() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.2])
    sets = [_candidate_set([0, 1]), _candidate_set([2, 3])]
    assert select_top1(scores, sets) == [1, 2]


def test_training_is_deterministic() -> None:
    features, labels, groups = _separable_data()
    config = _config()
    first = score_pairs(train_ranker(features, labels, groups, config), features, config)
    second = score_pairs(train_ranker(features, labels, groups, config), features, config)
    assert np.allclose(first, second, atol=0, rtol=0)


def test_wrong_feature_width_rejected() -> None:
    features, labels, groups = _separable_data(groups=2)
    with pytest.raises(ValueError, match="columns"):
        train_ranker(features[:, :3], labels, groups, _config())


def test_non_binary_labels_rejected() -> None:
    features, labels, groups = _separable_data(groups=2)
    labels[0] = 2
    with pytest.raises(ValueError, match="binary"):
        train_ranker(features, labels, groups, _config())


def test_scoring_empty_matrix_returns_empty() -> None:
    features, labels, groups = _separable_data(groups=2)
    booster = train_ranker(features, labels, groups, _config())
    empty = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    assert score_pairs(booster, empty, _config()).shape == (0,)


def test_select_top1_handles_empty_candidate_set() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_top1(np.zeros(0), [_candidate_set([])])


@pytest.mark.parametrize("sizes", [[], [-1, 5], [0, 4], [1.5, 2.5]])
def test_invalid_group_sizes_rejected(sizes) -> None:
    features, labels, _ = _separable_data(groups=2)
    with pytest.raises(ValueError, match="positive integers"):
        train_ranker(features, labels, sizes, _config())


@pytest.mark.parametrize("scores", [np.zeros(1), np.zeros(3), np.array([np.nan, 1.0]), np.array([np.inf, 1.0])])
def test_invalid_scores_rejected(scores) -> None:
    with pytest.raises(ValueError, match=r"candidate count|finite"):
        select_top1(scores, [_candidate_set([0, 1])])


def test_ensemble_size_honoured() -> None:
    features, labels, groups = _separable_data()
    ensemble = train_ensemble(features, labels, groups, _config(ensemble_size=3))
    assert len(ensemble.boosters) == 3
    assert len(ensemble.means) == 3
    assert len(ensemble.stds) == 3
    assert all(np.isfinite(value) for value in ensemble.means)
    assert all(np.isfinite(value) and value > 0 for value in ensemble.stds)


def test_members_differ_by_seed_only() -> None:
    features, labels, groups = _separable_data()
    config = _config(ensemble_size=2)
    ensemble = train_ensemble(features, labels, groups, config)
    for member, booster in enumerate(ensemble.boosters):
        assert int(booster.params["seed"]) == config.seed + member


def test_ensemble_scoring_is_row_independent() -> None:
    features, labels, groups = _separable_data(groups=4)
    config = _config(ensemble_size=3)
    ensemble = train_ensemble(features, labels, groups, config)
    full = score_ensemble(ensemble, features, config)
    first_group = score_ensemble(ensemble, features[:2], config)
    assert np.array_equal(full[:2], first_group)


def test_size_one_ensemble_matches_a_single_booster() -> None:
    features, labels, groups = _separable_data()
    config = _config(ensemble_size=1)
    ensemble = train_ensemble(features, labels, groups, config)
    single = train_ranker(features, labels, groups, config)
    scores = score_ensemble(ensemble, features, config)
    reference = score_pairs(single, features, config)
    assert np.array_equal(np.argsort(scores), np.argsort(reference))


def test_invalid_ensemble_size_rejected() -> None:
    features, labels, groups = _separable_data(groups=2)
    with pytest.raises(ValueError, match="ensemble_size"):
        train_ensemble(features, labels, groups, _config(ensemble_size=0))


def test_ensemble_scoring_of_empty_matrix_returns_empty() -> None:
    features, labels, groups = _separable_data(groups=2)
    config = _config(ensemble_size=2)
    ensemble = train_ensemble(features, labels, groups, config)
    empty = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    assert score_ensemble(ensemble, empty, config).shape == (0,)


def test_empty_ensemble_rejected() -> None:
    from port_entity_matching.ranker import RankerEnsemble

    features, _, _ = _separable_data(groups=2)
    with pytest.raises(ValueError, match="at least one member"):
        RankerEnsemble(boosters=(), means=(), stds=())


def test_constant_member_gets_a_unit_scale() -> None:
    features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.array([1, 0, 1, 0], dtype=np.int64)
    config = _config(ensemble_size=2)
    ensemble = train_ensemble(features, labels, [2, 2], config)
    assert ensemble.means == (0.0, 0.0)
    assert ensemble.stds == (1.0, 1.0)


def test_ensemble_uses_training_population_constants_and_arithmetic_mean() -> None:
    features, labels, groups = _separable_data()
    config = _config(ensemble_size=3)
    ensemble = train_ensemble(features, labels, groups, config)
    expected = np.zeros(len(features), dtype=np.float64)
    for booster, mean, std in zip(ensemble.boosters, ensemble.means, ensemble.stds, strict=True):
        raw = score_pairs(booster, features, config)
        assert mean == float(np.mean(raw))
        assert std == float(np.std(raw, ddof=0))
        expected += (raw - mean) / std
    np.testing.assert_array_equal(score_ensemble(ensemble, features, config), expected / 3)


@pytest.mark.parametrize(
    ("means", "stds", "message"),
    [
        ((), (1.0,), "equal lengths"),
        ((float("nan"),), (1.0,), "means must be finite"),
        ((float("inf"),), (1.0,), "means must be finite"),
        ((0.0,), (0.0,), "stds must be finite and positive"),
        ((0.0,), (-1.0,), "stds must be finite and positive"),
        ((0.0,), (float("nan"),), "stds must be finite and positive"),
        ((0.0,), (float("inf"),), "stds must be finite and positive"),
    ],
)
def test_invalid_ensemble_constants_rejected(means, stds, message) -> None:
    from port_entity_matching.ranker import RankerEnsemble

    features, labels, groups = _separable_data()
    booster = train_ranker(features, labels, groups, _config())
    with pytest.raises(ValueError, match=message):
        RankerEnsemble((booster,), means, stds)
