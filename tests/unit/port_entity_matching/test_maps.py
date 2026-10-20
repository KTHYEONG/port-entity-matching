"""Contract tests for the smoothed attribute compatibility maps."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.maps import AttributeMaps


def _index() -> CatalogIndex:
    frame = pd.DataFrame(
        [
            {
                "vessel_id": "V0",
                "vessel_name": "Alpha",
                "gross_tonnage": 100,
                "vessel_type": "Container",
                "flag_name": "Panama",
            },
            {
                "vessel_id": "V1",
                "vessel_name": "Beta",
                "gross_tonnage": 200,
                "vessel_type": "Container",
                "flag_name": "Panama",
            },
            {
                "vessel_id": "V2",
                "vessel_name": "Gamma",
                "gross_tonnage": 300,
                "vessel_type": "Bulk",
                "flag_name": "Liberia",
            },
            {
                "vessel_id": "V3",
                "vessel_name": "Delta",
                "gross_tonnage": 400,
                "vessel_type": "Tanker",
                "flag_name": "Korea",
            },
        ]
    )
    return CatalogIndex.build(frame)


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"vessel_type": "풀컨테이너선", "flag_name": "파나마", "target_vessel_id": "V0"},
            {"vessel_type": "풀컨테이너선", "flag_name": "파나마", "target_vessel_id": "V1"},
            {"vessel_type": "풀컨테이너선", "flag_name": "파나마", "target_vessel_id": "V1"},
            {"vessel_type": "벌크선", "flag_name": "라이베리아", "target_vessel_id": "V2"},
        ]
    )


def test_observed_pair_beats_unseen() -> None:
    maps = AttributeMaps.fit(_train(), _index())
    observed = maps.type_logprob("풀컨테이너선", "Container")
    unseen = maps.type_logprob("풀컨테이너선", "Bulk")
    assert observed > unseen


def test_missing_record_value_is_nan() -> None:
    maps = AttributeMaps.fit(_train(), _index())
    assert math.isnan(maps.flag_logprob(None, "Panama"))
    assert math.isnan(maps.type_logprob(None, "Container"))


def test_unseen_record_value_is_uniform() -> None:
    maps = AttributeMaps.fit(_train(), _index())
    uniform = math.log(1.0 / 3.0)
    assert maps.type_logprob("존재하지않는유형", "Container") == pytest.approx(uniform, rel=1e-9)
    assert maps.type_logprob("존재하지않는유형", "Bulk") == pytest.approx(uniform, rel=1e-9)


def test_logprior_follows_target_distribution() -> None:
    maps = AttributeMaps.fit(_train(), _index())
    assert maps.type_logprior("Container") > maps.type_logprior("Bulk")
    assert maps.type_logprior("Tanker") < maps.type_logprior("Container")


def test_missing_column_rejected() -> None:
    frame = pd.DataFrame([{"vessel_type": "벌크선", "target_vessel_id": "V2"}])
    with pytest.raises(ValueError, match="flag_name"):
        AttributeMaps.fit(frame, _index())


def test_target_outside_catalog_is_ignored() -> None:
    frame = pd.DataFrame(
        [
            {"vessel_type": "벌크선", "flag_name": "라이베리아", "target_vessel_id": "MISSING"},
        ]
    )
    maps = AttributeMaps.fit(frame, _index())
    assert maps.type_logprob("벌크선", "Bulk") == pytest.approx(math.log(1.0 / 3.0), rel=1e-9)


def test_json_round_trip(tmp_path) -> None:
    maps = AttributeMaps.fit(_train(), _index())
    path = tmp_path / "maps.json"
    maps.to_json(path)
    restored = AttributeMaps.from_json(path)
    assert restored.alpha == maps.alpha
    for record_type, catalog_type in (("풀컨테이너선", "Container"), ("벌크선", "Bulk"), ("없는유형", "Tanker")):
        assert restored.type_logprob(record_type, catalog_type) == maps.type_logprob(record_type, catalog_type)
    for record_flag, catalog_flag in (("파나마", "Panama"), ("라이베리아", "Liberia"), ("없음", "Panama")):
        assert restored.flag_logprob(record_flag, catalog_flag) == maps.flag_logprob(record_flag, catalog_flag)
    assert restored.type_logprior("Container") == maps.type_logprior("Container")
    assert restored.flag_logprior("Panama") == maps.flag_logprior("Panama")


@pytest.mark.parametrize("alpha", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_non_positive_alpha_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        AttributeMaps.fit(_train(), _index(), alpha=alpha)


def test_missing_target_values_are_skipped() -> None:
    frame = pd.DataFrame(
        [
            {"vessel_type": "벌크선", "flag_name": "라이베리아", "target_vessel_id": None},
            {"vessel_type": "벌크선", "flag_name": "라이베리아", "target_vessel_id": float("nan")},
            {"vessel_type": float("nan"), "flag_name": float("nan"), "target_vessel_id": "V2"},
            {"vessel_type": pd.NA, "flag_name": pd.NA, "target_vessel_id": "V1"},
            {"vessel_type": None, "flag_name": None, "target_vessel_id": "V2"},
        ]
    )
    maps = AttributeMaps.fit(frame, _index())
    assert maps.prior_total == 3
    assert maps.type_logprob("벌크선", "Bulk") == pytest.approx(math.log(1.0 / 3.0), rel=1e-9)
    assert math.isnan(maps.type_logprob(None, "Bulk"))


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"alpha": 1.0},
        {"alpha": 1.0, "type_counts": [], "type_totals": {}, "type_prior_counts": {}, "flag_counts": {},
         "flag_totals": {}, "flag_prior_counts": {}, "type_value_count": 1, "flag_value_count": 1, "prior_total": 0},
        {"alpha": 1.0, "type_counts": {}, "type_totals": [], "type_prior_counts": {}, "flag_counts": {},
         "flag_totals": {}, "flag_prior_counts": {}, "type_value_count": 1, "flag_value_count": 1, "prior_total": 0},
    ],
)
def test_malformed_json_rejected(tmp_path, payload: object) -> None:
    path = tmp_path / "maps.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"JSON object|malformed"):
        AttributeMaps.from_json(path)
