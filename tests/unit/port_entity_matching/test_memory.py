"""Contract tests for the call-sign identity memory."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.memory import CallSignMemory


def _index(count: int = 3) -> CatalogIndex:
    frame = pd.DataFrame(
        [
            {"vessel_id": f"V{position}", "vessel_name": f"Vessel {position}", "gross_tonnage": 100 + position}
            for position in range(count)
        ]
    )
    return CatalogIndex.build(frame)


def _train(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_unique_call_sign_is_memorized() -> None:
    index = _index()
    train = _train(
        [
            {"call_sign": "CS_A", "target_vessel_id": "V1"},
            {"call_sign": "CS_A", "target_vessel_id": "V1"},
        ]
    )
    memory = CallSignMemory.fit(train, index)
    assert memory.lookup("CS_A") == 1
    assert memory.lookup("CS_B") is None
    assert memory.lookup(None) is None
    assert len(memory) == 1


def test_conflicting_call_sign_is_excluded() -> None:
    index = _index()
    train = _train(
        [
            {"call_sign": "CS_B", "target_vessel_id": "V0"},
            {"call_sign": "CS_B", "target_vessel_id": "V1"},
        ]
    )
    memory = CallSignMemory.fit(train, index)
    assert memory.lookup("CS_B") is None


def test_target_missing_from_catalog_is_dropped() -> None:
    index = _index()
    train = _train([{"call_sign": "CS_C", "target_vessel_id": "UNKNOWN"}])
    memory = CallSignMemory.fit(train, index)
    assert memory.lookup("CS_C") is None
    assert len(memory) == 0


def test_missing_column_rejected() -> None:
    index = _index()
    with pytest.raises(ValueError, match="target_vessel_id"):
        CallSignMemory.fit(_train([{"call_sign": "CS_A"}]), index)


def test_missing_values_are_skipped() -> None:
    index = _index()
    train = _train(
        [
            {"call_sign": None, "target_vessel_id": "V0"},
            {"call_sign": np.nan, "target_vessel_id": "V0"},
            {"call_sign": "CS_D", "target_vessel_id": None},
            {"call_sign": "CS_E", "target_vessel_id": "V2"},
        ]
    )
    memory = CallSignMemory.fit(train, index)
    assert len(memory) == 1
    assert memory.lookup("CS_E") == 2


def test_json_payload_must_be_an_object(tmp_path) -> None:
    index = _index()
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(["CS_A"]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        CallSignMemory.from_json(path, index)


def test_json_round_trip(tmp_path) -> None:
    index = _index()
    train = _train([{"call_sign": "CS_A", "target_vessel_id": "V2"}])
    memory = CallSignMemory.fit(train, index)
    path = tmp_path / "memory.json"
    memory.to_json(path)
    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["CS_A"]

    restored = CallSignMemory.from_json(path, index)
    assert restored.lookup("CS_A") == memory.lookup("CS_A")
    assert len(restored) == len(memory)


def test_json_load_drops_unknown_vessel_ids(tmp_path) -> None:
    index = _index()
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"CS_A": "V1", "CS_B": "MISSING"}), encoding="utf-8")
    restored = CallSignMemory.from_json(path, index)
    assert restored.lookup("CS_A") == 1
    assert restored.lookup("CS_B") is None


def test_serialization_survives_catalog_reordering(tmp_path) -> None:
    frame = pd.DataFrame({"vessel_id": ["V0", "V1"], "vessel_name": ["Alpha", "Beta"], "gross_tonnage": [100, 200]})
    index = CatalogIndex.build(frame)
    memory = CallSignMemory.fit(pd.DataFrame({"call_sign": ["CS_B", "CS_A"], "target_vessel_id": ["V0", "V1"]}), index)
    path = tmp_path / "nested" / "memory.json"
    memory.to_json(path)
    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["CS_A", "CS_B"]
    restored = CallSignMemory.from_json(path, CatalogIndex.build(frame.iloc[::-1]))
    assert restored.lookup("CS_A") == 0
    assert restored.lookup("CS_B") == 1


def test_unknown_catalog_id_still_conflicts_with_known_target() -> None:
    train = pd.DataFrame({"call_sign": ["CS", "CS"], "target_vessel_id": ["V0", "UNKNOWN"]})
    assert CallSignMemory.fit(train, _index()).lookup("CS") is None


def test_literal_nan_call_sign_and_raw_vessel_id_round_trip(tmp_path) -> None:
    index = CatalogIndex.build(pd.DataFrame({"vessel_id": [" V0 "], "vessel_name": ["Alpha"], "gross_tonnage": [100]}))
    memory = CallSignMemory.fit(pd.DataFrame({"call_sign": ["nan"], "target_vessel_id": [" V0 "]}), index)
    assert memory.lookup("nan") == 0
    path = tmp_path / "memory.json"
    memory.to_json(path)
    assert CallSignMemory.from_json(path, index).lookup("nan") == 0
