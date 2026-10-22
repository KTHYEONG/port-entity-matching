"""Contract tests for the disk-backed candidate cache."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from port_entity_matching.candidates import generate_candidates
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.evaluation.cache import CandidateCache, retrieval_fingerprint
from port_entity_matching.records import QueryRecord


def _catalog_frame() -> pd.DataFrame:
    entries = [
        ("V00", "Glovis Spirit", 5000.0),
        ("V01", "Grand Ace", 1000.0),
        ("V02", "Calida Star", 61332.0),
        ("V03", "Donoussa", 162354.0),
        ("V04", "Sapsan", 28068.0),
        ("V05", "Shining Road", 2000.0),
    ]
    return pd.DataFrame(
        [
            {
                "vessel_id": vessel_id,
                "vessel_type": "Tanker",
                "vessel_name": name,
                "size": tonnage,
                "unit": "DWT",
                "deadweight_tonnage": tonnage,
                "gross_tonnage": tonnage,
                "flag_name": "Panama",
            }
            for vessel_id, name, tonnage in entries
        ]
    )


def _record(sample_id: str, name: str, tonnage: float | None = 1000.0) -> QueryRecord:
    return QueryRecord(
        sample_id=sample_id,
        vessel_name=name,
        call_sign=None,
        gross_tonnage=tonnage,
        international_gross_tonnage=tonnage,
        flag_name=None,
        vessel_type=None,
        arrival_year=None,
    )


def _records() -> list[QueryRecord]:
    names = [
        "글로비스 스피릿",
        "GRAND ACE",
        "칼리다 스타",
        "DONOUSSA",
        "삽산호",
        "SHINING ROAD",
        "글로비스호",
        "GRAND ACE 11",
        "칼리다",
        "DONOUSSA X",
        "SAPSAN",
        "샤이닝 로드",
        "GLOVIS SPIRIT",
        "그랜드 에이스",
        "CALIDA",
        "도누사",
        "SAPSAN X",
        "샤이닝",
        "GLOVIS",
        "그로비스",
    ]
    return [_record(f"T{i:05d}", name) for i, name in enumerate(names)]


def _assert_equal(left, right) -> None:
    assert len(left) == len(right)
    for first, second in zip(left, right, strict=True):
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.sources, second.sources)


def test_fetch_equals_generate(tmp_path: Path) -> None:
    catalog = _catalog_frame()
    index = CatalogIndex.build(catalog)
    config = MatchConfig(num_threads=1)
    records = _records()
    cache = CandidateCache.open(tmp_path / "cache", catalog, config)
    fetched = cache.fetch(records, index, config)
    _assert_equal(fetched, generate_candidates(records, index, None, config))
    assert cache.stats == (0, 20)


def test_second_open_hits(tmp_path: Path) -> None:
    catalog = _catalog_frame()
    index = CatalogIndex.build(catalog)
    config = MatchConfig(num_threads=1)
    records = _records()
    root = tmp_path / "cache"
    cache = CandidateCache.open(root, catalog, config)
    cache.fetch(records, index, config)
    cache.flush()
    reopened = CandidateCache.open(root, catalog, config)
    fetched = reopened.fetch(records, index, config)
    _assert_equal(fetched, generate_candidates(records, index, None, config))
    assert reopened.stats == (20, 0)


def test_sample_id_ignored(tmp_path: Path) -> None:
    catalog = _catalog_frame()
    index = CatalogIndex.build(catalog)
    config = MatchConfig(num_threads=1)
    first = _record("A", "GRAND ACE")
    second = dataclasses.replace(first, sample_id="B")
    cache = CandidateCache.open(tmp_path / "cache", catalog, config)
    cache.fetch([first], index, config)
    assert cache.stats == (0, 1)
    out = cache.fetch([second], index, config)
    assert cache.stats == (1, 1)
    assert len(out) == 1


def test_fingerprint_sensitivity(tmp_path: Path) -> None:
    _ = tmp_path
    catalog = _catalog_frame()
    config = MatchConfig(num_threads=1)
    base = retrieval_fingerprint(catalog, config)
    assert retrieval_fingerprint(catalog, dataclasses.replace(config, name_top_k=config.name_top_k + 1)) != base
    changed = catalog.copy()
    changed.loc[0, "gross_tonnage"] = float(changed.loc[0, "gross_tonnage"]) + 1.0
    assert retrieval_fingerprint(changed, config) != base
    renamed = catalog.rename(columns={"vessel_name": "flag_name", "flag_name": "vessel_name"})
    assert retrieval_fingerprint(renamed, config) != base
    for field, value in [
        ("lgbm_num_rounds", config.lgbm_num_rounds + 1),
        ("ensemble_size", config.ensemble_size + 1),
        ("augment_latin_rate", config.augment_latin_rate + 0.1),
        ("num_threads", config.num_threads + 1),
    ]:
        assert retrieval_fingerprint(catalog, dataclasses.replace(config, **{field: value})) == base


def test_partial_hits_keep_order(tmp_path: Path) -> None:
    catalog = _catalog_frame()
    index = CatalogIndex.build(catalog)
    config = MatchConfig(num_threads=1)
    records = _records()[:3]
    cache = CandidateCache.open(tmp_path / "cache", catalog, config)
    cache.fetch([records[1]], index, config)
    assert cache.stats == (0, 1)
    fetched = cache.fetch(records, index, config)
    _assert_equal(fetched, generate_candidates(records, index, None, config))
    assert cache.stats == (1, 3)


def test_data_dir_refused(tmp_path: Path) -> None:
    import pytest

    catalog = _catalog_frame()
    config = MatchConfig(num_threads=1)
    with pytest.raises(ValueError, match="data"):
        CandidateCache.open(tmp_path / "data" / "cache", catalog, config)


def test_concurrent_flush(tmp_path: Path) -> None:
    catalog = _catalog_frame()
    index = CatalogIndex.build(catalog)
    config = MatchConfig(num_threads=1)
    root = tmp_path / "cache"
    first = CandidateCache.open(root, catalog, config)
    second = CandidateCache.open(root, catalog, config)
    records = _records()
    first.fetch(records[:5], index, config)
    second.fetch(records[5:10], index, config)
    first.flush()
    second.flush()
    assert len(list((root / first.fingerprint).glob("shard_*.npz"))) == 2
    reopened = CandidateCache.open(root, catalog, config)
    fetched = reopened.fetch(records[:10], index, config)
    _assert_equal(fetched, generate_candidates(records[:10], index, None, config))
    assert reopened.stats == (10, 0)


def test_fetch_deduplicates_same_batch(tmp_path: Path) -> None:
    catalog = _catalog_frame()
    index = CatalogIndex.build(catalog)
    config = MatchConfig(num_threads=1)
    first = _record("A", "GRAND ACE")
    second = _record("B", "GRAND ACE")
    cache = CandidateCache.open(tmp_path / "cache", catalog, config)
    fetched = cache.fetch([first, second], index, config)
    _assert_equal(fetched, generate_candidates([first, second], index, None, config))
    assert cache.stats == (1, 1)


def test_fingerprint_tolerates_missing_source(tmp_path: Path, monkeypatch) -> None:
    import inspect as _inspect

    catalog = _catalog_frame()
    config = MatchConfig(num_threads=1)
    real = _inspect.getsourcefile

    def _raising(_module) -> str:
        raise TypeError("no source")

    monkeypatch.setattr(_inspect, "getsourcefile", _raising)
    with pytest.raises(ValueError, match="source unavailable"):
        retrieval_fingerprint(catalog, config)
    monkeypatch.setattr(_inspect, "getsourcefile", lambda _module: None)
    with pytest.raises(ValueError, match="source unavailable"):
        retrieval_fingerprint(catalog, config)
    monkeypatch.setattr(_inspect, "getsourcefile", real)
    assert len(retrieval_fingerprint(catalog, config)) == 64


def test_flush_ignores_bad_shard_stem(tmp_path: Path) -> None:
    catalog = _catalog_frame()
    index = CatalogIndex.build(catalog)
    config = MatchConfig(num_threads=1)
    cache = CandidateCache.open(tmp_path / "cache", catalog, config)
    bad = tmp_path / "cache" / cache.fingerprint / "shard_bad.npz"
    np.savez_compressed(bad, keys=np.array([], dtype="S64"))
    cache.fetch(_records()[:2], index, config)
    cache.flush()
    assert cache.stats == (0, 2)


def test_flush_retries_on_collision(tmp_path: Path, monkeypatch) -> None:
    import os as _os

    catalog = _catalog_frame()
    index = CatalogIndex.build(catalog)
    config = MatchConfig(num_threads=1)
    cache = CandidateCache.open(tmp_path / "cache", catalog, config)
    cache.fetch(_records()[:2], index, config)
    real_open = _os.open
    calls = {"count": 0}

    def _flaky(path, *args, **kwargs):
        if calls["count"] == 0:
            calls["count"] += 1
            raise FileExistsError(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(_os, "open", _flaky)
    cache.flush()
    assert calls["count"] == 1


def test_open_during_flush_sees_only_complete_shards(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    catalog = _catalog_frame()
    config = MatchConfig(num_threads=1)
    index = CatalogIndex.build(catalog)
    records = _records()
    first = CandidateCache.open(root, catalog, config)
    first.fetch(records[:2], index, config)
    first.flush()
    first.fetch(records[2:4], index, config)
    save = np.savez_compressed

    def observe(*args, **kwargs):
        reader = CandidateCache.open(root, catalog, config)
        reader.fetch(records[:2], index, config)
        assert reader.stats == (2, 0)
        assert len(list((root / first.fingerprint).glob("shard_*.npz"))) == 1
        return save(*args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", observe)
    first.flush()
    reader = CandidateCache.open(root, catalog, config)
    _assert_equal(reader.fetch(records[:4], index, config), generate_candidates(records[:4], index, None, config))
    assert reader.stats == (4, 0)


def test_failed_flush_can_be_retried(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    catalog = _catalog_frame()
    config = MatchConfig(num_threads=1)
    index = CatalogIndex.build(catalog)
    cache = CandidateCache.open(root, catalog, config)
    cache.fetch(_records()[:2], index, config)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as context:
        context.setattr(np, "savez_compressed", fail)
        with pytest.raises(OSError, match="disk full"):
            cache.flush()
    assert not list((root / cache.fingerprint).iterdir())
    cache.flush()
    reopened = CandidateCache.open(root, catalog, config)
    reopened.fetch(_records()[:2], index, config)
    assert reopened.stats == (2, 0)


def test_flush_handles_shard_published_after_directory_scan(tmp_path: Path, monkeypatch) -> None:
    import os

    root = tmp_path / "cache"
    catalog = _catalog_frame()
    config = MatchConfig(num_threads=1)
    index = CatalogIndex.build(catalog)
    first = CandidateCache.open(root, catalog, config)
    second = CandidateCache.open(root, catalog, config)
    first.fetch(_records()[:2], index, config)
    second.fetch(_records()[2:4], index, config)
    real_open = os.open
    triggered = False

    def publish_before_reservation(path, *args, **kwargs):
        nonlocal triggered
        if not triggered and str(path).endswith(".lock"):
            triggered = True
            second.flush()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", publish_before_reservation)
    first.flush()
    reopened = CandidateCache.open(root, catalog, config)
    reopened.fetch(_records()[:4], index, config)
    assert reopened.stats == (4, 0)


def test_fingerprint_directory_symlink_cannot_bypass_data_guard(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    protected = tmp_path / "data"
    protected.mkdir()
    catalog = _catalog_frame()
    config = MatchConfig()
    (root / retrieval_fingerprint(catalog, config)).symlink_to(protected, target_is_directory=True)
    with pytest.raises(ValueError, match="data directory"):
        CandidateCache.open(root, catalog, config)


def test_returned_arrays_cannot_mutate_cached_entries(tmp_path: Path) -> None:
    catalog = _catalog_frame()
    index = CatalogIndex.build(catalog)
    config = MatchConfig(num_threads=1)
    records = _records()[:2]
    cache = CandidateCache.open(tmp_path / "cache", catalog, config)
    returned = cache.fetch(records, index, config)
    returned[0].positions[:] = -1
    returned[0].sources[:] = 0
    _assert_equal(cache.fetch(records, index, config), generate_candidates(records, index, None, config))
