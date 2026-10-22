"""Disk-backed, content-addressed store of CandidateSets for one catalog and retrieval config."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import port_entity_matching.candidates as candidates_module
import port_entity_matching.catalog as catalog_module
import port_entity_matching.normalize as normalize_module
import port_entity_matching.records as records_module
from port_entity_matching.candidates import CandidateSet, generate_candidates
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig, config_to_dict
from port_entity_matching.evaluation.paths import guard_data_path
from port_entity_matching.records import QueryRecord

__all__ = ["CandidateCache", "retrieval_fingerprint"]

_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        "seed",
        "cv_splits",
        "num_threads",
        "batch_size",
        "query_chunk",
        "token_match_threshold",
        "band_low",
        "band_high",
        "near_tol",
        "ensemble_size",
    }
)


def _excluded(name: str) -> bool:
    return name in _EXCLUDED_FIELDS or name.startswith("lgbm_") or name.startswith("augment")


def retrieval_fingerprint(catalog: pd.DataFrame, config: MatchConfig) -> str:
    """Hex SHA-256 over the retrieval code, the catalog contents and the retrieval-relevant config fields."""
    digest = hashlib.sha256()
    for module in (candidates_module, catalog_module, normalize_module, records_module):
        try:
            source_file = inspect.getsourcefile(module)
        except TypeError as error:
            raise ValueError(f"Retrieval source unavailable: {module.__name__}") from error
        if source_file is None:
            raise ValueError(f"Retrieval source unavailable: {module.__name__}")
        digest.update(Path(source_file).read_bytes())
    catalog_hash = pd.util.hash_pandas_object(catalog, index=True).to_numpy()
    digest.update(json.dumps(list(catalog.columns), ensure_ascii=False).encode("utf-8"))
    digest.update(np.ascontiguousarray(catalog_hash).tobytes())
    relevant = {key: value for key, value in config_to_dict(config).items() if not _excluded(key)}
    digest.update(json.dumps(relevant, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    return digest.hexdigest()


def _record_key(record: QueryRecord) -> bytes:
    payload = dataclasses.asdict(record)
    payload.pop("sample_id", None)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().encode("ascii")


class CandidateCache:
    """Disk-backed store of CandidateSets keyed by record content."""

    def __init__(self, directory: Path, fingerprint: str) -> None:
        self._directory = directory
        self._fingerprint = fingerprint
        self._store: dict[bytes, tuple[NDArray[np.int64], NDArray[np.uint8]]] = {}
        self._pending: dict[bytes, tuple[NDArray[np.int64], NDArray[np.uint8]]] = {}
        self._hits = 0
        self._misses = 0

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the catalog and retrieval configuration."""
        return self._fingerprint

    @classmethod
    def open(cls, root: str | Path, catalog: pd.DataFrame, config: MatchConfig) -> CandidateCache:
        """Load or create the cache directory for this fingerprint under root."""
        resolved_root = guard_data_path(Path(root))
        fingerprint = retrieval_fingerprint(catalog, config)
        directory = guard_data_path(resolved_root / fingerprint)
        directory.mkdir(parents=True, exist_ok=True)
        cache = cls(directory, fingerprint)
        for shard in sorted(directory.glob("shard_*.npz")):
            with np.load(shard, allow_pickle=False) as data:
                keys = data["keys"]
                offsets = data["offsets"]
                positions = data["positions"]
                sources = data["sources"]
                for i in range(len(keys)):
                    key = bytes(keys[i])
                    start = int(offsets[i])
                    stop = int(offsets[i + 1])
                    cache._store[key] = (
                        np.ascontiguousarray(positions[start:stop], dtype=np.int64),
                        np.ascontiguousarray(sources[start:stop], dtype=np.uint8),
                    )
        return cache

    def fetch(self, records: Sequence[QueryRecord], index: CatalogIndex, config: MatchConfig) -> list[CandidateSet]:
        """Return generate_candidates output, computing and storing only cache misses."""
        keys = [_record_key(record) for record in records]
        out: list[CandidateSet | None] = [None] * len(records)
        miss_keys: list[bytes] = []
        miss_positions: list[int] = []
        seen_miss: dict[bytes, int] = {}
        for position, key in enumerate(keys):
            hit = self._store.get(key)
            if hit is None:
                hit = self._pending.get(key)
            if hit is not None:
                self._hits += 1
                out[position] = CandidateSet(hit[0].copy(), hit[1].copy())
                continue
            if key in seen_miss:
                self._hits += 1
                continue
            self._misses += 1
            seen_miss[key] = len(miss_keys)
            miss_keys.append(key)
            miss_positions.append(position)
        if miss_keys:
            miss_records = [records[position] for position in miss_positions]
            computed = generate_candidates(miss_records, index, None, config)
            for key, _record_position, candidate_set in zip(miss_keys, miss_positions, computed, strict=True):
                positions = np.array(candidate_set.positions, dtype=np.int64, copy=True)
                sources = np.array(candidate_set.sources, dtype=np.uint8, copy=True)
                self._pending[key] = (positions, sources)
                self._store[key] = (positions, sources)
            for position, key in enumerate(keys):
                if out[position] is None:
                    hit = self._store[key]
                    out[position] = CandidateSet(hit[0].copy(), hit[1].copy())
        return [candidate for candidate in out if candidate is not None]

    def flush(self) -> None:
        """Persist new entries as one additional shard file; no-op when nothing is new."""
        if not self._pending:
            return
        ordered = sorted(self._pending)
        keys = np.array(ordered, dtype="S64")
        arrays = [self._pending[key] for key in ordered]
        offsets = np.zeros(len(ordered) + 1, dtype=np.int64)
        for i, (positions, _sources) in enumerate(arrays):
            offsets[i + 1] = offsets[i] + len(positions)
        positions_all = (
            np.concatenate([pair[0].astype(np.int64) for pair in arrays]).astype(np.int64)
            if arrays
            else np.empty(0, dtype=np.int64)
        )
        sources_all = (
            np.concatenate([pair[1].astype(np.uint8) for pair in arrays]).astype(np.uint8)
            if arrays
            else np.empty(0, dtype=np.uint8)
        )
        existing = sorted(self._directory.glob("shard_*.npz"))
        next_number = 0
        if existing:
            numbers = []
            for shard in existing:
                try:
                    numbers.append(int(shard.stem.split("_")[1]))
                except (IndexError, ValueError):
                    continue
            if numbers:
                next_number = max(numbers) + 1
        while True:
            target = self._directory / f"shard_{next_number:04d}.npz"
            reservation = target.with_suffix(".lock")
            try:
                fd = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                next_number += 1
                continue
            os.close(fd)
            if target.exists():
                reservation.unlink()
                next_number += 1
                continue
            break
        try:
            fd2, tmp_name = tempfile.mkstemp(suffix=".npz", dir=str(self._directory))
            os.close(fd2)
            try:
                np.savez_compressed(tmp_name, keys=keys, offsets=offsets, positions=positions_all, sources=sources_all)
                os.replace(tmp_name, target)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp_name)
        finally:
            reservation.unlink()
        self._pending = {}

    @property
    def stats(self) -> tuple[int, int]:
        """(hits, misses) counted since open."""
        return (self._hits, self._misses)
