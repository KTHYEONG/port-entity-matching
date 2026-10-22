"""Catalog-derived features and sorted retrieval indexes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from port_entity_matching.normalize import (
    canonical_latin,
    digit_tokens,
    latin_phonetic_key,
    latin_phonetic_key_v2,
    latin_tokens,
    latin_word_keys,
    sorted_token_key,
)

__all__ = ["CatalogIndex"]

_REQUIRED_COLUMNS: tuple[str, ...] = ("vessel_id", "vessel_name", "gross_tonnage")


@dataclass(frozen=True, eq=False)
class CatalogIndex:
    """Catalog-side features and lookup structures; built once per run."""

    vessel_ids: tuple[str, ...]
    names: tuple[str, ...]
    canonical: list[str]
    tokens: list[frozenset[str]]
    digits: list[tuple[str, ...]]
    key_cons: list[str]
    key_vow: list[str]
    key2_cons: list[str]
    key2_vow: list[str]
    gt: NDArray[np.float64]
    vessel_type: tuple[str, ...]
    flag: tuple[str, ...]
    same_gt_count: NDArray[np.int64]
    token_df: dict[str, int]
    sorted_canonical: list[str]
    word_keys: list[tuple[str, ...]]
    sorted_key2_cons: list[str]
    word_key_df: dict[str, int]
    id_to_pos: dict[str, int]
    _gt_values: NDArray[np.float64] = field(
        init=False, repr=False, default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    _gt_positions: NDArray[np.int64] = field(
        init=False, repr=False, default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    _postings: dict[str, NDArray[np.int64]] = field(init=False, repr=False, default_factory=dict)
    _word_postings: dict[str, NDArray[np.int64]] = field(init=False, repr=False, default_factory=dict)

    @classmethod
    def build(cls, catalog: pd.DataFrame) -> CatalogIndex:
        """Index a catalog frame; raise ValueError on missing columns, empty frame or duplicate vessel_id."""
        missing = [column for column in _REQUIRED_COLUMNS if column not in catalog.columns]
        if missing:
            raise ValueError(f"Catalog frame is missing required columns: {', '.join(missing)}")
        if len(catalog) == 0:
            raise ValueError("Catalog frame is empty")

        vessel_ids = tuple(_as_text(value) for value in catalog["vessel_id"].tolist())
        duplicated = pd.Series(vessel_ids).duplicated().to_numpy()
        if bool(duplicated.any()):
            repeated = sorted({vessel_ids[position] for position in np.flatnonzero(duplicated)})
            raise ValueError(f"Catalog frame has duplicate vessel_id values: {', '.join(repeated[:5])}")

        names = tuple(_as_text(value) for value in catalog["vessel_name"].tolist())
        canonical = [canonical_latin(name) for name in names]
        tokens = [frozenset(latin_tokens(name)) for name in names]
        digits = [digit_tokens(name) for name in names]
        key_cons = [latin_phonetic_key(name, False) for name in names]
        key_vow = [latin_phonetic_key(name, True) for name in names]
        key2_cons = [latin_phonetic_key_v2(name, False) for name in names]
        key2_vow = [latin_phonetic_key_v2(name, True) for name in names]
        gt = _gross_tonnages(catalog["gross_tonnage"])
        vessel_type = (
            tuple(_as_text(value) for value in catalog["vessel_type"].tolist())
            if "vessel_type" in catalog.columns
            else tuple("" for _ in names)
        )
        flag = (
            tuple(_as_text(value) for value in catalog["flag_name"].tolist())
            if "flag_name" in catalog.columns
            else tuple("" for _ in names)
        )

        id_to_pos = {vessel_id: position for position, vessel_id in enumerate(vessel_ids)}
        postings_lists: dict[str, list[int]] = {}
        for position, row_tokens in enumerate(tokens):
            for token in row_tokens:
                postings_lists.setdefault(token, []).append(position)
        postings = {token: np.asarray(positions, dtype=np.int64) for token, positions in postings_lists.items()}
        token_df = {token: len(positions) for token, positions in postings_lists.items()}

        word_keys = [latin_word_keys(name) for name in names]
        sorted_canonical = [sorted_token_key(latin_tokens(name)) for name in names]
        sorted_key2_cons = [sorted_token_key(keys) for keys in word_keys]
        word_postings_lists: dict[str, list[int]] = {}
        for position, keys in enumerate(word_keys):
            for key in set(keys):
                word_postings_lists.setdefault(key, []).append(position)
        word_postings = {key: np.asarray(positions, dtype=np.int64) for key, positions in word_postings_lists.items()}
        word_key_df = {key: len(positions) for key, positions in word_postings_lists.items()}

        finite_positions = np.flatnonzero(np.isfinite(gt))
        order = np.argsort(gt[finite_positions], kind="stable")
        index = cls(
            vessel_ids=vessel_ids,
            names=names,
            canonical=canonical,
            tokens=tokens,
            digits=digits,
            key_cons=key_cons,
            key_vow=key_vow,
            key2_cons=key2_cons,
            key2_vow=key2_vow,
            gt=gt,
            vessel_type=vessel_type,
            flag=flag,
            same_gt_count=_same_gt_counts(gt),
            token_df=token_df,
            sorted_canonical=sorted_canonical,
            word_keys=word_keys,
            sorted_key2_cons=sorted_key2_cons,
            word_key_df=word_key_df,
            id_to_pos=id_to_pos,
        )
        object.__setattr__(index, "_gt_values", gt[finite_positions][order])
        object.__setattr__(index, "_gt_positions", finite_positions[order].astype(np.int64))
        object.__setattr__(index, "_postings", postings)
        object.__setattr__(index, "_word_postings", word_postings)
        return index

    def __len__(self) -> int:
        """Number of catalog rows."""
        return len(self.vessel_ids)

    def gt_range(self, low: float, high: float) -> NDArray[np.int64]:
        """Ascending int64 positions with low <= GT <= high (NaN GT never included)."""
        if np.isnan(low) or np.isnan(high) or low > high:
            return np.empty(0, dtype=np.int64)
        start = int(np.searchsorted(self._gt_values, low, side="left"))
        stop = int(np.searchsorted(self._gt_values, high, side="right"))
        return np.sort(self._gt_positions[start:stop])

    def gt_exact(self, value: float) -> NDArray[np.int64]:
        """Ascending int64 positions whose GT equals value exactly."""
        if not np.isfinite(value):
            return np.empty(0, dtype=np.int64)
        start = int(np.searchsorted(self._gt_values, value, side="left"))
        stop = int(np.searchsorted(self._gt_values, value, side="right"))
        return np.sort(self._gt_positions[start:stop])

    def token_postings(self, token: str) -> NDArray[np.int64]:
        """Ascending int64 positions whose latin_tokens contain token; empty array if unknown."""
        postings = self._postings.get(token)
        if postings is None:
            return np.empty(0, dtype=np.int64)
        return postings

    def word_key_postings(self, key: str) -> NDArray[np.int64]:
        """Ascending positions whose word_keys contain key; empty array if unknown."""
        postings = self._word_postings.get(key)
        if postings is None:
            return np.empty(0, dtype=np.int64)
        return postings

    def token_idf(self, token: str) -> float:
        """Log IDF of a Latin token over catalog rows."""
        return math.log((len(self) + 1) / (self.token_df.get(token, 0) + 1))

    def word_key_idf(self, key: str) -> float:
        """Log IDF of a word key over catalog rows."""
        return math.log((len(self) + 1) / (self.word_key_df.get(key, 0) + 1))


def _as_text(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def _gross_tonnages(column: pd.Series) -> NDArray[np.float64]:
    values = pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    return np.where(np.isfinite(values) & (values > 0), values, np.nan)


def _same_gt_counts(gt: NDArray[np.float64]) -> NDArray[np.int64]:
    counts = np.zeros(len(gt), dtype=np.int64)
    finite = np.isfinite(gt)
    if bool(finite.any()):
        _, inverse, occurrences = np.unique(gt[finite], return_inverse=True, return_counts=True)
        counts[finite] = occurrences[inverse]
    return counts
