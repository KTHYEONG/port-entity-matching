"""Append-only submission ledger with typed LB facts.

Development-only; never imported by ``submission/`` or ``pipeline.py``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, Literal, cast

from port_entity_matching.evaluation.lb.decode import Decoded, decode_in_windows
from port_entity_matching.evaluation.paths import guard_data_path

__all__ = [
    "DEFAULT_LEDGER_PATH",
    "ChannelSpec",
    "DecisionBand",
    "Fact",
    "FactKind",
    "Ledger",
    "LedgerEntry",
    "LedgerError",
    "Prediction",
    "Status",
]

FactKind = Literal["abs", "comp", "net"]
Status = Literal["registered", "packaged", "scored"]

DEFAULT_LEDGER_PATH: Final[Path] = Path("artifacts/eval/lb/ledger.jsonl")

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Prediction:
    lo: float
    hi: float
    source: str


@dataclass(frozen=True)
class DecisionBand:
    lo: float
    hi: float
    action: str


@dataclass(frozen=True)
class ChannelSpec:
    policy_key: str
    policy: Mapping[str, Any]
    kind: FactKind
    base: str | None
    window: tuple[float, float]
    prediction: Prediction | None
    decision: tuple[DecisionBand, ...]


@dataclass(frozen=True)
class LedgerEntry:
    entry_id: str
    created: str
    k: ChannelSpec
    l: ChannelSpec
    status: Status
    legacy: bool = False
    zip_sha256: str | None = None
    score: float | None = None
    decoded: Decoded | None = None


@dataclass(frozen=True)
class Fact:
    entry_id: str
    channel: Literal["K", "L"]
    policy_key: str
    kind: FactKind
    base: str | None
    value: float
    prediction_hit: bool | None


class LedgerError(ValueError):
    """Raised when a ledger mutation or query violates the append-only contract."""


def _channel_to_json(spec: ChannelSpec) -> dict[str, Any]:
    window = list(spec.window)
    prediction = (
        {"lo": spec.prediction.lo, "hi": spec.prediction.hi, "source": spec.prediction.source}
        if spec.prediction is not None
        else None
    )
    return {
        "policy_key": spec.policy_key,
        "policy": dict(spec.policy),
        "kind": spec.kind,
        "base": spec.base,
        "window": window,
        "prediction": prediction,
        "decision": [{"lo": b.lo, "hi": b.hi, "action": b.action} for b in spec.decision],
    }


def _channel_from_json(payload: Any) -> ChannelSpec:
    if not isinstance(payload, dict):
        raise LedgerError("Invalid channel spec payload")
    try:
        policy_key = str(payload["policy_key"])
        policy = dict(payload.get("policy", {}))
        kind = str(payload["kind"])
        base = payload.get("base")
        window_raw = payload["window"]
        window = (float(window_raw[0]), float(window_raw[1]))
        pred_raw = payload.get("prediction")
        prediction = (
            Prediction(lo=float(pred_raw["lo"]), hi=float(pred_raw["hi"]), source=str(pred_raw["source"]))
            if pred_raw is not None
            else None
        )
        decision = tuple(
            DecisionBand(lo=float(b["lo"]), hi=float(b["hi"]), action=str(b["action"]))
            for b in payload.get("decision", [])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LedgerError(f"Invalid channel spec payload: {error}") from None
    if kind not in ("abs", "comp", "net"):
        raise LedgerError(f"Invalid kind '{kind}'")
    return ChannelSpec(
        policy_key=policy_key, policy=policy, kind=cast(FactKind, kind), base=base, window=window,
        prediction=prediction, decision=decision,
    )


def _entry_to_json(entry: LedgerEntry, revision: int) -> dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "created": entry.created,
        "k": _channel_to_json(entry.k),
        "l": _channel_to_json(entry.l),
        "status": entry.status,
        "legacy": entry.legacy,
        "zip_sha256": entry.zip_sha256,
        "score": entry.score,
        "decoded": {"k": entry.decoded.k, "l": entry.decoded.l} if entry.decoded is not None else None,
        "revision": revision,
    }


def _entry_from_json(payload: Any) -> LedgerEntry:
    if not isinstance(payload, dict):
        raise LedgerError("Invalid ledger entry payload")
    try:
        decoded_raw = payload.get("decoded")
        decoded = (
            Decoded(k=int(decoded_raw["k"]), l=float(decoded_raw["l"])) if decoded_raw is not None else None
        )
        entry = LedgerEntry(
            entry_id=str(payload["entry_id"]),
            created=str(payload["created"]),
            k=_channel_from_json(payload["k"]),
            l=_channel_from_json(payload["l"]),
            status=cast(Status, str(payload["status"])),
            legacy=bool(payload.get("legacy", False)),
            zip_sha256=payload.get("zip_sha256"),
            score=float(payload["score"]) if payload.get("score") is not None else None,
            decoded=decoded,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LedgerError(f"Invalid ledger entry payload: {error}") from None
    if entry.status not in ("registered", "packaged", "scored"):
        raise LedgerError(f"Invalid status '{entry.status}'")
    return entry


def _check_channel(spec: ChannelSpec, name: str) -> None:
    if spec.kind not in ("abs", "comp", "net"):
        raise LedgerError(f"{name}: invalid kind '{spec.kind}'")
    if spec.kind == "abs":
        if spec.base is not None:
            raise LedgerError(f"{name}: abs kind must have base None")
    elif spec.base is None:
        raise LedgerError(f"{name}: kind '{spec.kind}' requires a base")
    lo, hi = spec.window
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise LedgerError(f"{name}: invalid window {(lo, hi)}")
    if spec.prediction is None:
        raise LedgerError(f"{name}: prediction is required")
    if not (math.isfinite(spec.prediction.lo) and math.isfinite(spec.prediction.hi) and spec.prediction.lo <= spec.prediction.hi):
        raise LedgerError(f"{name}: invalid prediction interval")
    if len(spec.decision) == 0:
        raise LedgerError(f"{name}: decision bands are required")
    for band in spec.decision:
        if not (math.isfinite(band.lo) and math.isfinite(band.hi) and band.lo <= band.hi):
            raise LedgerError(f"{name}: invalid band [{band.lo}, {band.hi}]")
    ordered = sorted(spec.decision, key=lambda b: (b.lo, b.hi))
    if [(b.lo, b.hi, b.action) for b in ordered] != [(b.lo, b.hi, b.action) for b in spec.decision]:
        raise LedgerError(f"{name}: decision bands must be sorted by lo")
    for prev, cur in pairwise(ordered):
        if cur.lo < prev.hi:
            raise LedgerError(f"{name}: decision bands overlap")
    pred_lo, pred_hi = spec.prediction.lo, spec.prediction.hi
    if not any(band.lo <= pred_lo <= band.hi for band in ordered):
        raise LedgerError(f"{name}: decision bands do not cover prediction [{pred_lo}, {pred_hi}]")
    cur_val = pred_lo
    for band in ordered:
        if cur_val < band.lo:
            raise LedgerError(f"{name}: decision bands do not cover prediction [{pred_lo}, {pred_hi}]")
        if band.lo <= cur_val <= band.hi:
            cur_val = band.hi
            if cur_val >= pred_hi:
                break
    if cur_val < pred_hi:
        raise LedgerError(f"{name}: decision bands do not cover prediction [{pred_lo}, {pred_hi}]")


def _channel_value(spec: ChannelSpec, decoded_count: float, base_abs: float | None) -> float:
    if spec.kind == "net":
        if base_abs is None:
            raise LedgerError(f"Missing scored base '{spec.base}'")
        return decoded_count - base_abs
    return decoded_count


def _action_for(spec: ChannelSpec, value: float) -> str:
    for band in spec.decision:
        if band.lo <= value <= band.hi:
            return band.action
    return "UNPLANNED"


class Ledger:
    def __init__(self, path: Path, order: list[str], current: dict[str, LedgerEntry], revisions: dict[str, int]):
        self._path = path
        self._order = list(order)
        self._current = dict(current)
        self._revisions = dict(revisions)

    @classmethod
    def load(cls, path: Path) -> Ledger:
        """Read the append-only JSONL ledger (missing file = empty); last revision per entry_id wins."""
        resolved = guard_data_path(Path(path))
        order: list[str] = []
        current: dict[str, LedgerEntry] = {}
        revisions: dict[str, int] = {}
        if not resolved.is_file():
            return cls(resolved, order, current, revisions)
        text = resolved.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            try:
                payload = json.loads(line)
            except ValueError:
                raise LedgerError(f"Corrupt ledger line {lineno}: invalid JSON") from None
            if not isinstance(payload, dict):
                raise LedgerError(f"Corrupt ledger line {lineno}: expected an object") from None
            try:
                revision = int(payload["revision"])
            except (KeyError, TypeError, ValueError):
                raise LedgerError(f"Corrupt ledger line {lineno}: missing revision") from None
            try:
                entry = _entry_from_json(payload)
            except LedgerError as error:
                raise LedgerError(f"Corrupt ledger line {lineno}: {error}") from None
            entry_id = entry.entry_id
            if entry_id not in current:
                order.append(entry_id)
            if revision >= revisions.get(entry_id, 0):
                current[entry_id] = entry
                revisions[entry_id] = revision
        return cls(resolved, order, current, revisions)

    def _append(self, entry: LedgerEntry) -> None:
        revision = self._revisions.get(entry.entry_id, 0) + 1
        line = json.dumps(_entry_to_json(entry, revision), ensure_ascii=False)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if entry.entry_id not in self._current:
            self._order.append(entry.entry_id)
        self._current[entry.entry_id] = entry
        self._revisions[entry.entry_id] = revision

    def register(self, entry: LedgerEntry) -> None:
        """Append a new registered entry; LedgerError on duplicate id or missing prediction/decision."""
        if entry.entry_id in self._current:
            raise LedgerError(f"Duplicate entry id '{entry.entry_id}'")
        if entry.status != "registered":
            raise LedgerError(f"Entry '{entry.entry_id}' must be registered")
        if entry.legacy:
            raise LedgerError(f"Entry '{entry.entry_id}' must not be legacy")
        if entry.zip_sha256 is not None or entry.score is not None or entry.decoded is not None:
            raise LedgerError(f"Entry '{entry.entry_id}' must not contain packaging or scoring results")
        _check_channel(entry.k, "K")
        _check_channel(entry.l, "L")
        self._append(entry)

    def attach_zip(self, entry_id: str, sha256: str) -> None:
        """Move a registered entry to packaged with the zip digest; LedgerError on wrong status."""
        if not _SHA_RE.match(sha256):
            raise LedgerError(f"Invalid sha256 '{sha256}'")
        entry = self._current.get(entry_id)
        if entry is None:
            raise LedgerError(f"Unknown entry id '{entry_id}'")
        if entry.status != "registered":
            raise LedgerError(f"Entry '{entry_id}' is {entry.status}, expected registered")
        self._append(
            LedgerEntry(
                entry_id=entry.entry_id, created=entry.created, k=entry.k, l=entry.l,
                status="packaged", legacy=entry.legacy, zip_sha256=sha256,
                score=entry.score, decoded=entry.decoded,
            )
        )

    def record_score(self, entry_id: str, score: float, allow_unpackaged: bool = False) -> tuple[Decoded, str, str]:
        """Decode, mark scored, and return (decoded, K action, L action); LedgerError/DecodeError on failure."""
        entry = self._current.get(entry_id)
        if entry is None:
            raise LedgerError(f"Unknown entry id '{entry_id}'")
        if entry.status == "scored":
            raise LedgerError(f"Entry '{entry_id}' is already scored")
        if entry.status != "packaged" and not (
            entry.status == "registered" and allow_unpackaged and entry.zip_sha256 is None
        ):
            raise LedgerError(f"Entry '{entry_id}' is {entry.status}, expected packaged")
        decoded = decode_in_windows(score, entry.k.window, entry.l.window)
        k_count = float(decoded.k)
        l_count = float(decoded.l)
        k_base = self._scored_base("K", entry.k.base, entry_id) if entry.k.kind == "net" else None
        l_base = self._scored_base("L", entry.l.base, entry_id) if entry.l.kind == "net" else None
        k_value = k_count if entry.k.kind != "net" else (k_count - k_base if k_base is not None else None)
        l_value = l_count if entry.l.kind != "net" else (l_count - l_base if l_base is not None else None)
        k_action = _action_for(entry.k, k_value) if k_value is not None else "UNPLANNED"
        l_action = _action_for(entry.l, l_value) if l_value is not None else "UNPLANNED"
        self._append(
            LedgerEntry(
                entry_id=entry.entry_id, created=entry.created, k=entry.k, l=entry.l,
                status="scored", legacy=entry.legacy, zip_sha256=entry.zip_sha256,
                score=float(score), decoded=decoded,
            )
        )
        return decoded, k_action, l_action

    def _scored_base(self, channel: Literal["K", "L"], base: str | None, exclude_id: str | None = None) -> float:
        if base is None:
            raise LedgerError("Missing base")
        for entry_id in reversed(self._order):
            if exclude_id is not None and entry_id == exclude_id:
                continue
            entry = self._current[entry_id]
            if entry.status != "scored" or entry.decoded is None:
                continue
            candidate = entry.k if channel == "K" else entry.l
            if candidate.policy_key == base and candidate.kind == "abs":
                return float(entry.decoded.k) if channel == "K" else float(entry.decoded.l)
        raise LedgerError(f"Missing scored base '{base}'")

    def seed(self, entry: LedgerEntry) -> None:
        """Append a legacy scored entry (pre-ledger history) without prediction requirements."""
        if entry.entry_id in self._current:
            raise LedgerError(f"Duplicate entry id '{entry.entry_id}'")
        if not entry.legacy or entry.status != "scored":
            raise LedgerError(f"Seed entry '{entry.entry_id}' must be legacy scored")
        if entry.decoded is None:
            raise LedgerError(f"Seed entry '{entry.entry_id}' requires decoded counts")
        for spec, name in ((entry.k, "K"), (entry.l, "L")):
            if spec.kind not in ("abs", "comp", "net"):
                raise LedgerError(f"{name}: invalid kind '{spec.kind}'")
            if spec.kind == "abs" and spec.base is not None:
                raise LedgerError(f"{name}: abs kind must have base None")
            if spec.kind in ("comp", "net") and spec.base is None:
                raise LedgerError(f"{name}: kind '{spec.kind}' requires a base")
            if spec.window[0] > spec.window[1]:
                raise LedgerError(f"{name}: invalid window")
        self._append(entry)

    def entries(self) -> list[LedgerEntry]:
        """Current revision of every entry in first-registration order."""
        return [self._current[entry_id] for entry_id in self._order]

    def _base_lookup(self, channel: Literal["K", "L"], base: str) -> float:
        for entry_id in reversed(self._order):
            entry = self._current[entry_id]
            if entry.status != "scored" or entry.decoded is None:
                continue
            candidate = entry.k if channel == "K" else entry.l
            if candidate.policy_key == base and candidate.kind == "abs":
                return float(entry.decoded.k) if channel == "K" else float(entry.decoded.l)
        raise LedgerError(f"Missing scored base '{base}'")

    def facts(self) -> list[Fact]:
        """Typed facts from scored entries (both channels); LedgerError if a net base is not scored."""
        out: list[Fact] = []
        for entry_id in self._order:
            entry = self._current[entry_id]
            if entry.status != "scored" or entry.decoded is None:
                continue
            for channel in ("K", "L"):
                spec = entry.k if channel == "K" else entry.l
                count = float(entry.decoded.k) if channel == "K" else float(entry.decoded.l)
                if spec.policy_key == "dummy":
                    continue
                if spec.kind in ("abs", "comp"):
                    value = count
                else:
                    if spec.base is None:
                        raise LedgerError("Missing scored base 'None'")
                    value = count - self._base_lookup(channel, spec.base)
                if entry.legacy or spec.prediction is None:
                    hit: bool | None = None
                else:
                    hit = spec.prediction.lo <= value <= spec.prediction.hi
                out.append(
                    Fact(
                        entry_id=entry_id, channel=channel,
                        policy_key=spec.policy_key, kind=spec.kind, base=spec.base,
                        value=value, prediction_hit=hit,
                    )
                )
        return out
