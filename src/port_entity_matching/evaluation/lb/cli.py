"""Ledger CLI: register, attach, record, facts, show and seed.

Development-only; never imported by ``submission/`` or ``pipeline.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from port_entity_matching.evaluation.lb.decode import DecodeError
from port_entity_matching.evaluation.lb.ledger import (
    DEFAULT_LEDGER_PATH,
    ChannelSpec,
    Decoded,
    Ledger,
    LedgerEntry,
    LedgerError,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lb", description="LB submission ledger.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register")
    reg.add_argument("--entry", type=Path, required=True)

    att = sub.add_parser("attach")
    att.add_argument("--id", dest="entry_id", required=True)
    att.add_argument("--zip", dest="zip_path", type=Path, required=True)

    rec = sub.add_parser("record")
    rec.add_argument("--id", dest="entry_id", required=True)
    rec.add_argument("--score", type=float, required=True)
    rec.add_argument("--unpackaged", action="store_true")

    facts = sub.add_parser("facts")
    facts.add_argument("--out", type=Path, required=True)

    sub.add_parser("show")

    seed = sub.add_parser("seed")
    seed.add_argument("--csv", dest="csv_path", type=Path, required=True)
    for command in (reg, att, rec, facts, seed, sub.choices["show"]):
        command.add_argument("--ledger", type=Path, default=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI over the ledger: register, attach, record, facts, show, seed."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            return _run_register(args)
        if args.command == "attach":
            return _run_attach(args)
        if args.command == "record":
            return _run_record(args)
        if args.command == "facts":
            return _run_facts(args)
        if args.command == "show":
            return _run_show(args)
        return _run_seed(args)
    except (LedgerError, DecodeError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2


def _parse_entry_file(path: Path) -> LedgerEntry:
    from port_entity_matching.evaluation.lb.ledger import _entry_from_json  # noqa: PLC2701

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LedgerError("Entry file must hold a JSON object")
    payload = dict(payload)
    payload["status"] = "registered"
    payload.pop("revision", None)
    return _entry_from_json(payload)


def _run_register(args: argparse.Namespace) -> int:
    ledger = Ledger.load(Path(args.ledger))
    entry = _parse_entry_file(Path(args.entry))
    ledger.register(entry)
    sys.stdout.write(f"registered id={entry.entry_id}\n")
    return 0


def _run_attach(args: argparse.Namespace) -> int:
    ledger = Ledger.load(Path(args.ledger))
    digest = hashlib.sha256()
    with open(args.zip_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    ledger.attach_zip(args.entry_id, digest.hexdigest())
    sys.stdout.write(f"attached id={args.entry_id} sha256={digest.hexdigest()}\n")
    return 0


def _run_record(args: argparse.Namespace) -> int:
    ledger = Ledger.load(Path(args.ledger))
    decoded, k_action, l_action = ledger.record_score(
        args.entry_id, float(args.score), allow_unpackaged=bool(args.unpackaged)
    )
    hits = {f.channel: f.prediction_hit for f in ledger.facts() if f.entry_id == args.entry_id}
    sys.stdout.write(
        f"id={args.entry_id} K={decoded.k} L={decoded.l} "
        f"K_action={k_action} L_action={l_action} K_hit={hits.get('K')} L_hit={hits.get('L')}\n"
    )
    return 0


def _run_facts(args: argparse.Namespace) -> int:
    from port_entity_matching.evaluation.paths import guard_data_path

    ledger = Ledger.load(Path(args.ledger))
    facts = ledger.facts()
    out = guard_data_path(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["entry_id", "channel", "policy_key", "kind", "base", "value", "prediction_hit"],
        )
        writer.writeheader()
        for fact in facts:
            writer.writerow(
                {
                    "entry_id": fact.entry_id,
                    "channel": fact.channel,
                    "policy_key": fact.policy_key,
                    "kind": fact.kind,
                    "base": fact.base if fact.base is not None else "",
                    "value": fact.value,
                    "prediction_hit": fact.prediction_hit,
                }
            )
    sys.stdout.write(f"facts={len(facts)} out={out}\n")
    return 0


def _run_show(args: argparse.Namespace) -> int:
    ledger = Ledger.load(Path(args.ledger))
    for entry in ledger.entries():
        if entry.decoded is not None:
            k_count: Any = entry.decoded.k
            l_count: Any = entry.decoded.l
        else:
            k_count, l_count = "-", "-"
        if entry.status == "scored":
            k_action, l_action = _scored_actions(ledger, entry)
        else:
            k_action, l_action = "-", "-"
        sys.stdout.write(
            f"{entry.entry_id} {entry.status} K={k_count} L={l_count} "
            f"K_action={k_action} L_action={l_action}\n"
        )
    return 0


def _scored_actions(ledger: Ledger, entry: LedgerEntry) -> tuple[str, str]:
    from port_entity_matching.evaluation.lb.ledger import _action_for  # noqa: PLC2701

    assert entry.decoded is not None
    facts = {f.channel: f for f in ledger.facts() if f.entry_id == entry.entry_id}
    k_value = facts["K"].value if "K" in facts else float(entry.decoded.k)
    l_value = facts["L"].value if "L" in facts else float(entry.decoded.l)
    return _action_for(entry.k, k_value), _action_for(entry.l, l_value)


def _run_seed(args: argparse.Namespace) -> int:
    ledger = Ledger.load(Path(args.ledger))
    with open(args.csv_path, encoding="utf-8", newline="") as handle:
        rows = [{k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in csv.DictReader(handle)]
    _seed_rows(ledger, rows)
    sys.stdout.write(f"seeded entries={len({r['entry_id'] for r in rows})}\n")
    return 0


def _seed_rows(ledger: Ledger, rows: Sequence[Mapping[str, str]]) -> None:
    grouped: OrderedDict[str, list[Mapping[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["entry_id"], []).append(row)
    abs_counts: dict[tuple[str, str], float] = {}
    for group in grouped.values():
        for row in group:
            if row["kind"] == "abs":
                value = float(row["value"])
                abs_counts[(row["channel"], row["policy_key"])] = value
    for entry_id, group in grouped.items():
        created = group[0]["created"]
        by_channel = {row["channel"]: row for row in group}
        specs: dict[str, ChannelSpec] = {}
        decoded_counts: dict[str, float] = {}
        for channel, width in (("K", 532.0), ("L", 1784.0)):
            crow = by_channel.get(channel)
            if crow is None:
                specs[channel] = ChannelSpec(
                    policy_key="dummy", policy={}, kind="abs", base=None,
                    window=(0.0, width), prediction=None, decision=(),
                )
                decoded_counts[channel] = 0.0
                continue
            kind = crow["kind"]
            base = crow["base"] or None
            value = float(crow["value"])
            if kind in ("abs", "comp"):
                decoded_count = value
            else:
                if base is None:
                    raise LedgerError(f"Seed row {entry_id}/{channel} needs a base")
                key = (channel, base)
                if key not in abs_counts:
                    raise LedgerError(f"Missing scored base '{base}'")
                decoded_count = value + abs_counts[key]
            specs[channel] = ChannelSpec(
                policy_key=crow["policy_key"], policy={}, kind=cast(Any, kind), base=base,
                window=(0.0, width), prediction=None, decision=(),
            )
            decoded_counts[channel] = decoded_count
        ledger.seed(
            LedgerEntry(
                entry_id=entry_id, created=created, k=specs["K"], l=specs["L"],
                status="scored", legacy=True,
                decoded=Decoded(k=int(decoded_counts["K"]), l=float(decoded_counts["L"])),
            )
        )
