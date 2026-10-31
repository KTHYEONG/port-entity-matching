"""Ledger CLI: register, attach, record, facts, show, seed, interval, gate, design, verdict.

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

import pandas as pd

from port_entity_matching.evaluation.lb.decode import DecodeError
from port_entity_matching.evaluation.lb.feasible import FeasibilityError
from port_entity_matching.evaluation.lb.ledger import (
    DEFAULT_LEDGER_PATH,
    ChannelSpec,
    Decoded,
    Fact,
    FactKind,
    Ledger,
    LedgerEntry,
    LedgerError,
)
from port_entity_matching.evaluation.lb.response import ResponseMatrix

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lb", description="LB submission ledger.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser(
        "register",
        help="Register a probe entry.",
        description=(
            "Probe workflow: the prediction of a new entry comes from "
            "`lb interval` on a gate-passing frame (source=\"feasible_lp\"); "
            "otherwise source=\"none\" with prediction [-N, N] and a decision "
            "table that states what each outcome changes."
        ),
    )
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

    interval = sub.add_parser("interval", help="Set-membership interval for one target key.")
    interval.add_argument("--frame", type=Path, required=True)
    interval.add_argument("--script", choices=("K", "L"), required=True)
    interval.add_argument("--kinds", required=True, help="KEY=KIND,... with KIND in abs|comp|net")
    interval.add_argument("--target", required=True)
    interval.add_argument("--unknown-process", action="append", default=[], help="Training process names")
    interval.add_argument("--ignore", action="append", default=[], help="Fact keys to skip")

    gate = sub.add_parser("gate", help="Leave-one-key-out validation gate.")
    gate.add_argument("--frame", type=Path, required=True)
    gate.add_argument("--script", choices=("K", "L"), required=True)
    gate.add_argument("--kinds", required=True, help="KEY=KIND,... with KIND in abs|comp|net")
    gate.add_argument("--held-out", required=True, help="KEY,... held-out keys")
    gate.add_argument("--ignore", action="append", default=[], help="Fact keys to skip")

    design = sub.add_parser("design", help="Rank candidate measurement policies.")
    design.add_argument("--frame", type=Path, required=True)
    design.add_argument("--script", choices=("K", "L"), required=True)
    design.add_argument("--kinds", required=True, help="KEY=KIND,... with KIND in abs|comp|net")
    design.add_argument("--candidates", required=True, help="KEY,... candidate keys")
    design.add_argument("--targets", required=True, help="KEY,... target keys")
    design.add_argument("--ignore", action="append", default=[], help="Fact keys to skip")

    verdict = sub.add_parser("verdict", help="LB verdict from public net and churn.")
    verdict.add_argument("--net", type=float, required=True)
    verdict.add_argument("--churn", type=float, required=True)
    verdict.add_argument("--threshold", type=float, default=0.9)
    for command in (reg, att, rec, facts, seed, interval, gate, design, sub.choices["show"]):
        command.add_argument("--ledger", type=Path, default=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI over the ledger: register, attach, record, facts, show, seed, interval, gate, design, verdict."""
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
        if args.command == "interval":
            return _run_interval(args)
        if args.command == "gate":
            return _run_gate(args)
        if args.command == "design":
            return _run_design(args)
        if args.command == "verdict":
            return _run_verdict(args)
        return _run_seed(args)
    except FeasibilityError as error:
        sys.stderr.write(f"LIBRARY INCONSISTENT: {error}\n")
        return 2
    except (LedgerError, DecodeError, ValueError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2


def _parse_kinds(spec: str) -> dict[str, FactKind]:
    kinds: dict[str, FactKind] = {}
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        key, sep, kind = item.partition("=")
        if not sep or not key or kind not in ("abs", "comp", "net"):
            raise LedgerError(f"Invalid kinds entry {item!r}: expected KEY=KIND with KIND in abs|comp|net")
        kinds[key] = cast("FactKind", kind)
    if not kinds:
        raise LedgerError("Empty --kinds: expected KEY=KIND,...")
    return kinds


def _split_keys(values: Sequence[str] | str) -> list[str]:
    parts = [values] if isinstance(values, str) else list(values)
    out: list[str] = []
    for part in parts:
        out.extend(item.strip() for item in part.split(","))
    return [item for item in out if item]


def _load_frame_and_facts(
    args: argparse.Namespace,
) -> tuple[ResponseMatrix, list[Fact]]:
    from port_entity_matching.evaluation.lb.response import response_matrix

    frame = pd.read_csv(Path(args.frame), dtype=str)
    kinds = _parse_kinds(str(args.kinds))
    facts = Ledger.load(Path(args.ledger)).facts()
    matrix = response_matrix(frame, args.script, kinds, "base")
    return matrix, facts


def _run_interval(args: argparse.Namespace) -> int:
    from port_entity_matching.evaluation.lb.feasible import classify, feasible_interval
    from port_entity_matching.evaluation.lb.response import circular_cells

    matrix, facts = _load_frame_and_facts(args)
    unknown = circular_cells(matrix, _split_keys(list(args.unknown_process)))
    interval = feasible_interval(matrix, facts, str(args.target), unknown_cells=unknown, ignore=_split_keys(list(args.ignore)))
    verdict = classify(interval)
    sys.stdout.write(
        f"target={args.target} lo={interval.lo} hi={interval.hi} "
        f"verdict={verdict} unknown_cells={len(unknown)}\n"
    )
    return 0


def _run_gate(args: argparse.Namespace) -> int:
    from port_entity_matching.evaluation.lb.feasible import loo_gate

    matrix, facts = _load_frame_and_facts(args)
    held_out = _split_keys(str(args.held_out))
    report = loo_gate(matrix, facts, held_out, ignore=_split_keys(list(args.ignore)))
    for key in held_out:
        interval = report.intervals[key]
        hit = interval.lo <= report.truth[key] <= interval.hi
        sys.stdout.write(f"{key} lo={interval.lo} hi={interval.hi} truth={report.truth[key]} hit={hit}\n")
    status = "GATE PASS" if report.passes else "GATE FAIL"
    sys.stdout.write(f"coverage={report.coverage:.4f} decisive={report.decisive} {status}\n")
    return 0 if report.passes else 3


def _run_design(args: argparse.Namespace) -> int:
    from port_entity_matching.evaluation.lb.design import rank_probes

    matrix, facts = _load_frame_and_facts(args)
    scores = rank_probes(
        matrix,
        facts,
        _split_keys(str(args.candidates)),
        _split_keys(str(args.targets)),
        ignore=_split_keys(list(args.ignore)),
    )
    for score in scores:
        sys.stdout.write(
            f"{score.candidate} expected_width={score.expected_width:.4f} "
            f"worst_width={score.worst_width:.4f} feasible_points={score.feasible_points}\n"
        )
    return 0


def _run_verdict(args: argparse.Namespace) -> int:
    from port_entity_matching.evaluation.lb.decision import lb_verdict, private_gain_probability

    prob = private_gain_probability(float(args.net), float(args.churn))
    verdict = lb_verdict(float(args.net), float(args.churn), float(args.threshold))
    sys.stdout.write(f"p_gain={prob:.4f} verdict={verdict}\n")
    return 0


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
        if entry.legacy:
            k_action, l_action = "LEGACY", "LEGACY"
        elif entry.status == "scored":
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
