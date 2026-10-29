"""Invariant tests for the LB submission ledger."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from port_entity_matching.evaluation.lb.decode import Decoded, _reconstructed
from port_entity_matching.evaluation.lb.ledger import (
    ChannelSpec,
    DecisionBand,
    Ledger,
    LedgerEntry,
    LedgerError,
    Prediction,
)

FULL_K = (440.0, 532.0)
FULL_L = (1680.0, 1784.0)

SEED_ROWS: list[tuple[str, str, str, str, str]] = [
    ("rev2syn", "K", "rev2syn", "abs", "505"),
    ("rev2syn", "L", "rev2syn", "abs", "1713"),
    ("rank2", "K", "rank2", "comp", "9"),
    ("rank2", "L", "rank2", "comp", "19"),
    ("rank3", "K", "rank3", "comp", "4"),
    ("rank3", "L", "rank3", "comp", "4.5"),
    ("k_namek", "K", "k_namek", "comp", "2"),
    ("tonex", "K", "tonex", "comp", "2"),
    ("tonex", "L", "tonex", "comp", "0"),
    ("k_rule", "K", "k_rule", "comp", "1"),
    ("k_fwdfz", "K", "k_fwdfz", "comp", "3"),
    ("k_rlinf", "K", "k_rlinf", "comp", "3"),
    ("k_tonsw", "K", "k_tonsw", "net", "-9"),
    ("aug", "K", "aug", "net", "0"),
    ("aug", "L", "aug", "net", "19.5"),
    ("wide", "K", "wide", "net", "-1"),
    ("wide", "L", "wide", "net", "19"),
    ("k_rl01", "K", "k_rl01", "net", "-8"),
    ("k_rl03", "K", "k_rl03", "net", "-20"),
    ("k_qwen", "K", "k_qwen", "net", "-1"),
    ("k_rr05", "K", "k_rr05", "net", "-3"),
    ("k_rr07", "K", "k_rr07", "net", "-5"),
    ("knn", "K", "knn", "net", "-14"),
    ("rev", "K", "rev", "net", "-6"),
    ("rev2", "K", "rev2", "net", "-3"),
    ("rev3syn", "K", "rev3syn", "net", "-1"),
    ("rev3synk", "K", "rev3synk", "net", "-4"),
    ("rev2synkf", "K", "rev2synkf", "net", "-5"),
    ("k8b_dual", "K", "k8b_dual", "net", "-1"),
    ("k_fp_namek", "K", "k_fp_namek", "abs", "386"),
    ("fp_tonname", "K", "fp_tonname", "abs", "254"),
    ("fp_tonname", "L", "fp_tonname", "abs", "655.5"),
    ("l_namel", "L", "l_namel", "comp", "1.5"),
    ("l_fz", "L", "l_fz", "comp", "28.5"),
    ("l_notin", "L", "l_notin", "comp", "16.5"),
    ("l_tonsw", "L", "l_tonsw", "net", "-12.5"),
    ("l_combo", "L", "l_combo", "net", "22.5"),
    ("l_big", "L", "l_big", "net", "22.5"),
    ("l_aug30", "L", "l_aug30", "net", "19"),
    ("l_combo2", "L", "l_combo2", "net", "22.5"),
    ("l_fp_namel", "L", "l_fp_namel", "abs", "1489"),
]


def _spec(
    key: str = "p",
    kind: str = "abs",
    base: str | None = None,
    window: tuple[float, float] = FULL_K,
    pred: tuple[float, float] | None = (440.0, 532.0),
    bands: tuple[tuple[float, float, str], ...] = ((440.0, 532.0, "ship"),),
) -> ChannelSpec:
    return ChannelSpec(
        policy_key=key,
        policy={},
        kind=kind,  # type: ignore[typeddict-item]
        base=base,
        window=window,
        prediction=Prediction(lo=pred[0], hi=pred[1], source="test") if pred is not None else None,
        decision=tuple(DecisionBand(lo=lo, hi=hi, action=action) for lo, hi, action in bands),
    )


def _entry(entry_id: str = "e1", k: ChannelSpec | None = None, l: ChannelSpec | None = None) -> LedgerEntry:
    return LedgerEntry(
        entry_id=entry_id,
        created="2026-10-29",
        k=k if k is not None else _spec(),
        l=l if l is not None else _spec(key="q", window=FULL_L, pred=(1680.0, 1784.0), bands=((1680.0, 1784.0, "ship"),)),
        status="registered",
    )


def _seed_base(ledger: Ledger) -> None:
    ledger.seed(
        LedgerEntry(
            entry_id="rev2syn",
            created="2026-10-29",
            k=ChannelSpec(policy_key="rev2syn", policy={}, kind="abs", base=None, window=(0.0, 532.0), prediction=None, decision=()),
            l=ChannelSpec(policy_key="rev2syn", policy={}, kind="abs", base=None, window=(0.0, 1784.0), prediction=None, decision=()),
            status="scored",
            legacy=True,
            decoded=Decoded(k=505, l=1713.0),
        )
    )


def test_register_requires_prediction(tmp_path: Path) -> None:
    ledger = Ledger.load(tmp_path / "ledger.jsonl")
    entry = _entry(k=_spec(pred=None))
    with pytest.raises(LedgerError):
        ledger.register(entry)
    assert not (tmp_path / "ledger.jsonl").exists()


@pytest.mark.parametrize(("pred", "bands"), [((0.0, 0.0), ((-2.0, -1.0, "x"),)), ((float("nan"), 1.0), ((0.0, 1.0, "x"),))])
def test_invalid_prediction_is_not_registered(tmp_path: Path, pred, bands) -> None:
    ledger = Ledger.load(tmp_path / "ledger.jsonl")
    with pytest.raises(LedgerError):
        ledger.register(_entry(k=_spec(pred=pred, bands=bands)))
    assert not (tmp_path / "ledger.jsonl").exists()


def test_blank_ledger_line_is_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="line 1"):
        Ledger.load(path)


def test_decision_bands_must_cover_prediction(tmp_path: Path) -> None:
    ledger = Ledger.load(tmp_path / "ledger.jsonl")
    entry = _entry(k=_spec(pred=(-5.0, 8.0), bands=((-5.0, 0.0, "a"), (1.0, 8.0, "b"))))
    with pytest.raises(LedgerError):
        ledger.register(entry)


def test_append_only_revisions(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger.load(path)
    ledger.register(_entry())
    first = path.read_bytes()
    ledger.attach_zip("e1", "ab" * 32)
    second = path.read_bytes()
    assert second.startswith(first)
    ledger.record_score("e1", 0.9525342223)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["revision"] for line in lines] == [1, 2, 3]
    assert json.loads(lines[0])["status"] == "registered"
    assert json.loads(lines[1])["status"] == "packaged"
    reloaded = Ledger.load(path)
    assert reloaded.entries()[0].status == "scored"
    assert path.read_bytes().startswith(second)


def test_net_fact_uses_base_abs_count(tmp_path: Path) -> None:
    ledger = Ledger.load(tmp_path / "ledger.jsonl")
    _seed_base(ledger)
    ledger.register(
        _entry(
            entry_id="net1",
            k=_spec(key="m", kind="net", base="rev2syn", pred=(-10.0, 0.0), bands=((-10.0, 0.0, "ship"),)),
            l=_spec(key="n", window=FULL_L, pred=(1730.0, 1740.0), bands=((1730.0, 1740.0, "ship"),)),
        )
    )
    ledger.attach_zip("net1", "cd" * 32)
    ledger.record_score("net1", 0.9497389072)
    facts = {f.channel: f for f in ledger.facts() if f.entry_id == "net1"}
    assert facts["K"].value == pytest.approx(-5.0)
    assert facts["K"].prediction_hit is True


def test_missing_base_fails_closed(tmp_path: Path) -> None:
    ledger = Ledger.load(tmp_path / "ledger.jsonl")
    ledger.register(
        _entry(
            entry_id="netx",
            k=_spec(key="m", kind="net", base="ghost", pred=(-10.0, 0.0), bands=((-10.0, 0.0, "ship"),)),
        )
    )
    ledger.attach_zip("netx", "ef" * 32)
    with pytest.raises(LedgerError, match="ghost"):
        ledger.record_score("netx", 0.9497389072)
    assert ledger.entries()[0].status == "packaged"


def test_unplanned_outcome_is_surfaced(tmp_path: Path) -> None:
    ledger = Ledger.load(tmp_path / "ledger.jsonl")
    _seed_base(ledger)
    ledger.register(
        _entry(
            entry_id="u1",
            k=_spec(key="m", kind="net", base="rev2syn", pred=(-5.0, 8.0), bands=((-5.0, 8.0, "ship"),)),
            l=_spec(key="n", window=FULL_L, pred=(1680.0, 1784.0), bands=((1680.0, 1784.0, "ship"),)),
        )
    )
    ledger.attach_zip("u1", "12" * 32)
    score = round(_reconstructed(496, 1700.0), 10)
    decoded, k_action, _ = ledger.record_score("u1", score)
    assert decoded.k == 496
    assert k_action == "UNPLANNED"
    assert ledger.entries()[-1].status == "scored"


def test_double_scoring_rejected(tmp_path: Path) -> None:
    ledger = Ledger.load(tmp_path / "ledger.jsonl")
    ledger.register(_entry())
    ledger.attach_zip("e1", "ab" * 32)
    ledger.record_score("e1", 0.9525342223)
    with pytest.raises(LedgerError):
        ledger.record_score("e1", 0.9525342223)


def test_corrupt_line_is_not_skipped(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger.load(path)
    ledger.register(_entry())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("not json\n")
    with pytest.raises(LedgerError, match="line 2"):
        Ledger.load(path)


def test_seed_history_round_trips(tmp_path: Path) -> None:
    from port_entity_matching.evaluation.lb.cli import main as lb_main

    csv_path = tmp_path / "seed.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entry_id", "created", "channel", "policy_key", "kind", "base", "value"])
        for entry_id, channel, key, kind, value in SEED_ROWS:
            writer.writerow([entry_id, "2026-10-29", channel, key, kind, "rev2syn" if kind != "abs" else "", value])
    ledger_path = tmp_path / "ledger.jsonl"
    assert lb_main(["--ledger", str(ledger_path), "seed", "--csv", str(csv_path)]) == 0
    facts = Ledger.load(ledger_path).facts()
    assert len(facts) == 41
    assert sum(1 for f in facts if f.channel == "K") == 25
    assert sum(1 for f in facts if f.channel == "L") == 16
    knn = next(f for f in facts if f.policy_key == "knn")
    assert knn.value == pytest.approx(-14.0)
    lfz = next(f for f in facts if f.policy_key == "l_fz")
    assert lfz.value == pytest.approx(28.5)
