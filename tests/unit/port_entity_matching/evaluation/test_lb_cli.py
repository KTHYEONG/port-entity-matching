"""Invariant tests for the LB CLI and the packaging gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from port_entity_matching.evaluation.lb.cli import main as lb_main
from port_entity_matching.evaluation.lb.ledger import (
    ChannelSpec,
    DecisionBand,
    Ledger,
    LedgerEntry,
    Prediction,
)


def _registered(ledger: Ledger, entry_id: str = "e1") -> None:
    ledger.register(
        LedgerEntry(
            entry_id=entry_id,
            created="2026-10-29",
            k=ChannelSpec(
                policy_key="p", policy={}, kind="abs", base=None, window=(440.0, 532.0),
                prediction=Prediction(lo=440.0, hi=532.0, source="t"),
                decision=(DecisionBand(lo=440.0, hi=532.0, action="ship"),),
            ),
            l=ChannelSpec(
                policy_key="q", policy={}, kind="abs", base=None, window=(1680.0, 1784.0),
                prediction=Prediction(lo=1680.0, hi=1784.0, source="t"),
                decision=(DecisionBand(lo=1680.0, hi=1784.0, action="ship"),),
            ),
            status="registered",
        )
    )


def _model_dir(path: Path) -> Path:
    from tools.build_submit import REQUIRED_ARTIFACTS

    path.mkdir(parents=True, exist_ok=True)
    for artifact in REQUIRED_ARTIFACTS:
        (path / artifact).write_text("{}", encoding="utf-8")
    return path


def test_packaging_refuses_without_registration(tmp_path: Path, capsys) -> None:
    from tools.build_submit import main

    out = tmp_path / "submit.zip"
    code = main(["--model-dir", str(_model_dir(tmp_path / "model")), "--out", str(out)])
    assert code == 2
    assert not out.exists()


def test_packaging_attaches_digest(tmp_path: Path) -> None:
    from tools.build_submit import main

    ledger_path = tmp_path / "ledger.jsonl"
    _registered(Ledger.load(ledger_path))
    out = tmp_path / "submit.zip"
    code = main([
        "--model-dir", str(_model_dir(tmp_path / "model")), "--out", str(out),
        "--ledger", str(ledger_path), "--ledger-id", "e1",
    ])
    assert code == 0
    assert out.is_file()
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    entry = Ledger.load(ledger_path).entries()[0]
    assert entry.status == "packaged"
    assert entry.zip_sha256 == digest


def test_unknown_id_refuses_before_building(tmp_path: Path) -> None:
    from tools.build_submit import main

    ledger_path = tmp_path / "ledger.jsonl"
    Ledger.load(ledger_path)
    out = tmp_path / "submit.zip"
    code = main([
        "--model-dir", str(_model_dir(tmp_path / "model")), "--out", str(out),
        "--ledger", str(ledger_path), "--ledger-id", "missing",
    ])
    assert code == 2
    assert not out.exists()


def test_record_prints_decoded_counts(tmp_path: Path, capsys) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger.load(ledger_path)
    _registered(ledger)
    ledger.attach_zip("e1", "ab" * 32)
    code = lb_main(["--ledger", str(ledger_path), "record", "--id", "e1", "--score", "0.9550020651"])
    assert code == 0
    assert "K=504 L=1735.5" in capsys.readouterr().out


def test_domain_errors_exit_2_without_traceback(tmp_path: Path, capsys) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger.load(ledger_path)
    _registered(ledger)
    ledger.attach_zip("e1", "ab" * 32)
    code = lb_main(["--ledger", str(ledger_path), "record", "--id", "e1", "--score", "0.5"])
    assert code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_attach_cli_round_trip(tmp_path: Path, capsys) -> None:
    import hashlib as _hashlib

    ledger_path = tmp_path / "ledger.jsonl"
    _registered(Ledger.load(ledger_path))
    blob = tmp_path / "submit.zip"
    blob.write_bytes(b"fake-zip-bytes")
    code = lb_main(["--ledger", str(ledger_path), "attach", "--id", "e1", "--zip", str(blob)])
    assert code == 0
    entry = Ledger.load(ledger_path).entries()[0]
    assert entry.status == "packaged"
    assert entry.zip_sha256 == _hashlib.sha256(b"fake-zip-bytes").hexdigest()
    assert "attached id=e1" in capsys.readouterr().out


def test_facts_cli_writes_typed_rows(tmp_path: Path, capsys) -> None:
    import csv as _csv

    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger.load(ledger_path)
    _registered(ledger)
    ledger.attach_zip("e1", "ab" * 32)
    ledger.record_score("e1", 0.9525342223)
    out = tmp_path / "facts.csv"
    assert lb_main(["--ledger", str(ledger_path), "facts", "--out", str(out)]) == 0
    with open(out, encoding="utf-8", newline="") as handle:
        rows = list(_csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["channel"] for row in rows} == {"K", "L"}
    assert {row["entry_id"] for row in rows} == {"e1"}
    assert float(next(row for row in rows if row["channel"] == "K")["value"]) == 505.0


def test_show_cli_lists_entries(tmp_path: Path, capsys) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = Ledger.load(ledger_path)
    _registered(ledger)
    ledger.attach_zip("e1", "ab" * 32)
    ledger.record_score("e1", 0.9525342223)
    assert lb_main(["--ledger", str(ledger_path), "show"]) == 0
    out = capsys.readouterr().out
    assert "e1 scored" in out
    assert "K=505" in out


def test_show_cli_handles_unscored_entries(tmp_path: Path, capsys) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    _registered(Ledger.load(ledger_path))
    assert lb_main(["show", "--ledger", str(ledger_path)]) == 0
    out = capsys.readouterr().out
    assert "e1 registered" in out
    assert "K=-" in out


def test_register_cli_rejects_non_object_entry(tmp_path: Path, capsys) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    entry_file = tmp_path / "entry.json"
    entry_file.write_text("[1, 2]", encoding="utf-8")
    assert lb_main(["--ledger", str(ledger_path), "register", "--entry", str(entry_file)]) == 2
    assert "error:" in capsys.readouterr().err


def test_seed_cli_fails_closed_on_bad_base(tmp_path: Path, capsys) -> None:
    import csv as _csv

    ledger_path = tmp_path / "ledger.jsonl"
    for base, name in (("", "empty"), ("ghost", "unknown")):
        csv_path = tmp_path / f"bad_{name}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = _csv.writer(handle)
            writer.writerow(["entry_id", "created", "channel", "policy_key", "kind", "base", "value"])
            writer.writerow(["bad", "2026-10-29", "K", "bad", "net", base, "-3"])
        assert lb_main(["--ledger", str(ledger_path), "seed", "--csv", str(csv_path)]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_register_cli_round_trip(tmp_path: Path, capsys) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    entry_file = tmp_path / "entry.json"
    entry_file.write_text(
        json.dumps({
            "entry_id": "cli1",
            "created": "2026-10-29",
            "k": {
                "policy_key": "p", "policy": {}, "kind": "abs", "base": None,
                "window": [440.0, 532.0],
                "prediction": {"lo": 440.0, "hi": 532.0, "source": "t"},
                "decision": [{"lo": 440.0, "hi": 532.0, "action": "ship"}],
            },
            "l": {
                "policy_key": "q", "policy": {}, "kind": "abs", "base": None,
                "window": [1680.0, 1784.0],
                "prediction": {"lo": 1680.0, "hi": 1784.0, "source": "t"},
                "decision": [{"lo": 1680.0, "hi": 1784.0, "action": "ship"}],
            },
            "status": "registered",
        }),
        encoding="utf-8",
    )
    assert lb_main(["--ledger", str(ledger_path), "register", "--entry", str(entry_file)]) == 0
    assert "registered id=cli1" in capsys.readouterr().out
