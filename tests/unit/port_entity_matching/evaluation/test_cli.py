"""Contract tests for the evaluation CLI: overrides, artifacts, stress table and comparisons."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from port_entity_matching.config import MatchConfig
from port_entity_matching.evaluation import cli as cli_module
from port_entity_matching.evaluation.cli import (
    _bucket_entry,
    _cast_value,
    _parse_scenarios,
    _stress_entry,
    _write_json,
    _write_oof,
    main,
    parse_overrides,
)
from port_entity_matching.evaluation.perturb import PERTURBATIONS
from port_entity_matching.evaluation.stats import OOF_COLUMNS


@pytest.fixture(scope="module", autouse=True)
def _isolated_cwd(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    # The default --cache-dir is relative; keep it out of the repository's artifacts/.
    # Module scope so that the module-scoped CLI fixtures also run inside the temp directory.
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("cwd"))
        yield

_VESSELS: list[tuple[str, str, str, int, str, str]] = [
    ("V00", "KAROS", "카로스", 1000, "Tanker", "Panama"),
    ("V01", "TOMOS", "토모스", 2000, "Container", "Liberia"),
    ("V02", "SAMOS", "사모스", 3000, "Tanker", "Malta"),
    ("V03", "PAKOS", "파코스", 4000, "Bulk", "Panama"),
    ("V04", "NOMOS", "노모스", 5000, "Tanker", "Greece"),
    ("V05", "KOMOS", "코모스", 6000, "Container", "Panama"),
    ("V06", "ROMOS", "로모스", 7000, "Tanker", "Cyprus"),
    ("V07", "SONIS", "소니스", 8000, "Container", "Panama"),
    ("V08", "PONOS", "포노스", 9000, "Tanker", "Marshall Islands"),
    ("V09", "TONIS", "토니스", 10000, "Bulk", "Panama"),
]


def _write_fixture(root: Path) -> tuple[Path, Path]:
    """Write a tiny catalog and training CSV with Korean and Latin rows."""
    catalog = pd.DataFrame(
        [
            {
                "vessel_id": vessel_id,
                "vessel_type": vessel_type,
                "vessel_name": name,
                "size": tonnage,
                "unit": "DWT",
                "deadweight_tonnage": tonnage,
                "gross_tonnage": tonnage,
                "flag_name": flag,
            }
            for vessel_id, name, _korean, tonnage, vessel_type, flag in _VESSELS
        ]
    )
    catalog_path = root / "vessel_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    train = pd.DataFrame(
        [
            {
                "sample_id": f"T{index:05d}",
                "vessel_name": korean if index % 2 == 0 else name,
                "call_sign": f"CS_{vessel_id}",
                "gross_tonnage": tonnage,
                "international_gross_tonnage": tonnage,
                "flag_name": "파나마" if flag == "Panama" else "라이베리아",
                "vessel_type": "유조선" if vessel_type == "Tanker" else "컨테이너선",
                "arrival_year": 2020,
                "target_vessel_id": vessel_id,
            }
            for index, (vessel_id, name, korean, tonnage, vessel_type, flag) in enumerate(_VESSELS)
        ]
    )
    train_path = root / "train.csv"
    train.to_csv(train_path, index=False)
    return train_path, catalog_path


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A tiny catalog/training pair with ten vessels, five Korean and five Latin rows."""
    root = tmp_path_factory.mktemp("tiny")
    return _write_fixture(root)


@pytest.fixture(scope="module")
def cv_run(tiny_dataset: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the cv subcommand once and return its output directory."""
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path_factory.mktemp("cv")
    assert (
        main(
            [
                "cv",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(out_dir),
                "--set",
                "ensemble_size=1",
            ]
        )
        == 0
    )
    return out_dir


def test_typed_overrides_are_cast_by_field_type() -> None:
    config = parse_overrides(["hangul_tonnage_window=0.08", "augment=false", "ensemble_size=2"])
    assert config.hangul_tonnage_window == pytest.approx(0.08)
    assert isinstance(config.hangul_tonnage_window, float)
    assert config.augment is False
    assert config.ensemble_size == 2
    assert isinstance(config.ensemble_size, int)
    defaults = MatchConfig()
    assert config.cv_splits == defaults.cv_splits
    assert config.seed == defaults.seed


def test_bool_overrides_accept_only_documented_literals() -> None:
    assert parse_overrides(["augment=TRUE"]).augment is True
    assert parse_overrides(["augment=0"]).augment is False
    with pytest.raises(ValueError, match="augment"):
        parse_overrides(["augment=yes"])


@pytest.mark.parametrize(
    ("pair", "key"),
    [("nope=1", "nope"), ("ensemble_size=two", "ensemble_size"), ("augment", "augment")],
)
def test_bad_overrides_are_rejected(pair: str, key: str) -> None:
    with pytest.raises(ValueError, match=key):
        parse_overrides([pair])


def test_float_override_with_a_non_numeric_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="tonnage_window"):
        parse_overrides(["tonnage_window=wide"])


def test_unsupported_field_types_are_rejected() -> None:
    with pytest.raises(ValueError, match="vessel_name"):
        _cast_value("vessel_name", "value", "text")


def test_cv_writes_artifacts(cv_run: Path, tiny_dataset: tuple[Path, Path]) -> None:
    oof_path = cv_run / "oof.csv"
    summary_path = cv_run / "summary.json"
    assert oof_path.is_file()
    assert summary_path.is_file()

    oof = pd.read_csv(oof_path)
    train_rows = len(pd.read_csv(tiny_dataset[0]))
    assert list(oof.columns) == list(OOF_COLUMNS)
    assert len(oof) == train_rows
    assert "scenario" not in oof.columns

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("config", "kor", "lat", "score", "ci", "elapsed_seconds"):
        assert key in summary
    assert summary["config"]["ensemble_size"] == 1
    assert summary["elapsed_seconds"] >= 0.0


def test_stress_writes_per_scenario_table(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path / "stress"
    assert (
        main(
            [
                "stress",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(out_dir),
                "--scenarios",
                "kor_tonnage_missing,lat_tonnage_missing",
                "--set",
                "ensemble_size=1",
            ]
        )
        == 0
    )
    assert (out_dir / "oof_long.csv").is_file()

    report = json.loads((out_dir / "stress.json").read_text(encoding="utf-8"))
    assert set(report) == {"kor_tonnage_missing", "lat_tonnage_missing"}
    for entry in report.values():
        assert entry["affected_rows"] > 0
        for field in ("acc_clean", "acc_perturbed", "delta"):
            assert field in entry


def test_compare_is_order_insensitive(cv_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    oof = pd.read_csv(cv_run / "oof.csv")
    shuffled_path = tmp_path / "oof_shuffled.csv"
    oof.sample(frac=1.0, random_state=11).to_csv(shuffled_path, index=False)

    report_path = tmp_path / "compare.json"
    exit_code = main(["compare", str(cv_run / "oof.csv"), str(shuffled_path), "--json", str(report_path)])
    assert exit_code == 0

    printed = capsys.readouterr().out
    assert "+0.0000" in printed
    assert "vessels_fixed=0" in printed
    assert "vessels_broken=0" in printed
    assert "rows_fixed=0" in printed
    assert "rows_broken=0" in printed

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["delta"] == pytest.approx(0.0, abs=1e-12)
    assert report["rows_fixed"] == 0
    assert report["rows_broken"] == 0


def test_unknown_scenario_is_reported(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train_path, catalog_path = tiny_dataset
    exit_code = main(
        [
            "stress",
            "--train",
            str(train_path),
            "--catalog",
            str(catalog_path),
            "--out-dir",
            str(tmp_path / "stress"),
            "--scenarios",
            "bogus",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "bogus" in captured.err
    assert "kor_tonnage_missing" in captured.err


def test_all_scenarios_are_run_by_default(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path / "stress_all"
    assert (
        main(
            [
                "stress",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(out_dir),
                "--set",
                "ensemble_size=1",
            ]
        )
        == 0
    )
    report = json.loads((out_dir / "stress.json").read_text(encoding="utf-8"))
    assert set(report) == set(PERTURBATIONS)


def test_a_scenario_without_affected_rows_is_reported(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path / "stress_empty"
    assert (
        main(
            [
                "stress",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(out_dir),
                "--scenarios",
                "kor_spaces_removed",
                "--set",
                "ensemble_size=1",
            ]
        )
        == 0
    )
    report = json.loads((out_dir / "stress.json").read_text(encoding="utf-8"))
    entry = report["kor_spaces_removed"]
    assert entry["affected_rows"] == 0
    assert entry["acc_clean"] is None
    assert entry["delta"] is None


def test_output_directory_is_created(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path / "nested" / "run"
    assert (
        main(
            [
                "cv",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(out_dir),
                "--set",
                "ensemble_size=1",
            ]
        )
        == 0
    )
    assert (out_dir / "oof.csv").is_file()


def test_writing_inside_a_data_directory_is_refused(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    exit_code = main(
        [
            "cv",
            "--train",
            str(train_path),
            "--catalog",
            str(catalog_path),
            "--out-dir",
            str(tmp_path / "data" / "eval"),
        ]
    )
    assert exit_code == 1


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "1e999"])
def test_nonfinite_overrides_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="tonnage_window"):
        parse_overrides([f"tonnage_window={raw}"])


@pytest.mark.parametrize("text", ["", ",", "all,bogus", "all,kor_tonnage_missing"])
def test_invalid_scenario_lists_are_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="Scenarios"):
        _parse_scenarios(text)


def test_duplicate_scenarios_run_only_once() -> None:
    assert _parse_scenarios("kor_tonnage_missing,kor_tonnage_missing") == ["kor_tonnage_missing"]


@pytest.mark.parametrize("bad_id", [None, "T00001"])
def test_invalid_training_ids_fail_before_cv(
    tmp_path: Path,
    bad_id: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_path, catalog_path = _write_fixture(tmp_path)
    train = pd.read_csv(train_path)
    train.loc[0, "sample_id"] = bad_id
    train.to_csv(train_path, index=False)
    assert (
        main(
            [
                "cv",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        == 1
    )
    assert "sample_id" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def _stress_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scenario": "clean", "sample_id": "A", "correct": 0, "target_retrieved": 1},
            {"scenario": "clean", "sample_id": "B", "correct": 1, "target_retrieved": 1},
            {"scenario": "clean", "sample_id": "C", "correct": 1, "target_retrieved": 1},
            {"scenario": "test", "sample_id": "C", "correct": 0, "target_retrieved": 0},
            {"scenario": "test", "sample_id": "B", "correct": 1, "target_retrieved": 1},
        ]
    )


def test_stress_accuracy_joins_only_affected_ids() -> None:
    report = _stress_entry(_stress_frame(), "test")
    assert report == {
        "affected_rows": 2,
        "acc_clean": 1.0,
        "acc_perturbed": 0.5,
        "delta": -0.5,
        "recall_perturbed": 0.5,
    }


def test_stress_rejects_unmatched_and_duplicate_ids() -> None:
    frame = _stress_frame()
    frame.loc[3, "sample_id"] = "missing"
    with pytest.raises(ValueError, match="matching clean"):
        _stress_entry(frame, "test")
    duplicated = pd.concat([_stress_frame(), _stress_frame().iloc[[0]]])
    with pytest.raises(ValueError, match="one-to-one"):
        _stress_entry(duplicated, "test")


@pytest.mark.parametrize("kind", ["csv", "json"])
def test_output_file_symlinks_cannot_bypass_data_guard(tmp_path: Path, kind: str) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    protected = data_dir / f"protected.{kind}"
    protected.write_text("unchanged", encoding="utf-8")
    output = tmp_path / f"output.{kind}"
    output.symlink_to(protected)
    if kind == "csv":
        with pytest.raises(ValueError, match="data directory"):
            _write_oof(output, pd.DataFrame({"value": [1]}))
    else:
        with pytest.raises(ValueError, match="data directory"):
            _write_json(output, {"value": 1})
    assert protected.read_text(encoding="utf-8") == "unchanged"


def test_argument_errors_exit_two() -> None:
    with pytest.raises(SystemExit) as error:
        main(["cv"])
    assert error.value.code == 2


def test_stress_writes_the_suite_summary(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path / "stress_suite"
    assert (
        main(
            [
                "stress",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(out_dir),
                "--scenarios",
                "kor_tonnage_missing,lat_tonnage_missing",
                "--set",
                "ensemble_size=1",
            ]
        )
        == 0
    )
    suite = json.loads((out_dir / "suite.json").read_text(encoding="utf-8"))
    assert set(suite) == {"all", "seen", "unseen"}
    for entry in suite.values():
        assert set(entry) == {"kor", "lat", "score", "n_kor", "n_lat"}
        assert 0.0 <= entry["score"] <= 1.0
        assert entry["n_kor"] >= 1
        assert entry["n_lat"] >= 1


def test_stress_serializes_a_missing_bucket_as_null(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    train = pd.read_csv(train_path)
    latin_path = tmp_path / "latin.csv"
    train.iloc[1::2].to_csv(latin_path, index=False)
    out_dir = tmp_path / "latin_stress"
    assert (
        main(
            [
                "stress",
                "--train",
                str(latin_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(out_dir),
                "--scenarios",
                "lat_tonnage_missing",
                "--set",
                "ensemble_size=1",
                "--set",
                "cv_splits=2",
            ]
        )
        == 0
    )
    suite = json.loads((out_dir / "suite.json").read_text(encoding="utf-8"))
    for entry in suite.values():
        assert entry["kor"] is None
        assert entry["score"] is None
        assert entry["n_kor"] == 0
        assert entry["lat"] is not None


def test_compare_accepts_a_scenario(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train_path, catalog_path = tiny_dataset
    out_dir = tmp_path / "stress_compare"
    assert (
        main(
            [
                "stress",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(out_dir),
                "--scenarios",
                "kor_tonnage_missing,lat_tonnage_missing",
                "--set",
                "ensemble_size=1",
            ]
        )
        == 0
    )
    long_path = out_dir / "oof_long.csv"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    long_frame = pd.read_csv(long_path)
    long_frame.sample(frac=1.0, random_state=5).to_csv(first, index=False)
    long_frame.to_csv(second, index=False)

    exit_code = main(
        [
            "compare",
            str(first),
            str(second),
            "--scenario",
            "lat_tonnage_missing",
        ]
    )
    assert exit_code == 0
    assert "+0.0000" in capsys.readouterr().out

    exit_code = main(
        [
            "compare",
            str(first),
            str(second),
            "--scenario",
            "bogus",
        ]
    )
    assert exit_code == 1


def test_cache_flags_keep_stress_identical(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.INFO)
    train_path, catalog_path = tiny_dataset
    cache_dir = tmp_path / "candidate_cache"
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    args = ["--set", "ensemble_size=1", "--cache-dir", str(cache_dir)]
    assert (
        main(["stress", "--train", str(train_path), "--catalog", str(catalog_path), "--out-dir", str(first_out), *args])
        == 0
    )
    assert (
        main(
            ["stress", "--train", str(train_path), "--catalog", str(catalog_path), "--out-dir", str(second_out), *args]
        )
        == 0
    )
    assert (first_out / "oof_long.csv").read_bytes() == (second_out / "oof_long.csv").read_bytes()
    assert "misses=0" in caplog.text
    third_out = tmp_path / "third"
    assert (
        main(
            [
                "stress",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(third_out),
                "--no-cache",
                "--set",
                "ensemble_size=1",
            ]
        )
        == 0
    )
    assert (first_out / "oof_long.csv").read_bytes() == (third_out / "oof_long.csv").read_bytes()


def test_cv_no_cache_matches_cached(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    train_path, catalog_path = tiny_dataset
    cached_out = tmp_path / "cached"
    assert (
        main(
            [
                "cv",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(cached_out),
                "--cache-dir",
                str(tmp_path / "c"),
                "--set",
                "ensemble_size=1",
            ]
        )
        == 0
    )
    plain_out = tmp_path / "plain"
    assert (
        main(
            [
                "cv",
                "--train",
                str(train_path),
                "--catalog",
                str(catalog_path),
                "--out-dir",
                str(plain_out),
                "--no-cache",
                "--set",
                "ensemble_size=1",
            ]
        )
        == 0
    )
    assert (cached_out / "oof.csv").read_bytes() == (plain_out / "oof.csv").read_bytes()


def _twin_args(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, *extra: str, rows: str = "2", out: str = "twin"
) -> list[str]:
    train_path, catalog_path = tiny_dataset
    return [
        "twin",
        "--train",
        str(train_path),
        "--catalog",
        str(catalog_path),
        "--out-dir",
        str(tmp_path / out),
        "--rows",
        rows,
        "--seed",
        "0",
        *extra,
    ]


def test_twin_writes_buckets_with_llm_disabled(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    out_dir = tmp_path / "twin"
    assert (
        main(
            _twin_args(
                tiny_dataset,
                tmp_path,
                "--set",
                "llm_enabled=false",
                "--set",
                "ensemble_size=1",
                "--no-cache",
            )
        )
        == 0
    )
    twin = json.loads((out_dir / "twin.json").read_text(encoding="utf-8"))
    assert len(twin) == 2
    assert all(set(entry) == {"src", "r"} for entry in twin.values())

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    for bucket in ("clean_K", "clean_L", "kor_nt", "r2k_wt", "r2k_nt"):
        entry = summary["buckets"][bucket]
        assert entry["size"] > 0
        assert 0 <= entry["correct"] <= entry["size"]
        assert entry["accuracy"] == entry["correct"] / entry["size"]

    oof = pd.read_csv(out_dir / "oof.csv")
    assert {"clean", "kor_nt", "r2k_wt", "r2k_nt"} <= set(oof["scenario"].unique())


def test_twin_reuses_existing_twin_file(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    first_args = _twin_args(tiny_dataset, tmp_path, "--set", "llm_enabled=false", "--set", "ensemble_size=1")
    assert main(first_args) == 0
    twin_path = tmp_path / "twin" / "twin.json"
    before = twin_path.read_bytes()
    assert main(first_args) == 0
    assert twin_path.read_bytes() == before


def test_twin_requires_llm_model_when_enabled(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_twin_args(tiny_dataset, tmp_path)) == 1
    assert "llm-model" in capsys.readouterr().err
    assert not (tmp_path / "twin").exists()


def test_twin_refuses_transliteration_cache_in_data(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(
        _twin_args(tiny_dataset, tmp_path, "--llm-model", "weights", "--llm-cache", str(tmp_path / "data" / "cache"))
    ) == 1
    assert "data directory" in capsys.readouterr().err
    assert not (tmp_path / "twin").exists()


def test_twin_summary_includes_scenarios_with_no_valid_renderings(
    tiny_dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    out_dir = tmp_path / "twin"
    out_dir.mkdir()
    (out_dir / "twin.json").write_text(
        json.dumps({"T00001": {"src": "TOMOS", "r": "LATIN", "q": ""}}), encoding="utf-8"
    )
    assert main(_twin_args(tiny_dataset, tmp_path, "--set", "llm_enabled=false", "--set", "ensemble_size=1")) == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    for name in ("r2k_wt", "r2k_nt", "q2k_wt", "q2k_nt"):
        assert summary["buckets"][name] == {"size": 0, "correct": 0, "accuracy": None}


def test_twin_with_llm_and_reverse_models(
    tiny_dataset: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeTransliterator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate(self, names: object) -> dict[str, tuple[str, ...]]:
            return {}

        def score_candidates(self, name: str, candidates: object) -> tuple[float, ...]:
            return tuple(0.0 for _ in list(candidates))  # type: ignore[arg-type]

    monkeypatch.setattr(cli_module, "CachedTransliterator", _FakeTransliterator)
    monkeypatch.setattr(
        cli_module, "reverse_llm", lambda weights_dir, shots: (lambda names: {str(n): "가나다" for n in names})
    )
    out_dir = tmp_path / "twin_llm"
    assert (
        main(
            _twin_args(
                tiny_dataset,
                tmp_path,
                "--llm-model",
                str(tmp_path / "llm"),
                "--reverse-model",
                str(tmp_path / "rev"),
                "--set",
                "ensemble_size=1",
                out="twin_llm",
            )
        )
        == 0
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert {"q2k_wt", "q2k_nt"} <= set(summary["buckets"])


def test_bucket_entry_handles_an_empty_frame() -> None:
    entry = _bucket_entry(pd.DataFrame({"correct": pd.Series([], dtype=bool)}))
    assert entry == {"size": 0, "accuracy": None, "correct": 0}
