"""Contract tests for training, vessel-disjoint cross-validation and the CLI."""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from port_entity_matching.augment import degraded_copies, notation_copies
from port_entity_matching.candidates import generate_candidates
from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.features import FEATURE_NAMES
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.ranker import score_ensemble
from port_entity_matching.records import records_from_frame
from port_entity_matching.train import (
    DIAGNOSTIC_COLUMNS,
    build_training_data,
    cross_validate,
    cross_validate_run,
    fit_bundle,
    fold_indices,
    main,
)

_CATALOG: list[tuple[str, str, float, str, str]] = [
    ("V00", "Glovis Spirit", 5000, "Tanker", "Panama"),
    ("V01", "Grand Ace 11", 1000, "Container", "Liberia"),
    ("V02", "Grand Ace 12", 1000, "Container", "Liberia"),
    ("V03", "Dilam", 168208, "Tanker", "Liberia"),
    ("V04", "Calida", 61332, "Tanker", "Malta"),
    ("V05", "Donoussa", 162354, "Tanker", "Marshall Islands"),
    ("V06", "Sapsan", 28068, "Tanker", "Singapore"),
    ("V07", "Shining Road", 2000, "Bulk", "Panama"),
    ("V08", "Heung-A", 5000, "Tanker", "Korea"),
    ("V09", "Marsgracht", 3000, "Bulk", "Netherlands"),
    ("V10", "Maasgracht", 3000, "Bulk", "Netherlands"),
    ("V11", "Kashima Maru", 8000, "Tanker", "Japan"),
]

_ROWS: list[tuple[str, str, str, float | None]] = [
    ("글로비스 스피릿", "V00", "Tanker", 5000),
    ("GRAND ACE 11", "V01", "Container", 1000),
    ("GRAND ACE 11", "V01", "Container", 1000),
    ("GRAND ACE 12", "V02", "Container", 1000),
    ("DILAN", "V03", "Tanker", 168208),
    ("CALIDA", "V04", "Tanker", 61332),
    ("DONOUSSA", "V05", "Tanker", 162354),
    ("SAPSAN", "V06", "Tanker", 28068),
    ("SHINING ROAD", "V07", "Bulk", 2000),
    ("흥아", "V08", "Tanker", 5000),
    ("MARSGRACHT", "V09", "Bulk", 3000),
    ("MAASGRACHT", "V10", "Bulk", 3000),
    ("카시마 마루", "V11", "Tanker", 8000),
    ("CALIDA", "V04", "Tanker", 61332),
    ("글로비스 스피릿", "V00", "Tanker", 5000),
]


def _config() -> MatchConfig:
    return MatchConfig(lgbm_num_rounds=20, lgbm_min_data_in_leaf=1, cv_splits=2, num_threads=1)


def _catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "vessel_id": vessel_id,
                "vessel_type": vessel_type,
                "vessel_name": vessel_name,
                "size": tonnage,
                "unit": "DWT",
                "deadweight_tonnage": tonnage,
                "gross_tonnage": tonnage,
                "flag_name": flag,
            }
            for vessel_id, vessel_name, tonnage, vessel_type, flag in _CATALOG
        ]
    )


def _train_frame() -> pd.DataFrame:
    rows = []
    for number, (name, target, vessel_type, tonnage) in enumerate(_ROWS):
        rows.append(
            {
                "sample_id": f"T{number:05d}",
                "vessel_name": name,
                "call_sign": f"CS_{target}_{number}",
                "gross_tonnage": tonnage,
                "international_gross_tonnage": tonnage,
                "flag_name": "파나마",
                "vessel_type": vessel_type,
                "arrival_year": 2020,
                "target_vessel_id": target,
            }
        )
    return pd.DataFrame(rows)


def test_training_data_well_formed() -> None:
    index = CatalogIndex.build(_catalog_frame())
    train = _train_frame()
    maps = AttributeMaps.fit(train, index)
    data = build_training_data(train, index, maps, _config())
    assert sum(data.group_sizes) == len(data.labels)
    assert data.features.shape[0] == len(data.labels)
    offset = 0
    for size in data.group_sizes:
        group = data.labels[offset : offset + size]
        assert int(np.count_nonzero(group == 1)) == 1
        offset += size
    assert data.recall_hangul == pytest.approx(1.0)
    assert data.recall_latin == pytest.approx(1.0)


def test_unretrieved_rows_excluded_and_counted() -> None:
    index = CatalogIndex.build(_catalog_frame())
    train = _train_frame()
    unknown = train.iloc[[0]].copy()
    unknown["target_vessel_id"] = "V_UNKNOWN"
    train = pd.concat([train, unknown], ignore_index=True)
    maps = AttributeMaps.fit(train, index)
    data = build_training_data(train, index, maps, MatchConfig(augment=False, num_threads=1))
    assert len(data.group_sizes) == len(_ROWS)
    assert data.recall_hangul < 1.0


def test_cv_is_vessel_disjoint() -> None:
    train = _train_frame()
    folds = fold_indices(train, 2)
    assert len(folds) == 2
    seen: list[pd.Index] = []
    for train_index, valid_index in folds:
        train_ids = set(train.iloc[train_index]["target_vessel_id"])
        valid_ids = set(train.iloc[valid_index]["target_vessel_id"])
        assert train_ids.isdisjoint(valid_ids)
        seen.append(pd.Index(valid_index))
    assert sorted(np.concatenate([index.to_numpy() for index in seen])) == list(range(len(train)))


def test_report_arithmetic() -> None:
    report = cross_validate(_train_frame(), _catalog_frame(), _config())
    assert report.weighted_score == pytest.approx(
        0.7 * report.hangul_accuracy + 0.3 * report.non_hangul_accuracy, rel=1e-12
    )
    assert report.n_hangul + report.n_non_hangul == len(_ROWS)
    assert 0.0 <= report.hangul_accuracy <= 1.0
    assert 0.0 <= report.non_hangul_accuracy <= 1.0


def test_diagnostics_written(tmp_path) -> None:
    path = tmp_path / "cv_oof.csv"
    cross_validate(_train_frame(), _catalog_frame(), _config(), path)
    frame = pd.read_csv(path)
    assert list(frame.columns) == list(DIAGNOSTIC_COLUMNS)
    assert len(frame) == len(_ROWS)
    assert set(frame["sample_id"]) == set(_train_frame()["sample_id"])


def test_fit_bundle_produces_ranker() -> None:
    bundle = fit_bundle(_train_frame(), _catalog_frame(), _config())
    assert bundle.memory.lookup("CS_V00_0") == 0
    assert bundle.maps.type_logprob("Tanker", "Tanker") > bundle.maps.type_logprob("Tanker", "Bulk")


def test_cli_fit_writes_bundle(tmp_path, monkeypatch) -> None:
    from port_entity_matching.transliterate import CachedTransliterator

    monkeypatch.setattr(CachedTransliterator, "translate", lambda self, names: dict.fromkeys(names, ("GLOVIS SPIRIT",)))
    train_path = tmp_path / "train.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    model_dir = tmp_path / "model"
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    (llm_dir / "config.json").write_text("{}", encoding="utf-8")
    _train_frame().to_csv(train_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    exit_code = main(
        ["--train", str(train_path), "--catalog", str(catalog_path), "--out", str(model_dir),
         "--llm-model", str(llm_dir)]
    )
    assert exit_code == 0
    for name in ("model_0.txt", "maps.json", "memory.json", "manifest.json"):
        assert (model_dir / name).is_file()


def test_cli_requires_work(tmp_path) -> None:
    train_path = tmp_path / "train.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    _train_frame().to_csv(train_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    with pytest.raises(SystemExit):
        main(["--train", str(train_path), "--catalog", str(catalog_path)])


def test_fold_indices_validates_input() -> None:
    train = _train_frame()
    with pytest.raises(ValueError, match="target_vessel_id"):
        fold_indices(train.drop(columns=["target_vessel_id"]), 2)
    with pytest.raises(ValueError, match="n_splits"):
        fold_indices(train, 1)
    with pytest.raises(ValueError, match="vessel-disjoint folds"):
        fold_indices(train, len(train) + 1)


def test_target_outside_candidates_is_excluded() -> None:
    index = CatalogIndex.build(_catalog_frame())
    train = _train_frame().iloc[[0]].copy()
    train["target_vessel_id"] = "V01"
    train["vessel_name"] = "CALIDA"
    maps = AttributeMaps.fit(train, index)
    data = build_training_data(train, index, maps, MatchConfig(name_top_k=1, num_threads=1))
    assert data.group_sizes == []
    assert data.labels.shape[0] == 0
    assert data.recall_latin == pytest.approx(0.0)


def test_module_entry_point_is_callable() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "port_entity_matching.train", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--diagnostics" in result.stdout


def test_cli_cross_validation_writes_diagnostics(tmp_path, monkeypatch) -> None:
    from port_entity_matching.transliterate import CachedTransliterator

    monkeypatch.setattr(CachedTransliterator, "translate", lambda self, names: dict.fromkeys(names, ("GLOVIS SPIRIT",)))
    train_path = tmp_path / "train.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    diagnostics_path = tmp_path / "cv.csv"
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    (llm_dir / "config.json").write_text("{}", encoding="utf-8")
    _train_frame().to_csv(train_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    exit_code = main(
        [
            "--train",
            str(train_path),
            "--catalog",
            str(catalog_path),
            "--cv",
            "--diagnostics",
            str(diagnostics_path),
            "--llm-model",
            str(llm_dir),
        ]
    )
    assert exit_code == 0
    assert diagnostics_path.is_file()


def test_raw_catalog_ids_are_preserved_in_labels() -> None:
    catalog = _catalog_frame()
    train = _train_frame()
    catalog["vessel_id"] = " " + catalog["vessel_id"] + " "
    train["target_vessel_id"] = " " + train["target_vessel_id"] + " "
    index = CatalogIndex.build(catalog)
    data = build_training_data(train, index, AttributeMaps.fit(train, index), MatchConfig(augment=False, num_threads=1))
    assert len(data.group_sizes) == len(train)
    report = cross_validate(train, catalog, _config())
    assert report.recall_hangul == 1.0
    assert report.recall_latin == 1.0


def test_cv_counts_unretrieved_targets_as_errors_and_ignores_call_signs(tmp_path) -> None:
    train = _train_frame()
    unknown = train.iloc[[0]].copy()
    unknown["target_vessel_id"] = "OUTSIDE_CATALOG"
    unknown["sample_id"] = "UNRETRIEVED"
    train = pd.concat([train, unknown], ignore_index=True)
    path = tmp_path / "oof.csv"
    report = cross_validate(train, _catalog_frame(), _config(), path)
    diagnostics = pd.read_csv(path).set_index("sample_id")
    assert diagnostics.loc["UNRETRIEVED", "target_retrieved"] == 0
    assert diagnostics.loc["UNRETRIEVED", "correct"] == 0
    assert diagnostics.loc["UNRETRIEVED", "pred_vessel_id"] in set(_catalog_frame()["vessel_id"])
    assert report.recall_hangul < 1.0
    train["call_sign"] = "CONTRADICTORY"
    assert cross_validate(train, _catalog_frame(), _config()) == report


def _mixed_frame() -> pd.DataFrame:
    rows = [
        ("M0", "글로비스 스피릿", "V00", 5000),
        ("M1", "카시마 마루", "V11", 8000),
        ("M2", "CALIDA", "V04", 61332),
        ("M3", "MARSGRACHT", "V09", 3000),
    ]
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "vessel_name": name,
                "call_sign": f"CS_{sample_id}",
                "gross_tonnage": tonnage,
                "international_gross_tonnage": tonnage,
                "flag_name": "파나마",
                "vessel_type": "Tanker",
                "arrival_year": 2020,
                "target_vessel_id": target,
            }
            for sample_id, name, target, tonnage in rows
        ]
    )


def test_augmentation_adds_copies_only_to_training() -> None:
    index = CatalogIndex.build(_catalog_frame())
    train = _mixed_frame()
    maps = AttributeMaps.fit(train, index)
    plain = build_training_data(train, index, maps, MatchConfig(augment=False, num_threads=1))
    augmented = build_training_data(
        train, index, maps, MatchConfig(augment=True, augment_notation=False, num_threads=1)
    )
    assert len(augmented.group_sizes) == len(plain.group_sizes) + 6
    assert augmented.recall_hangul == plain.recall_hangul
    assert augmented.recall_latin == plain.recall_latin


def test_notation_augmentation_adds_more_groups() -> None:
    index = CatalogIndex.build(_catalog_frame())
    train = _mixed_frame()
    maps = AttributeMaps.fit(train, index)
    without = build_training_data(
        train, index, maps, MatchConfig(augment=True, augment_notation=False, augment_hangul_notation=True, num_threads=1)
    )
    with_notation = build_training_data(
        train, index, maps, MatchConfig(augment=True, augment_notation=True, augment_hangul_notation=True, num_threads=1)
    )
    assert len(with_notation.group_sizes) > len(without.group_sizes)
    assert with_notation.recall_hangul == without.recall_hangul
    assert with_notation.recall_latin == without.recall_latin


def test_notation_ablation_is_exact() -> None:
    index = CatalogIndex.build(_catalog_frame())
    train = _train_frame()
    maps = AttributeMaps.fit(train, index)
    data = build_training_data(train, index, maps, MatchConfig(augment=True, augment_notation=False, num_threads=1))
    assert len(data.group_sizes) == 27
    assert set(data.group_sizes) == {12}
    assert hashlib.sha256(data.labels.tobytes()).hexdigest() == (
        "f07a77955422d4a4c6bd8ef8b58216365c5071e1c44c276e35212039c0208c6b"
    )
    assert hashlib.sha256(data.features[:, :53].copy().tobytes()).hexdigest() == (
        "7e6f3f72b6ee3c5b73f19cf14536c46ae2687d9698e4653464fb9e1ce27e22c7"
    )


def test_notation_copies_keep_their_source_targets() -> None:
    index = CatalogIndex.build(_catalog_frame())
    train = _mixed_frame()
    config = MatchConfig(augment=True, augment_notation=True, augment_latin_rate=1.0, num_threads=1)
    maps = AttributeMaps.fit(train, index)
    data = build_training_data(train, index, maps, config)

    records = records_from_frame(train)
    targets = [str(value) for value in train["target_vessel_id"].tolist()]
    degraded, degraded_sources = degraded_copies(records, config)
    notation, notation_sources = notation_copies(records, config)
    all_records = [*records, *degraded, *notation]
    all_targets = [
        *targets,
        *[targets[index] for index in degraded_sources],
        *[targets[index] for index in notation_sources],
    ]
    candidate_sets = generate_candidates(all_records, index, None, config)

    kept: list[int] = []
    for position, target in enumerate(all_targets):
        catalog_position = index.id_to_pos.get(target)
        if catalog_position is None:
            continue
        if not bool((candidate_sets[position].positions == catalog_position).any()):
            continue
        kept.append(position)
    assert len(kept) == len(data.group_sizes)

    offsets = np.cumsum([0, *data.group_sizes])
    group_of = {position: group for group, position in enumerate(kept)}
    notation_positions = range(len(records) + len(degraded), len(all_records))
    checked = 0
    for position in notation_positions:
        if position not in group_of:
            continue
        group = group_of[position]
        start = int(offsets[group])
        end = int(offsets[group + 1])
        catalog_position = index.id_to_pos[all_targets[position]]
        local = int(np.flatnonzero(candidate_sets[position].positions == catalog_position)[0])
        assert int(data.labels[start + local]) == 1
        assert int(np.count_nonzero(data.labels[start:end] == 1)) == 1
        checked += 1
    assert checked > 0


def test_maps_are_unaffected_by_augmentation(tmp_path) -> None:
    train = _mixed_frame()
    catalog = _catalog_frame()
    with_augmentation = fit_bundle(train, catalog, MatchConfig(augment=True, num_threads=1))
    without_augmentation = fit_bundle(train, catalog, MatchConfig(augment=False, num_threads=1))
    with_augmentation.maps.to_json(tmp_path / "with.json")
    without_augmentation.maps.to_json(tmp_path / "without.json")
    assert (tmp_path / "with.json").read_text(encoding="utf-8") == (tmp_path / "without.json").read_text(
        encoding="utf-8"
    )


def test_fit_bundle_is_deterministic() -> None:
    train = _mixed_frame()
    catalog = _catalog_frame()
    config = MatchConfig(lgbm_num_rounds=10, lgbm_min_data_in_leaf=1, num_threads=1, ensemble_size=2)
    first = fit_bundle(train, catalog, config)
    second = fit_bundle(train, catalog, config)
    from port_entity_matching.candidates import generate_candidates
    from port_entity_matching.features import build_features
    from port_entity_matching.pipeline import Matcher
    from port_entity_matching.records import records_from_frame

    queries = train.copy()
    queries["call_sign"] = None
    records = records_from_frame(queries)
    index = CatalogIndex.build(catalog)
    sets = generate_candidates(records, index, None, config)
    features = build_features(records, sets, index, first.maps, config)
    assert np.array_equal(
        score_ensemble(first.ranker, features, config),
        score_ensemble(second.ranker, features, config),
    )
    assert Matcher(index, first).predict_topk(records) == Matcher(index, second).predict_topk(records)


def _hangul_only_perturbation(record):
    from port_entity_matching.normalize import is_hangul_name

    if not is_hangul_name(record.vessel_name):
        return None
    return dataclasses.replace(record, gross_tonnage=None, international_gross_tonnage=None)


def test_cv_run_reproduces_the_clean_report_and_diagnostics(tmp_path) -> None:
    path = tmp_path / "cv.csv"
    report = cross_validate(_train_frame(), _catalog_frame(), _config(), path)
    run = cross_validate_run(_train_frame(), _catalog_frame(), _config())
    assert run.report == report
    expected = tmp_path / "expected.csv"
    run.oof.drop(columns=["scenario"]).loc[:, list(DIAGNOSTIC_COLUMNS)].to_csv(expected, index=False)
    assert path.read_bytes() == expected.read_bytes()


def test_scenario_rows_are_scored_with_the_fold_model(monkeypatch) -> None:
    from port_entity_matching import train as train_module

    calls: list[int] = []
    original = train_module.train_ensemble

    def counting_train_ensemble(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "train_ensemble", counting_train_ensemble)
    train = _train_frame()
    config = _config()
    run = cross_validate_run(train, _catalog_frame(), config, scenarios={"no_tonnage": _hangul_only_perturbation})

    hangul_rows = int((run.oof["scenario"] == "clean").sum())
    scenario_rows = int((run.oof["scenario"] == "no_tonnage").sum())
    assert hangul_rows == len(train)
    assert scenario_rows == int(run.oof[run.oof["scenario"] == "clean"]["is_hangul"].sum())
    assert len(calls) == config.cv_splits
    assert run.report == cross_validate_run(train, _catalog_frame(), config).report


def test_inapplicable_scenario_adds_no_rows() -> None:
    run = cross_validate_run(
        _train_frame(),
        _catalog_frame(),
        _config(),
        scenarios={"never": lambda record: None},
    )
    assert set(run.oof["scenario"]) == {"clean"}
    assert len(run.oof) == len(_ROWS)


def test_reserved_scenario_name_rejected() -> None:
    with pytest.raises(ValueError, match="clean"):
        cross_validate_run(_train_frame(), _catalog_frame(), _config(), scenarios={"clean": lambda record: record})


def test_oof_column_and_row_order() -> None:
    run = cross_validate_run(
        _train_frame(),
        _catalog_frame(),
        _config(),
        scenarios={"no_tonnage": _hangul_only_perturbation},
    )
    assert list(run.oof.columns) == ["scenario", *DIAGNOSTIC_COLUMNS]
    scenarios_in_order = list(dict.fromkeys(run.oof["scenario"]))
    assert scenarios_in_order[0] == "clean"
    assert scenarios_in_order[-1] == "no_tonnage"


def test_shipped_modules_stay_evaluation_free() -> None:
    package = Path(__file__).resolve().parents[3] / "src" / "port_entity_matching"
    for module in sorted(package.glob("*.py")):
        assert "port_entity_matching.evaluation" not in module.read_text(encoding="utf-8"), module.name


def test_injected_candidate_fn_used() -> None:
    from port_entity_matching import train as train_module
    from port_entity_matching.candidates import generate_candidates

    calls = {"count": 0}
    train = _train_frame()
    catalog = _catalog_frame()
    config = _config()
    scenarios = {"no_tonnage": _hangul_only_perturbation}
    baseline = cross_validate_run(train, catalog, config, scenarios=scenarios)
    index = CatalogIndex.build(catalog)
    original_features = train_module.build_features
    original_score = train_module.score_ensemble

    def _counting(records):
        calls["count"] += 1
        assert train_module.build_features is original_features
        assert train_module.score_ensemble is original_score
        return generate_candidates(records, index, None, config)

    injected = cross_validate_run(train, catalog, config, scenarios=scenarios, candidate_fn=_counting, index=index)
    assert calls["count"] == config.cv_splits * 3
    assert injected.oof.reset_index(drop=True).equals(baseline.oof.reset_index(drop=True))


def test_diagnostic_rows_without_candidate_fn() -> None:
    from port_entity_matching.maps import AttributeMaps
    from port_entity_matching.train import _diagnostic_rows

    train = _train_frame().iloc[:4]
    catalog = _catalog_frame()
    index = CatalogIndex.build(catalog)
    maps = AttributeMaps.fit(train, index)
    config = MatchConfig(lgbm_num_rounds=5, lgbm_min_data_in_leaf=1, num_threads=1)
    bundle = fit_bundle(train, catalog, config)
    records = records_from_frame(train)
    targets = [str(value) for value in train["target_vessel_id"].tolist()]
    rows = _diagnostic_rows(records, targets, index, maps, config, bundle.ranker, candidate_fn=None)
    assert len(rows) == len(records)


def test_translit_receives_hangul_names_only_once() -> None:
    from port_entity_matching.maps import AttributeMaps

    calls: list[tuple[str, ...]] = []

    def stub(names) -> dict[str, tuple[str, ...]]:
        calls.append(tuple(names))
        return {name: (name + " ALT",) for name in names}

    index = CatalogIndex.build(_catalog_frame())
    train = _train_frame()
    maps = AttributeMaps.fit(train, index)
    config = MatchConfig(augment=True, augment_notation=False, num_threads=1, lgbm_num_rounds=5, lgbm_min_data_in_leaf=1)
    build_training_data(train, index, maps, config, translit=stub)
    assert len(calls) == 1
    from port_entity_matching.normalize import is_hangul_name

    assert calls[0]
    assert all(is_hangul_name(name) for name in calls[0])
    assert "GRAND ACE 11" not in calls[0]
    originals = records_from_frame(train)
    copies, _ = degraded_copies(originals, config)
    expected = {record.vessel_name for record in [*originals, *copies] if is_hangul_name(record.vessel_name)}
    assert set(calls[0]) == expected
    assert len(calls[0]) == len(expected)


def test_disabled_channel_keeps_legacy_matrix() -> None:
    import numpy as np

    from port_entity_matching.maps import AttributeMaps

    def stub(names):  # pragma: no cover
        raise AssertionError("translit must not be called")

    index = CatalogIndex.build(_catalog_frame())
    train = _train_frame()
    maps = AttributeMaps.fit(train, index)
    config = MatchConfig(augment=False, num_threads=1, llm_enabled=False)
    data = build_training_data(train, index, maps, config, translit=stub)
    start = FEATURE_NAMES.index("llm_dl_sim")
    assert np.all(np.isnan(np.asarray(data.features[:, start : start + 6])))


def test_cv_transliterates_perturbed_names() -> None:
    import dataclasses

    from port_entity_matching.maps import AttributeMaps
    from port_entity_matching.records import QueryRecord

    seen: list[tuple[str, ...]] = []

    def stub(names) -> dict[str, tuple[str, ...]]:
        seen.append(tuple(names))
        return dict.fromkeys(names, ("PERTURBED ALT",))

    def perturb(record: QueryRecord) -> QueryRecord | None:
        return dataclasses.replace(record, vessel_name=" perturbed 호")

    train = _train_frame().iloc[:6].copy()
    catalog = _catalog_frame()
    config = MatchConfig(
        augment=False, num_threads=1, cv_splits=2, lgbm_num_rounds=5, lgbm_min_data_in_leaf=1,
    )
    cross_validate_run(train, catalog, config, scenarios={"pert": perturb}, translit=stub)
    assert any(any("perturbed" in name for name in call) for call in seen)


def test_fit_bundle_retains_fewshot() -> None:
    shots = (("글로비스 스피릿", "GLOVIS SPIRIT"),)
    bundle = fit_bundle(_train_frame(), _catalog_frame(), _config(), fewshot=shots)
    assert bundle.fewshot == shots


def test_cli_requires_llm_model_when_enabled(tmp_path) -> None:
    train_path = tmp_path / "train.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    _train_frame().to_csv(train_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    with pytest.raises(SystemExit):
        main(["--train", str(train_path), "--catalog", str(catalog_path), "--out", str(tmp_path / "model")])


def test_cv_logs_disabled_channel_once(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="port_entity_matching.train"):
        cross_validate_run(_train_frame(), _catalog_frame(), _config())
    assert caplog.messages.count("[ALGO] llm channel disabled for this run") == 1


def test_cv_does_not_reuse_clean_alts_for_untranslated_perturbations() -> None:
    from port_entity_matching.normalize import is_hangul_name

    observed = []
    index = CatalogIndex.build(_catalog_frame())
    config = dataclasses.replace(_config(), augment=False)

    def translit(names):
        return {name: ("GLOVIS SPIRIT",) for name in names if name != "새로운 이름"}

    def perturb(record):
        return dataclasses.replace(record, vessel_name="새로운 이름") if is_hangul_name(record.vessel_name) else None

    def candidates(records):
        observed.extend(record for record in records if record.vessel_name == "새로운 이름")
        return generate_candidates(records, index, None, config)

    cross_validate_run(
        _train_frame(), _catalog_frame(), config,
        scenarios={"changed": perturb}, candidate_fn=candidates, index=index, translit=translit,
    )
    assert observed
    assert all(record.alt_names == () for record in observed)
