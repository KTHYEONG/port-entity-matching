"""Contract tests for training, vessel-disjoint cross-validation and the CLI."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from port_entity_matching.catalog import CatalogIndex
from port_entity_matching.config import MatchConfig
from port_entity_matching.features import FEATURE_NAMES
from port_entity_matching.maps import AttributeMaps
from port_entity_matching.ranker import score_ensemble
from port_entity_matching.train import (
    DIAGNOSTIC_COLUMNS,
    build_training_data,
    cross_validate,
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


def test_cli_fit_writes_bundle(tmp_path) -> None:
    train_path = tmp_path / "train.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    model_dir = tmp_path / "model"
    _train_frame().to_csv(train_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    exit_code = main(["--train", str(train_path), "--catalog", str(catalog_path), "--out", str(model_dir)])
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


def test_cli_cross_validation_writes_diagnostics(tmp_path) -> None:
    train_path = tmp_path / "train.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    diagnostics_path = tmp_path / "cv.csv"
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
    augmented = build_training_data(train, index, maps, MatchConfig(augment=True, num_threads=1))
    assert len(augmented.group_sizes) == len(plain.group_sizes) + 6
    assert augmented.recall_hangul == plain.recall_hangul
    assert augmented.recall_latin == plain.recall_latin


def test_maps_are_unaffected_by_augmentation(tmp_path) -> None:
    train = _mixed_frame()
    catalog = _catalog_frame()
    with_augmentation = fit_bundle(train, catalog, MatchConfig(augment=True, num_threads=1))
    without_augmentation = fit_bundle(train, catalog, MatchConfig(augment=False, num_threads=1))
    with_augmentation.maps.to_json(tmp_path / "with.json")
    without_augmentation.maps.to_json(tmp_path / "without.json")
    assert (tmp_path / "with.json").read_text(encoding="utf-8") == (
        tmp_path / "without.json"
    ).read_text(encoding="utf-8")


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
