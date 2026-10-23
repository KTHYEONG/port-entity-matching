"""Contract tests for the inference pipeline and the submission writer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from port_entity_matching.config import MatchConfig
from port_entity_matching.pipeline import Matcher, run_submission
from port_entity_matching.records import QueryRecord, records_from_frame
from port_entity_matching.train import fit_bundle
from port_entity_matching.transliterate import Transliterator, TransliteratorUnavailable

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


def _config(**overrides: int | float) -> MatchConfig:
    defaults: dict[str, int | float] = {
        "lgbm_num_rounds": 20,
        "lgbm_min_data_in_leaf": 1,
        "cv_splits": 2,
        "num_threads": 1,
    }
    defaults.update(overrides)
    return MatchConfig(**defaults)


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


def _records(rows: list[dict[str, object]]) -> list[QueryRecord]:
    return records_from_frame(pd.DataFrame(rows))


def _record(sample_id: str, name: str, call_sign: str | None = None, tonnage: float | None = None) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "vessel_name": name,
        "call_sign": call_sign,
        "gross_tonnage": tonnage,
        "international_gross_tonnage": tonnage,
        "flag_name": "파나마",
        "vessel_type": "Tanker",
        "arrival_year": 2020,
    }


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    bundle = fit_bundle(_train_frame(), _catalog_frame(), _config())
    destination = tmp_path / "artifacts"
    bundle.save(destination)
    return destination


@pytest.fixture
def matcher(model_dir: Path) -> Matcher:
    return Matcher.from_dir(model_dir, _catalog_frame())


def test_exact_latin_match_is_predicted(matcher: Matcher) -> None:
    records = _records([_record("R1", "CALIDA", None, 61332)])
    assert matcher.predict(records) == ["V04"]


def test_hangul_record_is_predicted(matcher: Matcher) -> None:
    records = _records([_record("R2", "글로비스 스피릿", None, 5000)])
    assert matcher.predict(records) == ["V00"]


def test_memory_override_beats_the_ranker(matcher: Matcher) -> None:
    records = _records([_record("R3", "CALIDA", "CS_V00_0", 61332)])
    assert matcher.predict(records) == ["V00"]
    prediction = matcher.predict_topk(records, k=3)[0]
    assert prediction.memory_hit is True
    assert prediction.vessel_ids[0] == "V00"
    assert len(prediction.vessel_ids) == len(prediction.scores)


def test_predict_topk_rejects_invalid_k(matcher: Matcher) -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        matcher.predict_topk(_records([_record("R4", "CALIDA")]), k=0)


def test_predict_topk_without_memory_hit(matcher: Matcher) -> None:
    records = _records([_record("R5", "CALIDA", None, 61332)])
    prediction = matcher.predict_topk(records, k=3)[0]
    assert prediction.memory_hit is False
    assert prediction.sample_id == "R5"
    assert prediction.vessel_ids[0] == "V04"
    assert list(prediction.scores) == sorted(prediction.scores, reverse=True)


def test_predictions_are_row_independent(matcher: Matcher) -> None:
    rows = [
        _record("R0", "CALIDA", None, 61332),
        _record("R1", "글로비스 스피릿", None, 5000),
        _record("R2", "MARSGRACHT", None, 3000),
        _record("R3", "SHINING ROAD", None, 2000),
        _record("R4", "카시마 마루", None, 8000),
        _record("R5", "GRAND ACE 12", None, 1000),
        _record("R6", "DILAN", None, 168208),
        _record("R7", "흥아", None, 5000),
    ]
    records = _records(rows)
    batch = dict(zip([row["sample_id"] for row in rows], matcher.predict(records), strict=True))
    alone = {row["sample_id"]: matcher.predict(_records([row]))[0] for row in rows}
    subset = matcher.predict(_records(rows[:3]))
    order = [3, 0, 7, 5, 2, 6, 1, 4]
    shuffled = matcher.predict(_records([rows[position] for position in order]))
    assert batch == alone
    assert subset == [batch[row["sample_id"]] for row in rows[:3]]
    assert [batch[rows[position]["sample_id"]] for position in order] == shuffled


def test_chunking_does_not_change_predictions(model_dir: Path) -> None:
    catalog = _catalog_frame()
    bundle = fit_bundle(_train_frame(), catalog, _config())
    rows = [
        _record("R0", "CALIDA", None, 61332),
        _record("R1", "글로비스 스피릿", None, 5000),
        _record("R2", "MARSGRACHT", None, 3000),
        _record("R3", "SHINING ROAD", None, 2000),
        _record("R4", "카시마 마루", None, 8000),
    ]
    records = _records(rows)
    default_matcher = Matcher.from_dir(model_dir, catalog)
    chunked = Matcher(default_matcher.index, replace(bundle, config=replace(bundle.config, batch_size=2)))
    assert default_matcher.predict(records) == chunked.predict(records)


def test_submission_writes_expected_format(tmp_path: Path, model_dir: Path) -> None:
    test_path = tmp_path / "test.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    output_path = tmp_path / "nested" / "submission.csv"
    rows = [
        _record("S0", "CALIDA", None, 61332),
        _record("S1", "글로비스 스피릿", None, 5000),
        _record("S2", "MARSGRACHT", None, 3000),
    ]
    pd.DataFrame(rows).to_csv(test_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)

    written = run_submission(test_path, catalog_path, model_dir, output_path)
    assert written == output_path
    text = output_path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "sample_id,target_vessel_id"
    frame = pd.read_csv(output_path)
    assert list(frame.columns) == ["sample_id", "target_vessel_id"]
    assert frame["sample_id"].tolist() == ["S0", "S1", "S2"]
    assert set(frame["target_vessel_id"]) <= set(_catalog_frame()["vessel_id"])


def test_submission_is_deterministic(tmp_path: Path, model_dir: Path) -> None:
    test_path = tmp_path / "test.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    pd.DataFrame([_record("S0", "CALIDA", None, 61332)]).to_csv(test_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    run_submission(test_path, catalog_path, model_dir, first)
    run_submission(test_path, catalog_path, model_dir, second)
    assert first.read_bytes() == second.read_bytes()


def test_duplicate_sample_id_is_rejected(tmp_path: Path, model_dir: Path) -> None:
    test_path = tmp_path / "test.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    rows = [_record("S0", "CALIDA", None, 61332), _record("S0", "MARSGRACHT", None, 3000)]
    pd.DataFrame(rows).to_csv(test_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    with pytest.raises(ValueError, match="sample_id"):
        run_submission(test_path, catalog_path, model_dir, tmp_path / "out.csv")


def test_missing_test_file_raises(tmp_path: Path, model_dir: Path) -> None:
    catalog_path = tmp_path / "vessel_catalog.csv"
    _catalog_frame().to_csv(catalog_path, index=False)
    missing = tmp_path / "absent.csv"
    with pytest.raises(FileNotFoundError, match=r"absent\.csv"):
        run_submission(missing, catalog_path, model_dir, tmp_path / "out.csv")


def test_prediction_outside_the_catalog_is_rejected(
    tmp_path: Path, model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_path = tmp_path / "test.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    pd.DataFrame([_record("S0", "CALIDA", None, 61332)]).to_csv(test_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    monkeypatch.setattr(Matcher, "predict", lambda self, records: ["OUTSIDE_CATALOG"])
    with pytest.raises(RuntimeError, match="absent from the catalog"):
        run_submission(test_path, catalog_path, model_dir, tmp_path / "out.csv")


def test_empty_and_dirty_records_get_valid_ids(matcher: Matcher) -> None:
    rows = [
        _record("D0", ""),
        _record("D1", "   "),
        _record("D2", "X", None, None),
        _record("D3", "A" * 300, None, 61332),
    ]
    predictions = matcher.predict(_records(rows))
    known = set(_catalog_frame()["vessel_id"])
    assert len(predictions) == len(rows)
    assert set(predictions) <= known


def test_empty_input_predicts_nothing(matcher: Matcher) -> None:
    assert matcher.predict([]) == []
    assert matcher.predict_topk([]) == []


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_batch_size_is_rejected(matcher: Matcher, size: int) -> None:
    bundle = replace(matcher.bundle, config=replace(matcher.bundle.config, batch_size=size))
    with pytest.raises(ValueError, match="batch_size"):
        Matcher(matcher.index, bundle)


@pytest.mark.parametrize("ids", [[None], ["S0", "S0"]])
def test_invalid_submission_ids_do_not_create_output(tmp_path: Path, model_dir: Path, ids: list) -> None:
    test_path = tmp_path / "test.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    frame = pd.DataFrame([_record("ignored", "CALIDA") for _ in ids])
    frame["sample_id"] = ids
    frame.to_csv(test_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    output = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="sample_id"):
        run_submission(test_path, catalog_path, model_dir, output)
    assert not output.exists()


def test_empty_submission_has_header(tmp_path: Path, model_dir: Path) -> None:
    test_path = tmp_path / "test.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    pd.DataFrame(columns=list(_record("", ""))).to_csv(test_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    output = run_submission(test_path, catalog_path, model_dir, tmp_path / "out.csv")
    assert output.read_text(encoding="utf-8") == "sample_id,target_vessel_id\n"


def test_submission_logs_only_aggregate_values(
    tmp_path: Path, model_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    test_path = tmp_path / "test.csv"
    catalog_path = tmp_path / "vessel_catalog.csv"
    pd.DataFrame([_record("PRIVATE_SAMPLE", "CALIDA", "CS_V00_0", 61332)]).to_csv(test_path, index=False)
    _catalog_frame().to_csv(catalog_path, index=False)
    with caplog.at_level("INFO"):
        run_submission(test_path, catalog_path, model_dir, tmp_path / "PRIVATE_OUTPUT.csv")
    for value in ("PRIVATE_SAMPLE", "CALIDA", "CS_V00_0", "V00", "PRIVATE_OUTPUT"):
        assert value not in caplog.text
    assert "memory_hits=1" in caplog.text


def test_matcher_without_transliterator_matches_disabled_bundle(model_dir: Path) -> None:
    catalog = _catalog_frame()
    enabled = Matcher.from_dir(model_dir, catalog)
    assert enabled.transliterator is None
    disabled_bundle = replace(enabled.bundle, config=replace(enabled.bundle.config, llm_enabled=False))
    disabled = Matcher(enabled.index, disabled_bundle)
    rows = [_record("R0", "글로비스 스피릿", None, 5000), _record("R1", "CALIDA", None, 61332)]
    records = _records(rows)
    assert enabled.predict(records) == disabled.predict(records)


def test_precomputed_alts_are_not_regenerated(matcher: Matcher) -> None:
    import dataclasses

    seen: list[list[str]] = []

    class _Stub:
        def translate(self, names):  # type: ignore[no-untyped-def]
            seen.append(list(names))
            return dict.fromkeys(names, ("ALT",))

    matcher.transliterator = _Stub()  # type: ignore[method-assign]
    records = _records([_record("R0", "글로비스 스피릿", None, 5000)])
    records = [dataclasses.replace(record, alt_names=("EXISTING",)) for record in records]
    matcher.predict(records)
    assert seen == []


@pytest.mark.parametrize("topk", [False, True])
def test_llm_preserves_existing_alts_for_duplicate_names(matcher: Matcher, monkeypatch, topk: bool) -> None:
    import port_entity_matching.pipeline as pipeline

    records = _records([_record("R0", "흥아"), _record("R1", "흥아"), _record("R2", "CALIDA")])
    records[0] = replace(records[0], alt_names=("EXISTING",))
    stub = Mock(spec=Transliterator)
    stub.translate.return_value = {"흥아": ("HEUNG-A",)}
    matcher.transliterator = stub
    observed = []
    original = pipeline.generate_candidates

    def capture(rows, *args):
        observed.extend(rows)
        return original(rows, *args)

    monkeypatch.setattr(pipeline, "generate_candidates", capture)
    result = matcher.predict_topk(records) if topk else matcher.predict(records)
    assert len(result) == 3
    assert [record.alt_names for record in observed] == [("EXISTING",), ("HEUNG-A",), ()]
    assert records[1].alt_names == ()
    stub.translate.assert_called_once()
    assert stub.translate.call_args.args == (["흥아"],)
    assert stub.translate.call_args.kwargs == {}


def test_from_dir_unavailable_llm_falls_back(model_dir: Path, monkeypatch, caplog) -> None:
    matcher = Matcher.from_dir(model_dir, _catalog_frame())
    replace(matcher.bundle, fewshot=(("흥아", "HEUNG-A"),)).save(model_dir)
    stub = Mock(side_effect=TransliteratorUnavailable("PRIVATE_NAME"))
    monkeypatch.setattr(Transliterator, "load", stub)
    with caplog.at_level("WARNING"):
        fallback = Matcher.from_dir(model_dir, _catalog_frame())
    records = _records([_record("R0", "흥아")])
    assert fallback.predict(records) == matcher.predict(records)
    stub.assert_called_once()
    assert "llm unavailable reason=TransliteratorUnavailable" in caplog.text
    assert "PRIVATE_NAME" not in caplog.text


def test_disabled_llm_does_not_generate(matcher: Matcher) -> None:
    stub = Mock(spec=Transliterator)
    disabled = Matcher(matcher.index, replace(matcher.bundle, config=replace(matcher.bundle.config, llm_enabled=False)), stub)
    assert disabled.predict(_records([_record("R0", "흥아")]))
    stub.translate.assert_not_called()


def test_empty_llm_alts_log_counts_without_private_values(matcher: Matcher, caplog) -> None:
    records = _records([_record("PRIVATE_SAMPLE", "흥아")])
    expected = matcher.predict(records)
    stub = Mock(spec=Transliterator)
    stub.translate.return_value = {}
    matcher.transliterator = stub
    with caplog.at_level("INFO"):
        assert matcher.predict(records) == expected
    assert "generated=0" in caplog.text
    assert "흥아" not in caplog.text
    assert "PRIVATE_SAMPLE" not in caplog.text
