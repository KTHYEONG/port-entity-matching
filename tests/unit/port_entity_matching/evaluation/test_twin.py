"""Invariant tests for the twin evaluation package."""

from __future__ import annotations

import contextlib
import sys
import types
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from port_entity_matching.evaluation.hangulize import random_style, transliterate_latin
from port_entity_matching.evaluation.twin import build_twin, reverse_llm, twin_scenarios
from port_entity_matching.records import QueryRecord


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "T00001", "vessel_name": "GLOVIS SUN"},
            {"sample_id": "T00002", "vessel_name": "한진부산"},
            {"sample_id": "T00003", "vessel_name": "PACIFIC STAR"},
            {"sample_id": "T00004", "vessel_name": "글로비스"},
        ]
    )


def _record(sample_id: str, name: str) -> QueryRecord:
    return QueryRecord(
        sample_id=sample_id,
        vessel_name=name,
        call_sign=None,
        gross_tonnage=1000.0,
        international_gross_tonnage=900.0,
        flag_name=None,
        vessel_type=None,
        arrival_year=2020,
    )


def _fake_reverse(names: Sequence[str]) -> dict[str, str]:
    return {str(name): f"가짜{index}" for index, name in enumerate(names)}


def test_twin_is_deterministic() -> None:
    frame = _frame()
    first = build_twin(frame, 2, 0, None)
    second = build_twin(frame, 2, 0, None)
    assert first == second


def test_twin_skips_hangul_rows() -> None:
    twin = build_twin(_frame(), 2, 0, None)
    assert set(twin) <= {"T00001", "T00003"}
    assert len(twin) == 2
    for entry in twin.values():
        assert "src" in entry and "r" in entry


def test_too_many_rows_raises() -> None:
    with pytest.raises(ValueError, match="Latin rows"):
        build_twin(_frame(), 3, 0, None)


def test_empty_twin_has_no_rows() -> None:
    assert build_twin(_frame(), 0, 0, None) == {}


def test_reverse_optional() -> None:
    frame = _frame()
    without = build_twin(frame, 2, 0, None)
    assert all("q" not in entry for entry in without.values())

    with_q = build_twin(frame, 2, 0, _fake_reverse)
    assert all("q" in entry for entry in with_q.values())
    mapping = _fake_reverse(sorted({entry["src"] for entry in with_q.values()}))
    for entry in with_q.values():
        assert entry["q"] == mapping[entry["src"]]


def test_scenarios_blank_tonnage_and_skip_non_hangul() -> None:
    twin = {"T00001": {"src": "GLOVIS SUN", "r": "글로비스 선"}, "T00002": {"src": "PACIFIC", "r": "PACIFIC"}}
    scenarios = twin_scenarios(twin)
    assert set(scenarios) == {"kor_nt", "r2k_wt", "r2k_nt"}
    assert scenarios["r2k_wt"](_record("T00002", "PACIFIC")) is None
    renamed = scenarios["r2k_wt"](_record("T00001", "GLOVIS SUN"))
    assert renamed is not None
    assert renamed.vessel_name == "글로비스 선"
    assert renamed.gross_tonnage == 1000.0
    blanked = scenarios["r2k_nt"](_record("T00001", "GLOVIS SUN"))
    assert blanked is not None
    assert blanked.vessel_name == "글로비스 선"
    assert blanked.gross_tonnage is None
    assert blanked.international_gross_tonnage is None
    assert scenarios["kor_nt"](_record("T00003", "PACIFIC")) is None
    kor = scenarios["kor_nt"](_record("T00003", "한진부산"))
    assert kor is not None
    assert kor.gross_tonnage is None
    assert kor.international_gross_tonnage is None


def test_scenarios_skip_unknown_ids_and_missing_keys() -> None:
    twin = {"T00001": {"src": "GLOVIS SUN", "r": "글로비스 선"}, "T00009": {"src": "OTHER"}}
    scenarios = twin_scenarios(twin)
    assert scenarios["r2k_wt"](_record("MISSING", "GLOVIS SUN")) is None
    assert scenarios["r2k_nt"](_record("T00009", "OTHER")) is None


def test_scenarios_include_q_when_present() -> None:
    twin = {
        "T00001": {"src": "GLOVIS SUN", "r": "글로비스 선", "q": "글로비스 썬"},
        "T00002": {"src": "OTHER", "r": "OTHER"},
    }
    scenarios = twin_scenarios(twin)
    assert set(scenarios) == {"kor_nt", "r2k_wt", "r2k_nt", "q2k_wt", "q2k_nt"}
    renamed = scenarios["q2k_wt"](_record("T00001", "GLOVIS SUN"))
    assert renamed is not None
    assert renamed.vessel_name == "글로비스 썬"
    assert renamed.gross_tonnage == 1000.0
    blanked = scenarios["q2k_nt"](_record("T00001", "GLOVIS SUN"))
    assert blanked is not None
    assert blanked.vessel_name == "글로비스 썬"
    assert blanked.gross_tonnage is None
    assert blanked.international_gross_tonnage is None
    assert scenarios["q2k_wt"](_record("T00002", "OTHER")) is None


def test_hangulize_deterministic_and_passes_digits() -> None:
    style = {"a_as_ae": 0.5, "o_as_eo": 0.1, "final_stop_keep": 0.7, "ie_as_i": 0.8, "double_s": 0.1}
    first = transliterate_latin("GLOVIS SUN 7", np.random.default_rng(0), style)
    second = transliterate_latin("GLOVIS SUN 7", np.random.default_rng(0), style)
    assert first == second
    assert "7" in first
    assert any("가" <= char <= "힣" for char in first)
    assert random_style(np.random.default_rng(0)) == random_style(np.random.default_rng(0))


class _FakeTensor:
    """Minimal tensor stand-in supporting the shapes and device moves used by reverse_llm."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self._array.shape)

    def to(self, device: object) -> _FakeTensor:
        return self

    def __getitem__(self, key: object) -> np.ndarray:
        return self._array[key]  # type: ignore[index]


def _install_fake_stack(monkeypatch: pytest.MonkeyPatch, *, fail_thinking: bool) -> None:
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()  # type: ignore[attr-defined]
    torch_stub.inference_mode = lambda: contextlib.nullcontext()  # type: ignore[attr-defined]

    class _FakeTokenizer:
        def __init__(self) -> None:
            self.padding_side: str | None = None
            self._decoded = ["클로비스 선\nsecond line", "   "]

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> _FakeTokenizer:
            return cls()

        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            if fail_thinking and "enable_thinking" in kwargs:
                raise TypeError("thinking not supported")
            last = messages[-1]  # type: ignore[index]
            assert messages[1:3] == [  # type: ignore[index]
                {"role": "user", "content": "KAROS"},
                {"role": "assistant", "content": "가로스"},
            ]
            return f"prompt:{last['content']}"  # type: ignore[index]

        def __call__(self, prompts: object, **kwargs: object) -> dict[str, _FakeTensor]:
            assert self.padding_side == "left"
            assert kwargs == {"return_tensors": "pt", "padding": True}
            count = len(prompts)  # type: ignore[arg-type]
            array = np.zeros((count, 4), dtype=np.int64)
            return {"input_ids": _FakeTensor(array), "attention_mask": _FakeTensor(array)}

        def batch_decode(self, sequences: object, **kwargs: object) -> list[str]:
            count = len(sequences)  # type: ignore[arg-type]
            out, self._decoded = self._decoded[:count], self._decoded[count:]
            return out

    class _FakeModel:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> _FakeModel:
            return cls()

        def eval(self) -> None:
            return None

        def generate(self, **kwargs: object) -> object:
            assert kwargs["max_new_tokens"] == 32
            assert kwargs["do_sample"] is False
            assert kwargs["num_beams"] == 1
            inputs = kwargs["input_ids"]
            rows = inputs.shape[0]  # type: ignore[union-attr]
            return types.SimpleNamespace(sequences=np.zeros((rows, 6), dtype=np.int64))

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    transformers_stub.AutoModelForCausalLM = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)


def test_reverse_llm_reports_missing_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="stack unavailable"):
        reverse_llm("weights", [])


@pytest.mark.parametrize("fail_thinking", [False, True])
def test_reverse_llm_greedy_batches(monkeypatch: pytest.MonkeyPatch, fail_thinking: bool) -> None:
    _install_fake_stack(monkeypatch, fail_thinking=fail_thinking)
    reverse = reverse_llm("weights", [("가로스", "KAROS")], batch_size=1)
    out = reverse(["PACIFIC", "KAROS", "KAROS"])
    assert out == {"KAROS": "클로비스 선", "PACIFIC": ""}


@pytest.mark.parametrize("batch_size", [0, -1])
def test_reverse_llm_rejects_invalid_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        reverse_llm("weights", [], batch_size=batch_size)
