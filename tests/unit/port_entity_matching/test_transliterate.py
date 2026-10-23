"""Contract tests for the frozen LLM transliteration channel (core)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from port_entity_matching.config import MatchConfig
from port_entity_matching.records import QueryRecord
from port_entity_matching.transliterate import (
    SYSTEM_PROMPT,
    CachedTransliterator,
    Transliterator,
    TransliteratorUnavailable,
    attach_alt_names,
    select_fewshot,
)


def _train_frame() -> pd.DataFrame:
    rows = [
        ("글로비스 스피릿", "V00"),
        ("CALIDA", "V02"),
        ("그랜드 에이스", "V01"),
        ("칼리다 스타", "V02"),
        ("도누사", "V03"),
        ("삽산호", "V04"),
        ("샤이닝 로드", "V05"),
    ]
    return pd.DataFrame([{"vessel_name": name, "target_vessel_id": target} for name, target in rows])


def _catalog_frame() -> pd.DataFrame:
    entries = [
        ("V00", "GLOVIS SPIRIT"),
        ("V01", "GRAND ACE"),
        ("V02", "CALIDA STAR"),
        ("V03", "DONOUSSA"),
        ("V04", "SAPSAN"),
        ("V05", "SHINING ROAD"),
    ]
    return pd.DataFrame([{"vessel_id": vid, "vessel_name": name} for vid, name in entries])


def _record(sample_id: str, name: str) -> QueryRecord:
    return QueryRecord(
        sample_id=sample_id,
        vessel_name=name,
        call_sign=None,
        gross_tonnage=None,
        international_gross_tonnage=None,
        flag_name=None,
        vessel_type=None,
        arrival_year=None,
    )


def test_system_prompt_is_exact() -> None:
    assert SYSTEM_PROMPT == (
        "Korean port-call records often write a ship's name in Hangul, transliterated from its "
        "registered name (usually English words or romanized Chinese/Japanese/Korean). "
        "Recover the registered ship name in Latin letters. Keep numbers. "
        "Answer with the ship name only."
    )


def test_fewshot_selection_is_deterministic() -> None:
    train, catalog = _train_frame(), _catalog_frame()
    first = select_fewshot(train, catalog, 3, 0)
    second = select_fewshot(train, catalog, 3, 0)
    assert first == second
    assert len(first) == 3
    lookup = dict(zip(catalog["vessel_id"], catalog["vessel_name"], strict=True))
    train_targets = dict(zip(train["vessel_name"], train["target_vessel_id"], strict=True))
    for hangul, latin in first:
        assert latin == lookup[train_targets[hangul]]


def test_fewshot_uses_first_hangul_row_in_train_order() -> None:
    import numpy as np

    train = pd.DataFrame(
        {
            "vessel_name": ["LATIN", "가나", "다라", "마바", "중복", "사아", "자차"],
            "target_vessel_id": ["V02", "V04", "V01", "V03", "V04", "V02", "V00"],
        },
        index=[9, 8, 7, 6, 5, 4, 3],
    )
    chosen = set(np.random.default_rng(0).choice(["V00", "V01", "V02", "V03", "V04"], 3, replace=False))
    latin = dict(zip(_catalog_frame()["vessel_id"], _catalog_frame()["vessel_name"], strict=True))
    expected = tuple(
        (name, latin[target])
        for name, target in [
            ("가나", "V04"),
            ("다라", "V01"),
            ("마바", "V03"),
            ("사아", "V02"),
            ("자차", "V00"),
        ]
        if target in chosen
    )
    assert select_fewshot(train, _catalog_frame(), 3, 0) == expected


def test_fewshot_count_exceeding_vessels() -> None:
    train = pd.DataFrame(
        [
            {"vessel_name": "글로비스 스피릿", "target_vessel_id": "V00"},
            {"vessel_name": "그랜드 에이스", "target_vessel_id": "V01"},
        ]
    )
    catalog = pd.DataFrame(
        [{"vessel_id": "V00", "vessel_name": "GLOVIS SPIRIT"}, {"vessel_id": "V01", "vessel_name": "GRAND ACE"}]
    )
    assert len(select_fewshot(train, catalog, 12, 0)) == 2


def test_fewshot_empty_and_zero() -> None:
    catalog = _catalog_frame()
    assert select_fewshot(_train_frame(), catalog, 0, 0) == ()
    latin_only = pd.DataFrame([{"vessel_name": "CALIDA", "target_vessel_id": "V02"}])
    assert select_fewshot(latin_only, catalog, 3, 0) == ()


def test_fewshot_missing_columns() -> None:
    frame = pd.DataFrame([{"vessel_name": "글로비스 스피릿"}])
    assert select_fewshot(frame, _catalog_frame(), 3, 0) == ()


def test_attach_preserves_non_hangul_and_misses() -> None:
    import dataclasses

    records = [_record("L0", "CALIDA"), _record("H0", "글로비스 스피릿"), _record("H1", "그랜드 에이스")]
    records = [dataclasses.replace(record, alt_names=("EXISTING",)) for record in records]
    table = {"글로비스 스피릿": ("GLOVIS SPIRIT", "GLOVIS"), "CALIDA": ("IGNORED",)}
    out = attach_alt_names(records, table)
    assert [record.sample_id for record in out] == ["L0", "H0", "H1"]
    assert out[0].alt_names == ("EXISTING",)
    assert out[1].alt_names == ("GLOVIS SPIRIT", "GLOVIS")
    assert out[2].alt_names == ("EXISTING",)
    assert out[0] is records[0]
    assert out[2] is records[2]


def test_unavailable_without_weights(tmp_path: Path) -> None:
    with pytest.raises(TransliteratorUnavailable):
        Transliterator.load(tmp_path, [], MatchConfig())


def test_load_rejects_invalid_llm_config(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="llm_num_beams"):
        Transliterator.load(weights, [], MatchConfig(llm_num_beams=0))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_fewshot", -1),
        ("llm_num_beams", 0),
        ("llm_max_new_tokens", 0),
        ("llm_retrieval_beams", 6),
        ("llm_retrieval_beams", -1),
        ("llm_time_budget_s", 0.0),
    ],
)
def test_load_rejects_each_llm_bound(tmp_path: Path, field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        Transliterator.load(tmp_path, [], MatchConfig(**{field: value}))  # type: ignore[arg-type]


def test_module_import_has_no_torch_dependency() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import builtins\n"
            "original_import = builtins.__import__\n"
            "def blocked(name, *args, **kwargs):\n"
            "    if name.split('.')[0] in ('torch', 'transformers'):\n"
            "        raise ModuleNotFoundError(name)\n"
            "    return original_import(name, *args, **kwargs)\n"
            "builtins.__import__ = blocked\n"
            "import port_entity_matching.transliterate\n",
        ],
        check=True,
        env={"PYTHONPATH": "src", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True,
    )


def test_cache_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    class _Stub:
        def translate(self, names):  # type: ignore[no-untyped-def]
            seen.append(list(names))
            return {name: (f"{name}-LAT",) for name in names}

    monkeypatch.setattr(Transliterator, "load", lambda *args, **kwargs: _Stub())
    weights = _weights_with_config(tmp_path)
    cache = CachedTransliterator(tmp_path / "cache", weights, [], MatchConfig())
    first = cache.translate(["글로비스 스피릿", "그랜드 에이스"])
    second = cache.translate(["그랜드 에이스", "삽산호"])
    assert seen[0] == ["글로비스 스피릿", "그랜드 에이스"]
    assert seen[1] == ["삽산호"]
    assert second["그랜드 에이스"] == first["그랜드 에이스"]
    reopened = CachedTransliterator(tmp_path / "cache", tmp_path / "weights", [], MatchConfig())
    third = reopened.translate(["글로비스 스피릿"])
    assert third == {"글로비스 스피릿": first["글로비스 스피릿"]}
    assert len(seen) == 2


def test_cache_ignores_corrupt_trailing_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stub:
        def __init__(self) -> None:
            self.calls = 0

        def translate(self, names):  # type: ignore[no-untyped-def]
            self.calls += 1
            return dict.fromkeys(names, ("ALT",))

    stub = _Stub()
    monkeypatch.setattr(Transliterator, "load", lambda *args, **kwargs: stub)
    _weights_with_config(tmp_path)
    CachedTransliterator(tmp_path / "cache", tmp_path / "weights", [], MatchConfig()).translate(["글로비스 스피릿"])
    assert stub.calls == 1
    cache_file = next((tmp_path / "cache").glob("*.jsonl"))
    with open(cache_file, "a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write("not-json{\n")
    fresh = CachedTransliterator(tmp_path / "cache", tmp_path / "weights", [], MatchConfig())
    assert fresh.translate(["글로비스 스피릿"]) == {"글로비스 스피릿": ("ALT",)}
    assert stub.calls == 1


@pytest.mark.parametrize(
    "tail",
    [
        b'{"name":',
        b"\xed\x95",
        b'{"name": "bad", "alts": "not-a-list"}',
        b'{"name": 1, "alts": []}',
        b'{"name": "bad", "alts": [1]}',
        b"null",
    ],
)
def test_cache_recovers_after_truncated_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tail: bytes) -> None:
    seen: list[list[str]] = []

    class Stub:
        def translate(self, names):  # type: ignore[no-untyped-def]
            seen.append(list(names))
            return dict.fromkeys(names, ("ALT",))

    monkeypatch.setattr(Transliterator, "load", lambda *args: Stub())
    cache_dir = tmp_path / "cache"
    weights = _weights_with_config(tmp_path)
    config = MatchConfig()
    CachedTransliterator(cache_dir, weights, [], config).translate(["가나"])
    cache_file = next(cache_dir.glob("*.jsonl"))
    with cache_file.open("ab") as handle:
        handle.write(tail)
    reopened = CachedTransliterator(cache_dir, weights, [], config)
    assert reopened.translate(["가나", "다라"]) == {"가나": ("ALT",), "다라": ("ALT",)}
    assert seen == [["가나"], ["다라"]]
    assert CachedTransliterator(cache_dir, weights, [], config).translate(["다라"]) == {"다라": ("ALT",)}
    assert len(seen) == 2


def test_cache_fingerprint_tracks_settings_and_weights(tmp_path: Path) -> None:
    import dataclasses

    weights = _weights_with_config(tmp_path)
    config = MatchConfig()

    def fingerprint(shots=(), cfg=config):  # type: ignore[no-untyped-def]
        return CachedTransliterator(tmp_path / "cache", weights, shots, cfg)._fingerprint

    base = fingerprint()
    assert fingerprint((("가나", "GANA"),)) != base
    assert fingerprint(cfg=dataclasses.replace(config, llm_num_beams=6)) != base
    assert fingerprint(cfg=dataclasses.replace(config, llm_max_new_tokens=25)) != base
    assert fingerprint(cfg=dataclasses.replace(config, llm_time_budget_s=1.0)) == base
    (weights / "config.json").write_text("{}", encoding="utf-8")
    assert fingerprint() != base


def test_cache_requires_weights_config(tmp_path: Path) -> None:
    with pytest.raises(TransliteratorUnavailable, match=r"config\.json"):
        CachedTransliterator(tmp_path / "cache", tmp_path / "missing", [], MatchConfig())


@contextmanager
def _null_context():  # type: ignore[no-untyped-def]
    yield


class _FakeTensor:
    def __init__(self) -> None:
        self.shape = (1, 4)
        self.device = "cpu"

    def to(self, device: object) -> _FakeTensor:
        self.device = str(device)
        return self


class _FakeSequences:
    def __init__(self, rows: list[list[str]]) -> None:
        self._rows = rows
        self.shape = (len(rows),)

    def __getitem__(self, index: int) -> list[str]:
        return self._rows[index]


class _FakeTokenizer:
    def __init__(self, texts: list[str], with_mask: bool = True) -> None:
        self._texts = texts
        self._with_mask = with_mask
        self.padding_side = "right"
        self.chat_calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.chat_calls.append({"messages": messages, **kwargs})
        return "PROMPT:" + str(messages[-1]["content"])

    def __call__(self, prompt, return_tensors=None):  # type: ignore[no-untyped-def]
        payload: dict[str, _FakeTensor] = {"input_ids": _FakeTensor()}
        if self._with_mask:
            payload["attention_mask"] = _FakeTensor()
        return payload

    def decode(self, ids, skip_special_tokens=True) -> str:  # type: ignore[no-untyped-def]
        assert skip_special_tokens is True
        assert ids == ["generated"]
        return self._texts.pop(0) if self._texts else ""


class _FakeModel:
    def __init__(self, beams: int = 5) -> None:
        self.device = "cuda"
        self.beams = beams
        self.evaled = False
        self.generate_calls: list[dict[str, object]] = []

    def eval(self) -> _FakeModel:
        self.evaled = True
        return self

    def generate(self, input_ids, attention_mask=None, **kwargs):  # type: ignore[no-untyped-def]
        self.generate_calls.append(kwargs)
        count = int(kwargs.get("num_return_sequences", self.beams))
        assert input_ids.shape == (1, 4)
        rows = [["prompt"] * 4 + ["generated"] for _ in range(count)]
        return _FakeSequences(rows)


def _install_fake_stack(
    monkeypatch: pytest.MonkeyPatch, tokenizer: _FakeTokenizer, fail_model: bool = False
) -> _FakeModel:
    import sys
    import types

    model = _FakeModel()
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    torch_mod.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch_mod.inference_mode = _null_context  # type: ignore[attr-defined]
    transformers_mod = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs.get("local_files_only") is True
            return tokenizer

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs.get("local_files_only") is True
            assert kwargs.get("device_map") == "cuda"
            if fail_model:
                raise RuntimeError("boom")
            return model

    transformers_mod.AutoTokenizer = _FakeAutoTokenizer  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM = _FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    return model


def _weights_with_config(tmp_path: Path) -> Path:
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    (weights / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    return weights


def test_load_without_importable_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(TransliteratorUnavailable):
        Transliterator.load(tmp_path, [], MatchConfig())


def test_load_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    _install_fake_stack(monkeypatch, _FakeTokenizer([]))
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    with pytest.raises(TransliteratorUnavailable):
        Transliterator.load(_weights_with_config(tmp_path), [], MatchConfig())


def test_load_without_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    _install_fake_stack(monkeypatch, _FakeTokenizer([]))
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    with pytest.raises(TransliteratorUnavailable):
        Transliterator.load(tmp_path / "missing", [], MatchConfig())


def test_load_surfaces_model_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_stack(monkeypatch, _FakeTokenizer([]), fail_model=True)
    with pytest.raises(TransliteratorUnavailable):
        Transliterator.load(_weights_with_config(tmp_path), [], MatchConfig())


def test_translate_beam_cleanup_and_row_independence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    texts = [
        "SHIP A\ntrailing line",
        "SHIP A",
        "",
        "   ",
        "SHIP B",
        "SHIP C",
        "",
        "SHIP C ",
        "SHIP D",
        "",
    ]
    tokenizer = _FakeTokenizer(texts)
    model = _install_fake_stack(monkeypatch, tokenizer)
    shots = (("글로비스 스피릿", "GLOVIS SPIRIT"),)
    loaded = Transliterator.load(_weights_with_config(tmp_path), shots, MatchConfig())
    assert tokenizer.padding_side == "left"
    assert model.evaled is True
    out = loaded.translate(["그랜드 에이스", "CALIDA", "글로비스 스피릿"])
    assert set(out) == {"그랜드 에이스", "글로비스 스피릿"}
    assert out["그랜드 에이스"] == ("SHIP A", "SHIP B")
    assert out["글로비스 스피릿"] == ("SHIP C", "SHIP D")
    tokenizer._texts = ["SHIP A\ntrailing line", "SHIP A", "", "   ", "SHIP B"]
    assert loaded.translate(["그랜드 에이스", "그랜드 에이스"]) == {"그랜드 에이스": out["그랜드 에이스"]}
    first_call = tokenizer.chat_calls[0]
    assert first_call["add_generation_prompt"] is True
    assert first_call["enable_thinking"] is False
    messages = first_call["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1:3] == [
        {"role": "user", "content": "글로비스 스피릿"},
        {"role": "assistant", "content": "GLOVIS SPIRIT"},
    ]
    assert messages[-1] == {"role": "user", "content": "그랜드 에이스"}
    for call in model.generate_calls:
        assert call["num_beams"] == 5
        assert call["num_return_sequences"] == 5
        assert call["do_sample"] is False
        assert call["length_penalty"] == 0.0
        assert call["max_new_tokens"] == 24


def test_translate_honors_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer([])
    _install_fake_stack(monkeypatch, tokenizer)
    loaded = Transliterator.load(_weights_with_config(tmp_path), [], MatchConfig())
    assert loaded.translate(["글로비스 스피릿"], deadline=time.perf_counter() - 1.0) == {}
    assert tokenizer.chat_calls == []


def test_translate_without_attention_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer(["SOLO SHIP", "SOLO SHIP", "", "OTHER", ""])
    _install_fake_stack(monkeypatch, tokenizer, fail_model=False)
    tokenizer._with_mask = False
    loaded = Transliterator.load(_weights_with_config(tmp_path), [], MatchConfig())
    assert loaded.translate(["삽산호"]) == {"삽산호": ("SOLO SHIP", "OTHER")}


@pytest.mark.slow
def test_single_name_equals_batch_call() -> None:
    candidates = [
        os.environ.get("TRANSLIT_WEIGHTS", ""),
        "model/translit",
        "models/translit",
    ]
    weights = next((Path(p) for p in candidates if p and (Path(p) / "config.json").is_file()), None)
    if weights is None:
        pytest.skip("transliteration weights unavailable")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = MatchConfig()
    shots = select_fewshot(_train_frame(), _catalog_frame(), config.llm_fewshot, config.llm_fewshot_seed)
    model = Transliterator.load(weights, shots, config)
    names = ["글로비스 스피릿", "그랜드 에이스", "삽산호"]
    single = model.translate(names[:1])
    batch = model.translate(names)
    assert single[names[0]] == batch[names[0]]
    assert model.translate(names) == batch
    assert json.dumps(single, ensure_ascii=False)
