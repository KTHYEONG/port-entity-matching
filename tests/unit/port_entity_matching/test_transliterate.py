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
        ("llm_max_name_chars", 0),
        ("llm_rerank_weight", float("nan")),
        ("llm_rerank_weight", float("inf")),
        ("llm_rerank_weight", float("-inf")),
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
    assert fingerprint(cfg=dataclasses.replace(config, llm_max_name_chars=32)) != base
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
        self.pad_token_id = None
        self.chat_calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.chat_calls.append({"messages": messages, **kwargs})
        return "PROMPT:" + str(messages[-1]["content"]) + "<|eot|>"

    def encode(self, text, add_special_tokens=False):  # type: ignore[no-untyped-def]
        assert add_special_tokens is False
        if text == "<|eot|>":
            return [7]
        if text == "":
            return []
        return [(ord(c) % 90) + 10 for c in text[:8]]

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
        import types as _types

        self.device = "cuda"
        self.beams = beams
        self.evaled = False
        self.generate_calls: list[dict[str, object]] = []
        self.generation_config = _types.SimpleNamespace(eos_token_id=2, pad_token_id=None)

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
    first_call = tokenizer.chat_calls[1]
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


def test_translate_bounds_name_and_keeps_original_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = "가" * 80
    tokenizer = _FakeTokenizer(["SHIP"] * 5)
    _install_fake_stack(monkeypatch, tokenizer)
    loaded = Transliterator.load(_weights_with_config(tmp_path), [(name, "SHOT")], MatchConfig())
    assert loaded.translate([name]) == {name: ("SHIP",)}
    messages = tokenizer.chat_calls[1]["messages"]
    assert messages[1] == {"role": "user", "content": name[:64]}
    assert messages[-1] == {"role": "user", "content": name[:64]}


def test_elapsed_time_cannot_skip_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer(["SHIP"] * 10)
    model = _install_fake_stack(monkeypatch, tokenizer)
    loaded = Transliterator.load(_weights_with_config(tmp_path), [], MatchConfig())
    clock = iter((0.0, 3600.0))
    monkeypatch.setattr("port_entity_matching.transliterate.time.perf_counter", lambda: next(clock))
    names = ["가" * 80, "다" * 80]
    assert loaded.translate(names) == dict.fromkeys(names, ("SHIP",))
    assert len(model.generate_calls) == 2


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


class _EotTokenizer:
    def __init__(self, tail: str = "<|eot|>", tail_ids: tuple[int, ...] = (7,)) -> None:
        self._tail = tail
        self._tail_ids = tail_ids
        self.padding_side = "right"
        self.pad_token_id = None

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs.get("tokenize") is False
        return "PREFIX " + str(messages[1]["content"]) + self._tail

    def encode(self, text, add_special_tokens=False):  # type: ignore[no-untyped-def]
        assert add_special_tokens is False
        if text == self._tail:
            return list(self._tail_ids)
        if text == "":
            return []
        return [(ord(c) % 90) + 10 for c in text]


class _NoSentinelTokenizer(_EotTokenizer):
    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        return "PREFIX without content"


class _ScoreTokenizer:
    def __init__(self, eot_id: int = 99) -> None:
        self._eot = eot_id
        self.padding_side = "left"
        self.pad_token_id = None

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        return "PROMPT:" + str(messages[-1]["content"]) + "<|eot|>"

    def encode(self, text, add_special_tokens=False):  # type: ignore[no-untyped-def]
        if text == "<|eot|>":
            return [self._eot]
        if text.startswith("PROMPT:"):
            return [1, 2, 3]
        if text == "":
            return []
        out: list[int] = []
        for ch in text:
            if ch == " ":
                continue
            out.append((ord(ch) % 60) + 10)
        return out


class _ScoreOut:
    def __init__(self, logits, past=None):  # type: ignore[no-untyped-def]
        self.logits = logits
        self.past_key_values = past


class _ScoreModel:
    def __init__(self) -> None:
        import torch

        self._torch = torch
        self.device = "cpu"
        self.calls = 0

    def __call__(self, input_ids=None, past_key_values=None, attention_mask=None, use_cache=True):  # type: ignore[no-untyped-def]
        self.calls += 1
        torch = self._torch
        if past_key_values is None:
            past = ((torch.zeros(input_ids.shape[0], 1, 2, 2), torch.zeros(input_ids.shape[0], 1, 2, 2)),)
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 128)
            return _ScoreOut(logits, past)
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 128)
        logits[..., 10] += 1.0
        return _ScoreOut(logits, past_key_values)


def _scoring_translator() -> Transliterator:
    tok = _ScoreTokenizer()
    model = _ScoreModel()
    return Transliterator(model, tok, [], MatchConfig())  # type: ignore[arg-type]


def test_end_of_turn_id_from_template() -> None:
    from port_entity_matching.transliterate import end_of_turn_token_id

    assert end_of_turn_token_id(_EotTokenizer()) == 7


def test_end_of_turn_missing_sentinel_raises() -> None:
    from port_entity_matching.transliterate import TransliteratorUnavailable, end_of_turn_token_id

    with pytest.raises(TransliteratorUnavailable):
        end_of_turn_token_id(_NoSentinelTokenizer())


def test_load_adds_end_of_turn_to_eos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    tok = _EotTokenizer()
    model_holder: dict[str, object] = {}

    class _Gen:
        def __init__(self) -> None:
            self.eos_token_id = 2
            self.pad_token_id = None

    class _M:
        def __init__(self) -> None:
            self.device = "cpu"
            self.generation_config = _Gen()

        def eval(self):  # type: ignore[no-untyped-def]
            return self

    inst = _M()
    model_holder["inst"] = inst
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    torch_mod.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch_mod.inference_mode = _null_context  # type: ignore[attr-defined]
    transformers_mod = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            return tok

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            return inst

    transformers_mod.AutoTokenizer = _FakeAutoTokenizer  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM = _FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    loaded = Transliterator.load(_weights_with_config(tmp_path), [], MatchConfig())
    assert loaded._model.generation_config.eos_token_id == [2, 7]  # type: ignore[union-attr]


def test_score_candidates_preserves_order_and_duplicates() -> None:
    tr = _scoring_translator()
    out = tr.score_candidates("가나", ["B", "A", "B"])
    assert len(out) == 3
    assert out[0] == out[2]
    mapped = tr.score_candidates("가나", ["A", "B"])
    assert out[0] == mapped[1]
    assert out[1] == mapped[0]


def test_score_candidates_row_independent() -> None:
    tr = _scoring_translator()
    first = tr.score_candidates("가나", ["A", "B"])
    tr.score_candidates("다라", ["C", "D"])
    assert tr.score_candidates("가나", ["A", "B"]) == first


def test_score_candidates_empty_input() -> None:
    tr = _scoring_translator()
    assert tr.score_candidates("가나", []) == ()
    assert tr._model.calls == 0


@pytest.mark.parametrize("modern_cache", [False, True])
def test_candidate_scores_match_causal_full_forward(modern_cache: bool) -> None:
    import types

    import torch

    class Cache:
        def __init__(self) -> None:
            self.batch_size = 1

        def batch_repeat_interleave(self, repeats):  # type: ignore[no-untyped-def]
            self.batch_size *= repeats

    class Model:
        device = "cpu"

        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def __call__(self, input_ids, past_key_values=None, attention_mask=None, use_cache=True):  # type: ignore[no-untyped-def]
            assert torch.is_inference_mode_enabled()
            self.shapes.append(tuple(input_ids.shape))
            if past_key_values is None:
                assert input_ids.tolist() == [[1, 2, 3]]
                past = Cache() if modern_cache else ((torch.zeros(1, 1, 3, 2),),)
            else:
                cache_batch = past_key_values.batch_size if modern_cache else past_key_values[0][0].shape[0]
                assert cache_batch == 3
                assert input_ids.tolist() == [[99, 99, 99], [15, 99, 99], [16, 17, 99]]
                assert attention_mask.tolist() == [
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                ]
                past = past_key_values
            vocab = torch.arange(128)
            logits = -((vocab - (input_ids.unsqueeze(-1) + 1)) ** 2).to(torch.bfloat16)
            return types.SimpleNamespace(logits=logits, past_key_values=past)

    model = Model()
    tr = Transliterator(model, _ScoreTokenizer(), [], MatchConfig())
    actual = tr.score_candidates("가나", ["BC", "", "A", "BC"])
    expected = []
    for tokens in ([16, 17, 99], [99], [15, 99], [16, 17, 99]):
        full = torch.tensor([1, 2, 3, *tokens])
        logits = -((torch.arange(128) - (full.unsqueeze(-1) + 1)) ** 2).to(torch.bfloat16)
        log_probs = logits.float().log_softmax(-1)
        expected.append(sum(float(log_probs[2 + j, token]) for j, token in enumerate(tokens)) / len(tokens))
    assert actual == pytest.approx(expected, abs=1e-3)
    assert model.shapes == [(1, 3), (3, 3)]


def _load_with_tokenizer_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tok: object, model: object
) -> Transliterator:
    import sys
    import types

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    torch_mod.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch_mod.inference_mode = _null_context  # type: ignore[attr-defined]
    transformers_mod = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            return tok

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
            return model

    transformers_mod.AutoTokenizer = _FakeAutoTokenizer  # type: ignore[attr-defined]
    transformers_mod.AutoModelForCausalLM = _FakeAutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    return Transliterator.load(_weights_with_config(tmp_path), [], MatchConfig())


def test_end_of_turn_empty_tail_raises() -> None:
    from port_entity_matching.transliterate import TransliteratorUnavailable, end_of_turn_token_id

    with pytest.raises(TransliteratorUnavailable):
        end_of_turn_token_id(_EotTokenizer(tail_ids=()))


def test_rerank_weight_non_finite_type_rejected() -> None:
    from port_entity_matching.transliterate import _validate_llm_config

    with pytest.raises(ValueError, match="llm_rerank_weight"):
        _validate_llm_config(MatchConfig(llm_rerank_weight="bad"))  # type: ignore[arg-type]


def _gen_model(eos: object, pad: object):  # type: ignore[no-untyped-def]
    import types as _types

    class _Gen:
        def __init__(self) -> None:
            self.eos_token_id = eos
            self.pad_token_id = pad

    class _M:
        def __init__(self) -> None:
            self.device = "cpu"
            self.generation_config = _Gen()

        def eval(self):  # type: ignore[no-untyped-def]
            return self

    return _M()


def test_load_eos_variants_and_bad_pad(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    none_loaded = _load_with_tokenizer_model(tmp_path, monkeypatch, _EotTokenizer(), _gen_model(None, None))
    assert none_loaded._model.generation_config.eos_token_id == [7]  # type: ignore[union-attr]
    padded_tok = _EotTokenizer()
    padded_tok.pad_token_id = 5
    list_loaded = _load_with_tokenizer_model(tmp_path, monkeypatch, padded_tok, _gen_model([3, 2], None))
    assert list_loaded._model.generation_config.eos_token_id == [2, 3, 7]  # type: ignore[union-attr]
    assert list_loaded._model.generation_config.pad_token_id == 5  # type: ignore[union-attr]
    bad_tok = _EotTokenizer()
    bad_tok.pad_token_id = "bad"
    bad_pad = _load_with_tokenizer_model(tmp_path, monkeypatch, bad_tok, _gen_model(2, None))
    assert bad_pad._model.generation_config.pad_token_id == 7  # type: ignore[union-attr]


def test_score_cache_same_instance_second_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stub:
        def score_candidates(self, name, cands):  # type: ignore[no-untyped-def]
            return tuple(0.5 for _ in cands)

    monkeypatch.setattr(Transliterator, "load", lambda *args, **kwargs: _Stub())
    weights = _weights_with_config(tmp_path)
    cache = CachedTransliterator(tmp_path / "cache", weights, [], MatchConfig())
    assert cache.score_candidates("가나", ["A"]) == (0.5,)
    assert cache.score_candidates("다라", ["B"]) == (0.5,)
    assert cache.score_candidates("가나", []) == ()


def test_score_cache_skips_bad_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    class _Stub:
        def __init__(self) -> None:
            self.calls = 0

        def score_candidates(self, name, cands):  # type: ignore[no-untyped-def]
            self.calls += 1
            return tuple(1.0 for _ in cands)

    stub = _Stub()
    monkeypatch.setattr(Transliterator, "load", lambda *args, **kwargs: stub)
    weights = _weights_with_config(tmp_path)
    CachedTransliterator(tmp_path / "cache", weights, [], MatchConfig()).score_candidates("가나", ["A"])
    cache_file = next((tmp_path / "cache").glob("*.scores.jsonl"))
    with open(cache_file, "ab") as handle:
        handle.write(b"\n")
        handle.write(_json.dumps({"name": "가나", "cands": ["B", "A"], "scores": [1.0, 2.0]}).encode() + b"\n")
        handle.write(_json.dumps({"name": "가나", "cands": "bad", "scores": [1.0]}).encode() + b"\n")
    fresh = CachedTransliterator(tmp_path / "cache", weights, [], MatchConfig())
    assert fresh.score_candidates("가나", ["A"]) == (1.0,)
    assert stub.calls == 1


def test_score_cache_appends_without_trailing_newline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stub:
        def score_candidates(self, name, cands):  # type: ignore[no-untyped-def]
            return tuple(2.0 for _ in cands)

    monkeypatch.setattr(Transliterator, "load", lambda *args, **kwargs: _Stub())
    weights = _weights_with_config(tmp_path)
    cache = CachedTransliterator(tmp_path / "cache", weights, [], MatchConfig())
    assert cache.score_candidates("가나", ["A"]) == (2.0,)
    cache_file = next((tmp_path / "cache").glob("*.scores.jsonl"))
    with open(cache_file, "ab") as handle:
        handle.truncate(handle.tell() - 1)
    assert cache.score_candidates("다라", ["B"]) == (2.0,)
    assert CachedTransliterator(tmp_path / "cache", weights, [], MatchConfig()).score_candidates(
        "다라", ["B"]
    ) == (2.0,)


def test_score_cache_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class _Stub:
        def score_candidates(self, name, cands):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return tuple(float(len(c)) for c in cands)

    monkeypatch.setattr(Transliterator, "load", lambda *args, **kwargs: _Stub())
    weights = _weights_with_config(tmp_path)
    first = CachedTransliterator(tmp_path / "cache", weights, [], MatchConfig())
    assert first.score_candidates("가나", ["B", "A", "B"]) == (1.0, 1.0, 1.0)
    second = CachedTransliterator(tmp_path / "cache", weights, [], MatchConfig())
    assert second.score_candidates("가나", ["A", "B"]) == (1.0, 1.0)
    assert calls["n"] == 1


def test_score_cache_ignores_corrupt_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stub:
        def score_candidates(self, name, cands):  # type: ignore[no-untyped-def]
            return tuple(0.5 for _ in cands)

    monkeypatch.setattr(Transliterator, "load", lambda *args, **kwargs: _Stub())
    weights = _weights_with_config(tmp_path)
    CachedTransliterator(tmp_path / "cache", weights, [], MatchConfig()).score_candidates("가나", ["A"])
    cache_file = next((tmp_path / "cache").glob("*.scores.jsonl"))
    with open(cache_file, "ab") as handle:
        handle.write(b'{"name":')
    fresh = CachedTransliterator(tmp_path / "cache", weights, [], MatchConfig())
    assert fresh.score_candidates("가나", ["A"]) == (0.5,)


def test_rerank_config_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="llm_rerank_top_k"):
        Transliterator.load(tmp_path, [], MatchConfig(llm_rerank_top_k=0))
    with pytest.raises(ValueError, match="llm_rerank_weight"):
        Transliterator.load(tmp_path, [], MatchConfig(llm_rerank_weight=-0.1))
