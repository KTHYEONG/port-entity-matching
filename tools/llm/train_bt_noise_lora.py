"""Noise-robust back-translation LoRA for the Hangul -> Latin reader (the rev2syn model's LLM).

usage: uv run --with peft==0.17.1 python tools/llm/train_bt_noise_lora.py <out_dir> [exclude_train_targets 0|1]
       [epochs] [max_pairs]
rev2syn weights: artifacts/hf/kanana_bt_noise <- `... artifacts/hf/kanana_bt_noise 1 1.0` (~73 min, 12 GB GPU), then
"lora_variant": "bt_noise_r16_e1" added to its config.json so the transliteration cache fingerprint differs from other
merges of the same base.

Pairs are (Midm catalog rendering, catalog name) from artifacts/cache/rev_midm_catalog.jsonl, excluding train target
vessels; half the Hangul inputs get 1-2 Latin-train-style edits (syllable delete / duplicate / adjacent swap, jamo
substitution). Base: artifacts/hf/kanana-1.5-2.1b-instruct-2505. Loss only on the answer; the merged bf16 model is saved.
GPU training is not bit-reproducible; the shipped weights are the reference.
"""

import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from port_entity_matching.transliterate import SYSTEM_PROMPT, end_of_turn_token_id  # noqa: E402

BASE = "artifacts/hf/kanana-1.5-2.1b-instruct-2505"
RENDERINGS = "artifacts/cache/rev_midm_catalog.jsonl"
S0 = 0xAC00


def _syllables(s):
    return [i for i, c in enumerate(s) if "가" <= c <= "힣"]


def syl_op(s, op, rng):
    p = _syllables(s)
    if len(p) < 3:
        return s
    i = p[int(rng.integers(1, len(p)))]
    if op == "del":
        return s[:i] + s[i + 1 :]
    if op == "dup":
        return s[:i] + s[i] + s[i:]
    j = i - 1
    if not ("가" <= s[j] <= "힣"):
        return s
    return s[:j] + s[i] + s[j] + s[i + 1 :]


def jamo_sub(s, rng):
    p = _syllables(s)
    i = p[int(rng.integers(len(p)))]
    c = ord(s[i]) - S0
    cho, jung, jong = c // 588, (c % 588) // 28, c % 28
    part = rng.integers(3)
    if part == 0:
        cho = int(rng.integers(19))
    elif part == 1:
        jung = int(rng.integers(21))
    else:
        jong = int(rng.integers(28))
    return s[:i] + chr(S0 + cho * 588 + jung * 28 + jong) + s[i + 1 :]


def main() -> None:
    out_dir = sys.argv[1]
    exclude = sys.argv[2] == "1" if len(sys.argv) > 2 else True
    epochs = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    max_pairs = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
    train = pd.read_csv("data/raw/open/train.csv")
    catalog = pd.read_csv("data/raw/open/vessel_catalog.csv")
    skip = (
        set(catalog.set_index("vessel_id").loc[sorted(set(train.target_vessel_id))].vessel_name.astype(str))
        if exclude
        else set()
    )
    pairs = []
    with open(RENDERINGS, encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry["name"] in skip or not any("가" <= c <= "힣" for c in entry["hangul"]):
                continue
            pairs.append((entry["hangul"], entry["name"]))
    random.Random(0).shuffle(pairs)
    pairs = pairs[:max_pairs]
    noise_rng = np.random.default_rng(0)

    def noisy(h):
        for _ in range(1 if noise_rng.random() < 0.7 else 2):
            op = noise_rng.integers(4)
            h = jamo_sub(h, noise_rng) if op == 3 else syl_op(h, ("del", "dup", "swap")[op], noise_rng)
        return h

    pairs = [(noisy(h) if noise_rng.random() < 0.5 else h, latin) for h, latin in pairs]
    print(f"[DATA] pairs={len(pairs)} excluded_names={len(skip)}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    eot = end_of_turn_token_id(tok)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(
        model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM")
    )

    def encode(hangul, latin):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": hangul}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok(prompt, add_special_tokens=False).input_ids
        answer_ids = tok(latin, add_special_tokens=False).input_ids + [eot]
        return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    batch, accumulate = 8, 4
    steps = int(len(pairs) * epochs / (batch * accumulate))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / max(steps, 1))
    )
    model.train()
    started = time.time()
    pad = tok.pad_token_id if tok.pad_token_id is not None else eot
    for step in range(steps):
        for a in range(accumulate):
            base = (step * accumulate + a) * batch
            items = [encode(*pairs[(base + j) % len(pairs)]) for j in range(batch)]
            length = max(len(x) for x, _ in items)
            ids = torch.tensor([x + [pad] * (length - len(x)) for x, _ in items]).cuda()
            labels = torch.tensor([y + [-100] * (length - len(y)) for _, y in items]).cuda()
            mask = torch.tensor([[1] * len(x) + [0] * (length - len(x)) for x, _ in items]).cuda()
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss / accumulate
            loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"[ALGO] step={step}/{steps} loss={loss.item() * accumulate:.4f} elapsed_s={time.time() - started:.0f}", flush=True)
    model = model.merge_and_unload()
    model.eval()
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"[SYS] saved path={out_dir} elapsed_s={time.time() - started:.0f}")


if __name__ == "__main__":
    main()
