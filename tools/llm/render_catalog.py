"""Catalog-side precompute for the reverse channel: every distinct catalog vessel name -> clerk-style Hangul.

usage: uv run python tools/llm/render_catalog.py <weights_dir> <out_jsonl> [batch]

Greedy decoding, 12 few-shot pairs from train (MatchConfig.llm_fewshot / llm_fewshot_seed), resumable JSONL output
({"name", "hangul"}). Depends only on vessel_catalog.csv, train.csv few-shot pairs and public pretrained weights.
Files used by the rev2syn model (GPU, ~80 min each on an RTX-class 12 GB card):
  artifacts/cache/rev_midm_catalog.jsonl   <- artifacts/hf/Midm-2.0-Mini-Instruct, batch 48 (one resume at batch 16)
  artifacts/cache/rev_kanana_catalog.jsonl <- artifacts/hf/kanana-1.5-2.1b-instruct-2505, batch 48
Left-padded batches mean the batch size can change a few greedy outputs; the shipped files are the reference.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "src")
from port_entity_matching.config import MatchConfig  # noqa: E402
from port_entity_matching.transliterate import select_fewshot  # noqa: E402

DATA = "data/raw/open/"
SYSTEM = (
    "Korean port-call records write foreign ship names in Hangul as a Korean clerk would pronounce them "
    "(Korean loanword transliteration; romanized Korean/Chinese/Japanese words are written in their original reading). "
    "Keep numbers. Answer with the Hangul name only."
)


def main() -> None:
    weights, out_path = sys.argv[1], sys.argv[2]
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    train = pd.read_csv(DATA + "train.csv")
    catalog = pd.read_csv(DATA + "vessel_catalog.csv")
    config = MatchConfig()
    shots = select_fewshot(train, catalog, config.llm_fewshot, config.llm_fewshot_seed)
    messages = [{"role": "system", "content": SYSTEM}]
    for hangul, latin in shots:
        messages += [{"role": "user", "content": latin}, {"role": "assistant", "content": hangul}]
    names = sorted(set(catalog.vessel_name.dropna().astype(str)))
    done = set()
    try:
        with open(out_path, encoding="utf-8") as handle:
            done = {json.loads(line)["name"] for line in handle}
    except FileNotFoundError:
        pass
    names = [name for name in names if name not in done]
    tok = AutoTokenizer.from_pretrained(weights)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(weights, dtype=torch.bfloat16, device_map="cuda").eval()
    prompts = [
        tok.apply_chat_template(
            [*messages, {"role": "user", "content": name}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for name in names
    ]
    order = np.argsort([len(p) for p in prompts])
    started = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        for i in range(0, len(prompts), batch):
            idx = order[i : i + batch]
            enc = tok(
                [prompts[j] for j in idx],
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                add_special_tokens=False,
            ).to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **enc, max_new_tokens=32, do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id
                )
            texts = tok.batch_decode(generated[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
            for j, text in zip(idx, texts):
                out.write(json.dumps({"name": names[j], "hangul": text.strip().split("\n")[0]}, ensure_ascii=False) + "\n")
            if (i // batch) % 50 == 0:
                out.flush()
                print(f"[ALGO] rendered={i}/{len(prompts)} elapsed_s={time.time() - started:.0f}", flush=True)
    print(f"[ALGO] done rendered={len(prompts)} elapsed_s={time.time() - started:.0f}")


if __name__ == "__main__":
    main()
