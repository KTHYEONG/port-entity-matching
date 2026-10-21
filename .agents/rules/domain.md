# Vessel Matching Domain Directives

> **Goal: maximize the competition score (see `AGENTS.md` §2.5) with a deterministic, row-independent matcher. Experiment freely via `tools/evaluate.py`; keep `scratch/` for one-off probes only, and keep the submitted path reproducible and compliant.**

## 1. Task Model
- **Problem shape:** One port-call record row in, one `vessel_id` out, chosen from `vessel_catalog.csv` (~78k rows). Every row gets a prediction; there is no abstain in submitted output.
- **Row independence:** Production inference may use a row's own fields, the catalog, and artifacts fitted on `train.csv`/catalog. It must not use statistics or votes across test rows.
- **Metric alignment:** Optimize `0.7 * acc_korean + 0.3 * acc_non_korean` (split by the record's vessel-name script). Always report both buckets, not only the blended score.

## 2. Canonicalization
- **Deterministic & pure:** Normalization (Unicode NFC/NFKC, case folding, whitespace/punctuation handling, Hangul/romanization mapping, alias expansion) is a pure function with no hidden state or randomness.
- **Missing/dirty fields:** Treat missing specs as "unknown", not as a mismatch signal; typos and spelling variants are expected inputs.

## 3. Ambiguity Handling
- **Submission path:** Always emit the top-1 candidate; break exact score ties by a fixed, documented rule (e.g. stable catalog order) so results stay deterministic.
- **Diagnostics path:** Record top-k scores, margins, and matched-attribute evidence in dev/analysis tooling (`artifacts/eval/`, produced by `tools/evaluate.py`) to find error clusters and to support the finals presentation. Low-margin cases are an error-analysis target, not a reason to skip rows.

## 4. Validation & Leakage
- **Honest validation:** Hold out part of `train.csv` (group-aware where the same vessel recurs) and fit vocabularies, TF-IDF, embeddings, and thresholds on the training split only. Only labels from `train.csv` and `vessel_catalog.csv` may be used (no external vessel data).
- **Catalog caveat:** The hidden test may contain vessels or spellings absent from `train.csv`; do not rely on train-only vessel memorization as the sole signal. Check generalization on held-out vessels.
- **Determinism:** Same input and same artifacts give the same prediction on every run. Fix seeds and avoid nondeterministic ops in `script.py`.

## 5. Scale
- **Candidate generation:** Avoid brute-force all-pairs scoring on the full catalog per row when a blocking key, n-gram/inverted index, or ANN retrieval gives equal recall. Measure recall@k of the retrieval stage separately from final accuracy.
- **Batch processing:** Prefer vectorized or batched operations; the whole submission must fit the inference budget in `AGENTS.md` §2.5.

## 6. Tests
- **Minimal deterministic cases:** exact match, spelling variant, Hangul vs. romanized name, word-order or spacing change, missing attributes, and a near-duplicate lookalike vessel.
- **Contract tests:** row-independence (single row = full file = shuffled file) and submission format are mandatory; see `testing.md`.
