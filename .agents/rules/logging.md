---
paths:
  - "src/**"
---
# Logging & Diagnostic Directives

> **Logs are concise, structured, and diagnosable. Never suppress tracebacks, never log evaluation data contents, and never delete diagnostic evidence unilaterally.**

## 1. Integrity
- Keep full tracebacks and root causes for `ERROR`/`CRITICAL`.
- Keep enough numeric precision to diagnose (scores, margins, thresholds); do not round small non-zero values to zero.
- Include a stage or run identifier when a multi-stage run makes attribution ambiguous.
- **No sensitive content:** Never log credentials or tokens. In `script.py` (the evaluation path), do not write test row contents or predictions to logs; log only counts, timings, and shapes.

## 2. Format
- Prefix operational pipeline messages with one category tag: `logger.info("[<CATEGORY>] <event>")`.
  - `[SYS]`: runtime, config, environment, resource usage, timing.
  - `[DATA]`: loading, normalization, validation, splits.
  - `[ALGO]`: retrieval, features, scoring, reranking, training.
  - `[EVAL]`: validation metrics, error analysis, budget rehearsals, submission checks.
- Prefer key-value fields over prose. Summarize collections (count, shape, min/max, nulls, short head sample) rather than dumping them.
- Legacy untagged logs need not be refactored unless the surrounding code is being modified.

## 3. Levels
- `INFO`: one line per phase or milestone. `DEBUG`: fine-grained diagnostics, per-candidate scores, trial traces.
- Keep high-volume diagnostics (per-row candidates, hyperparameter trials) in files under `artifacts/eval/` (persistent run outputs) or `scratch/` (temporary), not the main log.

## 4. Lifecycle
- Persistent logs and evaluation run outputs go to `artifacts/eval/`; temporary diagnostics go under `scratch/`.
- Do not delete logs or evidence needed for an open investigation. Routine cleanup of temporary files via project scripts is fine.
