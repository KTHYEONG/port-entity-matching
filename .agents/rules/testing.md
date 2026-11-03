---
paths:
  - "tests/**"
  - "tools/**"
---
# Testing Directives & Quality Standards

> **Verify observable behavior and interface contracts, not internals. Keep tests small, deterministic, and fast. Competition-critical contracts are always tested.**

## 1. Paradigm: Invariant-Driven Development
- **Contracts first:** Prefer strict typing, explicit schemas, and domain invariants over test-first ritual. Define interface and invariants before code; avoid tests that freeze internal APIs.
- **Proportional process:** Bug fixes, localized changes, and probes: implement directly with targeted guard tests. Complex multi-component work: write a spec under `docs/specs/` first.
- **Behavior over internals:** Assert return values, outputs, and error conditions, not private details or mock call counts.
- **Minimal inputs:** Use tiny in-memory fixtures. Unit tests never load the full `data/` files.

## 2. Mandatory Competition Contract Tests
- **Row independence:** Predictions for a row are identical when run alone, within a subset, and within a shuffled file.
- **Determinism:** Two runs on the same input give identical output.
- **Submission format:** Output has exactly `sample_id,target_vessel_id`, covers every input `sample_id` in the input's order, and every `target_vessel_id` exists in the catalog.
- **Entry point:** `script.py` reads `data/test.csv` and `data/vessel_catalog.csv` and writes `output/submission.csv`, with no hardcoded row counts or ids and no network access.

## 3. Boundaries
- Cover empty values, NaN, missing columns' values, whitespace-only or very long names, mixed Hangul/Latin text, Unicode normalization variants, and duplicate catalog names.
- Use explicit tolerances (`rtol`/`atol`) for float comparisons.

## 4. Execution & Triage
- **Speed:** Default test runs are near-instant. Mark heavy runs (full-catalog inference, training, timing rehearsal of the 30-minute budget) with a marker and exclude them by default.
- **Focus:** Run targeted tests while iterating; run the full suite and `uv run python tools/verify.py` before concluding.
- **Isolation:** Use per-run temp directories (`tmp_path`); never wipe shared temp dirs. Tests are hermetic and read no ambient credentials or environment.
- **Instrumentation:** If a test fails only under coverage or profiling, check tracer overhead, timeouts, or fork interference before suspecting domain code.
- **Mocking:** Mock external boundaries (network, clock, filesystem) only. Do not mock domain calculations.

## 5. Quality
- Focus coverage on new logic and boundaries; prune dead defensive branches instead of writing vacuous tests.
- Never weaken assertions, delete valid tests, or skip failures to pass checks. Fix the root cause.
