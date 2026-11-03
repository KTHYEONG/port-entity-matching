---
paths:
  - "src/**"
---
# Performance & Resource Directives

> **Correctness first, then speed. The binding constraint is the evaluation server budget (`AGENTS.md` §2.5): 30 min inference, 26 GB RAM, L4 GPU 22.4 GiB, 6 vCPU, no internet. Never gain speed by silently dropping catalog entries, rows, or precision.**

## 1. Budget Discipline
- **Rehearse the budget:** Time the full `script.py` flow (load, preprocess, retrieve, rerank, save) on data sized like the real run, with a safety margin well under 30 minutes. Shorter inference time is also a tie-breaker.
- **Package size:** Keep model files and the ZIP small; ZIP size is the second tie-breaker. Prefer compact artifacts over bundling unused checkpoints.
- **Offline-safe:** Everything needed at inference is in the ZIP; no downloads at runtime.

## 2. Memory
- Keep peak memory comfortably below 26 GB (RAM) and 22 GiB (VRAM). Avoid unnecessary copies and full intermediate materializations; use batches.
- Release large buffers, file handles, and worker pools when done.

## 3. Algorithmic Efficiency
- Avoid per-row linear scans over the whole catalog. Precompute catalog-side indexes or features once, outside the per-row path (catalog features are row-independent and allowed).
- Avoid repeated parsing or deserialization inside hot loops; prefer vectorized/batched operations.
- Precompute only things that depend on the catalog or training data. Do not precompute anything across test rows.

## 4. Parallelism & Telemetry
- Use multiprocessing or GPU only where profiling shows a bottleneck and work units are large enough to amortize overhead; results must stay deterministic regardless of worker count.
- Long jobs (training, indexing, full inference) emit periodic progress lines.
