# Code Style & Engineering Standards

> **Respect existing layout and conventions. Keep typed contracts at boundaries. The submission path must stay simple, self-contained, and Python-only.**

## 1. Architecture & Scope
- Follow the existing `src/`, `tests/`, `tools/` layout; keep changes cohesive and avoid out-of-scope refactors.
- Treat validated code and contracts as authoritative for current behavior; when code and docs/specs disagree, investigate rather than assume either is right.

## 2. Typing & Configuration
- Use strong static typing on public interfaces, data models, and core logic; do not weaken existing type guarantees without justification.
- Keep tunable settings (thresholds, paths, model names, seeds) in explicit config, not scattered literals. Do not hardcode machine-specific paths or secrets.
- Use the repository's declared tooling (`uv`, `ruff`, `mypy`, `pytest`); do not add competing tools without a reason.

## 3. Submission Boundary
- **Python only:** No other languages in the submitted package.
- **Thin entry point:** `script.py` stays small and delegates to importable, tested modules in `model/` (or the packaged library), so the same code is exercised by tests and by the evaluation server.
- **Pinned dependencies:** `requirements.txt` lists only what inference needs, with versions that install within 10 minutes on Python 3.11 / CUDA 12.8.
- **Reproducibility:** Fix seeds; avoid reading ambient environment, wall-clock, or machine-dependent state in prediction logic.
- Experimental code (notebooks, exploratory scripts) lives under `scratch/` and is not imported by the submission path.
