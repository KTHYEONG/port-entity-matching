# Operating Manual: Port-Entity-Matching

> **Domain Identity:** High-precision, deterministic entity resolution, normalization, and linkage pipeline for port/maritime entities and cross-source record matching.

## 1. Domain Ground Truth & Hard Boundaries
- **Deterministic & Pure Canonicalization:** Entity normalization, alias expansion, and attribute parsing must be deterministic.
- **Explainable Linkages & Confidence Scoring:** Every linkage must output an audit-ready match score, matched attribute evidence, and decision threshold.
- **Fail-Closed on Ambiguity:** Ambiguous candidate ties or low-confidence matches must be partitioned into review queues rather than guessed silently.
- **Workspace Hygiene:** Exploratory experiments, scratch data probes, and temporary artifacts remain strictly isolated under `scratch/` and `artifacts/`. Never commit raw proprietary entity catalogs, unmasked PII, or large binary caches.

## 2. Autonomy & Execution Contract
- **Bias Toward Action & Diagnostic Autonomy:** For data queries, exploratory scratch diagnostics, and empirical root-cause isolation, execute immediately without asking for permission. When investigating matching anomalies or edge-case entities, proactively run scratch experiments under `scratch/` to discover ground truth. Never modify production pipeline code or commit in response to open-ended diagnostic queries.
- **Skills as On-Demand Tools:** Skills (`probe`, `spec`, `implement`, `check`, `refactor`, `commit`) are modular, independent utilities—NOT a mandatory sequential conveyor belt. When explicitly invoked via slash commands (`/probe`, `/spec`, etc.), execute only that targeted skill and halt for user review. Specs live under `docs/specs/` (gitignored).

## 3. Project Toolchain & Verification
Verify code changes against the project's native toolchains before concluding tasks:
- **Quality Gate:** `uv run python tools/verify.py`
- **Test Runner:** `uv run pytest`
- **Git Commits:** Run the project's `commit` skill.

## 4. Communication & Language
- **Natural Korean:** Converse, explain rationales, and report findings in Korean (한국어). Inside structured output cards, retain English keys/badges while writing descriptions in Korean.
- **Technical English:** System instructions, rules, specifications (`docs/specs/`), code, and docstrings are written in English.

## 5. Domain Rule Routing
- **Domain Invariants:** [domain.md](.agents/rules/domain.md) — *Entity resolution invariants, deterministic canonicalization, sub-quadratic blocking, and dual precision-recall thresholds.*
- **Testing & Quality:** [testing.md](.agents/rules/testing.md) — *Invariant-driven testing, boundary conditions, failure isolation, and diff-coverage.*
- **Architecture & Standards:** [code-style.md](.agents/rules/code-style.md) — *Module boundaries, strong static typing contracts, and toolchain alignment.*
- **Documentation & Comments:** [documentation.md](.agents/rules/documentation.md) — *Production docstrings, architecture specs, and non-obvious rationale.*
- **Performance & Optimization:** [performance.md](.agents/rules/performance.md) — *Sub-quadratic indexing/blocking, vectorized parsing, and resource budgets.*
- **Logging & Diagnostics:** [logging.md](.agents/rules/logging.md) — *Operational logging, category taxonomy, and credential redaction.*
