# Operating Manual: Port-Entity-Matching

> **Domain Identity:** Deterministic vessel entity-matching pipeline that links port arrival/departure records to a standard vessel catalog, built for the DACON port-challenge competition (§2.5).

## 1. Domain Ground Truth & Hard Boundaries
- **Deterministic & Pure Canonicalization:** Entity normalization, alias expansion, and attribute parsing must be deterministic.
- **Explainable Linkages (dev/analysis path):** Analysis tooling outputs match score, margin, and matched-attribute evidence for error analysis and the finals presentation. The submitted `script.py` emits only the required CSV.
- **Ambiguity:** The submission always emits a deterministic top-1 `vessel_id` per row (see §2.5); ambiguous or low-margin cases are surfaced in diagnostics, never skipped or silently randomized.
- **Workspace Hygiene:** Exploratory experiments, scratch data probes, and temporary artifacts remain strictly isolated under `scratch/` and `artifacts/`. Never commit raw proprietary entity catalogs, unmasked PII, or large binary caches.

## 2. Autonomy & Execution Contract
- **Bias Toward Action & Diagnostic Autonomy:** For data queries, exploratory scratch diagnostics, and empirical root-cause isolation, execute immediately without asking for permission. When investigating matching anomalies or edge-case entities, proactively run scratch experiments under `scratch/` to discover ground truth. Never modify production pipeline code or commit in response to open-ended diagnostic queries.
- **Skills as On-Demand Tools:** Skills (`probe`, `spec`, `implement`, `check`, `refactor`, `commit`) are modular, independent utilities—NOT a mandatory sequential conveyor belt. When explicitly invoked via slash commands (`/probe`, `/spec`, etc.), execute only that targeted skill and halt for user review. Specs live under `docs/specs/` (gitignored).

## 2.5 Competition Hard Rules (DACON "AI Technology Competition for Solving Port Challenges")
Source: https://dacon.io/competitions/official/236771 (organizer: MOF; host: Ulsan Port Authority). These rules take precedence over any conflicting rule below or in `.agents/rules/`. Violation can disqualify the team.
- **Task:** Input is one port-call record row; output is exactly one `vessel_id` from `vessel_catalog.csv`. Every test row must receive a prediction (no abstain, no review queue in the submitted output).
- **Metric:** `0.7 * accuracy(Korean vessel names) + 0.3 * accuracy(non-Korean vessel names)`; the split is by the record's vessel-name script. Public LB = 50% of test, private = 100%. Ties break by shorter inference time, then smaller submit ZIP.
- **Row independence (critical):** Predict each test row using only its own input plus `vessel_catalog.csv`/training artifacts. Never use other test rows or the overall test-file distribution (means, frequencies, ranks, group votes, cumulative features, post-hoc prediction adjustments). A row's prediction must be identical when run alone, in a subset, or in a shuffled file.
- **Data allowlist:** Train/validate only on the provided `train.csv` and `vessel_catalog.csv`. No external vessel DBs, external arrival/departure records, or external answer/mapping tables. Publicly available general pretrained weights are allowed; training/validating on outside labeled data is not.
- **Submission contract:** Python only. `submit.zip` root has `script.py` + `requirements.txt`; models live in `model/`. `script.py` reads `data/test.csv` and `data/vessel_catalog.csv` and writes `output/submission.csv` with columns `sample_id,target_vessel_id`. Never hardcode row counts or `sample_id`s.
- **Environment match:** Develop against the evaluation server's preinstalled stack (torch 2.7.1+cu128, pandas 2.0.3, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.5.2, lightgbm 4.6.0, catboost 1.2.8, rapidfuzz 3.14.3, Unidecode 1.4.0, transformers 4.57.6, etc.). Do not list or re-pin these in the submitted `requirements.txt`; add only packages inference truly needs. The zip must have no extra top-level folder, and `data/` is read-only.
- **Runtime budget:** Package install <= 10 min; inference <= 30 min total (load + preprocess + infer + save); ZIP <= 10 GB (32 GB unpacked). Server: NVIDIA L4 22.4 GiB, 6 vCPU, 26 GB RAM, Python 3.11, CUDA 12.8. No internet during inference, so no runtime downloads.
- **Reproducibility:** Code and models cannot change after the preliminary round; organizers re-run to verify. Outputs must be deterministic.
- **Finals rubric (100):** model performance 30, data understanding/preprocessing 20, model development process 25, error analysis and generalization to new vessel names/notations 15, field applicability 10. Keep data findings, experiment history, ablations, and error analyses recorded (see `documentation.md`), since 70 points depend on them.
- **Limits and schedule:** Max 5 submissions/day. Leaderboard closes 11.01, code verification 11.05 (2026).
- **Data handling:** Never log/record/leak/transmit private evaluation data or answers via logs, submission files, or any external channel. Do not attempt to restore original identifying information or redistribute competition data; use it only for this competition, and keep `data/` contents out of git.
- **Consent terms:** No manual labeling or human-predicted info on test/validation data. Do not share competition code or data privately with third parties (public sharing only via DACON). Use only original or properly licensed code, models, and tools. Full terms in `docs/architecture/competition.md`.

## 3. Project Toolchain & Verification
Verify code changes against the project's native toolchains before concluding tasks:
- **Quality Gate:** `uv run python tools/verify.py`
- **Test Runner:** `uv run pytest`
- **Git Commits:** Run the project's `commit` skill.

## 4. Communication & Language
- **Natural Korean:** Converse, explain rationales, and report findings in Korean (한국어). Inside structured output cards, retain English keys/badges while writing descriptions in Korean.
- **Technical English:** System instructions, rules, specifications (`docs/specs/`), code, and docstrings are written in English.

## 5. Domain Rule Routing
- **Competition Reference:** [competition.md](docs/architecture/competition.md) — *Full rules, schedule, submission contract, and data description.*
- **Domain Invariants:** [domain.md](.agents/rules/domain.md) — *Vessel matching task model, canonicalization, ambiguity handling, leakage-safe validation, and retrieval scale.*
- **Testing & Quality:** [testing.md](.agents/rules/testing.md) — *Invariant-driven testing, mandatory row-independence and submission-contract tests, boundaries.*
- **Architecture & Standards:** [code-style.md](.agents/rules/code-style.md) — *Module boundaries, typing, and the Python-only submission boundary.*
- **Documentation & Comments:** [documentation.md](.agents/rules/documentation.md) — *Docstrings, architecture docs, and competition documentation.*
- **Performance & Optimization:** [performance.md](.agents/rules/performance.md) — *30-minute inference budget, memory limits, and efficient candidate retrieval.*
- **Logging & Diagnostics:** [logging.md](.agents/rules/logging.md) — *Category taxonomy, structured logs, and no evaluation-data logging.*
