# Data Analysis & Design Rationale

Stable summary of data findings and the decisions they drive, organized by the finals rubric. Numbers come from `train.csv` (12,222 rows) and `vessel_catalog.csv` (78,595 rows). The EDA numbers were produced by one-off exploratory scripts, which have been retired; the figures of the current model are reproducible with `tools/evaluate.py`.

## 1. Data Understanding (rubric: data understanding & preprocessing)

### 1.1 Schema
| File | Key columns | Notes |
|---|---|---|
| train/test | `vessel_name`, `call_sign`, `gross_tonnage`, `international_gross_tonnage`, `flag_name` (Korean), `vessel_type` (Korean, 19 values), `arrival_year` (2015–2026) | test has no `target_vessel_id` |
| catalog | `vessel_id`, `vessel_name` (Latin, title case), `vessel_type` (English, 138 values), `flag_name` (English, 195 values), `gross_tonnage`, `deadweight_tonnage`, `size`, `unit` | `vessel_id` unique; names nearly unique (311 normalized-duplicate groups) |

### 1.2 Metric leverage
- Hangul-name rows: 9.7% of train (1,191 rows, 382 vessels) but 70% of the score. One Korean test row is worth ~20x a non-Korean row.
- No mixed-script names; script detection = "contains a Hangul syllable".

### 1.3 Name noise
- **Latin:** 64% exact after normalization; the rest differ by Levenshtein 1–2. Edit ops (catalog → record): deletion 1,403, duplicated-char insertion 706, other insertion 591, substitution 7; ~11% involve a transposition. Noise is per vessel (the same typo repeats in every row of a vessel).
- **Formatting:** hyphens instead of spaces, `M/V ` prefixes, Unicode Roman numerals (`Ⅱ`), NBSP in 18 catalog names. All are handled by NFKC and punctuation stripping.
- **Korean names** are Hangul renderings of the catalog name:
  - English loanwords: 리플렉션 → Reflection
  - Korean romanization: 흥아 → Heung-A
  - Chinese/Japanese names: 닝보 → Ningbo, 카시마 마루 → Kashima Maru
  - Spelled-out letters: 케이엠티씨 → KMTC, 에스티아이 → STI
  - Numbering: 제2 → No. 2, trailing 호 dropped, 투 → II
  - Token counts: 253/382 are 2→2 tokens; Hangul often concatenates tokens (샤이닝로드 → Shining Road).
- **Renames:** ~0.2% of Latin vessels carry a former name (BW WREN → catalog HAFNIA WREN, FAIRCHEM SUCCESS → BOW SUCCESS). Tonnage still matches exactly.

### 1.4 Tonnage noise
- Record/catalog ratio is either exactly 1 (~48% of Latin rows; 71.5% of Korean rows on IGT) or concentrated in two bands, 0.95–0.97 and 1.03–1.05: a synthetic ±3–5% perturbation plus a small natural ±1% component. The perturbation is per vessel (95.5% of vessels have a constant GT across rows).
- Korean records: domestic `gross_tonnage` is often smaller (ratio 0.5–0.9); `international_gross_tonnage` is the better key.
- Tonnage is missing in 7.2% of Latin rows and 0% of Korean rows.
- An exact IGT match leaves a median of 4 catalog candidates. A ±5% window holds ~1,160, so names must discriminate.

### 1.5 Other attributes
- Flag: 17% missing (Latin), 10% (Korean); even when present it agrees with the majority Korean→English mapping only 77% of the time. Weak signal.
- Vessel type: the top-3 catalog types per record type cover 85–100%, so it is a useful compatibility feature.
- Sister ships: ~6,800 catalog groups share the first name token and the same GT, so digits and fine spelling decide those.

### 1.6 Split structure and call sign
- `call_sign` is a salted hash, exactly 1:1 with `target_vessel_id` in train. It stays constant across renames (BW WREN / HAFNIA WREN) and across scripts (마린가스 / MARINE GAS).
- Rows per vessel: 1 (3,173 vessels), 2 (628), 3 (407), 4 (1,643). A random row-level split would thin 4-row vessels into many 3-row vessels, which is not observed. Train vessels therefore appear to keep all their rows, i.e. a vessel-level split. Overlap with test is unknown.

## 2. Model Development (rubric: model development process)

| Experiment | Korean acc | Non-Korean acc | Score | Notes |
|---|---|---|---|---|
| Official RF baseline, 5-fold vessel-disjoint GroupKFold | 0.252 | 0.982 | 0.471 | Korean recall@40 is only 0.686 (tonnage-only candidates) |
| Rule-based jamo phonetic key + tonnage window (no learning) | 0.968 | — | — | 0.986 when tonnage is exact, 0.923 when perturbed |
| Latin full-catalog Damerau cdist + tonnage bonus (no learning) | — | 0.9987 | — | residual errors are mostly renames |
| Matcher v1 Part 3: LightGBM lambdarank over 39 pair features, 5-fold vessel-disjoint CV | 0.9798 | 0.9993 | 0.9857 | retrieval recall 1.000 in both buckets |
| Matcher v2 Part 3: 46 pair features, ±10% Korean tonnage window, degraded-copy augmentation, 5-seed bag | 0.9824 | 0.9995 | 0.9875 | see §2.6 for the reconciliation with the prototype numbers |

Caveat: the Korean rules were written after inspecting train names, so 0.968 is optimistic. Learned components must be measured in vessel-disjoint folds.

### 2.1 Retrieval Stage (Matcher v1 Part 2)
Candidate generation (name similarity, tonnage, rare-token, call-sign memory) was measured separately from ranking, on 400 sampled train rows per script bucket (a one-off script, since retired). These are in-sample diagnostics: memory was fitted on all train labels, including the sampled rows, so total recall includes a memorization oracle and is not held-out retrieval performance. Held-out retrieval recall of the current model is reported by `tools/evaluate.py cv` in the `recall` field of `summary.json`.

| Bucket | Recall@400 | Name source | Tonnage source | Rare-token source | Mean candidates |
|---|---|---|---|---|---|
| Korean | 1.000 | 0.830 | 0.995 | 0.000 | 234 |
| Non-Korean | 1.000 | 1.000 | 0.475 | 0.613 | 34 |

The 1.000 total recall is expected with in-sample memory; it does not establish ranking headroom on unseen vessels. Source coverage suggests Korean rows depend more on the tonnage window than on the phonetic key. Non-Korean names have high sampled name-source coverage, while rare tokens provide an additional path for renames. Measure memory-off recall and vessel-disjoint validation before drawing generalization conclusions.

Runtime on the full catalog (78,595 rows): index build 1.5 s, candidates for all 12,222 train rows 90 s, peak RSS 0.25 GB — far inside the 30-minute inference budget.

### 2.2 Learned Ranker (Matcher v1 Part 3)
A LightGBM lambdarank model scores each (record, candidate) pair over 39 features: script-gated Latin string similarities (Damerau/Levenshtein/Jaro-Winkler, token-set, partial, prefix), Hangul jamo-phonetic similarities (consonant/vowel keys, length ratio, decoded acronym hit), tonnage agreement (log ratio, exact, ±2.5–5.5% band, ±1% near), digit agreement, smoothed vessel-type/flag compatibility tables, and row-local context (primary rank and gap, exact-match count, candidate count, retrieval source flags). Missing values stay NaN so LightGBM can split on them; all context features are computed inside a single record's candidate set, so a row's features never depend on other rows.

Initial implementation, vessel-disjoint 5-fold CV (`GroupKFold` on `target_vessel_id`, call-sign memory disabled because a disjoint fold cannot measure it):

| Metric | Value |
|---|---|
| Korean accuracy | 0.9798 (1,191 rows) |
| Non-Korean accuracy | 0.9993 (11,031 rows) |
| Weighted score (0.7/0.3) | 0.9857 |
| Retrieval recall (both buckets) | 1.000 |

Per-fold Korean accuracy ranges 0.963–1.000, so the result does not depend on a single fold. Out-of-fold diagnostics (score, runner-up, margin, retrieval flag) are written to `artifacts/cv_oof.csv` for error analysis. The fitted artifacts are small: `model.txt` 0.8 MB, `maps.json` 21 kB, `memory.json` 263 kB, which keeps the submission ZIP far below the size limit.

### 2.3 Inference Pipeline and Submission Packaging
The inference path (`src/port_entity_matching/pipeline.py`) binds a `CatalogIndex` to the fitted `ArtifactBundle` and evaluates records in fixed-size chunks: candidate retrieval → pair features → LightGBM scoring → top-1 selection. A record's candidate set, features and score depend only on that record, so a row's prediction is identical whether it is scored alone, inside a subset or in a shuffled file; chunk size only bounds memory. The call-sign memory is applied after ranking, so a memorized vessel wins regardless of the ranker score.

`submission/script.py` is a thin wrapper around `run_submission`: it reads `data/test.csv` and `data/vessel_catalog.csv` relative to the working directory and writes `output/submission.csv` with exactly `sample_id,target_vessel_id` in input order. `tools/build_submit.py` assembles the archive without a top-level folder (`script.py`, `requirements.txt`, `model/port_entity_matching/*.py`, `model/artifacts/*`), with sorted entries and a fixed timestamp so repeated builds are byte-identical; `requirements.txt` stays empty because every runtime package is preinstalled on the evaluation server. The built ZIP is 0.4 MB with 20 entries, and the logged peak RSS stays far below the 26 GB server limit.

### 2.4 Evidence Review: Ablations, Sensitivity and Stress Tests
Every design choice was re-tested under the same vessel-disjoint 5-fold CV (call-sign memory off) with a cached-feature harness in which experiments E0–E31 ran. That harness has been retired; its results are recorded here, and the scripts and logs are kept outside version control. The current model's CV, CIs, stress tests and paired comparisons are reproduced with:
- `tools/evaluate.py cv --out-dir artifacts/eval/<run>`
- `tools/evaluate.py stress --out-dir ...`
- `tools/evaluate.py compare A/oof.csv B/oof.csv`

The harness reproduces the official CV exactly (0.9857, 24 Korean + 8 non-Korean errors); an earlier `artifacts/cv_oof.csv` with 33 errors came from an older code version and was replaced. Deltas carry a 95% vessel-cluster bootstrap CI.

**Measurement noise.**
- Vessel bootstrap 95% CI: Korean 0.966–0.991, score 0.976–0.994. 382 Korean vessels make one vessel worth ~0.25 pp of Korean accuracy.
- Re-drawing the vessel folds (3 seeds) gives 0.9863–0.9875.
- Changing only the LightGBM seed moves Korean errors 23–29 (mean 25.2 ± 2.1 over 5 seeds).
- Consequence: differences below ~4 Korean errors are not interpretable from a single run.

**What carries the signal (information-type ablation, all redundant copies removed):**

| Removed information | Korean acc | Score | Δ score [95% CI] |
|---|---|---|---|
| none (full model) | 0.9798 | 0.9857 | — |
| Hangul phonetic similarity (kor_* + primary rank/gap) | 0.8262 | 0.8782 | −0.108 [−0.135, −0.080] |
| all tonnage information | 0.9076 | 0.9351 | −0.051 [−0.071, −0.032] |
| type/flag maps, digits, rare token, source flags, Latin sims (each alone) | 0.976–0.984 | 0.983–0.989 | within ±0.003, CIs include 0 |

Single-group ablations understate importance because the row-context features (`primary_rank`, `primary_gap`) duplicate the name similarity. Removing digit features appeared to help (+0.003), but every flipped row had no digits on either side, so the change is model variance, not a digit effect.

**Is a learned ranker needed?** (same candidates, same folds)

| Scorer | Korean acc | Score |
|---|---|---|
| Hand formula (prototype weights) | 0.9740 | 0.9814 |
| Logistic regression | 0.9731 | 0.9809 |
| LightGBM binary | 0.9790 | 0.9852 |
| LightGBM lambdarank | 0.9798 | 0.9857 |

LightGBM vs hand formula: 9 vessels fixed, 6 broken; Δ +0.004 [−0.005, +0.014]. The learned ranker is at least as good but its advantage is not statistically established on this data. Its practical value is combining NaN-aware signals without hand-tuned weights.

**Hangul rules** (Korean rows rebuilt with one rule removed or replaced):

| Variant | Korean errors | Δ score [95% CI] |
|---|---|---|
| Full rules | 24 | — |
| Plain Revised Romanization instead of sound classes | 63 | −0.023 [−0.038, −0.010] |
| No NG→N rewrite | 35 | −0.006 [−0.016, +0.002] |
| No letter-acronym decoding (touches 55/382 vessels) | 34 | −0.006 [−0.015, +0.001] |
| No 제N→NO N (8 vessels) | 27 | −0.002 |
| No N호 rule / no trailing-호 variant | 23 / 23 | +0.001 (noise) |
| ㅎ/H kept as `H` instead of dropped | 20 | +0.002 [−0.001, +0.007] |

The shared sound-class key space is the only design choice with a CI excluding zero.

**Candidate budget.**
- Korean: the target lies inside the ±6% tonnage window for 99.24% of rows (100% at ±10%). Inside the window, ranking by phonetic score puts it in the top 10 for 98.9% of rows. The name source rescues the 9 out-of-window rows.
- Latin: the name source alone gives recall@30 99.92%; exact tonnage (7 rows) and rare tokens (2 rows) rescue the rest.
- End-to-end, mean Korean candidates 35 / 65 / 124 / 233 / 452 all score 0.984–0.988 with recall ≥ 0.9966. Over 5 seeds, the tiny budget (35 Korean / 10 Latin candidates) scores 0.9861 vs 0.9850 for the default. The budget is not a limiting factor above ~35 candidates.

**Hyper-parameters** (5 seeds each):

| Setting | Korean errors | Mean score |
|---|---|---|
| Default (31 leaves) | 25.2 ± 2.1 | 0.9850 |
| 15 leaves | 22.4 ± 0.5 | 0.9867 |
| feature_fraction 1.0 | 24.8 ± 1.8 | 0.9852 |
| 5-seed averaged ensemble | 23 | 0.9863 |

Other settings: 200, 400 and 800 rounds are identical; 100 rounds is worse. Per-script models (+0.0006) do not differ from the shared model.

**Stress tests** (model trained on clean folds; held-out rows deliberately degraded):

| Perturbation | Recall | Accuracy |
|---|---|---|
| Korean tonnage ×1.08 | 0.845 | 0.980 → 0.801 |
| Korean tonnage ×1.12 | 0.838 | → 0.677 |
| Korean tonnage missing | 0.828 | → 0.682 |
| Korean name: one syllable deleted | 0.997 | → 0.794 |
| Korean spaces removed | 1.000 | → 0.971 |
| Korean flag + type missing | 1.000 | → 0.970 |
| Latin: one extra typo | 0.9998 | 0.9993 → 0.9915 |
| Latin tonnage missing | 0.9992 | → 0.9981 |

Korean accuracy depends on tonnage reaching the window and on the phonetic key being nearly intact. In train these hold (tonnage 100% within ±10%, never missing), so CV reflects the expected test distribution, but there is little margin outside it.

**Call-sign memory.**
- 0 conflicts in 12,222 rows; the call sign is stable across all 6 renames and script switches.
- Temporal holdout: memory fitted on earlier rows predicts the latest row of each multi-row vessel correctly for 2,678/2,678 rows. The ranker alone scores 0.977 (Korean) on the same rows.
- Expected score with memory: 0.9857 at 0% vessel overlap with test, 0.9900 at 30%, 0.9928 at 50%. It cannot lower the score unless a test call sign maps to a different vessel, which train never shows.

### 2.5 Second Review: Headroom, Feature Engineering, Model Families, Robustness
Same harness and CV as §2.4. Every comparison uses 5-seed averaged ensembles (z-scored LightGBM scores) so that seed noise (§2.4) does not decide the verdict. Baseline for this section: current production features with a 5-seed bag, 23 Korean / 7 non-Korean errors, score 0.9863.

**Headroom (where the remaining errors can be fixed).**
- Oracle over the ranked candidates: the Korean target is in the top 2 for 99.58% of rows and in the top 5 for 100%. Retrieval is not the bottleneck on clean data.
- Of the 23 Korean errors: 15 have a name signal that prefers the wrong vessel, 4 are worse on both name and tonnage, 2 are ranker mistakes (the correct vessel wins on the raw signals), 1 is indistinguishable (Shandong Fu Hui vs Fu Yi, same tonnage) and 1 is a name-vs-tonnage trade-off.
- Consequence: the lever is the name signal (features), not the model family or the candidate budget.
- Slice audit: no accuracy drift by build year, vessel type or flag. Errors are per vessel (15 of 19 error vessels have all their rows wrong). The 214 catalog rows without a gross tonnage are never train targets.

**Feature engineering (Korean name signal).**
A second phonetic key (key v2) adds three Korean-loanword rules to the shared sound-class key:
- non-rhotic R: R after a vowel and before a consonant or word end is silent (Marsgracht → 마스그라치);
- onset H: H before a vowel is kept on both sides, while coda H is dropped;
- Y as the vowel I when not followed by a vowel.

It contributes 7 features: consonant- and vowel-key similarity, Jaro-Winkler, partial ratio, Indel similarity, and the in-row rank/gap of the v2 primary score.

| Variant (5-seed bag) | Korean errors | Score | Δ vs baseline [95% CI] | Vessels fixed / broken |
|---|---|---|---|---|
| Baseline (39 features) | 23 | 0.9863 | — | — |
| + key v2 similarities only | 21 | 0.9875 | +0.0012 [−0.0057, +0.0075] | 5 / 2 |
| + key v2, all 7 features | 14 | 0.9916 | **+0.0053 [+0.0011, +0.0109]** | 8 / 1 |
| + same 7 metrics on the old key (no new rules) | 27 | 0.9840 | −0.0023 [−0.0074, +0.0001] | — |
| + token alignment (mean/min/cover/count diff) | 24 | 0.9857 | −0.0006 [−0.0066, +0.0039] | 3 / 1 |
| + fold-fitted Hangul→Latin token lexicon | 23 | 0.9863 | +0.0001 [−0.0062, +0.0050] | 6 / 2 |

- The gain comes from the rules, not from adding distance metrics: the same metrics on the old key are neutral to negative.
- Per-rule ablation:
  - Name-only rank-1 over the full catalog rises from 0.630 (old key) to 0.682 (264/382 vessels). The largest single contributor is Y-as-vowel, then onset H, then non-rhotic R.
  - End to end, dropping onset H removes the whole gain (23 errors) and dropping non-rhotic R leaves 19 errors.
  - A pinyin X→S rule is neutral and is dropped.
- Without tonnage (name only), Korean accuracy rises from 0.681 to 0.735, so the rules help most where tonnage cannot rescue the name.
- Final check, without the pinyin rule: +0.0047 [+0.0006, +0.0101], 15 Korean errors.

**Model families and ensembles** (all with key v2 features; error overlap = Jaccard of error-row sets):

| Ranker | Korean errors | Score |
|---|---|---|
| LightGBM lambdarank, 5 seeds | 14 | 0.9916 |
| CatBoost YetiRank (1 seed) | 13 | 0.9922 |
| CatBoost PairLogit (1 seed) | 15 | 0.9910 |
| LightGBM binary, 5 seeds | 19 | 0.9887 |
| sklearn HistGradientBoosting | 25 | 0.9851 |
| LightGBM rank_xendcg, 5 seeds | 41 | 0.9756 |

- The strong rankers make the same mistakes (Jaccard 0.74–0.86; lambdarank and YetiRank share 17 of 20 error rows).
- Fixed equal-weight blends (lambdarank + YetiRank, + binary, + PairLogit, all six) give Δ between −0.0024 and 0.0000 vs the lambdarank bag.
- This matches the headroom analysis: the residual errors are name-signal failures that no ranker can resolve. CatBoost's one-error lead is within seed noise and would add a second runtime library.
- Tree size: with key v2, 15 vs 31 leaves is Δ +0.0001. The earlier 15-leaf gain (§2.4) does not survive the better features.
- Seed averaging still pays: single seeds give 19/19/15/15/16 Korean errors (16.8 ± 1.8), while the 5-seed bag gives 15.

**Robustness to degraded inputs** (the §2.4 weakness).
Held-out Korean rows are degraded and the model is trained on the other folds. Augmentation adds degraded copies of training-fold Korean rows only: tonnage ×1.08, tonnage missing, one syllable deleted. The fold maps are refitted per fold, and the vessels stay disjoint from validation.

| Variant | Clean Korean errors (score) | Tonnage ×1.08 | Tonnage missing | Syllable deleted | *Unseen:* tonnage ×0.90 | *Unseen:* tonnage ×1.12 | *Unseen:* vowel swapped | *Unseen:* syllable inserted |
|---|---|---|---|---|---|---|---|---|
| A: old-key retrieval, ±6% window | 14 (0.9916) | 0.819 | 0.735 | 0.788 | 0.697 | 0.731 | 0.980 | 0.961 |
| B: key v2 retrieval, ±6% | 16 (0.9904) | 0.861 | 0.779 | 0.788 | — | — | — | — |
| C: key v2 retrieval, ±10% | 17 (0.9898) | 0.926 | 0.783 | 0.796 | 0.744 | 0.781 | 0.981 | 0.964 |
| D: key v2 retrieval, ±6% + augmentation | 16 (0.9904) | 0.890 | 0.855 | 0.917 | — | — | — | — |
| **E: key v2 retrieval, ±10% + augmentation** | **13 (0.9922)** | **0.977** | **0.856** | **0.921** | **0.835** | **0.882** | 0.979 | **0.982** |

- E is clean-neutral vs A: Δ +0.0006 [−0.0078, +0.0084].
- E roughly halves to quarters the error rate under every tonnage or syllable distortion, including the four distortions it was never trained on. The augmentation teaches the ranker to trust the name when tonnage disagrees, not to memorize one perturbation.
- Vowel swaps (ㅓ↔ㅗ, ㅐ↔ㅔ, ㅡ↔ㅜ, ㅣ↔ㅟ) were already harmless because the sound-class key merges those vowels.
- Remaining weakness: with tonnage missing, Korean recall is 0.887. Name-only retrieval misses about 11% of targets, which caps accuracy near 0.86.
- Harness note: E18 and E20 score clean and degraded copies in one fit per fold, so seed-bag z-scores are computed over the combined block. This is an affine change per seed and leaves rankings within a single model unchanged; production replaces those block constants with per-member training constants (§2.6).

### 2.6 Matcher v2 Part 3: degraded-copy augmentation, seed ensemble, artifacts

**Training-time augmentation.** Every Hangul training row produces three degraded copies: tonnage ×1.08, tonnage missing, and one Hangul syllable deleted (only when the name has at least three syllables). The copies inherit the target and every other field, and their `sample_id` carries a `#aug0/1/2` suffix. The RNG is seeded per call from `augment_seed`, so the copies are deterministic; the attribute maps stay fitted on the clean rows, and recall is measured on the clean rows only. Validation rows are never augmented. Full-training augmentation retrieval keeps the target for 99.3% (tonnage up), 88.8% (tonnage missing) and 100% (syllable deleted) of the copies. Copies with unretrieved targets are excluded from ranker training.

**Seed ensemble.** Five lambdarank members are fitted, member *k* with `seed = config.seed + k`. Each member's scores are standardized with constants fitted on that member's own training-pair scores (mean and population std), never on the scored batch: the prototype z-scored each seed over the whole validation block, which is row-dependent and therefore illegal at inference. E21 observed identical aggregate clean/stress accuracy with the two normalization schemes on the cached prototype setup. This is not a guarantee of identical ensemble rankings: changing member scales changes their relative weights.

**Artifacts.** Bundle version 0.2.0 stores one model file per member (`model_0.txt` … `model_4.txt`) and a manifest `ensemble` section with the file list, means and stds. Manifests without that section (v1 bundles) are rejected, so old artifacts cannot be loaded silently.

**Measured production CV** (vessel-disjoint 5-fold, `train.csv`): weighted score **0.9875** (Korean 0.9824 = 21 errors, non-Korean 0.9995), `recall_hangul` 1.0000, `recall_latin` 1.0000.

**Prototype comparison (partial diagnosis).** The E18/E19 harness numbers use earlier candidate/feature caches, including the pinyin X→S rule dropped in §2.5. Scratch runs E21/E26/E27 produced the following aggregate results; they do not fully isolate the production CV gap:

| Configuration (5-member bag, augmented training) | Korean errors | Score |
|---|---|---|
| Prototype E18 variant E as logged (stale caches, pinyin rule, seeds 42/7/123/1/2) | 13 | 0.9922 |
| Same harness, caches rebuilt from the current code | 15 | 0.9910 |
| Same harness, production seed set (42…46) | 16 | 0.9904 |
| Same harness, production training-pair order | 18 | 0.9893 |
| Production CV (per-fold augmentation draw) | 21 | 0.9875 |

- E21 changed only the seed set on the prototype caches and observed 13 → 16 Korean errors with training-fitted normalization. E26 rebuilt caches with current retrieval/features; the change cannot be attributed solely to removal of the pinyin rule.
- E26/E27 candidates equal production exactly: a row-by-row check of positions and source bits over all 12,222 rows found 0 differences. `tonnage_window` only affects Latin token retrieval, and E26/E27 built the Latin cache at the production default (0.06). The 1,160 differing rows in E30 came from running Latin rows at 0.10, a configuration neither E26/E27 nor production uses.
- E27 compared fold-ordered production diagnostics to input-ordered harness results without joining by `sample_id`, so its 50 differing flags were invalid. Joined by `sample_id`, production (21 Korean errors) and E27 (18) differ on 18 rows from 7 vessels, all with top-1/top-2 margins of 0.015–0.50. These are near-tie swaps, not a systematic gap.
- The only remaining production/harness difference is the augmentation RNG draw: per fold in production, once globally in the harness.

**Seed-noise study (E31)** — production setting (exact production candidates and features, production augmentation order, training-fitted normalization) vs Matcher v1, with four disjoint 5-member seed sets:

| Seed set | v1 Korean errors | v2 Korean errors | v2 score | Δ v2 − v1 [95% CI] |
|---|---|---|---|---|
| 42–46 | 23 | 18 | 0.9893 | +0.0030 [−0.0048, +0.0107] |
| 1042–1046 | 23 | 16 | 0.9904 | +0.0042 [−0.0039, +0.0128] |
| 2042–2046 | 23 | 19 | 0.9887 | +0.0024 [−0.0054, +0.0100] |
| 3042–3046 | 23 | 16 | 0.9904 | +0.0041 [−0.0040, +0.0128] |

- v2 beats v1 under every seed set: 16–19 vs 23 Korean errors, mean score 0.9897 vs 0.9863. No single vessel-bootstrap CI excludes zero, but the sign is consistent across all four seed sets.
- The v2 clean result moves by ±1.5 errors with the seed set alone. One 3–4-row vessel flipping moves the count by 3–4. The production CV (21 errors, 0.9875) sits one vessel below this range, which fits seed and augmentation-draw variation; no implementation defect was found.
- The spec's expected band of 0.991–0.992 was optimistic. It came from the best of five variants (E18 E, 13 errors) on stale caches that still included the pinyin rule. The selection-adjusted expectation for v2 is a clean CV of about 0.989–0.990.
- The case for v2 rests on two things: (a) a consistent but modest clean gain of about 6 Korean errors; (b) much higher robustness to degraded tonnage and names (§2.5, E18/E20). Clean CV alone does not carry it.
- The study is reproduced by rerunning `tools/evaluate.py cv --set seed=<base>` for each seed base and comparing the resulting out-of-fold files.

### 2.7 Leaderboard Gap (2026-10-22)

**The gap.** The v2 submission scores **0.8805** on the leaderboard, while vessel-disjoint CV reports **0.9875** (Korean 0.9824, non-Korean 0.9995). Two diagnostic submissions isolate the buckets:

| Diagnostic submission | LB score | Derived bucket |
|---|---|---|
| Latin (non-Korean) records only | 0.27393 | L = 0.913 |
| Korean records only (L known) | — | K = 0.867 |
| Memory off (call-sign override disabled) | 0.8805 | identical to the default submission |

The memory-off submission has the same reported LB score as the default one. This is consistent with disjoint test vessels, but an unchanged score alone proves neither byte-identical predictions nor zero call-sign hits: an override can agree with the ranker or changes can cancel in the aggregate.

**Ruled out.** Four shift hypotheses were measured and rejected:

| Hypothesis | Probe | Result |
|---|---|---|
| Serving path skew (train/serve mismatch) | fold-1 bundle through save/load + `Matcher` vs CV out-of-fold predictions | 0 differing predictions |
| Population shift (unseen / small vessels) | production model (memory off) on synthetic Latin records of catalog vessels unseen in train | 0.998 accuracy |
| Latin tonnage noise ±10–20% | CV stress scenario | 0.9965 |
| 2–4 character typos in Latin names | synthetic Latin records with 2–4 edits | 0.957 |

None of them reproduces a 0.87 Korean / 0.91 Latin accuracy pair.

**Token-level notation changes.** A fold-1 honest probe (train on fold 1's training rows, score fold 1's validation rows with token-level notation changes) shows the collapse:

| Distortion | Korean accuracy | Latin accuracy | Latin recall |
|---|---|---|---|
| tokens reversed | 0.47 | 0.42 | 0.73–0.78 |
| one token dropped | 0.69 | 0.49 | 0.73–0.78 |
| adjacent tokens swapped | 0.49 | 0.44 | 0.73–0.78 |

Token-level notation changes are absent from the training data: the per-vessel name noise is fixed (each vessel keeps one spelling per script), and 99.8% of the Latin vessels differ from their catalog name by at most two character edits. Cross-validation therefore cannot see token-level changes, and the clean CV score is blind to them.

**Decision.** Model selection stops relying on clean CV alone. The perturbation stress suite (§2.4, extended in this part) is the selection signal: `tools/evaluate.py stress` writes `stress.json` (per scenario) and `suite.json` (the seen / unseen / all suite scores), and a candidate is judged on the suite score rather than on the clean score alone.

**Token-level suite results (v2 baseline).** The nine new scenarios on the v2 model (`artifacts/eval/v2_suite`):

| Scenario | Affected rows | Perturbed accuracy | Perturbed recall |
|---|---|---|---|
| `kor_tokens_reversed` | 879 | 0.3925 | 0.8089 |
| `kor_token_dropped` | 879 | 0.6257 | 0.9772 |
| `kor_tokens_rotated` | 38 | 0.2895 | 0.8947 |
| `kor_tonnage_noise_10_20` | 1191 | 0.8102 | 0.8967 |
| `lat_tokens_reversed` | 8956 | 0.4864 | 0.7736 |
| `lat_token_dropped` | 8957 | 0.5224 | 0.7465 |
| `lat_tokens_rotated` | 1052 | 0.5162 | 0.6749 |
| `lat_heavy_typo` | 10965 | 0.9425 | 0.9815 |
| `lat_tonnage_noise_10_20` | 10238 | 0.9966 | 0.9991 |

The token-level scenarios confirm the fold-1 probe's qualitative weakness, not its exact aggregate values: Latin reversal/drop accuracy is 0.49–0.52, while Korean reversal is 0.3925, below the anticipated 0.45–0.60 band (fold 1 itself is 0.4726). Both retrieval and ranking fail: Latin token scenarios miss the target in 22–33% of rows, but their accuracy remains well below recall; Korean token-drop recall is 0.9772 despite accuracy of 0.6257. Ranking improvements can help already-retrieved targets, while retrieval improvements are needed to raise the recall ceiling. These stress results do not establish the cause or prevalence of distortions in the private test data.

The v2 baseline suite scores are:

| Suite group | kor | lat | score | n_kor | n_lat |
|---|---|---|---|---|---|
| all | 0.8201 | 0.8071 | 0.8162 | 14 | 8 |
| seen | 0.8079 | 0.8330 | 0.8154 | 9 | 6 |
| unseen | 0.8654 | 0.8194 | 0.8516 | 6 | 3 |

Each scenario is one distortion family of equal importance, so the group score is the unweighted mean of the per-scenario accuracies plus the clean item — never a row-count weighted average. These are the reference that the following parts have to beat.

**Notation fixes (measured).** Each step was measured with the full stress suite (5 folds, 20 scenarios):

| Run | Change | Clean score (K errors) | Suite all | Seen | Unseen |
|---|---|---|---|---|---|
| `v2_suite` | v2 baseline | 0.9875 (21) | 0.8162 | 0.8154 | 0.8516 |
| `p2_notation` | + notation and tonnage-noise augmentation | 0.9875 (21) | 0.8832 | 0.8674 | 0.9276 |
| `p3_retrieval` | + sorted-key and token-index retrieval, 7 order-invariant features | 0.9940 (10) | 0.9391 | 0.9185 | 0.9831 |
| `speed_p2` | + OSA prefilter, per-record augmentation RNG, 6 threads (new baseline) | 0.9917 (14) | 0.9370 | 0.9159 | 0.9818 |

- Token reversal and rotation are solved: Latin reversal accuracy 0.486 → 0.999, Korean reversal 0.392 → 0.953.
- The unseen families improve as well (0.8516 → 0.98), so the gain is not limited to the augmented families.
- The remaining weak spot is a dropped token:
  - Latin accuracy 0.658 with recall 0.836;
  - Korean accuracy 0.701 with recall 0.954.

**New baseline (`speed_p2`).** The −0.0023 clean change vs `p3_retrieval` is four Korean rows.
- Thread count is not the cause: the same fold trained with 4, 6 and 8 threads differs by at most 1e-12 in score.
- The OSA prefilter is not the cause: every Latin accuracy and recall is unchanged.
- What remains is the augmentation draw (per-record instead of sequential RNG). The seed study (E31) shows ±1.5 Korean errors from the draw alone, and one 3–4-row vessel flips the count by 3–4.

**Evaluation runtime.** A stage profile of the Part 3 code (3,440 s per stress run) showed the following costs:
- Latin retrieval: two full unrestricted Damerau–Levenshtein cdists against 78,595 catalog names (16.8 ms/row), one third of it spent on full `argsort`;
- re-retrieval: every training record was retrieved again in 4 of 5 folds and in every run;
- LightGBM fitting.

Changes:
- **Output-identical:**
  - top-k selection instead of a full sort;
  - batched Hangul window scoring;
  - a content-addressed candidate cache under `artifacts/cache/candidates/`, keyed by record content and fingerprinted by retrieval code, catalog contents and retrieval config.
  - Verified byte-identical OOF against `p3_retrieval`.
- **Output-changing:**
  - OSA prefilter (top 200) with Damerau–Levenshtein rescoring;
  - per-record augmentation RNG, so copies are fold-invariant and hit the cache;
  - `num_threads` 6.

| Run | Elapsed | Per fold: retrieve / features / fit / score (s) | Peak RSS |
|---|---|---|---|
| Part 3 code | 3,440 s | — | — |
| Exact changes, cold cache | 2,037 s | 150–314 / 41 / 155 / 17 | 1.9 GB |
| Exact changes, warm cache | 1,094 s | 0.4 / 42 / 155 / 18 | 2.6 GB |
| All changes, cold cache | 1,060 s | 20–58 / 42 / 123 / 13 | 2.1 GB |
| All changes, warm cache | 896 s | 0.4 / 42 / 123 / 13 | 2.5 GB |

- Fitting now dominates: one LightGBM member takes ≈ 23 s on 2.1 M pairs, and 8 threads give only 10%.
- Protocol:
  - screen candidates with `--set ensemble_size=1` on a warm cache;
  - confirm finalists with 5 members.
  - A retrieval change invalidates the cache automatically through the fingerprint.

**Retrieval config sweep (5 members, vs `speed_p2`).**

| Run | Change | Clean score (K errors) | Suite all | Seen | Unseen |
|---|---|---|---|---|---|
| `speed_p2` | baseline (window 0.10, rare-token df ≤ 50) | 0.9917 (14) | 0.9370 | 0.9159 | 0.9818 |
| `sweep_w020` | `hangul_tonnage_window=0.20` | 0.9934 (11) | 0.9454 | 0.9287 | 0.9824 |
| `sweep_w025` | `hangul_tonnage_window=0.25` | 0.9934 (11) | 0.9482 | 0.9292 | 0.9881 |
| `exp_w025_k300_rt5000` | + `hangul_window_keep=300`, `rare_token_max_df=5000` | 0.9905 (16) | 0.9466 | 0.9312 | 0.9807 |
| `exp_w025_rt5000` | + `rare_token_max_df=5000` (**adopted**) | 0.9910 (15) | 0.9477 | 0.9288 | 0.9869 |

- **Wider Korean tonnage window:**
  - With window 0.25, the Korean tonnage scenarios reach 0.97: ±10–20% noise goes 0.900 → 0.971, ×0.90 goes 0.901 → 0.971, and ×1.12 goes 0.909 → 0.973.
  - The trade-off: the 200-candidate keep budget inside the wider window loses Korean token-drop recall (0.954 → 0.883).
  - Raising the keep budget to 300 did not pay off.
- **Rare-token source:**
  - Error analysis of `lat_token_dropped` (1,471 unretrieved rows) found two causes:
    - 48% have a remaining token absent from the catalog vocabulary (a typo inside the token, which needs fuzzy token lookup);
    - most of the rest have a remaining token with document frequency 50–350, above the rare-token cap of 50.
  - The rare-token source already requires the ±6% Latin tonnage window, so raising the cap to 5000 adds few candidates.
  - Result: Latin token-drop recall 0.836 → 0.898, accuracy 0.657 → 0.685, with clean Latin accuracy unchanged.
- **Korean clean errors are noise here:** they move between 11 and 16 across runs whose Korean retrieval is identical, because Latin rows share the ranker's training data. Within the E31 band.
- **Adopted defaults:** `hangul_tonnage_window=0.25`, `rare_token_max_df=5000`.

**Final model (2026-10-22).**
- Retrained on all rows: 2,685,242 pairs, 21,665 groups, 5 members.
- Two retrains are byte-identical.
- ZIP: 25 entries, 3.1 MB, no `evaluation/`, smoke test passed.
- 3,000-row rehearsal: 15.2 s, peak RSS 388 MB.
- `artifacts/cv_oof.csv` now records the `exp_w025_rt5000` clean OOF, which is the regression guard's reference.

**Open after the final model:**
- Latin token-drop recall is 0.898 against the 0.97 target. The remaining half needs typo-tolerant token lookup, such as matching a query token to catalog tokens within one edit.
- Korean token-drop accuracy is 0.65 (recall 0.883).

### Decisions
- **No LLMs or large neural models.** CPU rules plus a learned transliteration cost and a GBDT ranker already reach the target. This keeps inference time and ZIP size small, which are the tie-breakers, and avoids synthetic-label compliance questions.
- **Call-sign memory: adopted as a deterministic override.** It is a lookup table fitted on `train.csv` (allowed training artifact, row-independent). The hash is a stable vessel identity: 1:1 in train and invariant to renames and script. It has no effect if test is vessel-disjoint and is exact otherwise. Vessel-disjoint CV cannot learn its weight, so it is applied as a rule after ranking rather than as a ranker feature. Its effect is measured by an on/off LB probe.
- **Matcher v2 candidates (evidence in §2.5; implemented in §2.6):**
  - adopt key v2 rules and features, without the pinyin rule;
  - adopt key v2 retrieval with a ±10% Korean tonnage window and degraded-copy training augmentation (variant E);
  - adopt the 5-seed LightGBM lambdarank bag.
- **Rejected after measurement:**
  - 15 leaves (no gain once key v2 is in);
  - model-family blends (CatBoost, HistGradientBoosting, binary objective);
  - the rank_xendcg objective;
  - token-alignment features;
  - the fold-fitted token lexicon.

## 3. Error Analysis (rubric: result interpretation & generalization)
Residual Korean errors of the rule prototype:
1. **Transliteration rule gaps:** initial ㅎ dropped (흥아 vs Heung-A), NG vs final ㅇ, vowel-initial loanwords (에이스 vs Ace).
2. **Digit mismatch among siblings:** Grand Ace 11 vs Grand Ace1.
3. **Same-GT sister ships with near-identical pronunciation:** 마스그라치 → Marsgracht vs Maasgracht; Shandong Fu Hui vs Fu Yi. Partly irreducible.
4. **Tonnage outside ±6%** (~0.8%): needs name-only retrieval over the full catalog.

Residual Latin errors are renames, recoverable via exact tonnage plus a shared rare token plus type compatibility.

### 3.1 Residuals of the learned ranker
The out-of-fold run leaves 32 errors over 12,222 rows (24 Korean rows over 13 vessels, 8 non-Korean rows). Clusters:

1. **Korean numbering forms** — `제2 오션파이오니어` → `Ocean Pioneer No. 2` is beaten by the sibling `No.3 Ocean Pioneer`; the `제N` rule and the `No. N` catalog form disagree about digit placement. `씨와이 인터오션 투` → `CY Interocean II` is beaten by `CY Interocean I` for the same reason.
2. **Similar names in the phonetic key space** — `마스그라치` (Marsgracht) vs Maasgracht, `산동 푸 후이` (Shandong Fu Hui) vs Shandong Fu Yi, `샨동 푸 유안` (Shandong Fu Yuan) vs Shandong Fu En. These errors do not establish irreducibility; external vessel registries are outside the competition allowlist.
3. **Other Korean-name confusions** — `그레이트 호프` (Great Hope) vs Egor Letov, `스카이 글로리` (Sky Glory) vs Sky Aurora, `메이플 하버` (Maple Harbour) vs V Bravo, `글로비스 코러스` (Glovis Chorus) vs Glovis Corona. Syllable-level similarities are a possible experiment, not a demonstrated remedy.
4. **Korean name variants of the same vessel** — `신간` (Singan) vs Sungkiang, `흥아영` (Heung-A Young) vs An Hong, `펭슈하이` (Feng Xiu Hai) vs Feng Shou Hai: the Korean spelling differs from the romanized catalog name.
5. **Digit disagreement despite an explicit digit feature** — `그랜드 에이스11` (Grand Ace 11) vs Grand Ace1. `digit_tokens` already extracts `11` from the concatenated spelling; the ranker nevertheless prefers the wrong candidate.
6. **Non-Korean name changes and look-alikes** (8 rows) — examples include `BOW SUCCESS` → Fairchem Success, `HAFNIA WREN` → BW Wren, `HAFNIA MERLIN` → BW Merlin, `NAVIG8 VICTORIA` → Hafnia Viridian, and the spelling confusion `DIIJLAH` → Dijilah vs Dijlah. All targets were retrieved; retrieval failure is not the cause of these ranking errors.

Error margins are informative for the field workflow: 6 of the 24 Korean errors have a top-1/top-2 margin below 1.3, i.e. a low-margin review queue would catch roughly a quarter of them, while the remaining errors have a large margin and need better features rather than better ranking.

## 4. Field Applicability
- Operator workflow: top-k candidate recommendation with score margin; low-margin rows go to a human review queue in the field (not in the competition output).
- Degraded-input robustness (§2.5) matters more in the field than in CV: manual entry produces missing or misreported tonnage and dropped syllables, which train rarely shows. Variant E keeps clean accuracy while holding 0.84–0.98 Korean accuracy under such errors.
- Call-sign memory mirrors field practice: once a record is confirmed, its call sign resolves future records instantly, including after renames.
