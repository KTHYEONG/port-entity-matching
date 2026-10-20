# Data Analysis & Design Rationale

Stable summary of data findings and the decisions they drive, organized by the finals rubric. Numbers come from `train.csv` (12,222 rows) and `vessel_catalog.csv` (78,595 rows); reproduction scripts live under `scratch/` (`eda*.py`, `proto_*.py`, `eval_baseline.py`).

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

Caveat: the Korean rules were written after inspecting train names, so 0.968 is optimistic. Learned components must be measured in vessel-disjoint folds.

### 2.1 Retrieval Stage (Matcher v1 Part 2)
Candidate generation (name similarity, tonnage, rare-token, call-sign memory) was measured separately from ranking, on 400 sampled train rows per script bucket (`scratch/eval_candidates.py`). These are in-sample diagnostics: memory was fitted on all train labels, including the sampled rows, so total recall includes a memorization oracle and is not held-out retrieval performance.

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

### Decisions
- **No LLMs or large neural models.** CPU rules plus a learned transliteration cost and a GBDT ranker already reach the target. This keeps inference time and ZIP size small, which are the tie-breakers, and avoids synthetic-label compliance questions.
- **Call-sign memory: adopted as a deterministic override.** It is a lookup table fitted on `train.csv` (allowed training artifact, row-independent). The hash is a stable vessel identity: 1:1 in train and invariant to renames and script. It has no effect if test is vessel-disjoint and is exact otherwise. Vessel-disjoint CV cannot learn its weight, so it is applied as a rule after ranking rather than as a ranker feature. Its effect is measured by an on/off LB probe.

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
- Call-sign memory mirrors field practice: once a record is confirmed, its call sign resolves future records instantly, including after renames.
