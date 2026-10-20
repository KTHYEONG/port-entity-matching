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

Caveat: the Korean rules were written after inspecting train names, so 0.968 is optimistic. Learned components must be measured in vessel-disjoint folds.

### 2.1 Retrieval Stage (Matcher v1 Part 2)
Candidate generation (name similarity, tonnage, rare-token, call-sign memory) was measured separately from ranking, on 400 sampled train rows per script bucket (`scratch/eval_candidates.py`). These are in-sample diagnostics: memory was fitted on all train labels, including the sampled rows, so total recall includes a memorization oracle and is not held-out retrieval performance.

| Bucket | Recall@400 | Name source | Tonnage source | Rare-token source | Mean candidates |
|---|---|---|---|---|---|
| Korean | 1.000 | 0.830 | 0.995 | 0.000 | 234 |
| Non-Korean | 1.000 | 1.000 | 0.475 | 0.613 | 34 |

The 1.000 total recall is expected with in-sample memory; it does not establish ranking headroom on unseen vessels. Source coverage suggests Korean rows depend more on the tonnage window than on the phonetic key. Non-Korean names have high sampled name-source coverage, while rare tokens provide an additional path for renames. Measure memory-off recall and vessel-disjoint validation before drawing generalization conclusions.

Runtime on the full catalog (78,595 rows): index build 1.5 s, candidates for all 12,222 train rows 90 s, peak RSS 0.25 GB — far inside the 30-minute inference budget.

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

## 4. Field Applicability
- Operator workflow: top-k candidate recommendation with score margin; low-margin rows go to a human review queue in the field (not in the competition output).
- Call-sign memory mirrors field practice: once a record is confirmed, its call sign resolves future records instantly, including after renames.
