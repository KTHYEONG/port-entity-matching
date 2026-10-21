# Competition Reference: AI Technology Competition for Solving Port Challenges

Source: https://dacon.io/competitions/official/236771 (overview, rules, evaluation, data, agreement tabs). Captured 2026-10-20; re-check the official page for changes. Participation is individual; team, prize, and administrative details are intentionally omitted.

## Task
- Match incomplete port arrival/departure records to a standard vessel catalog. Input: one record row. Output: one catalog `vessel_id`. Misspellings and missing specs must be tolerated by combining vessel name with other specs.
- Format: online preliminary (code submission, scored on a hidden test set on DACON servers), then finals (presentation of development process, methodology, experiments, field applicability); final ranking combines both.

## Schedule (2026, KST)
| Date | Event |
|---|---|
| 10.12 | Competition start |
| 11.01 | Leaderboard submission deadline |
| 11.02 | Competition end |
| 11.05 | Code verification deadline |
| 11.06 | Finalist announcement |
| 11.09 | Finals presentation PDF due |
| 11.10 | Finals presentation |
| 11.12 | Winners announced |

## Evaluation
- Score = `0.7 * accuracy(Korean vessel names) + 0.3 * accuracy(non-Korean vessel names)`; bucket by the record's vessel-name script.
- Public LB uses 50% of test; Private uses 100%. Preliminary ranking uses the score at competition end; finalists (top 6, up to 9) are confirmed after code verification (reproducibility, execution format, rule compliance).
- Finals presentation rubric (100 points):
  - Model performance (30): converted Private leaderboard score at competition end.
  - Data understanding and preprocessing (20): grasp of the provided data's characteristics and suitability of preprocessing.
  - Model development (25): design of the matching model and the process of improving performance.
  - Result interpretation and generalization (15): error analysis and ability to handle new vessel names/notations.
  - Practical applicability (10): field applicability and model operation/advancement plans.
- Tie-breakers: shorter inference time, then smaller submission ZIP, then earlier achievement of the score.
- Only scores measured by the evaluation server are official. Leaderboard inference time varies with server load, so keep a safety margin under the time limit.

## Rules
- Max 5 submissions per day. Python only (R/Python for model code per DACON terms).
- Data allowlist: only `train.csv` and `vessel_catalog.csv` for additional training/validation. External vessel DBs, external arrival/departure records, and external answer/mapping tables are prohibited. Publicly available general pretrained weights are allowed; training or validating on outside labeled data is not. (The generic DACON terms allow disclosed external data, but this competition's stricter rule governs.)
- Row independence: each test row is predicted from its own input and official data only. No use of other test rows or overall test distribution (means, frequencies, ranks, group votes, cumulative features, post-hoc adjustments). Same row must give the same prediction alone, in a subset, or in a shuffled file; organizers may re-run to verify.
- No manual labeling or human-predicted information on test/validation data.
- Code and models cannot be replaced after the preliminary round.
- Prohibited: leaking evaluation data through the submission feature, logging/recording/transmitting private evaluation data or answers, restoring original identifying information, redistributing competition data, privately sharing competition code or data with third parties (public sharing only via DACON). Violations can disqualify.
- Winners must provide reproducible code and grant the sponsor a 5-year non-exclusive license (internal research, development, pilot operation); use only original or properly licensed code, models, and tools (AutoML allowed with a proper license).

## Submission Contract
- `submit.zip` root: `script.py`, `requirements.txt`; model files under `model/`.
- `script.py` reads `data/test.csv` and `data/vessel_catalog.csv` and writes `output/submission.csv` with columns `sample_id,target_vessel_id`, covering every test row. Do not hardcode row counts or ids.
- Structure must match exactly (names included); an extra top-level folder inside the zip causes an install error. Layout: `model/` (weights), `script.py`, `requirements.txt` (must be installable via `pip install -r requirements.txt` and list every package inference needs beyond the preinstalled ones).
- The server adds `data/` (read-only, contains the real test data and competition data) and `output/` (must receive `submission.csv`). The guide text also mentions an `open/` directory; trust the rules tab (`data/test.csv`, `data/vessel_catalog.csv`) and have `script.py` fail clearly if a path is missing.
- Error accounting: install errors (structure mismatch, package install failure) do not count toward the daily limit; any error after `script.py` starts counts.
- Limits: package install <= 10 min; inference <= 30 min (load, preprocess, infer, save); ZIP <= 10 GB; unpacked <= 32 GB.
- Server: Ubuntu 22.04.5, NVIDIA L4 (22.4 GiB VRAM), 6 vCPU, 26 GB RAM, Python 3.11.17, CUDA 12.8. Internet is blocked at inference, so no runtime downloads.

## Preinstalled Evaluation Packages
Prefer these versions and leave them out of `requirements.txt`; pinning other versions of the listed packages can cause install errors.
`torch==2.7.1+cu128`, `pandas==2.0.3`, `numpy==1.26.4`, `scipy==1.15.3`, `scikit-learn==1.5.2`, `joblib==1.5.2`, `threadpoolctl==3.7.0`, `lightgbm==4.6.0`, `catboost==1.2.8`, `rapidfuzz==3.14.3`, `Unidecode==1.4.0`, `transformers==4.57.6`, `accelerate==1.9.0`, `safetensors==0.6.2`, `sentencepiece==0.2.0`, `tqdm==4.66.5`.
System: git, git-lfs, build-essential, cmake, ninja-build, libomp, libgomp, gfortran, zip/unzip/p7zip. No internet after install.

## Data
- `train.csv`: 12,222 rows x 9 columns (inputs and answers).
- `vessel_catalog.csv`: 78,595 rows x 8 columns (vessel specs, prediction candidates).
- `sample_submission.csv`: format example only. `test.csv` is hidden (8 columns).
- Local copy: `data/open_data.zip` (contains `open.zip` and `baseline_submit.zip`); `data/*` is gitignored.
