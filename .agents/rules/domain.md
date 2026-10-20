# Entity Matching & Resolution Core Directives

> **The primary directive is engineering rigorous, reproducible entity matching pipelines that guarantee data integrity, deterministic canonicalization, and explainable resolution without phantom linkages.**

## 1. Ground Truth & Resolution Invariants
- **Deterministic Canonicalization:** Normalization rules (case folding, whitespace collapse, punctuation stripping, phonetic/alias mapping) must be deterministic and pure functions. Identical entity inputs must produce identical canonical tokens.
- **Explainable Linkages:** Every matched entity pair must carry an interpretable similarity score, matched attributes, and the rule or model version that produced the linkage. Black-box unexplainable links are prohibited.
- **Fail-Closed on Ambiguity:** When an entity match falls below the acceptance threshold or within a conflict margin (e.g. multiple candidates with near-identical top scores), fail closed: mark as ambiguous or flag for manual review rather than arbitrarily selecting one.
- **Workspace Hygiene:** Exploratory matching probes and candidate evaluations stay under `scratch/`. Never commit raw proprietary entity datasets, caches, or scraped catalogs.

## 2. Invariant & Contract Boundaries
- **Idempotence:** Re-running the matching pipeline on unchanged inputs must produce bit-for-bit identical entity clusters and IDs.
- **Partitioning & Clustering Sanity:** When grouping entities into clusters (e.g. connected components or hierarchical clustering), ensure transitive relations do not create oversized spurious mega-clusters without strict diameter or density guards.
- **Leakage Prevention:** Normalization statistics, TF-IDF vocabularies, or embedding centroids must be fitted strictly on reference/training sets, never leaking evaluation entity sets.

## 3. Performance & Scalability
- **Sub-quadratic Blocking:** Never perform $O(N \times M)$ brute-force cartesian comparisons on unblocked corpora. Employ blocking keys, sorted neighborhood, LSH, or inverted index pruning to bound pairwise comparison budgets.
- **Vectorized Preprocessing:** Process text transformations and attribute extractions via vectorized operations or bounded batch generators.

## 4. Evaluation & Verification
- **Dual Metric Verification:** Evaluate linkages simultaneously on Precision, Recall, and Pairwise F1. Never optimize Recall at the catastrophic cost of Precision.
- **Deterministic Synthetic Tests:** Unit tests verify matching edge cases using minimal deterministic entity pairs: exact duplicates, typographic errors, word order swaps, missing attributes, and distinct false-positive lookalikes.
