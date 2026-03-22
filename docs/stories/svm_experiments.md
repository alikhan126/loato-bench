# LOATO-42: Run SVM Experiments Across All Embeddings

## Summary

Run SVM (RBF kernel) experiments for both Standard CV and LOATO protocols across all 5 embedding models, completing the full 5×4 experiment matrix. SVM was deferred from Sprint 3 due to O(n²) computational cost on 69K samples. This story applies PCA dimensionality reduction as a preprocessing step to make SVM feasible within hardware constraints.

---

## Type

Task

## Priority

Medium

## Sprint

4A

## Story Points

3

## Labels

`experiments`, `classifiers`, `svm`, `sprint-4a`

---

## Background

The experiment matrix calls for 5 embeddings × 4 classifiers. Sprint 3 completed 5×3 = 30 runs (LogReg, XGBoost, MLP). SVM was skipped because:

- **SVM RBF kernel** is O(n²) in training time and memory (pairwise kernel matrix)
- **69K samples** → ~4.7 billion kernel computations per fold
- **E5-Mistral (4096d)** makes this even worse — high dimensionality inflates kernel computation
- **18GB Apple Silicon Mac** — likely OOM or multi-hour runtimes per fold

### Why PCA

PCA to 256 dimensions is the standard mitigation:
- Reduces kernel computation significantly (4096d → 256d)
- Keeps all 69K samples (results stay comparable to other classifiers)
- PCA is a linear transform — preserves the most informative variance
- Well-established practice in SVM literature for high-dimensional embeddings
- 256d retains >95% variance for most embedding models in this range

### Why not alternatives

| Alternative | Reason rejected |
|-------------|-----------------|
| Subsample to ~10K | Results not comparable — other classifiers trained on full 69K |
| LinearSVC | Loses non-linear capacity, which is the whole point of including RBF SVM |
| Skip entirely | Experiment matrix in thesis proposal specifies 4 classifiers |

---

## Acceptance Criteria

- [x] SVM runs complete for **Standard CV** (5 embeddings × 5 folds = 25 runs)
- [x] SVM runs complete for **LOATO** (5 embeddings × 6 folds = 30 runs)
- [x] SVM runs complete for **Direct→Indirect** (5 embeddings × 1 fold = 5 runs)
- [x] PCA (128d) + Nystroem(500) kernel approximation applied for tractability on 68K samples
- [x] Results saved as `{experiment}_{embedding}_svm.json` (15 files)
- [ ] Results uploaded to HF Hub (`alikhan126/loato-bench-artifacts`)
- [x] All results tables updated (15 → 20 rows, now includes SVM)
- [x] All existing tests still pass (773 tests)

---

## Implementation Plan

### Step 1: Add PCA preprocessing to SVM pipeline

**File**: `src/loato_bench/classifiers/svm.py`

- Add `PCA(n_components=256)` to the sklearn Pipeline, between `StandardScaler` and `SVC`
- Pipeline becomes: `StandardScaler → PCA(256) → SVC(RBF)`
- Make `n_components` configurable via constructor param (default 256)
- If input dim ≤ 256, skip PCA (e.g., MiniLM is only 384d — PCA to 256 still helps, but for future-proofing)

### Step 2: Update SVM config

**File**: `configs/classifiers/svm.yaml`

- Add `pca_components: 256` to hyperparams section
- Update sweep config if needed (PCA components could be a sweep param: [128, 256, 512])

### Step 3: Run Standard CV experiments

Run sequentially (one embedding at a time to avoid memory pressure):

```
minilm → bge_large → instructor → openai_small → e5_mistral
```

Each run: 5 folds, ~69K samples, PCA to 256d before SVM fit.

**Expected runtime**: ~10-30 min per embedding (after PCA reduction), ~1-2.5 hours total.

### Step 4: Run LOATO experiments

Same order, same approach. LOATO has 5-6 folds depending on category sizes.

**Expected runtime**: Similar to Standard CV, ~1-2.5 hours total.

### Step 5: Upload results to HF Hub

- Upload new JSON result files to `results/experiments/` on HF Hub
- Update `loato_all_results.json` and `standard_cv_all_results.json` aggregates if they exist

### Step 6: Update README

- Add 5 SVM rows to the results table (one per embedding)
- Note PCA preprocessing in the table footnote or SVM row
- Update experiment count from 30 to 40 runs

### Step 7: Update tests

- Add/update test for SVM pipeline to verify PCA step is present
- Test that SVM works with input dim > 256 (PCA applied) and dim ≤ 256 (PCA still applied or skipped)

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SVM still too slow after PCA | Low | High | Monitor first fold runtime. If >30 min, abort and subsample to 30K |
| PCA loses critical signal | Low | Medium | Check that Standard CV F1 is still >0.95. If massive drop, try 512d |
| Memory pressure on E5-Mistral | Medium | Medium | E5 embeddings are 4096d × 69K = ~1.1 GB. PCA fit needs this in memory. Should be fine on 18GB but monitor |
| Results not comparable to non-PCA classifiers | Low | Low | Document PCA in results. PCA is standard preprocessing — reviewers expect it for SVM on high-dim data |

---

## Dependencies

- All 5 embedding `.npz` files must be available locally (run `download_artifacts.py` if needed)
- Split files must be present (`data/splits/`)
- No external API calls needed (SVM is local sklearn)

---

## Notes

- SVM `probability=True` is already set (enables `predict_proba` via Platt scaling) — this doubles training time but is needed for AUC-ROC/AUC-PR metrics
- If runtime is acceptable without PCA on smaller embeddings (MiniLM 384d), consider running those without PCA and only using PCA for ≥1024d. Keep it simple for now — PCA everywhere for consistency.
- The sweep config has `C: [0.1, 1, 10, 100]` and `gamma: ["scale", "auto", 0.001, 0.01]` — sweeps are **not** part of this story. Default hyperparams only. Sweeps are a separate story if needed.
