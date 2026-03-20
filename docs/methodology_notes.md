# Methodology Notes

Questions and reasoning raised during capstone advising sessions. These clarify
design choices in the LOATO evaluation protocol and should be reflected in the
final write-up.

---

## 1. How is baseline F1 calculated?

**Baseline F1** comes from **standard 5-fold stratified cross-validation** — the
model sees all attack categories during training (random splits, no category
held out). This gives the "best case" F1 where the model has encountered
examples of every attack type.

**LOATO F1** is the macro F1 averaged across the K leave-one-out folds, where
each fold holds out an entire attack category the model has never seen during
training.

The generalization gap is then:

```
ΔF1 = Standard_CV_F1 − LOATO_F1
```

Standard CV is the baseline because it answers: *"How well does the model
perform when it has seen all attack types?"* LOATO answers: *"How well does it
generalize to novel, unseen attack types?"*

---

## 2. If ΔF1 ≈ 0, does that mean the model is smart — or that attack categories are too similar?

**It could be either.** Distinguishing between the two is a key analytical
challenge of this work.

### Interpretation A: Model is genuinely robust

The model learned abstract features of "injection-ness" rather than memorizing
category-specific patterns.

Evidence that would support this:
- High F1 across diverse held-out categories
- UMAP embeddings showing injection/benign separation regardless of category
- Feature importance (SHAP) shows the model relies on different features per
  fold, indicating it can adapt to structurally different attacks

### Interpretation B: Attack categories are too similar

The held-out category overlaps heavily with training categories, so the model
is not truly tested on anything novel.

Evidence that would support this:
- High inter-category cosine similarity in embedding space
- Taxonomy places semantically similar attacks in different categories
- SHAP shows the model uses the same few features regardless of which category
  is held out

### How we distinguish them in our experiments

1. **Per-fold breakdown** — If ΔF1 ≈ 0 for *all* folds uniformly, categories
   may be too similar. If ΔF1 ≈ 0 for most folds but one drops sharply (e.g.,
   holding out `obfuscation_encoding`), the model is genuinely robust to some
   attack types but not others. Per-fold variance is informative.

2. **Inter-category embedding distances** — Compute centroid distances between
   categories in embedding space. If all categories cluster tightly, they share
   too much signal. If they are well-separated yet ΔF1 remains low, the model
   is genuinely generalizing. *(Sprint 4B: UMAP + pairwise centroid distances)*

3. **Feature importance / SHAP analysis** — If the model shifts which features
   matter depending on the held-out fold, categories carry meaningfully
   different signals. If it always relies on the same features, the categories
   are redundant from the model's perspective. *(Sprint 4B)*

4. **Lexical + semantic contamination check** — Already completed in Sprint
   2A-05. If contamination between categories is high, a low ΔF1 is less
   impressive because training data already contains near-duplicates of the
   held-out category.

### Suggested write-up framing

> "We validate that low ΔF1 reflects genuine cross-attack generalization rather
> than category redundancy by examining (a) per-fold variance in LOATO F1,
> (b) inter-category embedding centroid distances, and (c) SHAP feature
> importance stability across folds."

---

*Last updated: 2026-03-19*
