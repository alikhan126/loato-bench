# Direct→Indirect Transfer Findings (LOATO-4A-01 + 4A-02)

**Date**: 2026-03-21
**Experiment**: Train on direct injections, test on indirect injections
**Dataset**: 68,845 samples (balanced labeled_v1.parquet)
**Split**: `data/splits/direct_indirect_split.json` (flat train/test, 80/20 benign split)

## Results — Macro F1

| Embedding | LogReg | SVM* | XGBoost | MLP |
|-----------|--------|------|---------|-----|
| minilm (384d) | 0.2903 | 0.2647 | 0.2207 | 0.2477 |
| bge_large (1024d) | 0.2711 | 0.2071 | 0.2143 | 0.2602 |
| instructor (1024d) | 0.3196 | **0.5231** | 0.2263 | 0.3422 |
| openai_small (1536d) | 0.4081 | 0.2180 | 0.2259 | **0.4130** |
| e5_mistral (4096d) | 0.3248 | 0.2056 | 0.2112 | 0.2521 |

*\*SVM uses Nystroem(500) RBF kernel approximation + PCA(128) for tractability on 68K samples (LOATO-4A-02).*

## Results — AUC-ROC

| Embedding | LogReg | SVM* | XGBoost | MLP |
|-----------|--------|------|---------|-----|
| minilm | 0.8338 | 0.8727 | 0.8754 | 0.8162 |
| bge_large | 0.7579 | 0.7353 | 0.8751 | 0.7994 |
| instructor | 0.9366 | 0.9230 | 0.9270 | 0.9635 |
| openai_small | 0.8713 | 0.8814 | 0.9034 | 0.9507 |
| e5_mistral | 0.7971 | 0.6842 | 0.7990 | 0.8608 |

## Key Observations

1. **Massive generalization gap**: F1 0.21–0.41 vs standard CV 0.90+ — classifiers trained on direct injections largely fail on indirect ones. This is the headline finding. SVM follows the same pattern (F1 0.21–0.26), with one notable exception.

2. **Best combo**: openai_small × MLP (F1=0.413) and openai_small × LogReg (F1=0.408). OpenAI embeddings capture more transferable features.

3. **Instructor × SVM outlier**: instructor × SVM achieves F1=0.523 — the highest transfer F1 across all 20 combinations. This is surprising given SVM's poor Standard CV performance (0.896). The Nystroem approximation may act as implicit regularization, and instruction-tuned embeddings provide better semantic separation. However, this result should be interpreted cautiously as it may not generalize to exact-kernel SVM.

4. **XGBoost and SVM collapse**: XGBoost consistently collapses (~0.21 F1), and SVM is similarly poor (0.21–0.26) except for the instructor outlier. Both overfit to surface patterns.

5. **AUC-ROC >> F1**: AUC ranges 0.68–0.96 while F1 is 0.21–0.52. Classifiers rank indirect injections somewhat correctly but decision thresholds are miscalibrated for the shifted distribution. Threshold analysis (LOATO-4B-02, see `docs/findings_master.md` §6.4) confirms: oracle threshold recalibration improves mean F1 from 0.29 → 0.56, but still falls far short of in-distribution performance (0.96). The gap is fundamentally about representation, not calibration.

6. **Instructor embedding stands out on AUC**: instructor × MLP achieves AUC-ROC 0.9635 despite F1=0.342, suggesting instruction-tuned embeddings capture injection semantics better but still can't cleanly separate at default threshold.

## Comparison with Fomin (2026)

Fomin reports 7–37% detection rates for indirect injections on production guardrails (PromptGuard 2, LlamaGuard), but this was observational — not a controlled train-on-direct/test-on-indirect experiment. Our 0.21–0.52 F1 range is consistent with Fomin's finding but provides the first controlled measurement across 20 embedding-classifier combinations.

## Implications

- **Deployment risk**: A classifier scoring 0.97 F1 on standard CV may score 0.21 on indirect injections — a deployment-critical blind spot
- **Threshold recalibration** (quantified in LOATO-4B-02): Oracle thresholds improve mean F1 from 0.29 → 0.56; 10% labeled holdout calibration recovers ~100% of oracle improvement. SVM benefits most (oracle F1 up to 0.88 for Instructor × SVM). See `docs/findings_master.md` §6.4
- **LLM baseline confirms architectural gap**: GPT-4o scores 0.71 F1 on indirect injections (LOATO-4A-03) vs the best embedding classifier at 0.41 (0.52 with SVM outlier) — a +0.19–0.30 F1 advantage. The generalization failure is architectural (pattern matching vs reasoning), not data-driven. See `docs/findings_llm_baseline.md`

## Artifacts

- 20 result JSONs: `results/experiments/direct_indirect_*.json`
- W&B: runs tagged `transfer_type:direct_indirect` in `loato-bench` project
- HF Hub: PR submitted to `alikhan126/loato-bench-artifacts`
- Threshold analysis: `analysis/transfer_threshold_analysis.json`, `analysis/figures/transfer_*.{png,pdf}`
- Citation: Fomin (2026) in `docs/references.bib`
- Positioning: `docs/related_work_fomin.md`
