# LLM Zero-Shot Baseline Findings (LOATO-4A-03)

**Date**: 2026-03-21
**Experiment**: GPT-4o zero-shot binary classification (no training, no few-shot examples)
**Samples**: 500 per test pool (stratified, seed=42)
**Total cost**: ~$0.80 (276K tokens)

## Results

| Test Pool | F1 | Accuracy | Precision | Recall | AUC-ROC | AUC-PR |
|-----------|-----|----------|-----------|--------|---------|--------|
| Standard CV | 0.8528 | 0.8640 | 0.9671 | 0.7000 | 0.8820 | 0.8445 |
| Direct→Indirect | 0.7105 | 0.7240 | 0.9916 | 0.6334 | 0.8404 | 0.9173 |

## Comparison: Embedding Classifiers vs GPT-4o

### Standard CV (familiar attack types)

| System | F1 | Notes |
|--------|-----|-------|
| Best embedding classifier (standard CV) | **0.95–0.97** | 5 emb × 3 clf, trained |
| GPT-4o zero-shot | 0.8528 | No training |

Embedding classifiers outperform GPT-4o by +0.10–0.12 F1 when attacks match training distribution. Cheap classifiers win on familiar ground.

### Direct→Indirect (novel attack surface)

| System | F1 | Notes |
|--------|-----|-------|
| Best embedding classifier (openai_small × MLP) | 0.4130 | Trained on direct only |
| Best calibrated embedding classifier (Instructor × SVM) | 0.8815 | Threshold-tuned on 10% indirect holdout |
| GPT-4o zero-shot | **0.7105** | No training |

GPT-4o outperforms the best embedding classifier by **+0.30 F1** on indirect injections. This is the headline result.

## Key Observations

1. **Pattern matching vs reasoning**: Embedding classifiers memorize surface patterns ("ignore previous instructions") but fail on novel attack vectors. GPT-4o reasons about intent, detecting indirect injections embedded in retrieved context even without training examples.

2. **Precision is near-perfect**: GPT-4o achieves 0.97–0.99 precision in both pools — it almost never flags benign text as an injection. The weakness is recall (0.63–0.70): it misses some attacks, particularly subtle indirect injections.

3. **The generalization gap is architectural**: Embedding classifiers drop from 0.97 to 0.41 F1 (a 0.56 gap) when moving from standard CV to indirect injections. GPT-4o drops from 0.85 to 0.71 (a 0.14 gap). The LLM's gap is 4x smaller, confirming the limitation is in the pattern-matching approach, not the data.

4. **Cost-performance tradeoff**: At ~$0.80 for 1,000 calls, GPT-4o is orders of magnitude more expensive per query than an embedding classifier (~$0 after training). The thesis contribution is quantifying exactly where cheap classifiers break down and by how much.

5. **GPT-4o is not perfect either**: 0.71 F1 on indirect injections means ~30% of attacks still get through. Neither approach solves the problem completely — this motivates layered defenses.

6. **Threshold recalibration narrows but doesn't close the gap**: Threshold analysis (LOATO-4B-02) shows that with oracle threshold tuning, the best embedding classifier (Instructor × SVM) reaches 0.88 F1 — surpassing GPT-4o's 0.71. However, this requires labeled indirect injection data for calibration. Without such data (the realistic deployment scenario), embedding classifiers remain at 0.21–0.52 F1. Mean calibrated F1 across all 20 combos is 0.56, still below GPT-4o's 0.71. See `docs/findings_master.md` §6.4.

## Implications for the Thesis

- **Main argument validated**: Standard CV F1 is misleading for deployment safety. A 0.97 F1 classifier misses 60–80% of indirect injections. LOATO and transfer experiments expose this blind spot; the LLM baseline quantifies the cost of closing it.
- **Practical recommendation**: Use cheap classifiers as a first layer (high precision, fast, free). Escalate uncertain cases to an LLM for reasoning-based detection. At 10% escalation, cost is $0.08/1K queries (80x cheaper than LLM-only) with +0.019 F1 gain on novel attacks. Threshold recalibration (if labeled indirect data is available) can further improve the first layer from 0.29 → 0.56 mean F1, reducing LLM escalation volume. See `docs/findings_master.md` §5.5 for the full cost-performance regime map.
- **Fomin (2026) alignment**: Fomin reported 7–37% detection rates for indirect injections on production guardrails. Our embedding classifiers (21–41% F1) and GPT-4o (71% F1) bracket and extend this finding with controlled, reproducible measurements.

## Artifacts

- Result JSONs: `results/llm_baseline/llm_baseline_*_gpt_4o.json`
- Per-sample logs: `results/llm_baseline/llm_baseline_*_gpt_4o.jsonl`
- W&B: 2 runs tagged `llm_baseline` in `loato-bench` project
- Zero-shot prompt: `src/loato_bench/evaluation/llm_baseline.py::ZERO_SHOT_SYSTEM_PROMPT`
