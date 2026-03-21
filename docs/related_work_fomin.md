# Related Work: Fomin (2026) Positioning

## Citation

Fomin, M. (2026). *When Benchmarks Lie: Evaluating Malicious Prompt Classifiers Under True Distribution Shift.* arXiv:2602.14161.

## Positioning Paragraph (for Related Work chapter)

Fomin (2026) proposes Leave-One-Dataset-Out (LODO) evaluation across 18 prompt injection datasets, revealing that standard same-source train-test splits inflate aggregate AUC by 8.4 percentage points, with per-dataset accuracy gaps ranging from 1% to 25%. LOATO-Bench is complementary: it holds out attack *categories* within a unified taxonomy rather than entire datasets, measuring whether classifiers generalize across attack types (e.g., training on instruction override but testing on obfuscation) rather than across data sources. The two approaches answer different questions --- LODO asks "does your classifier work on a dataset it hasn't seen?" while LOATO asks "does your classifier detect an attack type it hasn't seen?" Both independently confirm that standard evaluation overestimates real-world performance. Additionally, LOATO-Bench evaluates lightweight embedding-based classifiers (sentence-transformers + LogReg/XGBoost/MLP) relevant to production deployment, while Fomin evaluates LLM internal activations (Llama-3.1-8B layer 31 + SAE features) and production guardrails (PromptGuard 2, LlamaGuard). Notably, Fomin reports 7--37% detection rates for indirect injections on production guardrails but does not run a controlled direct-to-indirect transfer experiment; LOATO-Bench provides exactly this, finding F1 scores of 0.21--0.41 when classifiers trained on direct injections are evaluated on indirect ones.

## Sharpened Contributions (for Introduction)

1. **LOATO protocol** --- attack-type-level holdout evaluation (distinct from LODO's dataset-level holdout), revealing which attack categories resist cross-type generalization
2. **Direct-to-indirect transfer collapse** --- F1 of 0.21--0.41 across 15 embedding-classifier combinations, the first controlled measurement of this deployment-critical blind spot
3. **Embedding classifier focus** --- systematic comparison of 5 embedding models x 4 classifiers, a production-relevant lightweight stack not covered by Fomin's LLM activation probes
4. **Template homogeneity analysis** --- contamination and category-level findings as a mechanistic explanation for the generalization gap (Delta-F1)

## Key Differentiators Table

| Dimension | Fomin (LODO) | LOATO-Bench |
|---|---|---|
| Holdout unit | Entire dataset | Attack category |
| # Datasets | 18 | 5 (unified + deduplicated) |
| Taxonomy | None (dataset-level) | 7-category v1.0 |
| Detection approach | LLM activations, production guardrails | Embedding + classical ML |
| Direct-to-indirect | 7-37% on guardrails (observational) | 0.21-0.41 F1 (controlled experiment) |
| Inflation finding | 8.4pp AUC | TBD (Standard CV vs LOATO Delta-F1) |
| Production relevance | Guardrail audit | Lightweight classifier deployment |
