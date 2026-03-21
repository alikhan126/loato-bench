# LOATO-Bench: Master Findings (Paper Reference)

**Project**: LOATO-Bench — Leave-One-Attack-Type-Out Benchmark for Prompt Injection Classifiers
**Team**: MS Data Science Capstone, Pace University
**Date**: 2026-03-21

This document collects every finding, method, and design decision in the order work was done. It is the single reference for writing the thesis — everything a reviewer would ask is answered here.

---

## Research Question

**Can embedding-based classifiers generalize to novel attack types they have never seen during training?**

Everyone builds prompt injection classifiers, tests them with standard cross-validation, gets 0.95+ F1, and ships them. But standard CV leaks attack types into training — every fold sees examples of every attack category. In the real world, attackers invent new techniques the classifier has never encountered. Does it still work?

---

## Phase 0 — Motivation & Problem Statement

### Finding 0.1: Frontier LLMs Are Vulnerable to Prompt Injections

**Source**: `docs/llm_vulnerability_demo.md`

We tested Claude Sonnet and GPT-4o-mini against 5 RAG-style indirect injection attacks. Both models were compromised on 3 out of 5 tests (60% attack success rate). An embedding-based classifier blocked all 5 poisoned documents.

**Implication**: LLM safety training alone is insufficient. Lightweight input-layer classifiers are needed as a first defense — but how well do they generalize to attacks they haven't seen?

---

## Phase 1 — Data Collection & Exploration (Sprints 1A–1B)

### 1.1 Datasets Used

We unified **9 public datasets** — 5 injection-focused and 4 benign augmentation sources.

#### Injection Datasets (5 sources)

| # | Dataset | HuggingFace ID | License | Raw Samples | After Cleaning | What It Contains |
|---|---------|----------------|---------|-------------|----------------|------------------|
| 1 | Deepset | `deepset/prompt-injections` | MIT | 662 | 659 | Small, manually labeled. Both benign + injection. High quality. |
| 2 | HackAPrompt | `hackaprompt/hackaprompt-dataset` | Apache 2.0 | ~12K | 4,671 | Competition entries (successful injections only). Multi-level attacks against GPT-3.5/FlanT5. Filtered to `correct=True, error=False`. No benign. |
| 3 | GenTel-Bench | `GenTelLab/gentelbench-v1` | CC-BY-4.0 | 177K → 10K cap | ~2,068 | Originally content-harm, NOT injection-specific. Required injection-confidence scoring + filtering (see Finding 1.3). |
| 4 | PINT/Gandalf | `lakera/gandalf_ignore_instructions` | Apache 2.0 | 1,000 | 999 | Gandalf challenge — all injection attempts designed to extract a secret password. |
| 5 | Open-Prompt | `guychuk/open-prompt-injection` | MIT | ~33.6K | 24,286 | Largest source. Primarily **indirect** injection attacks (injected into task context). Each row yields 1 injection + 1 benign (deduplicated). |

#### Benign Augmentation Datasets (4 sources)

Added to fix severe class imbalance (originally 94% injection / 6% benign before augmentation).

| # | Dataset | HuggingFace ID | License | Samples | Origin | Prompt Style |
|---|---------|----------------|---------|---------|--------|--------------|
| 6 | Dolly 15K | `databricks/databricks-dolly-15k` | CC-BY-SA-3.0 | ~14,738 | Human-written (5K Databricks employees) | Task instructions (summarize, classify, brainstorm) |
| 7 | OASST1 | `OpenAssistant/oasst1` | Apache 2.0 | ~7,974 | Human-written (13.5K volunteers) | Conversational questions (how-to, explain, help) |
| 8 | WildChat | `allenai/WildChat-nontoxic` | ODC-BY | ~7,524 | Human-written (real ChatGPT sessions) | Naturalistic chat — the most realistic benign prompts |
| 9 | Alpaca | `yahma/alpaca-cleaned` | CC-BY-NC-4.0 | ~7,994 | Synthetic (GPT-3 text-davinci-003) | Diverse instructions (translate, write, generate) |

**Why 4 benign sources?** Prevents the classifier from learning surface-level shortcuts. If all benign samples were instructions (Dolly), the model might learn "instructions = benign" rather than the actual injection/benign distinction.

### 1.2 Preprocessing Pipeline

```
Step 1: Download (9 loaders)
  → 80,438 raw samples

Step 2: Harmonize
  → NFC Unicode normalization
  → Exact deduplication (SHA-256 hash on text)
  → Near-deduplication (MinHash LSH, Jaccard threshold=0.90, word 5-grams)
  → Language detection (fasttext lid.176.bin)
  → 68,845 samples (unified_dataset.parquet)

Step 3: Taxonomy Labeling (3-tier system)
  → Tier 1: Source-specific mappings (e.g., Open-Prompt "jailbreak" → C2)
  → Tier 2: Regex patterns (e.g., "ignore previous" → C1)
  → Tier 3: LLM-assisted (GPT-4o-mini) for the ~47% unmapped samples
  → 68,845 samples, 100% categorized (labeled_v1.parquet)

Step 4: Split Generation
  → Standard 5-fold CV (stratified on label)
  → LOATO 6-fold (5 held-out categories + 1 benign-only)
  → Direct→Indirect (flat train/test)
  → 4 split JSON files + SHA-256 manifest
```

**Deduplication calibration note**: Initial MinHash threshold (0.85, word 3-grams) was too aggressive — dropped dataset to 25K (50% loss). After reviewing community standards (Lee et al. 2022, BigCode, SlimPajama), adjusted to 0.90/5-grams. Recovered ~12K legitimate samples while still removing true near-duplicates.

### 1.3 GenTel Quality Gate

GenTel-Bench (177K raw) is a content-harm dataset, not injection-specific. Our heuristic injection-confidence scoring (keyword matching, range 0–1):
- 62% of GenTel scored below 0.3 — general harmful content, not injection
- Average score: 0.25
- After filtering (threshold=0.4, cap 5K): **71% of GenTel removed**

Without this gate, the dataset would be dominated by non-injection noise.

### 1.4 Post-Cleaning Dataset

| Label | Count | Percentage |
|-------|-------|------------|
| Benign (0) | 40,003 | 58.1% |
| Injection (1) | 28,842 | 41.9% |
| **Total** | **68,845** | **100%** |

**By source** (top 5): Open-Prompt 24,286 · Dolly 14,738 · Alpaca 7,994 · OASST 7,974 · WildChat 7,524

### 1.5 EDA Key Findings

**Source**: `docs/eda_findings.md`, `docs/eda.md`

1. **Class imbalance** (pre-augmentation): 90% attack / 10% benign → addressed by 4 benign augmentation datasets
2. **Taxonomy coverage** after Tier 1+2: ~53% of injections mapped → ~47% required Tier 3 LLM labeling
3. **Three categories too small** for LOATO: Information Extraction (48), Obfuscation (17), Context Manipulation (12) → resolved in Sprint 2A via taxonomy restructuring
4. **Almost all English**: 98% (36,996/37,773). Cross-lingual experiments excluded (<300 per non-English language)
5. **Few outliers**: 3 samples >10K chars, 27 mostly-uppercase samples → not actionable

---

## Phase 2 — Taxonomy & Splits (Sprint 2A)

### 2.1 Taxonomy v1.0 — 7 Attack Categories

**Source**: `docs/taxonomy_spec_v1.0.md`, `configs/final_categories.json`

Final taxonomy after 3-tier labeling:

| ID | Slug | Category Name | LOATO Eligible | Samples | Mechanism |
|----|------|---------------|:-:|---|---|
| C1 | `instruction_override` | Instruction Override | Yes | 18,108 | Direct commands to ignore/replace system instructions |
| C2 | `jailbreak_roleplay` | Jailbreak / Roleplay | Yes | 622 | Persona adoption to bypass safety (DAN, evil mode) |
| C3 | `obfuscation_encoding` | Obfuscation / Encoding | Yes | 544 | Hiding attacks in Base64, ROT13, pig latin, etc. |
| C4 | `information_extraction` | Information Extraction | Yes | 896 | Extracting system prompts, training data, confidential info |
| C5 | `social_engineering` | Social Engineering | Yes | 303 | Emotional manipulation, urgency, authority claims |
| C6 | `context_manipulation` | Context Manipulation | No (188) | 188 | Indirect injection via retrieved context — excluded from LOATO (too few samples) |
| C7 | `other` | Other / Multi-Strategy | No | 8,355 | Multi-technique or unclassifiable — excluded from LOATO |

**5 LOATO-eligible categories** (C1–C5), each with ≥200 samples.

**Migration from 8-slug draft**: Old `context_manipulation` (system prompt extraction) → C4; old `indirect_injection` → C6; old `payload_splitting` (<50 samples) → C7.

### 2.2 Split Design

| Split Type | Structure | Purpose |
|------------|-----------|---------|
| **Standard 5-fold CV** | Stratified on label, all categories in every fold | Baseline — "how well does it perform when it's seen everything?" |
| **LOATO 6-fold** | 5 held-out categories (C1–C5) + 1 benign-only fold | Primary contribution — "how well does it generalize to unseen attack types?" |
| **Direct→Indirect** | Flat train/test — direct injections for training, indirect for test | Transfer experiment — "can a classifier trained on direct attacks detect indirect ones?" |
| **Cross-lingual** | Excluded | <300 samples per non-English language — documented as limitation |

**LOATO test sets**: Held-out attack category + 20% benign samples (so the test set has both classes).

**Contamination check** (Sprint 2A-05): Lexical (Jaccard) + semantic (cosine similarity) contamination between train/test splits verified to be minimal.

---

## Phase 3 — Embeddings & Classifiers (Sprints 1B, 2B)

### 3.1 Embedding Models (5)

| Key | Model | Dim | Library | Notes |
|-----|-------|-----|---------|-------|
| minilm | all-MiniLM-L6-v2 | 384 | sentence-transformers | Fastest, smallest. Good baseline. |
| bge_large | BGE-large-en-v1.5 | 1024 | sentence-transformers | Requires query prefix. |
| instructor | Instructor-large | 768 | InstructorEmbedding | Instruction-tuned — you can tell it what the text *is*. CPU fallback on MPS. |
| openai_small | text-embedding-3-small | 1536 | OpenAI API | Closed-source, API-based. Best transfer performance. |
| e5_mistral | E5-Mistral-7B (GGUF Q4) | 4096 | llama-cpp-python | Largest model. Metal acceleration on Apple Silicon. |

**Why these 5?** Span the spectrum from tiny/fast (MiniLM, 384d) to large/expensive (E5-Mistral, 4096d). Mix of open-source and proprietary. Different training paradigms (contrastive, instruction-tuned, decoder-based).

All embeddings cached as `.npz` files. Reproducible via `loato-bench embed run --all`.

### 3.2 Classifiers (4)

| Classifier | Type | Implementation | Key Details |
|------------|------|----------------|-------------|
| LogReg | Logistic Regression | sklearn `LogisticRegression` | Linear baseline. Fast, interpretable. |
| SVM | Support Vector Machine (RBF) | sklearn `SVC(probability=True)` | Non-linear. Slow with 4096d (E5-Mistral needs PCA). |
| XGBoost | Gradient Boosted Trees | `xgboost.XGBClassifier` | Ensemble method. Prone to overfitting surface patterns. |
| MLP | Multi-Layer Perceptron | sklearn `MLPClassifier` (2 hidden layers, early stopping) | Neural network. Best non-linear capacity. |

**All wrap sklearn pipelines**: `StandardScaler → Classifier`. This ensures features are normalized regardless of embedding dimension.

### 3.3 Metrics

| Metric | Role |
|--------|------|
| **Macro F1** (primary) | Treats both classes equally. Critical because dataset is 58/42% split — accuracy alone would be misleading. |
| Accuracy | Overall correctness |
| Precision | Of predicted injections, how many are real? |
| Recall | Of real injections, how many did we catch? |
| AUC-ROC | Ranking quality (threshold-independent) |
| AUC-PR | Ranking quality for the positive class |

**Generalization gap**: `ΔF1 = Standard_CV_F1 − LOATO_F1`. Positive means the model does worse on unseen attack types.

**Statistical methods**: Bootstrap 95% CIs (10K resamples), McNemar's test, Friedman + Nemenyi, Cohen's d.

---

## Phase 4 — Core Experiments (Sprint 3)

### Finding 4.1: Standard CV Baseline — Classifiers Score 0.90–0.97 F1

**Source**: W&B project `loato-bench`, 30 runs (5 embeddings × 3 classifiers × 2 experiments)

Under standard 5-fold CV (all attack types present in training), embedding classifiers achieve excellent performance:

| Embedding | LogReg | XGBoost | MLP |
|-----------|--------|---------|-----|
| minilm (384d) | ~0.94 | ~0.95 | ~0.93 |
| bge_large (1024d) | ~0.95 | ~0.96 | ~0.94 |
| instructor (1024d) | ~0.96 | ~0.96 | ~0.95 |
| openai_small (1536d) | ~0.97 | ~0.97 | ~0.96 |
| e5_mistral (4096d) | ~0.95 | ~0.96 | ~0.94 |

*Note: These are approximate from W&B. Pull exact numbers for the paper.*

**These numbers look deployment-ready — but they're misleading.** Standard CV guarantees the model has seen examples of every attack category during training. In deployment, new attack types emerge that the model has never encountered.

### Finding 4.2: LOATO Reveals the Generalization Gap

**Source**: W&B project `loato-bench`, LOATO runs

When a single attack category is held out during training, F1 drops on the held-out type. The generalization gap varies by category:

- Some categories (C1: Instruction Override) have low ΔF1 — other categories contain similar patterns
- Others (C3: Obfuscation, C5: Social Engineering) show large drops — they use unique strategies not present in other categories

**Key insight**: Standard CV overstates real-world performance. LOATO exposes *which specific attack types* a classifier is blind to.

---

## Phase 5 — Transfer Experiments (Sprint 4A)

### Finding 5.1: Direct→Indirect Transfer Collapse (LOATO-4A-01)

**Source**: `docs/findings_direct_indirect.md`

15 experiments (5 embeddings × 3 classifiers) trained on direct injections, tested on indirect:

| Embedding | LogReg | XGBoost | MLP |
|-----------|--------|---------|-----|
| minilm (384d) | 0.2903 | 0.2207 | 0.2477 |
| bge_large (1024d) | 0.2711 | 0.2143 | 0.2602 |
| instructor (1024d) | 0.3196 | 0.2263 | 0.3422 |
| openai_small (1536d) | **0.4081** | 0.2259 | **0.4130** |
| e5_mistral (4096d) | 0.3248 | 0.2112 | 0.2521 |

**Headline**: F1 collapses from 0.90–0.97 (standard CV) to **0.21–0.41** on indirect injections. Classifiers memorize direct injection surface patterns ("ignore previous instructions") but cannot recognize the same malicious intent when it's delivered indirectly (embedded in retrieved documents, tool outputs).

**Key details**:
- **XGBoost worst**: ~0.21 F1 across all embeddings — tree-based models overfit hardest to surface patterns
- **AUC-ROC >> F1**: AUC ranges 0.76–0.96 while F1 is 0.21–0.41. Classifiers *rank* indirect injections somewhat correctly but decision thresholds are miscalibrated. Threshold tuning could partially close the gap.
- **OpenAI embeddings most transferable**: F1 0.41 (best) vs 0.21 (worst). Proprietary embeddings capture more abstract injection features.
- **Fomin (2026) consistency**: Fomin reported 7–37% detection rates on production guardrails (PromptGuard 2, LlamaGuard). Our 0.21–0.41 F1 is consistent but provides the first controlled measurement across 15 combinations.

### Finding 5.2: GPT-4o Zero-Shot Baseline (LOATO-4A-03)

**Source**: `docs/findings_llm_baseline.md`

GPT-4o evaluated zero-shot (no training, no few-shot examples) on 500 stratified samples from each test pool:

| Test Pool | GPT-4o F1 | Best Embedding Classifier F1 | Who Wins | Gap |
|-----------|-----------|-------------------------------|----------|-----|
| Standard CV | 0.8528 | 0.95–0.97 | **Classifiers** | +0.10–0.12 |
| Direct→Indirect | **0.7105** | 0.4130 | **GPT-4o** | **+0.30** |

**Headline**: On familiar attacks, cheap classifiers beat GPT-4o. On novel attack surfaces, GPT-4o's reasoning provides +0.30 F1 advantage — confirming the generalization failure is **architectural** (pattern matching vs reasoning), not a data problem.

**Key details**:
- **Precision near-perfect**: GPT-4o achieves 0.97–0.99 precision — almost never false-positives
- **Weakness is recall**: 0.63–0.70 — still misses ~30% of indirect attacks
- **The gap shrinks with reasoning**: Classifiers drop 0.56 F1 (from 0.97 to 0.41) going standard→indirect. GPT-4o drops 0.14 (from 0.85 to 0.71). The LLM's gap is **4x smaller**.
- **Cost**: ~$0.80 for 1,000 samples (~$0.0008/query) vs ~$0/query for embedding classifiers after training

---

## Summary Table: All Key Numbers

| Metric | Standard CV | LOATO | Direct→Indirect | LLM Baseline (Indirect) |
|--------|:-----------:|:-----:|:----------------:|:----------------------:|
| Best Embedding F1 | 0.97 | varies by fold | 0.41 | — |
| GPT-4o F1 | 0.85 | — | — | 0.71 |
| Who wins | Classifiers +0.12 | — | GPT-4o +0.30 | — |
| Cost per query | ~$0 (after training) | ~$0 | ~$0 | ~$0.0008 |

---

## Methodology Q&A (Anticipated Reviewer Questions)

### Q1: How do you calculate baseline F1?

**Baseline F1** = Standard 5-fold stratified cross-validation. The model sees all attack categories during training (random splits, no category held out). This gives the "best case" F1 where the model has encountered examples of every attack type.

**LOATO F1** = Macro F1 averaged across K leave-one-out folds. Each fold holds out an entire attack category the model has never seen during training.

**Generalization gap** = `ΔF1 = Standard_CV_F1 − LOATO_F1`. A positive gap means the model performs worse on unseen attack types.

Standard CV answers: *"How well does the model perform when it has seen everything?"*
LOATO answers: *"How well does it generalize to novel, unseen attack types?"*

### Q2: If ΔF1 ≈ 0, does that mean the model is smart — or that attack categories are too similar?

**It could be either.** Distinguishing between the two is a key analytical challenge.

**Interpretation A — Model is genuinely robust**: It learned abstract features of "injection-ness" rather than memorizing category-specific patterns. Evidence: high F1 across diverse held-out categories, UMAP showing injection/benign separation regardless of category, SHAP showing different features matter per fold.

**Interpretation B — Categories are too similar**: The held-out category overlaps heavily with training categories, so the model isn't truly tested on anything novel. Evidence: high inter-category cosine similarity in embedding space, SHAP showing the model uses the same few features regardless of fold.

**How we distinguish them:**

1. **Per-fold breakdown** — If ΔF1 ≈ 0 uniformly across all folds, categories may be too similar. If most folds have ΔF1 ≈ 0 but one drops sharply (e.g., holding out `obfuscation_encoding`), the model is genuinely robust to some types but not others. The variance across folds is informative.

2. **Inter-category embedding distances** — Compute centroid distances between categories. If all categories cluster tightly, they share too much signal. If they're well-separated yet ΔF1 is low, the model is genuinely generalizing. *(Sprint 4B: UMAP + pairwise centroid distances)*

3. **SHAP feature importance** — If the model shifts which features matter depending on the held-out fold, categories carry meaningfully different signals. If it always relies on the same features, the categories are redundant. *(Sprint 4B)*

4. **Contamination check** — Already completed (Sprint 2A-05). Lexical + semantic contamination between train/test splits verified to be minimal.

**Write-up framing**: *"We validate that low ΔF1 reflects genuine cross-attack generalization rather than category redundancy by examining (a) per-fold variance in LOATO F1, (b) inter-category embedding centroid distances, and (c) SHAP feature importance stability across folds."*

### Q3: Why Macro F1 instead of accuracy?

Dataset is 58% benign / 42% injection. A classifier that always predicts "benign" would score 58% accuracy. Macro F1 treats both classes equally — it's the harmonic mean of precision and recall, averaged across classes. This is standard practice for imbalanced binary classification in security.

### Q4: Why not fine-tune the LLM instead of using zero-shot?

The point of the LLM baseline is to measure what **reasoning alone** provides — no training, no examples, no domain adaptation. This isolates the architectural advantage (understanding intent vs matching patterns). If we fine-tuned, we'd be measuring the combined effect of reasoning + training, which wouldn't answer whether the generalization gap is architectural.

### Q5: Why only 500 samples for the LLM baseline?

Cost constraint (~$0.80 per 1,000 calls to GPT-4o). The sample is **stratified** to preserve label and category distribution from the full test set. 500 samples is sufficient for stable F1 estimates (bootstrap CIs would show precision). For comparison, Fomin (2026) used even smaller test sets.

### Q6: Aren't the 5 embedding models too diverse to compare fairly?

That's the point. We span from tiny/fast (MiniLM, 384d, open-source) to large/expensive (E5-Mistral, 4096d, open-source) to proprietary (OpenAI, 1536d). If the generalization gap exists across *all* of them, the problem is fundamental to the embedding+classifier approach, not a quirk of one model. The fact that OpenAI embeddings transfer best (F1 0.41 vs 0.21) but still fail badly confirms this.

### Q7: Why is XGBoost so bad on transfer?

XGBoost builds decision trees that split on specific feature thresholds. It's excellent at memorizing exact patterns in training data but terrible at extrapolating. Direct injections have distinctive surface patterns ("ignore previous instructions", "you are now DAN"). XGBoost learns these exact splits. When indirect injections arrive with completely different surface text (malicious instructions buried in a document), the decision boundaries don't transfer. LogReg and MLP, which learn smoother decision boundaries, handle the shift slightly better.

### Q8: Why is AUC-ROC high but F1 low for transfer experiments?

AUC-ROC measures ranking quality — "can the classifier tell that injections are *more likely* to be injections than benign text?" F1 measures hard classification — "at the default 0.5 threshold, does it get the right answer?"

The classifiers **rank** indirect injections somewhat correctly (they get higher scores than benign text) but the **decision threshold is wrong** because it was calibrated on direct injections, which have a different score distribution. This suggests **threshold recalibration** or **Platt scaling** could partially close the F1 gap without retraining.

### Q9: How does this compare to Fomin (2026)?

Fomin's LODO (Leave-One-Dataset-Out) holds out entire *datasets* and reports 7–37% detection rates on production guardrails. Our LOATO holds out *attack categories within* datasets. The approaches are complementary:

| | Fomin (LODO) | Ours (LOATO) |
|---|---|---|
| Holdout unit | Entire dataset | Attack category |
| What it tests | Cross-dataset generalization | Cross-attack-type generalization |
| Classifiers tested | Production guardrails (PromptGuard 2, LlamaGuard) | 5 embeddings × 3 classifiers (controlled) |
| Our advantage | Finer-grained, isolates which *attack type* fails | |
| Their advantage | Tests production systems as-is | |

Our 0.21–0.41 F1 on indirect injections is consistent with Fomin's 7–37% detection rates.

### Q10: What are the limitations?

1. **C6 (Context Manipulation) excluded from LOATO** — only 188 samples. This is the most deployment-relevant category (indirect injection). Its underrepresentation in public datasets is itself a finding.
2. **Cross-lingual experiments excluded** — <300 samples per non-English language. LOATO-Bench is English-only.
3. **LLM baseline is GPT-4o only** — budget constraint. Claude would be a valuable second data point.
4. **Taxonomy is researcher-defined** — 7 categories may not capture all real-world attack types. The LLM labeling (Tier 3) introduces model bias.
5. **Standard CV numbers are approximate** — Sprint 3 results were logged to W&B but not saved as local JSON. Will pull exact numbers for the paper.
6. **SVM deferred** — SVM with 4096d E5-Mistral embeddings is prohibitively slow without PCA. SVM results are incomplete.

---

## Thesis Arguments (Ordered Narrative)

1. **Prompt injection is a real, demonstrated threat** — Frontier LLMs are compromised 60% of the time by indirect injections (Phase 0)

2. **Embedding classifiers look great on paper** — 0.95–0.97 F1 under standard CV suggests they're deployment-ready (Phase 4)

3. **Standard CV is a misleading evaluation** — It guarantees every attack type appears in training. Real deployments face novel attacks.

4. **LOATO exposes the blind spot** — Holding out one attack type reveals which categories a classifier can't generalize to. The gap varies by category — some attacks are genuinely harder to generalize to (Phase 4)

5. **The gap is catastrophic for indirect injections** — F1 collapses to 0.21–0.41 when testing on indirect attacks not seen during training. A classifier scoring 0.97 on your test set may score 0.21 in the real world (Phase 5)

6. **This is an architectural limitation, not a data problem** — GPT-4o, with zero training, scores 0.71 F1 on the same test set where the best trained classifier scores 0.41. Reasoning > pattern matching for novel threats (Phase 5)

7. **Practical recommendation** — Use cheap classifiers as a high-precision first layer (they rarely false-positive). Escalate uncertain cases to LLMs for reasoning-based detection. Standard CV is insufficient for evaluating deployment safety — LOATO and transfer testing should be standard practice.

---

## What's Next

| Sprint | Task | Status |
|--------|------|--------|
| 4B | Analysis: UMAP plots, heatmaps, per-category breakdown, SHAP feature importance | Upcoming |
| 5 | Thesis write-up | Final |

---

## Artifact Locations

| What | Where |
|------|-------|
| Final dataset | `data/processed/labeled_v1.parquet` (68,845 samples) |
| Splits | `data/splits/*.json` (4 split files + manifest) |
| Embeddings | `data/embeddings/` (gitignored, reproducible via `loato-bench embed run`) |
| Taxonomy spec | `docs/taxonomy_spec_v1.0.md`, `configs/final_categories.json` |
| Sprint 3 results | W&B project `loato-bench` (standard_cv + loato runs) |
| Sprint 4A-01 results | `results/experiments/direct_indirect_*.json` (15 files) |
| Sprint 4A-03 results | `results/llm_baseline/llm_baseline_*.json` (2 files + per-sample JSONL logs) |
| EDA outputs | `results/eda/figures/*.png`, `results/eda/*.json` |
| Fomin positioning | `docs/related_work_fomin.md`, `docs/references.bib` |
| Vulnerability demo | `docs/llm_vulnerability_demo.md` |
| Dataset documentation | `docs/datasets.md` |
| EDA findings | `docs/eda_findings.md` |
| Methodology notes | `docs/methodology_notes.md` |
| This document | `docs/findings_master.md` |

---

## Reproducibility

```bash
# Full pipeline from scratch
uv run loato-bench data download          # Download 9 datasets
uv run loato-bench data harmonize         # NFC → dedup → lang detect
uv run loato-bench data label             # 3-tier taxonomy labeling
uv run loato-bench data split             # Generate evaluation splits
uv run loato-bench embed run --all        # Compute 5 embedding models
uv run loato-bench train run --all --experiment standard_cv   # Standard CV
uv run loato-bench train run --all --experiment loato          # LOATO
uv run loato-bench train run --all --experiment direct_indirect # Transfer
uv run loato-bench analyze llm-baseline --model gpt-4o --samples 500 --test-pool both
```

**Requirements**: Python 3.12, uv, HuggingFace token (WildChat access), OpenAI API key, ~$1 for LLM labeling + baseline.
**Hardware**: Apple Silicon Mac (18GB). MPS preferred, CPU fallback.
**Seed**: 42 everywhere (`seed_everything()`).
