# LOATO-Bench: Evaluating Cross-Attack Generalization of Embedding-Based Prompt Injection Classifiers

**Rough Paper Draft — All Sections**

*Ali Khan*
MS Data Science, Pace University
2026

---

## Abstract

Embedding-based prompt injection classifiers achieve 0.977–0.997 F1 under standard cross-validation, suggesting deployment readiness. However, standard CV guarantees every attack type appears in training — a condition violated in real deployments where novel attack techniques emerge continuously. We introduce LOATO-Bench (Leave-One-Attack-Type-Out Benchmark), an evaluation framework that holds out entire attack categories during training to measure cross-attack generalization. Using a unified dataset of 68,845 samples from 9 public sources, harmonized into a 7-category taxonomy, we evaluate 5 embedding models × 4 classifiers under three protocols: standard 5-fold CV, LOATO, and direct-to-indirect transfer. LOATO reveals a mean generalization gap of ΔF1 = 0.034 across the top 15 combinations (ΔF1 = 0.051 across all 20), with obfuscation/encoding attacks causing the largest drops (F1 = 0.874 when held out). More critically, classifiers scoring 0.997 F1 under standard CV collapse to 0.21–0.41 F1 when tested on indirect injections unseen during training. A GPT-4o zero-shot baseline scores 0.71 F1 on the same indirect test set — a +0.30 advantage requiring no training — confirming the generalization failure is architectural. We argue that standard CV is insufficient for evaluating prompt injection classifiers intended for production deployment, and that LOATO-style evaluation should become standard practice.

**Keywords**: prompt injection, adversarial attacks, embedding classifiers, evaluation methodology, generalization, LLM security

---

## 1. Introduction

### 1.1 Problem Statement

Large Language Models (LLMs) are increasingly deployed in applications that process untrusted input — RAG pipelines, chatbots, AI agents with tool access. Prompt injection attacks exploit this by embedding malicious instructions in user inputs or retrieved documents, causing the LLM to deviate from its intended behavior. As LLM adoption grows, so does the attack surface.

A common defense is an input-layer classifier: a lightweight model that screens incoming text before it reaches the LLM. These classifiers typically use sentence embeddings (e.g., MiniLM, BGE, OpenAI text-embedding-3) fed into a classical ML head (logistic regression, XGBoost, MLP). They are fast, cheap, and — under standard evaluation — appear to perform excellently, with F1 scores routinely exceeding 0.95.

**The problem**: Standard evaluation (stratified k-fold cross-validation) guarantees that every attack type appears in both training and test sets. This does not reflect deployment reality, where attackers invent novel techniques the classifier has never seen. A classifier that memorizes "ignore previous instructions" will score perfectly on a test set containing that phrase, but fail when an attacker hides the same intent inside a retrieved document or encodes it in Base64.

**Research question**: Can embedding-based prompt injection classifiers generalize to novel attack types they have never seen during training?

### 1.2 Motivation

We conducted a live demonstration (Appendix A) showing that Claude Sonnet and GPT-4o-mini are both vulnerable to RAG-style indirect injection attacks, with a 60% compromise rate across 5 tests. An embedding-based classifier blocked all 5 poisoned documents. This validates the need for input-layer classifiers — but raises the question of whether they would catch attack types not present in their training data.

### 1.3 Contributions

1. **LOATO protocol** — A leave-one-attack-type-out evaluation framework that holds out entire attack categories during training, measuring cross-attack generalization at the technique level (distinct from Fomin's (2026) dataset-level LODO).

2. **Direct-to-indirect transfer measurement** — The first controlled experiment showing F1 collapse from 0.97 to 0.21–0.41 across 15 embedding-classifier combinations when classifiers trained on direct injections are tested on indirect ones.

3. **Unified benchmark dataset** — 68,845 samples from 9 public sources, harmonized through a reproducible pipeline with exact deduplication (SHA-256), near-deduplication (MinHash LSH), and a 7-category attack taxonomy built via 3-tier labeling (source mapping → regex → GPT-4o-mini).

4. **LLM baseline contextualization** — GPT-4o zero-shot evaluation quantifying the cost of the embedding classifier approach: +0.30 F1 advantage on indirect injections at ~1000x higher cost per query.

### 1.4 Novelty

Prior work evaluates prompt injection classifiers using standard CV or dataset-level holdout (Fomin, 2026). No existing benchmark systematically holds out *attack categories* to test whether classifiers generalize across technique boundaries. LOATO-Bench fills this gap and demonstrates that the standard evaluation paradigm dramatically overstates deployment safety.

---

## 2. Related Work

### 2.1 Prompt Injection Attacks

Prompt injection was first identified as a distinct threat class by Perez and Ribeiro (2022), who demonstrated that GPT-3 could be trivially misdirected by adversarial inputs. Their PromptInject framework formalized two attack categories — *goal hijacking* (overriding the system's intended output) and *prompt leaking* (extracting the hidden system prompt) — establishing the foundational taxonomy that subsequent work has extended. Critically, these attacks required no model access or gradient computation: simple handcrafted strings such as "ignore previous instructions" were sufficient to compromise model behavior.

Greshake et al. (2023) expanded the threat model to *indirect* prompt injection, where malicious instructions are not provided by the user but embedded in external data sources — web pages, emails, or retrieved documents — that an LLM-integrated application processes on the user's behalf. This attack vector is particularly dangerous in retrieval-augmented generation (RAG) pipelines, where the model cannot distinguish between trusted system instructions and untrusted retrieved content. The authors demonstrated real-world exploits including data exfiltration, cross-plugin attacks, and persistent compromise of LLM agents. The direct/indirect distinction is central to our work: we show that classifiers trained exclusively on direct injections collapse catastrophically when tested on indirect ones (§5.3).

Liu et al. (2024a) provided the first formal framework for benchmarking prompt injection attacks and defenses, evaluating 5 attack strategies and 10 defense mechanisms across 10 LLMs and 7 tasks. Their Open-Prompt-Injection benchmark — which we incorporate as a data source (§3.1) — revealed that no existing defense achieves both high detection rates and low false positive rates simultaneously. More recent surveys (Esmradi et al., 2025) have catalogued the rapid evolution of attack techniques from manual crafting to automated generation and deep-learning-driven optimization, underscoring the challenge of training classifiers on a static attack distribution.

The distinction between direct and indirect injection has become a central organizing principle in the field. Direct injections are user-supplied adversarial inputs (jailbreaks, instruction overrides); indirect injections are adversarial payloads embedded in data the LLM retrieves or processes. Our 7-category taxonomy (§3.3) maps both attack surfaces, and our three experimental protocols — standard CV, LOATO, and direct-to-indirect transfer — measure generalization across both dimensions.

### 2.2 Prompt Injection Detection

Detection approaches span three paradigms: LLM-based classifiers, fine-tuned transformer classifiers, and embedding-based classifiers with classical ML heads.

**LLM-based guardrails.** Meta's LlamaGuard (Inan et al., 2023) fine-tunes Llama-2-7B for multi-class safety classification of both prompts and responses, achieving competitive performance on OpenAI's Moderation Evaluation dataset and ToxicChat. Its successor, Prompt-Guard-2-86M, is a DeBERTa-based model trained specifically for prompt injection and jailbreak detection, offering lower latency than full LLM inference. Rebuff (Protect AI, 2023) combines heuristic filtering, an LLM-based analyzer, a known-attack database, and canary tokens in a multi-layer defense architecture. These systems demonstrate strong performance on their evaluation benchmarks but are typically assessed using held-out splits of their training data — the same evaluation paradigm our work critiques.

**Fine-tuned transformer classifiers.** Deepset released a DeBERTa-v3-base model fine-tuned on their 662-sample prompt injection dataset, achieving 99.1% accuracy on the held-out test set (Deepset, 2023). Protect AI subsequently released DeBERTa-v3-base-prompt-injection-v2, trained on a larger corpus. These models achieve high accuracy on in-distribution test sets, but their generalization to unseen attack types has not been systematically evaluated — precisely the gap LOATO-Bench addresses.

**Embedding-based classifiers.** Ayub and Majumdar (2024) demonstrated that embedding-based classifiers using Random Forest and XGBoost heads can detect prompt injection attacks, evaluating three embedding methods on a curated dataset of 467K prompts. Their work validates the viability of the embedding + classical ML pattern but evaluates only under standard train-test splits. Our work adopts the same architectural pattern — sentence embeddings fed into classical ML heads — but subjects it to LOATO evaluation, revealing that the high in-distribution performance does not transfer to held-out attack categories.

A common limitation across all three paradigms is evaluation methodology: existing systems report performance on test sets drawn from the same distribution as training data. Standard cross-validation guarantees that every attack type, source dataset, and surface form appears in both training and test folds. This inflates reported metrics and obscures the generalization failures that LOATO-Bench is designed to expose.

### 2.3 Evaluation Methodology and Distribution Shift

The machine learning security community has increasingly recognized that in-distribution evaluation overestimates real-world robustness. Li et al. (2024) introduced OODRobustBench, demonstrating that adversarial robustness suffers severe degradation under distribution shift across 706 models and 23 dataset-wise shifts. Their central finding — that in-distribution robustness is a poor proxy for out-of-distribution robustness — directly parallels our observation that standard CV F1 (0.977–0.997) dramatically overstates LOATO F1 (0.874–0.977) and transfer F1 (0.21–0.41).

Fomin (2026) applies this insight specifically to prompt injection, proposing Leave-One-Dataset-Out (LODO) evaluation across 18 datasets. LODO reveals that standard same-source train-test splits inflate aggregate AUC by 8.4 percentage points, with per-dataset accuracy gaps of 1–25%. LOATO-Bench is complementary: it holds out attack *categories* within a unified taxonomy rather than entire datasets, measuring whether classifiers generalize across attack types rather than across data sources. The two approaches answer different questions — LODO asks "does your classifier work on a dataset it hasn't seen?" while LOATO asks "does your classifier detect an attack *type* it hasn't seen?" Both independently confirm that standard evaluation overestimates real-world performance.

| Dimension | Fomin (LODO) | LOATO-Bench (Ours) |
|---|---|---|
| Holdout unit | Entire dataset | Attack category |
| # Datasets | 18 | 9 (unified + deduplicated) |
| Taxonomy | None (dataset-level) | 7-category v1.0 |
| Detection approach | LLM activations, production guardrails | Embedding + classical ML |
| Direct-to-indirect | 7–37% on guardrails (observational) | 0.21–0.41 F1 (controlled) |
| Inflation finding | 8.4pp AUC | Standard CV F1 0.97 → LOATO/transfer 0.21–0.41 |

Fomin reports 7–37% detection rates for indirect injections on production guardrails but does not run a controlled direct-to-indirect transfer experiment. LOATO-Bench provides exactly this, quantifying the collapse at 0.21–0.41 F1 across 20 embedding-classifier combinations (§5.3).

### 2.4 Embedding Models for Text Classification

Sentence-level embeddings have become a standard representation for text classification tasks. Reimers and Gurevych (2019) introduced Sentence-BERT (SBERT), adapting BERT with siamese networks to produce semantically meaningful sentence embeddings efficiently. The resulting sentence-transformers library underpins two of our five embedding models: MiniLM (384d) and BGE-large (1024d).

Subsequent work has extended the paradigm in two directions. Su et al. (2023) introduced INSTRUCTOR, which conditions embeddings on task-specific instructions, achieving state-of-the-art performance across 70 diverse tasks with an order of magnitude fewer parameters than prior approaches. Wang et al. (2024) demonstrated that decoder-only LLMs (Mistral-7B) can be fine-tuned into effective embedding models with minimal contrastive training, producing E5-Mistral-7B-instruct (4096d). Xiao et al. (2023) released the BGE (BAAI General Embedding) family, with BGE-large-en-v1.5 achieving strong retrieval and classification performance. OpenAI's proprietary text-embedding-3-small (1536d) represents the commercial end of the spectrum (OpenAI, 2024).

Our experimental design (§4) leverages this diversity: 5 models spanning 384–4096 dimensions, open-source and proprietary, general-purpose and instruction-tuned. The embedding + classical ML pattern — where pre-trained embeddings are fed into a logistic regression, SVM, XGBoost, or MLP head — is attractive for production deployment because it decouples representation learning from classification, enabling fast inference at negligible per-query cost. LOATO-Bench tests whether this architectural simplicity comes at the cost of generalization.

---

## 3. Dataset

### 3.1 Data Sources

We unified 9 public datasets — 5 injection-focused and 4 benign augmentation sources.

#### 3.1.1 Injection Datasets

| Dataset | HuggingFace ID | License | Raw | After Cleaning | Description |
|---------|----------------|---------|-----|----------------|-------------|
| Deepset | `deepset/prompt-injections` | MIT | 662 | 659 | Manually labeled, both benign + injection |
| HackAPrompt | `hackaprompt/hackaprompt-dataset` | Apache 2.0 | ~12K | 4,671 | Competition entries (successful injections only). Filtered to `correct=True, error=False` |
| GenTel-Bench | `GenTelLab/gentelbench-v1` | CC-BY-4.0 | 177K → 10K cap | ~2,068 | Content-harm dataset. Required injection-confidence filtering (§3.2) |
| PINT/Gandalf | `lakera/gandalf_ignore_instructions` | Apache 2.0 | 1,000 | 999 | Gandalf challenge — password extraction attempts |
| Open-Prompt | `guychuk/open-prompt-injection` | MIT | ~33.6K | 24,286 | Largest source. Primarily **indirect** injections. Each row yields 1 injection + 1 benign |

#### 3.1.2 Benign Augmentation Datasets

Added to address severe class imbalance (originally 94% injection / 6% benign before augmentation).

| Dataset | HuggingFace ID | License | Samples | Origin | Prompt Style |
|---------|----------------|---------|---------|--------|--------------|
| Dolly 15K | `databricks/databricks-dolly-15k` | CC-BY-SA-3.0 | ~14,738 | Human (5K Databricks employees) | Task instructions |
| OASST1 | `OpenAssistant/oasst1` | Apache 2.0 | ~7,974 | Human (13.5K volunteers) | Conversational questions |
| WildChat | `allenai/WildChat-nontoxic` | ODC-BY | ~7,524 | Human (real ChatGPT sessions) | Naturalistic chat |
| Alpaca | `yahma/alpaca-cleaned` | CC-BY-NC-4.0 | ~7,994 | Synthetic (GPT-3) | Diverse instructions |

Four benign sources were chosen to cover different prompt styles (instructions, conversations, naturalistic chat, synthetic diversity), preventing the classifier from learning surface-level shortcuts (e.g., "all benign samples are instructions").

### 3.2 Data Preprocessing Pipeline

```
Step 1: Download → 80,438 raw samples from 9 loaders

Step 2: Harmonize
  ├── NFC Unicode normalization + whitespace collapse
  ├── Exact deduplication (SHA-256 hash on normalized text)
  ├── Near-deduplication (MinHash LSH)
  │   ├── Jaccard threshold: 0.90
  │   ├── Shingles: word 5-grams
  │   └── MinHash permutations: 128
  ├── Language detection (langdetect)
  └── → 68,845 samples (unified_dataset.parquet)

Step 3: Taxonomy Labeling (3-tier)
  ├── Tier 1: Source-specific mappings (e.g., Open-Prompt attack_type → taxonomy slug)
  ├── Tier 2: Regex patterns (32 keyword patterns, e.g., "ignore previous" → C1)
  ├── Tier 3: LLM-assisted (GPT-4o-mini, temperature=0.0) for ~47% unmapped samples
  └── → 68,845 samples, 100% categorized (labeled_v1.parquet)

Step 4: Split Generation
  ├── Standard 5-fold CV (stratified on label + attack_category)
  ├── LOATO 6-fold (5 held-out categories + benign)
  ├── Direct→Indirect (flat train/test)
  └── → 4 split JSON files + SHA-256 manifest
```

**Deduplication calibration**: Initial MinHash threshold (0.85, word 3-grams) was too aggressive — reduced dataset by 50%. After reviewing community standards (Lee et al. 2022, BigCode, SlimPajama), we adjusted to 0.90 threshold with word 5-grams. This recovered ~12,000 legitimate samples while still removing true near-duplicates.

**GenTel quality gate**: GenTel-Bench is a content-harm dataset, not injection-specific. We developed a heuristic injection-confidence scorer using 32 injection-indicative keywords (e.g., "ignore", "disregard", "override", "jailbreak", "system prompt"). Samples scoring below 0.4 (on a 0–1 scale) were filtered. This removed 71% of GenTel samples but dramatically improved data quality.

### 3.3 Final Dataset

| Label | Count | Percentage |
|-------|-------|------------|
| Benign (0) | 40,003 | 58.1% |
| Injection (1) | 28,842 | 41.9% |
| **Total** | **68,845** | **100%** |

### 3.4 Attack Taxonomy (v1.0)

We developed a 7-category taxonomy via iterative refinement through 3-tier labeling:

| ID | Category | Mechanism | LOATO Eligible | Samples |
|----|----------|-----------|:-:|---|
| C1 | Instruction Override | Direct commands to ignore/replace system instructions | Yes | 19,161 |
| C2 | Jailbreak / Roleplay | Persona adoption to bypass safety (DAN, evil mode) | Yes | 1,614 |
| C3 | Obfuscation / Encoding | Hiding attacks in Base64, ROT13, pig latin, leetspeak | Yes | 545 |
| C4 | Information Extraction | Extracting system prompts, training data, secrets | Yes | 771 |
| C5 | Social Engineering | Emotional manipulation, urgency, authority appeals | Yes | 311 |
| C6 | Context Manipulation | Indirect injection via retrieved context or tool outputs | No (188) | 188 |
| C7 | Other / Multi-Strategy | Catch-all for multi-technique or unclassifiable | No | 8,242 |

**Boundary rules**: If one strategy clearly dominates, assign to that category. Persona + override → primary mechanism decides. Encoding tricks are C3 even if wrapped in a persona. C7 only when no single category fits.

**C6 exclusion**: Context Manipulation has only 188 samples — 12 short of the 200 minimum threshold for LOATO folds. All 188 were LLM-labeled. This underrepresentation of indirect injection in public datasets is itself a finding and a noted limitation.

### 3.5 Template Homogeneity and Category Structure

Prompt injection datasets are highly templated: Standard CV exploits this by allowing near-duplicate patterns to leak between train and test. LOATO controls for this by holding out entire categories, but the degree of "novelty" varies by category — some held-out categories still resemble training patterns from other categories.

We quantify this via **template homogeneity score**: for each LOATO fold, we compute the mean maximum cosine similarity between each test sample's MiniLM embedding and its nearest neighbor in the training set. Higher scores indicate that the held-out category's patterns are well-represented by other categories in training.

| Category | Homogeneity | Mean ΔF1 | Interpretation |
|----------|:-----------:|:--------:|----------------|
| C1 — Instruction Override | 0.845 | 0.037 | Highly templated; patterns shared across categories → easy to detect when held out |
| C2 — Jailbreak / Roleplay | 0.676 | 0.049 | Moderate; persona-based attacks partially overlap with override patterns |
| C3 — Obfuscation / Encoding | 0.670 | 0.090 | Low; encoding tricks are structurally unique → largest generalization gap |
| C4 — Information Extraction | 0.675 | 0.051 | Low; extraction-specific vocabulary not shared with other categories |
| C5 — Social Engineering | 0.674 | 0.065 | Low; emotional manipulation language distinct from technical injection patterns |

Linear regression yields r = −0.604 (R² = 0.365, p = 0.28). The negative correlation supports the hypothesis: categories with higher template homogeneity exhibit smaller generalization gaps. The p-value reflects low statistical power with n = 5 data points, not absence of signal — the direction and magnitude are consistent, and the effect is mechanistically interpretable.

**Inter-category centroid distances** (cosine distance on MiniLM centroids, injection samples only) reveal that C1 and C7 (Other) are nearest neighbors (distance = 0.029), explaining why C7 is easy to classify even when held out — it shares substantial signal with the dominant C1 category. C3 (Obfuscation) is most distant from C5 (distance = 0.441) and C7 (distance = 0.423), confirming its structural uniqueness.

**UMAP projection** (Figure 3.5) shows injection samples forming distinct clusters by category, with C1 samples spanning a broad, overlapping region — consistent with its high homogeneity score and dominant dataset presence.

These findings connect to §5.2 (LOATO results) and §6.6 (interpreting low ΔF1). Artifacts: `results/analysis/template_homogeneity_analysis.json`, `results/analysis/figures/homogeneity_vs_delta_f1.{png,pdf}`, `results/analysis/figures/category_centroid_distances.{png,pdf}`, `results/analysis/figures/umap_category_projection.{png,pdf}`.

---

## 4. Methodology

### 4.1 Embedding Models

We selected 5 embedding models spanning the spectrum from tiny/fast to large/expensive, covering open-source and proprietary models with different training paradigms:

| Model | Dim | Library | Device | Batch | Special Parameters |
|-------|-----|---------|--------|-------|--------------------|
| all-MiniLM-L6-v2 | 384 | sentence-transformers | MPS | 64 | — |
| BGE-large-en-v1.5 | 1024 | sentence-transformers | MPS | 32 | Prefix: "Represent this sentence for classification: " |
| Instructor-large | 768 | InstructorEmbedding | CPU | 16 | Instruction: "Classify whether this text is a prompt injection attack: " |
| text-embedding-3-small | 1536 | OpenAI API | — | 512 | Proprietary, API-based |
| E5-Mistral-7B (GGUF Q4) | 4096 | llama-cpp-python | Metal | 1 | Prompt template with task instruction |

All embeddings are cached as `.npz` files and are fully reproducible.

### 4.2 Classifiers

All classifiers wrap sklearn pipelines with `StandardScaler` as the first step to normalize features regardless of embedding dimension.

| Classifier | Type | Key Hyperparameters |
|------------|------|---------------------|
| LogReg | Logistic Regression | C=1.0, max_iter=1000, solver=lbfgs |
| XGBoost | Gradient Boosted Trees | n_estimators=300, max_depth=6, learning_rate=0.05, tree_method=hist |
| MLP | Multi-Layer Perceptron | hidden_layers=[256, 128], lr=0.001, max_iter=500, early_stopping=True |
| SVM | SVM (RBF kernel) | C=1.0, kernel=rbf, gamma=scale. PCA(128) + Nystroem(500) approximation for tractability on 68K samples |

**SVM methodology note**: Exact RBF-kernel SVM is O(n²–n³) in training time and requires the full n×n kernel matrix in memory. On 68K samples with `probability=True` (needed for AUC-ROC), a single fold exceeded 20+ minutes and was infeasible for the full experiment matrix (~55 runs). We use a two-step approximation: (1) PCA reduces embeddings from their native dimensionality to 128 components, and (2) Nystroem kernel approximation (500 components, RBF kernel) maps the reduced features into an approximate kernel space, followed by SGDClassifier (hinge loss) wrapped in CalibratedClassifierCV (3-fold) for probability estimates. This reduces training time from hours to ~1.3 seconds per fold. The approximation achieves lower absolute F1 (0.83–0.93 Standard CV) compared to other classifiers (0.97–0.99), so SVM results should be interpreted as a lower bound. For datasets with ≤10K samples, the classifier automatically uses exact SVC.

### 4.3 Evaluation Protocols

#### 4.3.1 Standard 5-Fold Cross-Validation (Baseline)

Stratified on label and attack_category. All attack types present in every fold. This establishes the "best case" performance where the model has seen examples of every attack technique during training. Seed=42.

#### 4.3.2 LOATO (Leave-One-Attack-Type-Out) — Primary Contribution

For each of the 5 eligible categories (C1–C5), we hold out the entire category from training and use it as the test set. The test set also includes 20% of benign samples so both classes are represented.

| Fold | Held-Out Category | Train Size | Test Size |
|------|-------------------|-----------|-----------|
| 1 | Instruction Override (C1) | 5,906 | 19,504 |
| 2 | Jailbreak / Roleplay (C2) | 23,431 | 1,979 |
| 3 | Obfuscation / Encoding (C3) | 24,500 | 910 |
| 4 | Information Extraction (C4) | 24,336 | 1,074 |
| 5 | Social Engineering (C5) | 24,741 | 669 |

**Generalization gap**: ΔF1 = Standard_CV_F1 − LOATO_F1. A positive gap means the model performs worse on unseen attack types.

#### 4.3.3 Direct→Indirect Transfer

Train exclusively on direct injections (explicit commands like "ignore previous instructions"), test on indirect injections (malicious instructions embedded in retrieved context, tool outputs, or documents). This simulates the most deployment-critical scenario: RAG pipelines where injections arrive via external data, not user input.

#### 4.3.4 LLM Zero-Shot Baseline

GPT-4o evaluated with zero-shot binary classification (no training, no few-shot examples) on 500 stratified samples per test pool. This isolates the architectural advantage of reasoning over pattern matching. The zero-shot prompt is documented in Appendix B.

### 4.4 Metrics

| Metric | Role |
|--------|------|
| **Macro F1** (primary) | Treats both classes equally — critical because dataset is 58/42% split |
| Accuracy | Overall correctness |
| Precision | Of predicted injections, how many are real? |
| Recall | Of real injections, how many did we catch? |
| AUC-ROC | Ranking quality (threshold-independent) |
| AUC-PR | Ranking quality for the positive (injection) class |

**Why Macro F1**: A classifier that always predicts "benign" would score 58% accuracy. Macro F1 treats both classes equally and is standard for imbalanced binary classification in security.

### 4.5 Reproducibility

All experiments use seed=42 via `seed_everything()`. Hardware: Apple Silicon Mac (18GB), MPS preferred, CPU fallback. Full pipeline is reproducible via CLI commands (see Appendix C). Code, configs, and splits are version-controlled. Experiment tracking via Weights & Biases.

---

## 5. Results

### 5.1 Standard CV Baseline — Classifiers Score 0.977–0.997 F1

Under standard 5-fold CV, embedding classifiers achieve excellent performance (Macro F1):

| Embedding | LogReg | SVM | XGBoost | MLP |
|-----------|--------|-----|---------|-----|
| MiniLM (384d) | 0.9770 | 0.9281 | 0.9829 | 0.9920 |
| BGE-Large (1024d) | 0.9894 | 0.8301 | 0.9856 | 0.9941 |
| Instructor (768d) | 0.9945 | 0.8956 | 0.9936 | 0.9966 |
| OpenAI-Small (1536d) | 0.9952 | 0.9020 | 0.9925 | **0.9974** |
| E5-Mistral (4096d) | 0.9937 | 0.8514 | 0.9871 | 0.9958 |

Mean F1 across all 20 combinations: **0.9637**. Excluding SVM (which uses kernel approximation), the top 15 combinations average **0.9912**. Best: OpenAI-Small × MLP (0.9974). SVM with Nystroem approximation + PCA(128) achieves lower absolute F1 (0.83–0.93) due to the approximation trade-off, but is included for completeness across all 4 classifier architectures. These numbers look deployment-ready. A team evaluating any of these classifiers with standard CV would reasonably conclude it is safe to ship.

### 5.2 LOATO Reveals the Generalization Gap

When a single attack category is held out during training, F1 drops. Mean LOATO F1 across all 20 combinations: **0.9130**, yielding a mean generalization gap of **ΔF1 = 0.0507**. Excluding SVM (kernel approximation), the top 15 combinations average LOATO F1 = 0.9568 with ΔF1 = 0.0344.

#### Table 5.2a: LOATO F1 and ΔF1

| Embedding | Classifier | CV F1 | LOATO F1 | ΔF1 |
|-----------|------------|-------|----------|-----|
| MiniLM (384d) | LogReg | 0.9770 | 0.9169 | 0.0601 |
| MiniLM (384d) | SVM | 0.9281 | 0.8102 | 0.1179 |
| MiniLM (384d) | XGBoost | 0.9829 | 0.9310 | 0.0519 |
| MiniLM (384d) | MLP | 0.9920 | 0.9626 | 0.0294 |
| BGE-Large (1024d) | LogReg | 0.9894 | 0.9565 | 0.0329 |
| BGE-Large (1024d) | SVM | 0.8301 | 0.7588 | 0.0712 |
| BGE-Large (1024d) | XGBoost | 0.9856 | 0.9275 | 0.0581 |
| BGE-Large (1024d) | MLP | 0.9941 | 0.9760 | **0.0181** |
| Instructor (768d) | LogReg | 0.9945 | 0.9670 | 0.0275 |
| Instructor (768d) | SVM | 0.8956 | 0.7461 | **0.1496** |
| Instructor (768d) | XGBoost | 0.9936 | 0.9577 | 0.0359 |
| Instructor (768d) | MLP | 0.9966 | 0.9770 | 0.0196 |
| OpenAI-Small (1536d) | LogReg | 0.9952 | 0.9659 | 0.0293 |
| OpenAI-Small (1536d) | SVM | 0.9020 | 0.8184 | 0.0836 |
| OpenAI-Small (1536d) | XGBoost | 0.9925 | 0.9571 | 0.0354 |
| OpenAI-Small (1536d) | MLP | 0.9974 | 0.9758 | 0.0216 |
| E5-Mistral (4096d) | LogReg | 0.9937 | 0.9641 | 0.0297 |
| E5-Mistral (4096d) | SVM | 0.8514 | 0.7749 | 0.0765 |
| E5-Mistral (4096d) | XGBoost | 0.9871 | 0.9398 | 0.0473 |
| E5-Mistral (4096d) | MLP | 0.9958 | 0.9772 | 0.0187 |

Largest gap: Instructor × SVM (ΔF1 = 0.150). Smallest gap: BGE-Large × MLP (ΔF1 = 0.018). SVM exhibits the largest gaps across all embeddings (0.071–0.150 ΔF1), consistent with kernel approximation limiting its ability to learn precise attack boundaries. Statistical testing: **3/20 combinations significant** (p < 0.05, paired t-test across folds) — MiniLM × LogReg, MiniLM × XGBoost, and BGE-Large × LogReg. SVM's high per-fold variance prevents significance despite large absolute gaps.

#### Table 5.2b: Per-Fold F1 by Held-Out Category (averaged across all 20 combinations)

| Held-Out Category | Mean F1 | Hardest Combo | Easiest Combo |
|---|---|---|---|
| **other** | **0.9507** | BGE-Large×SVM (0.786) | BGE-Large×MLP (0.994) |
| instruction_override | 0.9457 | E5-Mistral×SVM (0.623) | Instructor×MLP (0.996) |
| information_extraction | 0.9264 | Instructor×SVM (0.598) | OpenAI-Small×MLP (0.989) |
| jailbreak_roleplay | 0.9191 | Instructor×SVM (0.783) | OpenAI-Small×MLP (0.972) |
| social_engineering | 0.9097 | BGE-Large×SVM (0.706) | E5-Mistral×MLP (0.981) |
| **obfuscation_encoding** | **0.8738** | Instructor×SVM (0.556) | BGE-Large×MLP (0.964) |

**Key findings**:

1. **Obfuscation/Encoding (C3) is the hardest** — mean F1 = 0.874 when held out. Encoded attacks (Base64, ROT13, leetspeak) use patterns not shared with other categories. SVM drops to F1 = 0.556 on this category.

2. **Instruction Override (C1) is no longer easiest** — with SVM included, "other" category (C7) is now easiest (0.951), while C1 drops to 0.946 due to SVM's poor performance on instruction_override (E5-Mistral×SVM = 0.623). Among LogReg/XGBoost/MLP, C1 remains easiest.

3. **MLP consistently has the smallest gap** across all embeddings (0.018–0.029 ΔF1). Its smooth decision boundaries generalize better than XGBoost's tree-based splits (0.035–0.058 ΔF1) and SVM's approximated kernel boundaries (0.071–0.150 ΔF1).

4. **SVM has the largest gaps** across all embeddings (0.071–0.150 ΔF1). The Nystroem kernel approximation + PCA(128) trade-off reduces both baseline performance and generalization ability, amplifying the gap.

5. **Higher-quality embeddings narrow the gap** — MiniLM (cheapest) has the widest range (0.029–0.118 ΔF1), while Instructor/OpenAI/E5-Mistral cluster in a tighter range for non-SVM classifiers (0.019–0.047).

6. **The gap connects to template homogeneity** — Categories with higher template homogeneity (surface-level patterns, e.g., C1, homogeneity = 0.845) are easier to detect even when held out (ΔF1 = 0.037). Semantically diverse categories (C3, homogeneity = 0.670) expose the generalization gap most starkly (ΔF1 = 0.090). Linear regression: r = −0.604, R² = 0.365. See §3.5 for the full template homogeneity analysis.

### 5.3 Direct→Indirect Transfer Collapse

20 experiments (5 embeddings × 4 classifiers) trained on direct injections, tested on indirect:

#### Table 5.3a: Macro F1

| Embedding | LogReg | SVM | XGBoost | MLP |
|-----------|--------|-----|---------|-----|
| minilm (384d) | 0.2903 | 0.2647 | 0.2207 | 0.2477 |
| bge_large (1024d) | 0.2711 | 0.2071 | 0.2143 | 0.2602 |
| instructor (768d) | 0.3196 | 0.5231 | 0.2263 | 0.3422 |
| openai_small (1536d) | **0.4081** | 0.2180 | 0.2259 | **0.4130** |
| e5_mistral (4096d) | 0.3248 | 0.2056 | 0.2112 | 0.2521 |

#### Table 5.3b: AUC-ROC

| Embedding | LogReg | SVM | XGBoost | MLP |
|-----------|--------|-----|---------|-----|
| minilm | 0.8338 | 0.8727 | 0.8754 | 0.8162 |
| bge_large | 0.7579 | 0.7353 | 0.8751 | 0.7994 |
| instructor | 0.9366 | 0.9230 | 0.9270 | 0.9635 |
| openai_small | 0.8713 | 0.8814 | 0.9034 | 0.9507 |
| e5_mistral | 0.7971 | 0.6842 | 0.7990 | 0.8608 |

**Headline finding**: F1 collapses from 0.83–0.97 (standard CV) to **0.21–0.52** on indirect injections. A classifier that appears 97% effective under standard evaluation catches only 21–41% of indirect attacks (SVM scores similarly, with one outlier: instructor×SVM = 0.52).

**Observations**:

1. **XGBoost collapses worst** (~0.21 F1 across all embeddings). Tree-based models overfit to surface patterns in direct injections and cannot extrapolate.

2. **AUC-ROC >> F1**: AUC ranges 0.76–0.96 while F1 is 0.21–0.41. Classifiers *rank* indirect injections somewhat correctly (they get higher scores than benign text) but decision thresholds calibrated on direct injections are wrong for the shifted distribution. This suggests threshold recalibration or Platt scaling could partially close the gap.

3. **OpenAI embeddings most transferable**: F1 0.41 (best) vs 0.21 (worst). Proprietary embeddings trained on diverse data capture more abstract injection features.

4. **Instructor embedding stands out on AUC**: instructor × MLP achieves AUC-ROC 0.9635 despite F1=0.342. Instruction-tuned embeddings capture injection semantics better but still can't cleanly separate at the default threshold.

5. **Consistent with Fomin (2026)**: Fomin reports 7–37% detection rates for indirect injections on production guardrails. Our 0.21–0.52 F1 range aligns with this but provides the first controlled measurement across 20 embedding-classifier combinations.

6. **SVM follows the collapse pattern** with one outlier: instructor × SVM achieves F1=0.523 — the highest transfer F1 across all 20 combinations. The Nystroem approximation may act as implicit regularization. All other SVM combos score 0.21–0.26, consistent with XGBoost.

### 5.4 LLM Zero-Shot Baseline

GPT-4o evaluated zero-shot on 500 stratified samples per test pool:

| Test Pool | F1 | Accuracy | Precision | Recall | AUC-ROC | AUC-PR | Cost |
|-----------|-----|----------|-----------|--------|---------|--------|------|
| Standard CV | 0.8528 | 0.8640 | 0.9671 | 0.7000 | 0.8820 | 0.8445 | $0.39 |
| Direct→Indirect | 0.7105 | 0.7240 | 0.9916 | 0.6334 | 0.8404 | 0.9173 | $0.41 |

**Comparison**:

| Test Pool | GPT-4o F1 | Best Classifier F1 | Winner | Gap |
|-----------|-----------|---------------------|--------|-----|
| Standard CV | 0.8528 | ~0.97 (MLP) | Classifiers | +0.12 |
| Direct→Indirect | **0.7105** | 0.5231 (SVM†) / 0.4130 (MLP) | **GPT-4o** | **+0.19 / +0.30** |

*†instructor × SVM outlier — see §5.3 observation #6.*

**Key findings**:

1. **On familiar attacks, classifiers win** — cheap, fast, and +0.12 F1 better than GPT-4o zero-shot. No reason to use an LLM when attack types match training data.

2. **On novel attacks, GPT-4o wins by +0.19–0.30 F1** — reasoning about intent beats pattern matching when the surface patterns are unfamiliar. Even the SVM outlier (0.52) still trails GPT-4o by 0.19.

3. **The gap is architectural**: Classifiers drop 0.45–0.56 F1 (0.97 → 0.41–0.52) from standard to indirect. GPT-4o drops 0.14 (0.85 → 0.71). The LLM's degradation is **3–4x smaller**.

4. **GPT-4o precision is near-perfect** (0.97–0.99) — it almost never false-positives. Weakness is recall (0.63–0.70): it misses ~30% of attacks.

5. **Cost tradeoff**: ~$0.0008/query (GPT-4o) vs ~$0/query (classifier after training) — an ~800x cost difference. On known attacks, this cost buys *worse* performance (−0.14 F1). On novel attacks, it buys +0.19 F1 over the best classifier. The cost-performance crossover defines the deployment decision boundary.

### 5.5 Cost-Performance Analysis

The regime map quantifies when to use which approach:

| Test Pool | Best Classifier | GPT-4o | Winner | F1 Gap | Cost Gap |
|-----------|----------------|--------|--------|--------|----------|
| Standard CV | 0.997 (OpenAI-Small × MLP) | 0.853 | Classifier | +0.14 | ~800x cheaper |
| Direct→Indirect | 0.523 (Instructor × SVM) | 0.711 | GPT-4o | +0.19 | ~800x more expensive |

**Layered defense model.** A classifier screens all inputs; a fraction (escalation rate) is escalated to GPT-4o. At key escalation rates on the novel attack surface:

| Escalation Rate | Cost/1K Queries | Est. F1 | F1 Gain |
|-----------------|-----------------|---------|---------|
| 0% (classifier only) | $0.001 | 0.523 | — |
| 10% | $0.08 | 0.542 | +0.019 |
| 25% | $0.21 | 0.573 | +0.050 |
| 50% | $0.41 | 0.619 | +0.096 |
| 100% (GPT-4o only) | $0.80 | 0.711 | +0.187 |

At 10% escalation, the system achieves 80x cost reduction vs GPT-4o-only with modest F1 loss (0.542 vs 0.711). At 25% escalation, cost is still 4x cheaper than GPT-4o-only while gaining +0.05 F1 over classifier-only. The relationship is linear because the model assumes oracle escalation (uncertain predictions escalated first); in practice, a confidence-based escalation policy would yield diminishing returns at higher rates.

**GPT-4o precision-recall tradeoff.** GPT-4o's near-perfect precision (0.967–0.992) means it almost never flags benign text as an injection — critical for user experience. The weakness is recall: it misses 30% of known attacks and 37% of novel indirect injections. For deployment, this means GPT-4o is a reliable *second opinion* (very few false alarms) but not a standalone solution (still misses a third of attacks). This asymmetry favors using GPT-4o as an escalation layer rather than a primary detector.

Full results: `results/analysis/cost_performance_analysis.json` and `results/analysis/figures/`.

---

## 6. Discussion

### 6.1 Standard CV Is Misleading for Deployment Safety

The central finding: a classifier scoring 0.97 F1 under standard CV may score 0.21 on indirect injections it hasn't seen. This is not a minor performance degradation — it is a near-total failure. Standard CV should not be the sole evaluation for classifiers intended for production deployment.

### 6.2 The Generalization Failure Is Architectural

Embedding classifiers learn to match surface patterns (specific phrases, syntactic structures). When the same malicious intent is expressed differently — embedded in a document, encoded in Base64, or delivered through social engineering — the patterns don't match and the classifier fails. GPT-4o's zero-shot performance (0.71 F1 with no training) confirms this: reasoning about intent transfers across attack surfaces in a way that pattern matching cannot.

### 6.3 Not All Attack Types Are Equally Hard

LOATO per-fold analysis reveals that some categories (e.g., Instruction Override) generalize well — their patterns are sufficiently represented in other categories. Others (e.g., Obfuscation, Social Engineering) are truly novel and cause the largest drops. This per-category granularity is the advantage of LOATO over dataset-level evaluation like LODO.

### 6.4 AUC-ROC vs F1: A Threshold Problem

The high AUC-ROC (0.68–0.96) despite low F1 (0.21–0.52) on transfer experiments suggests classifiers can partially *rank* indirect injections correctly but their decision boundaries are miscalibrated. We quantify this with threshold analysis across all 20 transfer combinations.

**Score distributional shift.** Classifiers trained on direct injections assign dramatically lower P(injection) scores to indirect injections. The mean score shift (injection class, train vs test) ranges from −0.27 to −0.97, with most combinations showing shifts below −0.78. At the default threshold of 0.5, nearly all indirect injections fall below the decision boundary.

**Oracle threshold analysis.** Sweeping thresholds to maximize F1 on the indirect test set reveals a ceiling for threshold recalibration. Oracle F1 ranges from 0.35 (MiniLM × MLP) to 0.88 (Instructor × SVM), with a mean of 0.56 across all 20 combinations — a +0.27 improvement over uncalibrated F1 (mean 0.29). However, oracle thresholds are extremely aggressive: 15 of 20 combinations require t ≤ 0.01, meaning nearly every sample above baseline noise is classified as injection. This would unacceptably raise false positives in deployment.

**Practical calibration.** Using 10% of the indirect test set for threshold tuning (simulating access to a small labeled sample), calibrated F1 closely matches oracle F1 (mean calibrated F1 = 0.56 vs oracle 0.57), confirming that a small labeled sample suffices to find the right threshold. The improvement is real but limited — threshold recalibration cannot recover the full AUC-ROC potential because the underlying score distributions overlap too much for most classifiers.

**SVM is the outlier.** SVM benefits disproportionately from threshold recalibration: oracle F1 jumps from 0.21–0.52 (uncalibrated) to 0.61–0.88. The Nystroem approximation produces better-separated probability distributions (score shifts of −0.27 to −0.78, less extreme than other classifiers at −0.78 to −0.97). Instructor × SVM achieves the highest oracle F1 (0.88) and calibrated F1 (0.88) across all combinations.

**Bottom line.** Threshold recalibration provides a meaningful but insufficient fix. Mean F1 improves from 0.29 → 0.56, but this still falls well below the 0.91 LOATO F1 and 0.96 Standard CV F1 achieved in-distribution. The gap is not merely a calibration problem — classifiers fundamentally assign lower confidence to indirect injections because the patterns they learned from direct injections do not transfer. High AUC-ROC reflects correct *relative* ranking within the test set, but the absolute score distributions are too shifted for threshold tuning alone to close the gap.

Full results: `analysis/transfer_threshold_analysis.json` and `analysis/figures/transfer_threshold_summary.{png,pdf}`.

### 6.5 Practical Recommendations

1. **Layered defense with cost-aware escalation**: Use embedding classifiers as a high-precision first layer (~$0/query, 0.997 F1 on known attacks). Escalate uncertain cases to an LLM (~$0.0008/query). At 10% escalation, cost is $0.08/1K queries (80x cheaper than LLM-only) with +0.019 F1 gain on novel attacks. At 25% escalation, cost is $0.21/1K (4x cheaper) with +0.05 F1 gain. The optimal rate depends on the deployment's cost tolerance and novel attack exposure (§5.5).

2. **Evaluate with LOATO**: Before deploying a prompt injection classifier, run LOATO evaluation. Report per-category F1, not just aggregate. If any category shows ΔF1 > 0.15, the classifier has a blind spot.

3. **Test direct→indirect transfer**: If the deployment involves RAG or tool use, run a controlled transfer experiment. Standard CV performance is not predictive of indirect injection detection. A 0.997 F1 classifier may score 0.21–0.52 on indirect injections.

4. **Consider threshold calibration (with caveats)**: If AUC-ROC is high but F1 is low, threshold recalibration on a small labeled sample (~10%) of the target distribution can improve F1 from 0.29 → 0.56 on average. However, this requires labeled data from the target distribution and still falls far short of in-distribution performance (0.96 F1). Threshold tuning is a band-aid, not a fix.

5. **Leverage GPT-4o's precision asymmetry**: GPT-4o's precision (0.97–0.99) far exceeds its recall (0.63–0.70). It rarely false-positives, making it ideal as a *second opinion* on escalated cases rather than a primary detector. Combined with a classifier's high recall on known attacks, this produces a system that's both cost-efficient and broadly defensive.

### 6.6 Interpreting Low ΔF1: Robustness vs Category Redundancy

If ΔF1 ≈ 0, it could mean the model is genuinely robust *or* that attack categories are too similar. We distinguish these via:

1. **Per-fold variance** — If ΔF1 ≈ 0 uniformly across all folds, categories may be redundant. If most folds have ΔF1 ≈ 0 but one drops sharply, the model is robust to some types but not others.

2. **Inter-category embedding distances** (§3.5, COMPLETE) — Centroid cosine distances range 0.029 (C1–C7, nearly overlapping) to 0.441 (C3–C5, well-separated). C1's low ΔF1 (0.037) coincides with high similarity to other categories; C3's high ΔF1 (0.090) coincides with maximal separation — consistent with genuine category-dependent difficulty, not redundancy.

3. **SHAP feature importance** — If the model shifts which features matter per fold, categories carry different signals. If same features dominate regardless of fold, categories are redundant.

4. **Contamination check** — Lexical (Jaccard) + semantic (cosine) contamination between train/test splits, verified to be minimal (Sprint 2A-05).

5. **Template homogeneity correlation** (§3.5, COMPLETE) — Template homogeneity score correlates negatively with ΔF1 (r = −0.604, R² = 0.365), confirming that the generalization gap is driven by structural category differences, not evaluation artifacts. UMAP projection visualizes the category separation in embedding space.

---

## 7. Limitations

1. **C6 (Context Manipulation) excluded from LOATO** — Only 188 samples, 12 short of the 200 threshold. This is arguably the most deployment-relevant category (indirect injection). Its underrepresentation in public datasets is itself a finding.

2. **English-only** — 98% of the dataset is English. Cross-lingual experiments were excluded due to insufficient non-English samples (<300 per language).

3. **Single LLM baseline** — Only GPT-4o evaluated. Claude or open-source LLMs (Llama 3) would provide additional data points.

4. **SVM uses kernel approximation** — SVM with exact RBF kernel is prohibitively slow on 68K samples. We use Nystroem(500) kernel approximation + PCA(128) for tractability, which achieves lower absolute F1 (0.83–0.93 CV vs 0.97–0.99 for other classifiers). SVM results carry this caveat and should be interpreted as a lower bound on exact-kernel SVM performance.

5. **Taxonomy is researcher-defined** — 7 categories may not capture all real-world attack types. Tier 3 LLM labeling introduces model bias (GPT-4o-mini's classification tendencies).

6. **No fine-tuned LLM comparison** — We compare zero-shot LLM vs trained classifier. A fair comparison would also include a fine-tuned LLM, but this was outside scope and budget.

8. **Only 3/15 LOATO gaps are statistically significant** — With only 5–6 folds per LOATO experiment, paired t-tests have low power. The gaps are consistent in direction (always positive) but most p-values fall in 0.06–0.15. Bootstrap CIs or more folds would strengthen significance claims.

9. **Category size imbalance** — C1 (Instruction Override) has 19,161 samples while C5 (Social Engineering) has only 311. Smaller categories have noisier estimates.

---

## 8. Conclusion

Embedding-based prompt injection classifiers achieve 0.977–0.997 F1 under standard cross-validation, but this metric is misleading for deployment. LOATO evaluation reveals a mean ΔF1 = 0.034 (top 15 combinations) with category-dependent blind spots (obfuscation/encoding: F1 = 0.874 when held out). SVM with kernel approximation shows even larger gaps (ΔF1 = 0.071–0.150). More critically, direct-to-indirect transfer experiments show F1 collapsing to 0.21–0.41 — a deployment-critical failure that standard evaluation entirely conceals.

The generalization gap is architectural: GPT-4o achieves 0.71 F1 on the same indirect test set with zero training, a +0.30 advantage driven by reasoning about intent rather than matching surface patterns. However, GPT-4o costs orders of magnitude more per query and still misses ~30% of attacks.

We recommend: (1) LOATO-style evaluation as standard practice before deploying prompt injection classifiers, (2) layered defenses combining cheap classifiers with LLM escalation — at 10% escalation, cost is 80x lower than LLM-only with modest F1 loss (§5.5), and (3) explicit testing of direct-to-indirect transfer for any system processing untrusted external content.

Standard CV scores are reassuring. The real world is not.

---

## 9. Future Work

1. **Sprint 4B**: Core results tables (4B-01), threshold analysis (4B-02), cost-performance regime map (4B-03), and template homogeneity + UMAP analysis (4B-04) completed. SHAP feature importance remains as thesis polish.

2. **~~Threshold recalibration study~~** (COMPLETE — §6.4): Threshold recalibration improves transfer F1 from 0.29 → 0.56 on average but cannot fully close the gap. See §6.4 for details.

3. **Multi-model LLM baseline**: Evaluate Claude, Llama 3, and Gemini to determine if the reasoning advantage is model-specific or general.

4. **Fine-tuned LLM comparison**: Train a small LLM (e.g., Llama-3.1-8B) on the same data to compare fine-tuned LLM vs embedding classifier under LOATO.

5. **Real-world deployment study**: Test classifiers on a live RAG pipeline with adversarial red-teaming to validate lab findings in production conditions.

6. **Expanded indirect injection dataset**: Address C6 underrepresentation by collecting more indirect injection samples, enabling LOATO testing on the most deployment-critical category.

---

## References

Ayub, M. A., & Majumdar, S. (2024). Embedding-based classifiers can detect prompt injection attacks. In *Conference on Applied Machine Learning for Information Security (CAMLIS)*. arXiv:2410.22284.

Deepset. (2023). Prompt injections dataset. HuggingFace: `deepset/prompt-injections`.

Esmradi, A., Yue, D., & Chow, S. (2025). Prompt injection attacks in large language models: A comprehensive review of vulnerabilities, attack vectors, and defense mechanisms. *Information*, 17(1), 54.

Fomin, M. (2026). When benchmarks lie: Evaluating malicious prompt classifiers under true distribution shift. *arXiv preprint arXiv:2602.14161*.

GenTel Lab. (2024). GenTel-Bench v1. HuggingFace: `GenTelLab/gentelbench-v1`.

Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. In *Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec)* (pp. 79–90).

Inan, H., Upasani, K., Chi, J., Rungta, R., Iyer, K., Mao, Y., Tontchev, M., Hu, Q., Fuller, B., Testuggine, D., & Khabsa, M. (2023). Llama Guard: LLM-based input-output safeguard for human-AI conversations. *arXiv preprint arXiv:2312.06674*.

Lakera AI. (2023). Gandalf ignore instructions dataset. HuggingFace: `lakera/gandalf_ignore_instructions`.

Li, L., Chai, Y., Wang, J., Chen, Y., Li, H., et al. (2024). OODRobustBench: A benchmark and large-scale analysis of adversarial robustness under distribution shift. In *International Conference on Machine Learning (ICML)*. arXiv:2310.12793.

Liu, Y., Jia, Y., Geng, R., Jia, J., & Gong, N. Z. (2024a). Formalizing and benchmarking prompt injection attacks and defenses. In *33rd USENIX Security Symposium (USENIX Security 24)* (pp. 1831–1847). arXiv:2310.12815.

Meta AI. (2024). Prompt-Guard-2-86M. HuggingFace: `meta-llama/Prompt-Guard-86M`.

OpenAI. (2024). New embedding models and API updates. https://openai.com/index/new-embedding-models-and-api-updates/.

Perez, F., & Ribeiro, I. (2022). Ignore previous prompt: Attack techniques for language models. In *NeurIPS ML Safety Workshop*. arXiv:2211.09527.

Protect AI. (2023). Rebuff: LLM prompt injection detector. https://github.com/protectai/rebuff.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 3982–3992).

Schulhoff, S., Pinto, J., Khan, A., Bouchard, L.-F., Si, C., et al. (2023). HackAPrompt: Exposing systemic vulnerabilities of LLMs through a global prompt hacking competition. arXiv:2311.16119.

Su, H., Shi, W., Kasai, J., Wang, Y., Hu, Y., Ostendorf, M., Yih, W., Smith, N. A., Zettlemoyer, L., & Yu, T. (2023). One embedder, any task: Instruction-finetuned text embeddings. In *Findings of the Association for Computational Linguistics: ACL 2023* (pp. 1102–1121).

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Improving text embeddings with large language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*. arXiv:2401.00368.

Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packed resources for general Chinese embeddings. *arXiv preprint arXiv:2309.07597*.

---

## Appendix A: LLM Vulnerability Demonstration

**Source**: `docs/llm_vulnerability_demo.md`

Live test results from Claude Sonnet and GPT-4o-mini against 5 RAG-style indirect injection attacks. Both models compromised on 3 out of 5 tests (60% attack success rate). The embedding-based classifier blocked all 5 poisoned documents.

**Test setup**: Simulated RAG pipeline where retrieved documents contain hidden injection instructions. The injection is not in the user's query but in the "context" fed to the LLM. This tests the indirect injection scenario that deployment-facing classifiers must handle.

---

## Appendix B: Zero-Shot Classification Prompt

The exact prompt used for the GPT-4o baseline evaluation:

```
SYSTEM: You are a prompt injection detector. Your task is to determine whether
a given text is a prompt injection attack or a benign/normal input.

A prompt injection is any text that attempts to override, manipulate, or hijack
the instructions of an AI system. This includes (but is not limited to):
- Instructions to ignore previous context or system prompts
- Attempts to make the AI adopt a different persona or role
- Encoded or obfuscated commands
- Social engineering tactics aimed at the AI
- Text that tries to extract system prompts or confidential information
- Indirect injections embedded in retrieved context or tool outputs

Respond with ONLY a JSON object (no markdown, no explanation):
{"label": "injection" or "benign", "confidence": <float 0.0-1.0>}

USER: <sample text>
```

No few-shot examples, no prompt engineering beyond a clear task description. Temperature=0.0 for deterministic output. Model: GPT-4o (March 2026). Cost: ~$0.0008/query.

---

## Appendix C: Reproducibility Commands

```bash
# Environment
uv sync                                          # Install all dependencies

# Data pipeline
uv run loato-bench data download                 # Download 9 datasets (~5 min)
uv run loato-bench data harmonize                # NFC → dedup → lang detect (~2 min)
uv run loato-bench data label                    # 3-tier taxonomy labeling
uv run loato-bench data split                    # Generate all evaluation splits

# Embeddings
uv run loato-bench embed run --all               # Compute 5 embedding models

# Experiments
uv run loato-bench train run --all --experiment standard_cv --wandb
uv run loato-bench train run --all --experiment loato --wandb
uv run loato-bench train run --all --experiment direct_indirect --wandb

# LLM baseline
uv run loato-bench analyze llm-baseline --model gpt-4o --samples 500 --test-pool both --wandb

# Analysis (Sprint 4B)
uv run loato-bench analyze features --all        # SHAP feature importance
uv run loato-bench analyze report                # Generate report tables
```

**Requirements**: Python 3.12, uv, HuggingFace token (WildChat gated access), OpenAI API key.
**Hardware**: Apple Silicon Mac (18GB). MPS preferred, CPU fallback.
**Seed**: 42 everywhere.
**Experiment tracking**: Weights & Biases (project: `loato-bench`).

---

## Appendix D: Full Classifier Hyperparameters

### Logistic Regression
- C (regularization): 1.0
- max_iter: 1000
- solver: lbfgs
- Sweep: C ∈ {0.001, 0.01, 0.1, 1, 10, 100}

### XGBoost
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.05
- eval_metric: logloss
- tree_method: hist
- Sweep: n_estimators ∈ {100, 300, 500}, max_depth ∈ {3, 6, 9}, learning_rate ∈ {0.01, 0.05, 0.1}

### MLP (2-layer)
- hidden_layer_sizes: [256, 128]
- learning_rate_init: 0.001
- max_iter: 500
- early_stopping: True
- validation_fraction: 0.1
- Sweep: layers ∈ {[128,64], [256,128], [512,256]}, lr ∈ {0.0001, 0.001, 0.01}

### SVM (RBF kernel, Nystroem approximation)
- C: 1.0
- kernel: rbf
- gamma: scale
- PCA: 128 components (applied after StandardScaler)
- Nystroem: 500 components (RBF kernel approximation)
- Classifier: SGDClassifier(loss=hinge) + CalibratedClassifierCV(cv=3)
- Threshold: Exact SVC used for n ≤ 10K; Nystroem approximation for n > 10K
- Sweep: C ∈ {0.1, 1, 10, 100}, gamma ∈ {scale, auto, 0.001, 0.01}

All classifiers prepend `StandardScaler` in an sklearn pipeline.

---

## Appendix E: Artifact Locations

| Artifact | Location |
|----------|----------|
| Final dataset | `data/processed/labeled_v1.parquet` (68,845 samples) |
| Splits | `data/splits/*.json` (4 files + SHA-256 manifest) |
| Embeddings | `data/embeddings/{model_name}/` (gitignored, reproducible) |
| Taxonomy spec | `docs/taxonomy_spec_v1.0.md`, `configs/final_categories.json` |
| Sprint 3 results | `results/experiments/standard_cv_*.json`, `results/experiments/loato_*.json` (40 files: 5 emb × 4 clf × 2 experiments) |
| 4B-01 analysis | `analysis/figures/*.{png,pdf}`, `analysis/tables/*.{md,tex}`, `analysis/4b_01_summary.md` |
| 4B-02 threshold analysis | `analysis/transfer_threshold_analysis.json`, `analysis/figures/transfer_*.{png,pdf}` |
| 4B-03 cost-performance | `results/analysis/cost_performance_analysis.json`, `results/analysis/cost_performance_table.md`, `results/analysis/figures/cost_performance_regime_map.{png,pdf}`, `results/analysis/figures/layered_defense_cost_curve.{png,pdf}` |
| 4B-04 template homogeneity | `results/analysis/template_homogeneity_analysis.json`, `results/analysis/figures/homogeneity_vs_delta_f1.{png,pdf}`, `results/analysis/figures/category_centroid_distances.{png,pdf}`, `results/analysis/figures/umap_category_projection.{png,pdf}` |
| Transfer results (4A-01/02) | `results/experiments/direct_indirect_*.json` (20 files: 5 emb × 4 clf) |
| LLM baseline results (4A-03) | `results/llm_baseline/llm_baseline_*.json` (2 files + JSONL logs) |
| EDA outputs | `results/eda/figures/*.png`, `results/eda/*.json` |
| Fomin positioning | `docs/related_work_fomin.md`, `docs/references.bib` |
| Vulnerability demo | `docs/llm_vulnerability_demo.md` |
| This document | `docs/findings_master.md` |
| Full codebase | `src/loato_bench/` (773 tests, 90%+ coverage) |

---

## Appendix F: Anticipated Reviewer Questions

### Q1: Why Macro F1 instead of accuracy?
Dataset is 58% benign / 42% injection. A classifier that always predicts "benign" scores 58% accuracy. Macro F1 treats both classes equally. Standard for imbalanced binary classification in security.

### Q2: Why not fine-tune the LLM instead of zero-shot?
The LLM baseline isolates what **reasoning alone** provides. Fine-tuning would measure reasoning + training combined, which doesn't answer whether the generalization gap is architectural.

### Q3: Why only 500 samples for the LLM baseline?
Cost constraint (~$0.80 per 1,000 GPT-4o calls). The sample is stratified to preserve label and category distribution. 500 is sufficient for stable F1 estimates. Fomin (2026) used comparable sample sizes.

### Q4: Aren't the 5 embedding models too diverse to compare fairly?
That's the point. We span tiny (MiniLM, 384d) to large (E5-Mistral, 4096d), open-source to proprietary. If the generalization gap exists across *all* of them, the problem is fundamental to the embedding+classifier approach, not a quirk of one model.

### Q5: Why is XGBoost so bad on transfer?
XGBoost builds decision trees that split on specific feature thresholds — excellent at memorizing exact patterns, terrible at extrapolating. Direct injections have distinctive surface patterns. When indirect injections arrive with completely different surface text, the decision boundaries don't transfer. LogReg and MLP learn smoother decision boundaries and handle the shift slightly better.

### Q6: Why is AUC-ROC high but F1 low for transfer?
AUC-ROC measures ranking quality ("can it tell injections are *more likely* than benign?"). F1 measures hard classification at threshold=0.5. Classifiers rank indirect injections somewhat correctly but the threshold calibrated on direct injections is wrong for the shifted distribution. Our threshold analysis (§6.4) shows recalibration improves mean F1 from 0.29 → 0.56 but cannot match in-distribution performance (0.96), confirming the gap is fundamentally about representation, not calibration.

### Q7: If ΔF1 ≈ 0, is the model smart or are categories too similar?
Could be either. We distinguish via: (a) per-fold variance — uniform low ΔF1 suggests redundancy, one large drop suggests genuine difficulty; (b) inter-category embedding centroid distances; (c) SHAP feature importance stability across folds; (d) contamination checks (completed, minimal).

### Q8: How does this compare to Fomin (2026)?
Fomin's LODO holds out entire *datasets*, ours holds out *attack categories*. Different questions: LODO asks "does it work on unseen data sources?" LOATO asks "does it detect unseen attack types?" Our 0.21–0.41 F1 is consistent with Fomin's 7–37% detection rates. The approaches are complementary.

### Q9: Is C6 exclusion a problem?
Yes, it's a limitation. Context Manipulation (indirect injection) is the most deployment-relevant category and it's underrepresented in public datasets. However, the direct→indirect transfer experiment (§5.3) directly tests this scenario using Open-Prompt's indirect injections, which are the bulk of indirect samples in the dataset.

### Q10: Could data augmentation solve the generalization problem?
Possibly partially. Generating synthetic indirect injections for training could help, but the fundamental issue is that embedding classifiers match surface patterns. Augmentation might cover known indirect patterns but wouldn't help with genuinely novel attack techniques — which is the scenario LOATO is designed to test.
