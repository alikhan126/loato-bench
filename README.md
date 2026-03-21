# LOATO-Bench

**Cross-Attack Generalization of Embedding-Based Prompt Injection Classifiers**

MS Data Science Capstone — Pace University

## Overview

LOATO-Bench studies whether embedding-based prompt injection classifiers trained on *known* attack types can detect *unseen* attack categories. The core contribution is a **LOATO (Leave-One-Attack-Type-Out)** evaluation protocol applied to 5 embedding models × 4 classifiers on a unified benchmark of ~69K samples from 9 public datasets.

### Research Questions

1. **RQ1**: How well do embedding-based classifiers generalize to unseen prompt injection attack types?
2. **RQ2**: Which embedding × classifier combinations show the strongest cross-attack generalization?
3. **RQ3**: Can classifiers trained on direct injections detect indirect injection attacks?
4. **RQ4**: How well do English-trained classifiers transfer to non-English prompt injections?

## Architecture

```
5 Datasets → Unified Benchmark → 5 Embedding Models → 4 Classifiers → LOATO Evaluation
```

### Datasets

The dataset was deliberately balanced (~58% benign / 42% injection) to prevent classifiers from learning a trivial "predict injection always" shortcut. See `docs/datasets.md` for full details on each source.

**Injection Sources (5 datasets):**

| Dataset | Samples | Notes |
|---------|---------|-------|
| [Open-Prompt-Injection](https://huggingface.co/datasets/guychuk/open-prompt-injection) | ~24K | Indirect injection attacks, largest source |
| [HackAPrompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) | ~4.7K | Successful attacks only |
| [PINT / Gandalf](https://huggingface.co/datasets/lakera/gandalf_ignore_instructions) | ~1K | Password extraction attacks |
| [Deepset](https://huggingface.co/datasets/deepset/prompt-injections) | ~260 | Mixed benign/injection |
| GenTel-Bench | — | Currently excluded (stale HF cache) |

**Benign Augmentation Sources (4 datasets):**

| Dataset | Samples | Why chosen |
|---------|---------|------------|
| [Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | ~14.7K | Human-written instructions (diverse tasks) |
| [Alpaca (cleaned)](https://huggingface.co/datasets/yahma/alpaca-cleaned) | ~8K | Synthetic instructions (GPT-3, diverse) |
| [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | ~8K | Real human chat prompts (English filtered) |
| [WildChat](https://huggingface.co/datasets/allenai/WildChat-nontoxic) | ~7.5K | Real ChatGPT sessions (most realistic) |

**Post-harmonization totals:** 68,845 samples (40,017 benign / 28,828 injection)

### Embedding Models

| Model | Dim | Library |
|-------|-----|---------|
| all-MiniLM-L6-v2 | 384 | sentence-transformers |
| BGE-large-en-v1.5 | 1024 | sentence-transformers |
| Instructor-large | 768 | InstructorEmbedding |
| text-embedding-3-small | 1536 | OpenAI API |
| E5-Mistral-7B (GGUF Q4) | 4096 | llama-cpp-python |

### Classifiers

- **Logistic Regression** — linear baseline
- **SVM (RBF)** — non-linear kernel method
- **XGBoost** — gradient-boosted trees
- **MLP** — 2-layer neural network (sklearn)

### Evaluation Protocols

| Protocol | Description |
|----------|-------------|
| **Standard 5-Fold CV** | Stratified by label + attack category |
| **LOATO** | Train on K-1 attack types, test on held-out type |
| **Direct → Indirect** | Train on direct injections, test on indirect |

> **Note:** Cross-lingual evaluation (train English → test non-English) was considered but deemed out of scope — only ~777 non-English samples exist across 30+ languages, with no single language reaching the 300-sample minimum for statistical validity. This data gap is documented as a limitation and future work direction.

## Data Pipeline

The benchmark dataset is built from 5 public sources through a multi-stage pipeline:

```
9 raw datasets → harmonize (dedup + normalize) → unified_dataset.parquet (68,845 samples)
                                                          ↓
                                              3-tier taxonomy labeling
                                              (source maps → regex → GPT-4o-mini)
                                              [benign samples skip this step]
                                                          ↓
                                                labeled_v1.parquet
                                                          ↓
                                              4 experiment split files
```

**Harmonization** (Sprint 1A): Each of the 9 dataset loaders produces `UnifiedSample` records. The harmonizer applies NFC unicode normalization, exact deduplication (SHA-256), near-deduplication (MinHash LSH, Jaccard 0.90, word 5-grams), and language detection. This reduces ~80K raw samples to ~69K after cross-source dedup.

**Taxonomy labeling** (Sprint 2A): Samples are assigned to 7 attack categories using a 3-tier system. Tier 1 maps known source-specific labels (e.g., Open-Prompt "jailbreak" → `jailbreak_roleplay`). Tier 2 applies regex patterns for common signals (e.g., "ignore previous" → `instruction_override`). Tier 3 uses GPT-4o-mini via OpenAI's Batch API for samples that Tiers 1+2 couldn't classify (~30% of injections). The raw LLM outputs are preserved as an audit trail.

**Split generation** (Sprint 2A): Four sets of train/test index files are generated — one per evaluation protocol. Splits store integer indices referencing rows in `labeled_v1.parquet`, not the data itself, keeping them portable and deterministic (seed=42).

## Data & Reproducibility

### Artifacts on Hugging Face Hub

Pre-computed embeddings (~2.2 GB), experiment results, dataset files, and splits are hosted on Hugging Face Hub for full reproducibility without re-running the pipeline:

**Repo: [alikhan126/loato-bench-artifacts](https://huggingface.co/datasets/alikhan126/loato-bench-artifacts)**

Download everything with one command:

```bash
uv run python scripts/download_artifacts.py
```

Or download selectively:

```bash
uv run python scripts/download_artifacts.py --only embeddings  # ~2.2 GB
uv run python scripts/download_artifacts.py --only results     # experiment JSONs
uv run python scripts/download_artifacts.py --only data        # parquets + splits
```

#### What's on HF Hub

| Artifact | Size | Contents |
|----------|------|----------|
| `embeddings/` | ~2.2 GB | Pre-computed embeddings for all 5 models (`.npz` + metadata) |
| `results/experiments/` | ~256 KB | 30 experiment result JSONs (5 models × 3 classifiers × 2 protocols) |
| `data/processed/` | ~16 MB | `labeled_v1.parquet` + `unified_dataset.parquet` |
| `data/splits/` | ~6 MB | All 4 split index files + manifest |

### What's in Git

Only small metadata and config files are committed to the repo:

| File | Purpose |
|------|---------|
| `data/labeling/coverage_report.json` | Labeling coverage statistics |
| `data/labeling/labeling_report.json` | Labeling pipeline summary |
| `configs/final_categories.json` | Taxonomy v1.0 export (7 categories) |

### What's NOT in Git (download from HF Hub)

| Artifact | How to get it |
|----------|---------------|
| `data/processed/` | `uv run python scripts/download_artifacts.py --only data` |
| `data/splits/` | `uv run python scripts/download_artifacts.py --only data` |
| `data/labeling/llm_labels_raw.jsonl` | `uv run python scripts/download_artifacts.py --only data` |
| `data/embeddings/` | `uv run python scripts/download_artifacts.py --only embeddings` |
| `results/` | `uv run python scripts/download_artifacts.py --only results` |
| `data/raw/` | `uv run loato-bench data download` |
| `.env` | Copy `.env.example` and fill in your keys |

### Attack Taxonomy (v1.0)

7 attack categories (6 LOATO-eligible), defined in `src/loato_bench/data/taxonomy_spec.py`. C7 is excluded from LOATO because it's a catch-all — holding out a grab-bag category doesn't test meaningful generalization.

| ID | Category | Mechanism | LOATO |
|----|----------|-----------|:-----:|
| C1 | Instruction Override | Directly tells the model to ignore/replace its instructions | Yes |
| C2 | Jailbreak / Roleplay | Adopts a fictional persona to bypass safety constraints | Yes |
| C3 | Obfuscation / Encoding | Encodes malicious payload (Base64, ROT13, leetspeak) to evade filters | Yes |
| C4 | Information Extraction | Attempts to extract system prompts, training data, or secrets | Yes |
| C5 | Social Engineering | Uses emotional manipulation, urgency, or authority claims | Yes |
| C6 | Context Manipulation | Injects via external content (documents, tool outputs, hidden text) | Yes |
| C7 | Other / Multi-Strategy | Catch-all for multi-strategy or unclassifiable attacks | No |

This was consolidated from an earlier 8-category draft: `context_manipulation` (system prompt extraction) merged into C4, `indirect_injection` became C6, and `payload_splitting` (<50 samples) merged into C7. See `OLD_SLUG_TO_NEW` in `taxonomy_spec.py` for the full migration map.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Apple Silicon Mac (MPS backend) recommended, CPU works too

### Installation

```bash
# 1. Clone
git clone <repo-url>
cd loato-bench

# 2. Install Python dependencies
uv sync

# 3. Set up environment
cp .env.example .env
# Edit .env: add HF_TOKEN (required for artifact download)
#            add OPENAI_API_KEY and WANDB_API_KEY (only if re-running pipelines)

# 4. Download pre-computed artifacts (embeddings, results, data)
uv run python scripts/download_artifacts.py
```

### Special Installs

E5-Mistral GGUF model (optional, for Metal acceleration):

```bash
CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python
bash scripts/setup_e5_gguf.sh
```

## Usage

### CLI

```bash
# Show all commands
uv run loato-bench --help

# Data pipeline
uv run loato-bench data download
uv run loato-bench data harmonize
uv run loato-bench data split

# Compute embeddings
uv run loato-bench embed run --all
uv run loato-bench embed run --model minilm

# Train classifiers
uv run loato-bench train run --all --experiment standard_cv
uv run loato-bench train run --all --experiment loato
uv run loato-bench train run --embedding minilm --classifier xgboost --experiment loato

# Hyperparameter sweeps
uv run loato-bench sweep run --all

# Analysis
uv run loato-bench analyze features --all
uv run loato-bench analyze llm-baseline --samples 500
uv run loato-bench analyze report
```

### Justfile (run full pipeline)

```bash
# Requires: https://github.com/casey/just
just all          # Run everything end-to-end
just data         # Data pipeline only
just embed        # Embeddings only
just train-loato  # LOATO experiments only
```

### Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
loato-bench/
├── pyproject.toml
├── .gitignore              # Data & results gitignored (hosted on HF Hub)
├── Justfile
├── configs/
│   ├── data/               # sources.yaml, taxonomy.yaml (Tier 1+2 rules)
│   ├── embeddings/         # one YAML per embedding model
│   ├── classifiers/        # one YAML per classifier
│   ├── experiments/        # one YAML per experiment protocol
│   └── final_categories.json  # ★ Taxonomy v1.0 export (generated from code)
├── src/loato_bench/
│   ├── cli.py              # Typer CLI entrypoints
│   ├── data/               # Dataset loaders, harmonization, taxonomy, splits
│   │   ├── taxonomy_spec.py    # Taxonomy v1.0 — single source of truth for categories
│   │   └── llm_labeler.py      # GPT-4o-mini batch labeling (Tier 3)
│   ├── embeddings/         # Embedding model implementations + cache
│   ├── classifiers/        # Classifier implementations (LogReg, SVM, XGB, MLP)
│   ├── evaluation/         # LOATO protocol, metrics, transfer, statistical tests
│   ├── analysis/           # EDA, visualization, SHAP, report generation
│   ├── tracking/           # W&B integration
│   └── utils/              # Config loading, device selection, reproducibility
├── data/                   # Gitignored — download from HF Hub
│   ├── processed/          # labeled_v1.parquet, unified_dataset.parquet
│   ├── splits/             # Split index files + fold parquets
│   ├── labeling/           # Audit trail: reports + raw LLM outputs
│   ├── raw/                # Downloaded source datasets
│   └── embeddings/         # .npz embedding caches per model
├── results/                # (gitignored) all experiment outputs
├── docs/                   # EDA guide, taxonomy spec, dataset docs, methodology notes
├── notebooks/              # Interactive analysis (EDA, embeddings, results)
├── scripts/                # Setup scripts (e.g., GGUF model download)
└── tests/                  # pytest suite (712 tests, 90%+ coverage)
```

All data files are hosted on [HF Hub](https://huggingface.co/datasets/alikhan126/loato-bench-artifacts) — run `uv run python scripts/download_artifacts.py` to get them.

## Experiment Matrix

| Experiment | Models | Classifiers | Folds | Total Runs |
|-----------|--------|-------------|-------|------------|
| Standard 5-Fold CV | 5 | 4 | 5 | 100 |
| LOATO | 5 | 4 | 6-8 | 120-160 |
| Direct → Indirect | 5 | 4 | 1 | 20 |
| Feature Analysis (SHAP) | 5 | 1 | 1 | 5 |
| LLM Baseline | 2 | — | 1 | 2 |
| **Total** | | | | **~250-290** |

## Key Metrics

- **Macro F1 (LOATO)**: Primary metric — average F1 across held-out categories
- **ΔF1 = Standard_F1 − LOATO_F1**: The generalization gap
- **Degradation %**: ΔF1 / Standard_F1 × 100
- Bootstrap 95% confidence intervals (10,000 resamples)
- McNemar's test, Friedman test + Nemenyi post-hoc for statistical comparisons

## Results (Sprint 3 — Complete)

5 embedding models × 3 classifiers (SVM deferred — O(n²) on 69K samples), sorted by LOATO F1:

| Model × Classifier | Standard CV | LOATO F1 | ΔF1 |
|---------------------|------------|----------|-----|
| e5_mistral × MLP | 0.996 | **0.977** | 0.019 |
| instructor × MLP | **0.997** | 0.977 | 0.020 |
| bge_large × MLP | 0.994 | 0.976 | **0.018** |
| openai_small × MLP | 0.997 | 0.976 | 0.022 |
| instructor × LogReg | 0.995 | 0.967 | 0.028 |
| openai_small × LogReg | 0.995 | 0.966 | 0.029 |
| e5_mistral × LogReg | 0.994 | 0.964 | 0.030 |
| minilm × MLP | 0.992 | 0.963 | 0.029 |
| instructor × XGBoost | 0.994 | 0.958 | 0.036 |
| openai_small × XGBoost | 0.993 | 0.957 | 0.035 |
| bge_large × LogReg | 0.989 | 0.956 | 0.033 |
| e5_mistral × XGBoost | 0.987 | 0.940 | 0.047 |
| minilm × XGBoost | 0.983 | 0.931 | 0.052 |
| bge_large × XGBoost | 0.986 | 0.927 | 0.058 |
| minilm × LogReg | 0.977 | 0.917 | 0.060 |

**Key findings:**
1. **MLP generalizes best** — smallest ΔF1 across all embeddings (0.018–0.029)
2. **XGBoost generalizes worst** — largest gaps (0.035–0.058), likely overfitting to category-specific tree splits
3. **Best overall: e5_mistral × MLP** — highest LOATO F1 (0.977) with ΔF1=0.019
4. **Top-4 are all MLPs** — the classifier matters more than the embedding for generalization
5. **Embedding dimension ≠ better generalization** — instructor (768d) ties e5_mistral (4096d)
6. **All models achieve >0.91 LOATO F1** — embedding-based classifiers generalize well to unseen attack types

## Hardware

- **Primary**: Apple Silicon Mac (M3 Pro, 18GB RAM, MPS backend)
- **E5-Mistral**: ~8 hours for 69K embeddings via llama-cpp-python with Metal (GGUF Q4)
- **Other models**: Minutes each (sentence-transformers on MPS, OpenAI API)

## Progress

- [x] **Sprint 0** — Scaffolding: ABCs, CLI, configs, CI pipeline
- [x] **Sprint 1A** — Data pipeline + EDA: 5 dataset loaders, harmonization, quality gate, taxonomy Tiers 1+2, EDA with docs
- [x] **Sprint 1B** — Embedding pipeline: 5 models implemented + cached, W&B integration
- [x] **Sprint 2A** — Taxonomy finalization: Tier 3 LLM labeling (GPT-4o-mini), 7-category v1.0, split generation, data artifacts on HF Hub
- [x] **Sprint 2B** — Classifier implementations (LogReg, SVM, XGBoost, MLP) + training pipeline + benign dataset augmentation (4 new sources, 68.8K balanced samples)
- [x] **Sprint 3** — Core experiments: Standard CV + LOATO across all 5 embeddings × 3 classifiers (30 runs complete)
- [ ] **Sprint 4A** — Transfer experiments: direct→indirect, SVM (with PCA), LLM baseline
- [ ] **Sprint 4B** — Analysis & visualization: UMAP, heatmaps, SHAP, final report
- [ ] **Sprint 5** — Integration + thesis write-up

## Experiment Tracking

All experiments are logged to [Weights & Biases](https://wandb.ai/). Run naming convention:

```
{experiment}_{embedding}_{classifier}_{fold}
```

## License

Academic use — Pace University MS Data Science Capstone.
