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
| **Cross-lingual** | Train on English, test on non-English |

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

### Why Git LFS?

Several data files (parquet datasets, split indices, labeling outputs) are between 1–7MB each. GitHub rejects files over 100MB and warns above 50MB, and committing multi-megabyte binaries directly bloats the repo's clone size permanently (they can't be garbage-collected from git history). Git LFS replaces these files with lightweight pointers in the repo while storing the actual content on a separate LFS server. This keeps `git clone` fast while still versioning the data.

### Git LFS Setup

**Git LFS must be installed before cloning**, otherwise you'll get 130-byte pointer files instead of actual data.

```bash
# Install Git LFS (one-time per machine)
brew install git-lfs   # macOS
git lfs install        # configures git hooks

# Then clone normally — LFS files download automatically
git clone <repo-url>
```

If you already cloned without LFS, retroactively fetch the real files:

```bash
git lfs install
git lfs pull
```

### What's Tracked and Why

We commit only the files that are **hard to reproduce** (require API calls, manual review, or represent the exact dataset the experiments run on) or that serve as an **audit trail** for the committee. Everything that's cheap to regenerate from code is gitignored.

#### Datasets (LFS)

| File | Size | Why it's tracked |
|------|------|------------------|
| `data/processed/unified_dataset.parquet` | ~8MB | The harmonized benchmark before labeling — proves the dedup/filtering pipeline output. Regenerating requires downloading all 9 source datasets and re-running harmonization. |
| `data/processed/labeled_v1.parquet` | ~8MB | The final labeled dataset (68,845 samples) that all experiments run on. This is the single artifact that must be identical across all experiment runs for results to be comparable. |

#### Splits (LFS)

| File | Size | Why it's tracked |
|------|------|------------------|
| `data/splits/standard_cv_folds.json` | 2.3MB | 5-fold stratified CV indices. Exact fold assignments matter for reproducibility — even with the same seed, library version differences could produce different splits. |
| `data/splits/loato_splits.json` | 3.3MB | LOATO fold indices (6 folds, one per eligible category). This is the core evaluation protocol — the primary contribution of the thesis. |
| `data/splits/direct_indirect_split.json` | 340KB | Direct→indirect transfer experiment indices. |
| `data/splits/crosslingual_split.json` | 341KB | English→non-English transfer experiment indices. |
| `data/splits/split_manifest.json` | 1KB | SHA-256 checksums of all split files and `labeled_v1.parquet`. Allows anyone to verify data integrity without re-running the pipeline — if the checksums match, the experiments are running on the exact same data. |

#### Labeling Audit Trail (LFS + Git)

| File | Size | Storage | Why it's tracked |
|------|------|---------|------------------|
| `data/labeling/llm_labels_raw.jsonl` | 5MB | LFS | Every raw GPT-4o-mini response for Tier 3 labeling. This is the audit trail — it proves what the model was asked, what it returned, and how those responses were mapped to categories. Required for committee review and error analysis. |
| `data/labeling/coverage_report.json` | 2KB | Git | Summary of how many samples each tier labeled. Shows that Tier 3 LLM was only used where Tiers 1+2 couldn't classify. |
| `data/labeling/labeling_report.json` | 2KB | Git | Pipeline run summary (timestamps, sample counts, error rates). |

#### Config Exports (Git)

| File | Why it's tracked |
|------|------------------|
| `configs/final_categories.json` | Machine-readable export of taxonomy v1.0 (7 categories, LOATO eligibility flags). Ensures configs and code stay in sync — generated from `taxonomy_spec.py`. |

### What's NOT Tracked and Why

| Path | Why it's gitignored | How to reproduce |
|------|---------------------|------------------|
| `data/raw/` | Source datasets are publicly available and large (~200MB+). No reason to duplicate them in the repo. | `uv run loato-bench data download` |
| `data/embeddings/` | `.npz` caches are deterministic given the same model + dataset. They're also large (hundreds of MB across 5 models). | `uv run loato-bench embed run --all` |
| `data/labeling/batch_requests.jsonl` | 87MB batch API request file. Too large even for LFS, and fully reproducible from code + `unified_dataset.parquet`. | Re-run labeling pipeline |
| `data/labeling/batch_id_mapping.json` | OpenAI batch job ID mapping — ephemeral, only useful during the API call. | Re-run labeling pipeline |
| `results/` | All experiment outputs (trained models, metrics, figures). These are the *results* of the research, regenerated by running experiments. | Re-run experiments |
| `.env` | Contains `OPENAI_API_KEY` and `WANDB_API_KEY`. Never commit secrets. | Copy `.env.example` and fill in your keys |

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
- [Git LFS](https://git-lfs.com/) — required for data files (see [Git LFS Setup](#git-lfs-setup) above)
- Apple Silicon Mac (MPS backend) recommended, CPU works too

### Installation

```bash
# 1. Ensure Git LFS is installed (data files won't download without it)
git lfs install

# 2. Clone (LFS files download automatically during clone)
git clone <repo-url>
cd loato-bench

# 3. Install Python dependencies
uv sync

# 4. Set up API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and WANDB_API_KEY

# 5. Verify data integrity (optional — checks SHA-256 hashes)
cat data/splits/split_manifest.json
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
├── .gitattributes          # Git LFS tracking patterns (*.parquet, splits, etc.)
├── .gitignore              # Broad /data/ ignore + selective negation rules
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
├── data/                   # Mostly gitignored — only specific files tracked
│   ├── processed/          # ★ labeled_v1.parquet, unified_dataset.parquet (LFS)
│   ├── splits/             # ★ 4 split index JSONs + integrity manifest (LFS)
│   ├── labeling/           # ★ Audit trail: reports + raw LLM outputs (LFS + Git)
│   ├── raw/                # (gitignored) downloaded source datasets
│   └── embeddings/         # (gitignored) .npz embedding caches per model
├── results/                # (gitignored) all experiment outputs
├── docs/                   # EDA guide, taxonomy spec, dataset docs, methodology notes
├── notebooks/              # Interactive analysis (EDA, embeddings, results)
├── scripts/                # Setup scripts (e.g., GGUF model download)
└── tests/                  # pytest suite (712 tests, 90%+ coverage)
```

★ = committed to repo for reproducibility/audit (large files via Git LFS, see above)

## Experiment Matrix

| Experiment | Models | Classifiers | Folds | Total Runs |
|-----------|--------|-------------|-------|------------|
| Standard 5-Fold CV | 5 | 4 | 5 | 100 |
| LOATO | 5 | 4 | 6-8 | 120-160 |
| Direct → Indirect | 5 | 4 | 1 | 20 |
| Cross-lingual | 5 | 4 | 1 | 20 |
| Feature Analysis (SHAP) | 5 | 1 | 1 | 5 |
| LLM Baseline | 2 | — | 1 | 2 |
| **Total** | | | | **~270-310** |

## Key Metrics

- **Macro F1 (LOATO)**: Primary metric — average F1 across held-out categories
- **ΔF1 = Standard_F1 − LOATO_F1**: The generalization gap
- **Degradation %**: ΔF1 / Standard_F1 × 100
- Bootstrap 95% confidence intervals (10,000 resamples)
- McNemar's test, Friedman test + Nemenyi post-hoc for statistical comparisons

## Initial Results (Sprint 2B/3)

Preliminary results from 4 embedding models × 3 classifiers (SVM deferred, E5-Mistral in progress):

| Model × Classifier | Standard CV | LOATO F1 | ΔF1 |
|---------------------|------------|----------|-----|
| instructor × MLP | **0.997** | **0.977** | 0.020 |
| bge_large × MLP | 0.994 | 0.976 | **0.018** |
| openai_small × MLP | 0.997 | 0.976 | 0.022 |
| minilm × MLP | 0.992 | 0.963 | 0.029 |
| instructor × LogReg | 0.995 | 0.967 | 0.028 |
| openai_small × LogReg | 0.995 | 0.966 | 0.029 |
| bge_large × LogReg | 0.989 | 0.957 | 0.033 |
| openai_small × XGBoost | 0.993 | 0.957 | 0.035 |
| instructor × XGBoost | 0.994 | 0.958 | 0.036 |
| minilm × XGBoost | 0.983 | 0.931 | 0.052 |
| bge_large × XGBoost | 0.986 | 0.928 | 0.058 |
| minilm × LogReg | 0.977 | 0.917 | 0.060 |

**Key findings so far:**
1. **MLP generalizes best** — smallest ΔF1 across all embeddings (0.018–0.029)
2. **XGBoost generalizes worst** — largest gaps (0.035–0.058), likely overfitting to category-specific tree splits
3. **Best overall: instructor × MLP** — highest LOATO F1 (0.977) with tiny 0.020 gap
4. **Embedding dimension ≠ better generalization** — instructor (768d) beats openai_small (1536d)
5. **All models achieve >0.91 LOATO F1** — embedding-based classifiers generalize reasonably well to unseen attack types

## Hardware

- **Primary**: Apple Silicon Mac (18GB RAM, MPS backend)
- **Fallback**: Google Colab / Kaggle free GPU for heavy models (E5-Mistral)
- **Estimated total compute**: ~2-3 hours for all embeddings + training runs

## Progress

- [x] **Sprint 0** — Scaffolding: ABCs, CLI, configs, CI pipeline
- [x] **Sprint 1A** — Data pipeline + EDA: 5 dataset loaders, harmonization, quality gate, taxonomy Tiers 1+2, EDA with docs
- [x] **Sprint 1B** — Embedding pipeline: 5 models implemented + cached, W&B integration
- [x] **Sprint 2A** — Taxonomy finalization: Tier 3 LLM labeling (GPT-4o-mini), 7-category v1.0, split generation, data artifacts in Git LFS
- [x] **Sprint 2B** — Classifier implementations (LogReg, SVM, XGBoost, MLP) + training pipeline + benign dataset augmentation (4 new sources, 68.8K balanced samples)
- [x] **Sprint 3** — Core experiments: Standard CV + LOATO evaluation (24/32 runs complete, initial results above)
- [ ] **Sprint 4A** — Transfer experiments: direct→indirect, cross-lingual, LLM baseline
- [ ] **Sprint 4B** — Analysis & visualization: UMAP, heatmaps, SHAP, final report
- [ ] **Sprint 5** — Integration + thesis write-up

## Experiment Tracking

All experiments are logged to [Weights & Biases](https://wandb.ai/). Run naming convention:

```
{experiment}_{embedding}_{classifier}_{fold}
```

## License

Academic use — Pace University MS Data Science Capstone.
