# LOATO-Bench

**Cross-Attack Generalization of Embedding-Based Prompt Injection Classifiers**

MS Data Science Capstone — Pace University

## Overview

LOATO-Bench studies whether embedding-based prompt injection classifiers trained on *known* attack types can detect *unseen* attack categories. The core contribution is a **LOATO (Leave-One-Attack-Type-Out)** evaluation protocol applied to 5 embedding models × 4 classifiers on a unified benchmark of ~20K+ samples from 5 public datasets.

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

| Dataset | Samples | Notes |
|---------|---------|-------|
| [Deepset Prompt Injections](https://huggingface.co/datasets/deepset/prompt-injections) | ~662 | Binary labels, simplest source |
| [HackAPrompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) | ~5K (filtered) | Successful attacks only, injection-only |
| [GenTel-Bench](https://huggingface.co/datasets/GenTelLab/gentelbench-v1) | Subset | Quality-gated for genuine injection techniques |
| [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection) | Varies | Indirect injection samples |
| [PINT / Gandalf](https://huggingface.co/datasets/lakera/gandalf_ignore_instructions) | ~1.3K+ | Multilingual prompt injections |

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

## Data & Reproducibility

### Git LFS

This repo uses **Git LFS** for large data files. Install it before cloning:

```bash
# Install Git LFS (one-time)
brew install git-lfs   # macOS
git lfs install

# Clone (LFS files download automatically)
git clone <repo-url>
```

### Tracked Data Artifacts

Key data files are committed for reproducibility and audit:

| File | Size | Storage | Purpose |
|------|------|---------|---------|
| `data/processed/labeled_v1.parquet` | 5.4MB | LFS | Final labeled dataset (32,683 samples, taxonomy v1.0) |
| `data/processed/unified_dataset.parquet` | 6.5MB | LFS | Pre-labeling harmonized dataset |
| `data/splits/*.json` (4 files) | 6.3MB | LFS | Train/test index files for all 4 experiment protocols |
| `data/splits/split_manifest.json` | 1KB | LFS | SHA-256 checksums for reproducibility verification |
| `data/labeling/llm_labels_raw.jsonl` | 5MB | LFS | Raw GPT-4o-mini labeling output (audit trail) |
| `data/labeling/coverage_report.json` | 2KB | Git | Labeling coverage statistics |
| `data/labeling/labeling_report.json` | 2KB | Git | Labeling pipeline summary |
| `configs/final_categories.json` | 2KB | Git | Taxonomy v1.0 category definitions |

### What's NOT Tracked (gitignored)

| Path | Reproducible via |
|------|------------------|
| `data/raw/` | `uv run loato-bench data download` |
| `data/embeddings/` | `uv run loato-bench embed run --all` |
| `results/` | Re-run experiments |
| `.env` | Copy from `.env.example` |

### Attack Taxonomy (v1.0)

7 categories (6 LOATO-eligible), defined in `src/loato_bench/data/taxonomy_spec.py`:

| ID | Category | Mechanism |
|----|----------|-----------|
| C1 | Instruction Override | "Ignore previous instructions" |
| C2 | Jailbreak / Roleplay | Persona adoption to bypass safety |
| C3 | Obfuscation / Encoding | Base64, ROT13, leetspeak evasion |
| C4 | Information Extraction | System prompt / training data extraction |
| C5 | Social Engineering | Emotional manipulation, authority claims |
| C6 | Context Manipulation | Indirect injection via documents/tools |
| C7 | Other / Multi-Strategy | Catch-all (not LOATO-eligible) |

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Git LFS](https://git-lfs.com/) for large data files
- Apple Silicon Mac (MPS backend) recommended, CPU works too

### Installation

```bash
git clone <repo-url>
cd loato-bench

# Install all dependencies
uv sync

# Copy and fill in API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and WANDB_API_KEY
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
├── .gitattributes        # Git LFS tracking rules
├── Justfile
├── configs/
│   ├── data/             # sources.yaml, taxonomy.yaml
│   ├── embeddings/       # one YAML per embedding model
│   ├── classifiers/      # one YAML per classifier
│   ├── experiments/      # one YAML per experiment protocol
│   └── final_categories.json  # Taxonomy v1.0 export
├── src/loato_bench/
│   ├── cli.py            # Typer CLI entrypoints
│   ├── data/             # Dataset loaders, harmonization, taxonomy, splits
│   │   ├── taxonomy_spec.py   # Taxonomy v1.0 (single source of truth)
│   │   └── llm_labeler.py     # GPT-4o-mini batch labeling
│   ├── embeddings/       # Embedding model implementations + cache
│   ├── classifiers/      # Classifier implementations (LogReg, SVM, XGB, MLP)
│   ├── evaluation/       # LOATO protocol, metrics, transfer, statistical tests
│   ├── analysis/         # Visualization, SHAP analysis, report generation
│   ├── tracking/         # W&B integration
│   └── utils/            # Config loading, device selection, reproducibility
├── data/                 # Selectively tracked (see Data & Reproducibility above)
│   ├── processed/        # ★ labeled_v1.parquet, unified_dataset.parquet (LFS)
│   ├── splits/           # ★ 4 split JSONs + manifest (LFS)
│   ├── labeling/         # ★ Audit trail (reports + raw labels)
│   ├── raw/              # (gitignored) source datasets
│   └── embeddings/       # (gitignored) .npz caches
├── results/              # (gitignored) models, metrics, figures
├── docs/                 # EDA guide, taxonomy spec
├── notebooks/            # EDA, embedding viz, results analysis, SHAP
├── scripts/              # Setup scripts (e.g., GGUF model download)
└── tests/                # pytest test suite
```

★ = committed to repo (Git LFS for large files)

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

## Hardware

- **Primary**: Apple Silicon Mac (18GB RAM, MPS backend)
- **Fallback**: Google Colab / Kaggle free GPU for heavy models (E5-Mistral)
- **Estimated total compute**: ~2-3 hours for all embeddings + training runs

## Tracking

All experiments are logged to [Weights & Biases](https://wandb.ai/). Run naming convention:

```
{experiment}_{embedding}_{classifier}_{fold}
```

## License

Academic use — Pace University MS Data Science Capstone.
