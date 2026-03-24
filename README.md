# LOATO-Bench

**Evaluating Cross-Attack Generalization of Embedding-Based Prompt Injection Classifiers**

Ali Khan & Ahmad Mukhtar — MS Data Science Capstone, Pace University (2026)

[![CI](https://github.com/alikhan126/loato-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/alikhan126/loato-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

## TL;DR

Embedding-based prompt injection classifiers score **0.997 F1** under standard cross-validation — but collapse to **0.21–0.41 F1** when tested on indirect injections they've never seen. LOATO-Bench is an evaluation framework that exposes this gap.

## Overview

Standard cross-validation guarantees every attack type appears in training. This doesn't reflect reality — attackers invent novel techniques. LOATO-Bench evaluates classifiers under three protocols:

| Protocol | What it tests | Key finding |
|----------|--------------|-------------|
| **Standard 5-Fold CV** | In-distribution performance | 0.977–0.997 F1 (looks great) |
| **LOATO** | Generalization to unseen attack types | Mean ΔF1 = 0.051 gap |
| **Direct → Indirect** | Transfer to indirect injections | F1 collapses to 0.21–0.41 |

GPT-4o zero-shot achieves **0.71 F1** on the indirect test set — a +0.30 advantage over the best classifier — confirming the failure is architectural, not a tuning problem.

## Key Results

**5 embeddings × 4 classifiers = 20 combinations**, evaluated on 68,845 samples from 9 public datasets:

| Metric | Value |
|--------|-------|
| Best Standard CV F1 | 0.997 (OpenAI-Small × MLP) |
| Mean LOATO F1 | 0.913 (all 20), 0.957 (excl. SVM) |
| Mean ΔF1 (generalization gap) | 0.051 |
| Hardest category (C3: Obfuscation) | 0.874 F1 when held out |
| Direct → Indirect transfer | 0.21–0.41 F1 (collapse) |
| GPT-4o zero-shot (indirect) | 0.71 F1 |

**MLP generalizes best** (ΔF1 = 0.018–0.029). **SVM generalizes worst** (ΔF1 = 0.071–0.150, Nystroem approximation). The classifier matters more than the embedding.

## Dataset

68,845 samples from 9 public sources, unified through deduplication (SHA-256 + MinHash LSH) and a 7-category attack taxonomy built via 3-tier labeling.

**Injection sources (5):**

| Dataset | Samples | Type |
|---------|---------|------|
| [Open-Prompt-Injection](https://huggingface.co/datasets/guychuk/open-prompt-injection) | ~24K | Indirect injection attacks |
| [HackAPrompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset) | ~4.7K | Competition-winning attacks |
| [GenTel-Bench](https://huggingface.co/datasets/GenTelLab/gentelbench-v1) | ~5K | Content-harm (filtered for injection) |
| [PINT / Gandalf](https://huggingface.co/datasets/lakera/gandalf_ignore_instructions) | ~1K | Password extraction |
| [Deepset](https://huggingface.co/datasets/deepset/prompt-injections) | ~260 | Mixed benign/injection |

**Benign augmentation (4):**

| Dataset | Samples | Source |
|---------|---------|--------|
| [Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | ~14.7K | Human-written instructions |
| [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned) | ~8K | Synthetic instructions |
| [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | ~8K | Real human chat prompts |
| [WildChat](https://huggingface.co/datasets/allenai/WildChat-nontoxic) | ~7.5K | Real ChatGPT sessions |

**Final split:** 40,003 benign (58.1%) / 28,842 injection (41.9%)

### Attack Taxonomy (v1.0)

| ID | Category | N | LOATO |
|----|----------|---|:-----:|
| C1 | Instruction Override | 19,161 | Yes |
| C2 | Jailbreak / Roleplay | 1,614 | Yes |
| C3 | Obfuscation / Encoding | 545 | Yes |
| C4 | Information Extraction | 771 | Yes |
| C5 | Social Engineering | 311 | Yes |
| C6 | Context Manipulation | 188 | — |
| C7 | Other / Multi-Strategy | 8,242 | — |

## Models

### Embeddings

| Model | Dim | Type |
|-------|-----|------|
| all-MiniLM-L6-v2 | 384 | sentence-transformers |
| BGE-large-en-v1.5 | 1024 | sentence-transformers |
| Instructor-large | 768 | InstructorEmbedding |
| text-embedding-3-small | 1536 | OpenAI API |
| E5-Mistral-7B (GGUF Q4) | 4096 | llama-cpp-python |

### Classifiers

All use `StandardScaler` preprocessing. Seed = 42 everywhere.

- **Logistic Regression** (C=1.0, LBFGS)
- **SVM** (RBF, Nystroem 500 + PCA 128 for tractability)
- **XGBoost** (300 trees, depth 6, lr 0.05)
- **MLP** (256-128 hidden, early stopping)

## Setup

```bash
# Clone
git clone https://github.com/alikhan126/loato-bench.git
cd loato-bench

# Install
uv sync

# Environment
cp .env.example .env
# Add HF_TOKEN, OPENAI_API_KEY, WANDB_API_KEY

# Download pre-computed artifacts
uv run python scripts/download_artifacts.py

# Install pre-commit hooks
uv run pre-commit install
```

## Usage

```bash
# Full help
uv run loato-bench --help

# Data pipeline
uv run loato-bench data download
uv run loato-bench data harmonize
uv run loato-bench data label
uv run loato-bench data split

# Embeddings
uv run loato-bench embed run --all

# Training
uv run loato-bench train run --all --experiment standard_cv
uv run loato-bench train run --all --experiment loato
uv run loato-bench train run --all --experiment direct_indirect

# Analysis
uv run loato-bench analyze llm-baseline --model gpt-4o --samples 500
uv run loato-bench analyze template-homogeneity
uv run loato-bench analyze report

# Tests
uv run pytest tests/ -v
```

## Project Structure

```
loato-bench/
├── src/loato_bench/
│   ├── cli.py                 # Typer CLI
│   ├── data/                  # Loaders, harmonization, taxonomy, splits
│   ├── embeddings/            # 5 embedding model implementations + cache
│   ├── classifiers/           # LogReg, SVM, XGBoost, MLP
│   ├── evaluation/            # LOATO protocol, metrics, statistical tests
│   ├── analysis/              # EDA, visualization, template homogeneity
│   └── utils/                 # Config, device, reproducibility
├── paper/                     # IEEE conference paper (LaTeX)
├── configs/                   # YAML configs for models, classifiers, experiments
├── scripts/                   # Standalone scripts (splits, artifacts, CMs)
├── tests/                     # 773 tests, 90%+ coverage
├── data/                      # (gitignored) download from HF Hub
├── results/                   # (gitignored) experiment outputs
└── docs/                      # Research documentation
```

## Pre-computed Artifacts

All data, embeddings, and results are on [HF Hub](https://huggingface.co/datasets/alikhan126/loato-bench-artifacts):

```bash
uv run python scripts/download_artifacts.py              # Everything
uv run python scripts/download_artifacts.py --only embeddings  # ~2.2 GB
uv run python scripts/download_artifacts.py --only results     # Experiment JSONs
uv run python scripts/download_artifacts.py --only data        # Parquets + splits
```

## Reproducibility

All experiments use seed 42. Hardware: Apple Silicon Mac (18 GB), MPS preferred.

```bash
uv sync
uv run loato-bench data download
uv run loato-bench data harmonize
uv run loato-bench data label
uv run loato-bench data split
uv run loato-bench embed run --all
uv run loato-bench train run --all --experiment standard_cv
uv run loato-bench train run --all --experiment loato
uv run loato-bench train run --all --experiment direct_indirect
uv run loato-bench analyze llm-baseline --model gpt-4o --samples 500
uv run loato-bench analyze report
```

## Citation

If you use LOATO-Bench in your research:

```bibtex
@inproceedings{khan2026loato,
  title={LOATO-Bench: Evaluating Cross-Attack Generalization
         of Embedding-Based Prompt Injection Classifiers},
  author={Khan, Ali and Mukhtar, Ahmad},
  year={2026},
  institution={Pace University}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
