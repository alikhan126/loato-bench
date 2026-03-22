# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

MS Data Science capstone (Pace University) studying cross-attack generalization of embedding-based
prompt injection classifiers. Core contribution: LOATO (Leave-One-Attack-Type-Out) evaluation —
train on K-1 attack types, test on the held-out type. 5 embedding models × 4 classifiers on a
unified benchmark of ~20K+ samples from 5 public datasets.

## Commands

```bash
uv sync                                    # Install deps
uv run loato-bench --help                  # CLI help
uv run pytest tests/ -v                    # Run tests
uv run pytest tests/test_eda.py -v         # Run single test file
uv run pytest tests/test_eda.py::TestLoadParquetSafely::test_loads_valid_parquet_file -v  # Single test
uv run pytest tests/ --cov=loato_bench --cov-report=term  # Coverage report
uv run ruff check src/ tests/              # Lint
uv run ruff format src/ tests/             # Format
uv run mypy src/loato_bench/               # Type check
```

### Data & EDA Commands

```bash
# Data pipeline
uv run loato-bench data download           # Download raw datasets
uv run loato-bench data harmonize          # Harmonize to UnifiedSample format

# EDA (Sprint 1A - COMPLETE)
uv run loato-bench analyze eda             # Run full EDA pipeline
uv run loato-bench analyze eda --output-dir=results/my_eda --no-log-wandb

# Interactive EDA
uv run jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### Embedding & Training Commands

```bash
# Embeddings (Sprint 1B - COMPLETE)
uv run loato-bench embed run --all         # All 5 models
uv run loato-bench embed run --model minilm

# Training (Sprint 2B+)
uv run loato-bench train run --all --experiment loato
uv run loato-bench train run --embedding minilm --classifier xgboost --experiment loato

# Sweeps
uv run loato-bench sweep run --all

# Analysis
uv run loato-bench analyze features --all
uv run loato-bench analyze llm-baseline --samples 500
uv run loato-bench analyze report
```

## Project Layout

```
src/loato_bench/
  cli.py              # Typer CLI — entrypoint is app_entry()
  data/
    base.py           # UnifiedSample dataclass + DatasetLoader ABC
    harmonize.py      # Deduplication (exact SHA-256 + MinHash LSH), language detection
    taxonomy.py       # 3-tier mapping (Tier 1: source, Tier 2: regex, Tier 3: LLM)
    deepset.py, gentel.py, hackaprompt.py, open_prompt.py, pint.py  # Dataset loaders
  embeddings/
    base.py           # EmbeddingModel ABC (name, dim, encode)
    minilm.py, bge_large.py, instructor.py, openai_small.py, e5_mistral.py
    cache.py          # .npz caching in data/embeddings/{model_name}/
  classifiers/
    base.py           # Classifier ABC (name, fit, predict, predict_proba)
    # logreg.py, svm.py, xgboost.py, mlp.py — all wrap sklearn pipelines with StandardScaler
  analysis/           # Sprint 1A EDA - COMPLETE (107 tests, 90%+ coverage)
    eda.py            # Core statistics, text properties, label/source/language distributions
    quality.py        # GenTel quality gate (injection confidence scoring), data integrity
    visualization.py  # 5 plot types (label, source, text length, language, attack category)
  evaluation/
    # LOATO protocol, metrics, transfer experiments, statistical tests
  tracking/
    wandb_utils.py    # W&B integration (init_run, log_metrics, log_eda_artifacts)
  utils/
    config.py         # Pydantic models + YAML loader; PROJECT_ROOT, CONFIGS_DIR, DATA_DIR
    device.py         # get_device() → CPU/MPS/CUDA
    reproducibility.py # seed_everything(42)
configs/              # YAML configs for embeddings, classifiers, experiments, data, analysis
  analysis/eda.yaml   # EDA params (GenTel filtering, text analysis, visualization, split feasibility)
  data/taxonomy.yaml  # Attack category definitions + regex patterns
data/                 # Mostly gitignored — selective files tracked via LFS (see below)
  processed/          # labeled_v1.parquet, unified_dataset.parquet (LFS-tracked)
  splits/             # 4 split index JSONs + split_manifest.json (LFS-tracked)
  labeling/           # Audit trail: coverage/labeling reports, llm_labels_raw.jsonl
  raw/                # (gitignored) downloaded source datasets
  embeddings/         # (gitignored) .npz caches per model
  review/             # (future, Sprint 2A-03) error_rate_report.csv, manual_overrides.csv
results/              # (gitignored) models, metrics, figures
  eda/                # EDA outputs: figures/*.png, *.json reports
docs/                 # Documentation (NEW - Sprint 1A)
  eda.md              # Comprehensive EDA guide (28KB) — goals, methodology, findings, how-to
  taxonomy_spec_v1.0.md # Taxonomy v1.0 specification (7 categories)
  README.md           # Docs navigation
notebooks/            # Jupyter notebooks
  01_exploratory_data_analysis.ipynb  # Interactive EDA (50+ cells)
tests/                # pytest suite (107 tests passing)
  test_eda.py, test_quality.py, test_visualization.py, test_taxonomy.py  # EDA test suite
```

## Architecture Contracts (ABCs)

**UnifiedSample** (`data/base.py`): Every dataset loader produces these.
- Fields: text, label (0=benign, 1=injection), source, attack_category, original_category,
  language, is_indirect, metadata

**DatasetLoader** (`data/base.py`): `load() -> list[UnifiedSample]`

**EmbeddingModel** (`embeddings/base.py`): Properties `name`, `dim`. Method `encode(texts, batch_size) -> NDArray[float32]`

**Classifier** (`classifiers/base.py`): Property `name`. Methods `fit(X, y)`, `predict(X)`, `predict_proba(X)`

## Key Design Decisions

### Data Pipeline (Sprint 1A - COMPLETE)
- **Harmonization**: NFC unicode → exact dedup (SHA-256) → near dedup (MinHash LSH, Jaccard 0.90, word 5-grams) → language detection
- **GenTel Quality Gate**: Heuristic injection confidence scoring (0-1) using keyword matching. Threshold=0.4, cap at 5K samples
- **Taxonomy**: 3-tier mapping system
  - Tier 1: Source-specific mappings (e.g., Open-Prompt "jailbreak" → `jailbreak_roleplay`)
  - Tier 2: Regex patterns (e.g., "ignore previous" → `instruction_override`)
  - Tier 3: LLM-assisted (GPT-4o-mini, Sprint 2A, for ambiguous cases only)
- **EDA validates**: Data quality, taxonomy coverage (~70% after Tier 1+2), split feasibility (LOATO needs ≥200/category)

### Embeddings & Classifiers
- Embeddings cached as .npz in `data/embeddings/{model_name}/`
- All classifiers wrap sklearn pipelines: `StandardScaler → classifier`
- Splits stored as JSON index files in `data/splits/`
- LOATO test sets: held-out attack category + 20% benign samples

### Configuration & Reproducibility
- Configs: per-model/per-classifier YAMLs loaded via `utils/config.py`
- W&B run naming: `{experiment}_{embedding}_{classifier}_{fold}`
- Seed: 42 everywhere (via `seed_everything()`)
- Hardware: Apple Silicon Mac (18GB), MPS preferred, CPU fallback

## Embedding Models

| Key       | Model                   | Dim  | Notes                                    |
|-----------|-------------------------|------|------------------------------------------|
| minilm    | all-MiniLM-L6-v2        | 384  | sentence-transformers, MPS               |
| bge_large | BGE-large-en-v1.5       | 1024 | sentence-transformers, prefix required    |
| instructor| Instructor-large        | 768  | InstructorEmbedding, CPU fallback         |
| openai_small | text-embedding-3-small | 1536 | OpenAI API, tenacity retries           |
| e5_mistral| E5-Mistral-7B GGUF Q4   | 4096 | llama-cpp-python, Metal                  |

## Classifiers

LogReg, SVM (RBF, probability=True), XGBoost, MLP (2-layer, sklearn MLPClassifier, early stopping)

## Experiments

| Experiment       | What                                                    | ~Runs |
|------------------|---------------------------------------------------------|-------|
| standard_cv      | Stratified 5-fold CV                                    | 100   |
| loato            | Leave-one-attack-type-out (primary contribution)        | 120-160 |
| direct_indirect  | Train direct only → test indirect                       | 20    |
| ~~crosslingual~~ | ~~Train English only → test non-English~~ (out of scope — <300 samples per language) | — |

## Metrics

Primary: Macro F1 (LOATO). Also: accuracy, precision, recall, AUC-ROC, AUC-PR.
Generalization gap: ΔF1 = Standard_F1 − LOATO_F1.
Stats: bootstrap 95% CIs (10K resamples), McNemar, Friedman + Nemenyi, Cohen's d.

## Sprint Status

- [x] Sprint 0: Scaffolding, ABCs, CLI, configs
- [x] Sprint 1A: Data pipeline + EDA (5/5 loaders, harmonization, quality gate, taxonomy Tier 1+2, EDA complete with docs)
- [x] Sprint 1B: Embedding pipeline (5 models, cache, W&B utils) — all implemented + tested
- [x] Sprint 2A: Taxonomy finalization (Tier 3 LLM labeling, 7-category v1.0) + split generation + data artifacts tracked in Git LFS
- [x] Sprint 2B: Classifier implementations + training pipeline + balanced dataset (68.8K samples)
- [x] Sprint 3: Core experiments (Standard CV + LOATO) — 5 embeddings × 3 classifiers (30 runs)
- [x] Sprint 4A: Transfer experiments (direct→indirect), SVM (with PCA + Nystroem), LLM baseline — 5 emb × 4 clf × 3 experiments (60 runs)
- [ ] Sprint 4B: Analysis & visualization (UMAP, heatmaps, SHAP, report)
- [ ] Sprint 5: Integration + write-up

## EDA (Sprint 1A) — Key Points for Future Work

**Status**: ✅ COMPLETE (107 tests passing, 90%+ coverage, mypy + ruff clean)

**What was validated**:
1. **GenTel Quality**: 177K samples → need filtering via injection confidence scores (threshold=0.4, cap 5K)
2. **Taxonomy Coverage**: ~70% mapped with Tier 1+2, rest needs Tier 3 LLM (Sprint 2A)
3. **Split Feasibility**: Checked LOATO (need ≥200/category), direct→indirect (need ≥500 indirect), cross-lingual (need ≥300/language)
4. **Data Integrity**: Validated schema, identified outliers, documented quality issues

**Key Outputs**:
- `results/eda/figures/*.png` — 5 publication-ready plots (150 DPI)
- `results/eda/gentel_quality_report.json` — Filtering recommendations
- `results/eda/taxonomy_coverage.json` — Category coverage + merge suggestions
- `results/eda/data_integrity.json` — Validation warnings
- `docs/eda.md` — Comprehensive 28KB guide with methodology, findings (TBD), recommendations (TBD)
- `notebooks/01_exploratory_data_analysis.ipynb` — Interactive analysis (50+ cells)

**For Sprint 2A**: Use EDA findings to:
1. Apply GenTel filtering (threshold from EDA)
2. Merge small taxonomy categories (merge plan from EDA)
3. Apply Tier 3 LLM to unmapped samples (scope: ~30% of samples)
4. Generate LOATO splits (validated categories only)

## Taxonomy v1.0 (Finalized — Sprint 2A)

7 attack categories (C1–C7), defined in `src/loato_bench/data/taxonomy_spec.py` and exported to `configs/final_categories.json`:

| ID | Slug | Name | LOATO Eligible |
|----|------|------|:--------------:|
| C1 | `instruction_override` | Instruction Override | Yes |
| C2 | `jailbreak_roleplay` | Jailbreak / Roleplay | Yes |
| C3 | `obfuscation_encoding` | Obfuscation / Encoding | Yes |
| C4 | `information_extraction` | Information Extraction | Yes |
| C5 | `social_engineering` | Social Engineering | Yes |
| C6 | `context_manipulation` | Context Manipulation / Indirect Injection | Yes |
| C7 | `other` | Other / Multi-Strategy | No |

**Migration from 8-slug draft**: Old `context_manipulation` (system prompt extraction) → C4; old `indirect_injection` → C6; old `payload_splitting` (<50 samples) → C7. See `OLD_SLUG_TO_NEW` in `taxonomy_spec.py`.

## Git LFS & Data Tracking

### What's Tracked (committed to repo)

Large binary/data files are stored via **Git LFS** (`.gitattributes`):

| File | Size | Storage | Purpose |
|------|------|---------|---------|
| `data/processed/labeled_v1.parquet` | 5.4MB | LFS | Final labeled dataset (32,683 samples) |
| `data/processed/unified_dataset.parquet` | 6.5MB | LFS | Pre-labeling harmonized dataset |
| `data/splits/standard_cv_folds.json` | 2.3MB | LFS | Stratified 5-fold CV indices |
| `data/splits/loato_splits.json` | 3.3MB | LFS | LOATO fold indices (6 folds) |
| `data/splits/crosslingual_split.json` | 341KB | LFS | Cross-lingual split indices |
| `data/splits/direct_indirect_split.json` | 340KB | LFS | Direct→indirect split indices |
| `data/splits/split_manifest.json` | 1KB | LFS | SHA-256 checksums of all splits |
| `data/labeling/llm_labels_raw.jsonl` | 5MB | LFS | Raw GPT-4o-mini labeling output (audit trail) |

Small metadata files tracked via regular Git:

| File | Purpose |
|------|---------|
| `data/labeling/coverage_report.json` | Labeling coverage statistics |
| `data/labeling/labeling_report.json` | Labeling pipeline summary |
| `configs/final_categories.json` | Taxonomy v1.0 export (7 categories) |
| `docs/taxonomy_spec_v1.0.md` | Taxonomy specification document |

### What's Gitignored (NOT committed)

| Path | Why |
|------|-----|
| `data/raw/` | Downloaded source datasets (reproducible via `loato-bench data download`) |
| `data/embeddings/` | `.npz` caches (reproducible via `loato-bench embed run`) |
| `data/processed/*` (except two parquets) | Intermediate processing artifacts |
| `data/labeling/*` (except 3 audit files) | Batch request files, intermediate processing |
| `results/` | All experiment outputs (models, metrics, figures) |
| `.env` | API keys (OPENAI, WANDB) |

### Pre-wired for Future Tracking (Sprint 2A-03)

These paths have `.gitignore` negation rules ready but files don't exist yet:
- `data/review/error_rate_report.csv` — Manual review error analysis
- `data/review/manual_overrides.csv` — Human corrections to LLM labels

### LFS Patterns (`.gitattributes`)

```
*.parquet    → LFS (all parquet files)
*.npy        → LFS (numpy arrays)
*.npz        → LFS (compressed numpy, currently gitignored but LFS-ready)
data/splits/*.json → LFS (split index files are large)
data/labeling/llm_labels_raw.jsonl → LFS
```

### Adding New Data Files

To track a new data file:
1. If >1MB, ensure the extension/path is in `.gitattributes` for LFS
2. Add a negation rule to `.gitignore` (e.g., `!/data/new_dir/file.ext`)
3. Use `git add -f data/path/to/file` (force-add past gitignore)
4. If it contains SHA-256 hashes, exclude its path from detect-secrets in `.pre-commit-config.yaml`
5. Update `split_manifest.json` if it's a new split file

### Gotcha: detect-secrets

`data/` is excluded from detect-secrets scanning (`.pre-commit-config.yaml`) because SHA-256 hashes and hex strings in data files trigger false positives. This is intentional — actual secrets (API keys) are in `.env` which is gitignored.

## Conventions

- Python 3.12, ruff for linting/formatting (line-length=100)
- Package managed with uv (pyproject.toml, uv.lock)
- Type hints: 100% coverage (mypy strict mode)
- Docstrings: Google-style format
- Tests: pytest, aim for 90% coverage (currently exempt: CLI, classifiers, evaluation)
- Security: Path traversal prevention, input validation, no eval/exec, API keys from env only
- Hardware: Apple Silicon Mac (18GB), MPS preferred, CPU fallback

## Known Gotchas

- Instructor-large needs PYTORCH_ENABLE_MPS_FALLBACK=1 or falls back to CPU
- SVM is slow with 4096-dim (E5-Mistral) — may need PCA to 256d
- GenTel-Bench categories are content harm, not injection technique — use quality gate (implemented in `analysis/quality.py`)
- HackAPrompt is injection-only (no benign samples) — needs MinHash near-dedup, cap 5K
- llama-cpp-python needs `CMAKE_ARGS="-DGGML_METAL=on"` for Metal acceleration
- Coverage exemptions: `cli.py`, `classifiers/*`, `evaluation/*` are currently stub/incomplete (Sprint 2B+)
- EDA `analysis/*` modules are NOW COVERED (no longer exempt) — maintain 90%+ coverage

## Code Quality Standards (Enforced — Two Layers)

### Layer 1: Pre-commit Hooks (local, every commit)

Pre-commit runs automatically on `git commit`. Hooks installed via:
```bash
uv run pre-commit install           # One-time setup (already done)
uv run pre-commit run --all-files   # Manual run on all files
```

**Hooks** (in order):
1. **File hygiene** — trailing whitespace, end-of-file fix, YAML/TOML check, large file check, merge conflicts, debug statements
2. **Ruff lint** — auto-fixes and fails on remaining errors (`--fix --exit-non-zero-on-fix`)
3. **Ruff format** — auto-formats Python and Jupyter files
4. **mypy** — type checking (strict mode, via `uv run mypy`)
5. **detect-secrets** — prevents accidental secret commits (baseline: `.secrets.baseline`)

### Layer 2: CI Pipeline (GitHub Actions, every PR)

```bash
# These mirror pre-commit but run independently on CI
uv run mypy                                    # Type check job
uv run ruff check src/ tests/                  # Lint job
uv run ruff format --check src/ tests/         # Lint job
uv run pytest tests/ -v --tb=short --cov       # Test job
```

### Quick Commands

```bash
# Tests (must pass with ≥90% coverage for non-exempt modules)
uv run pytest tests/ --cov=loato_bench --cov-report=term

# Full QA (pre-commit + tests)
uv run pre-commit run --all-files && uv run pytest tests/ -v
```

## Important Files to Check

**Before modifying data pipeline**:
- `src/loato_bench/data/base.py` — UnifiedSample schema (don't break this)
- `src/loato_bench/data/harmonize.py` — Dedup + language detection pipeline
- `src/loato_bench/data/taxonomy_spec.py` — Taxonomy v1.0 (TAXONOMY_V1, single source of truth)
- `src/loato_bench/data/llm_labeler.py` — GPT-4o-mini batch labeling pipeline
- `configs/data/taxonomy.yaml` — Category definitions + regex patterns (Tier 1+2)
- `configs/final_categories.json` — Machine-readable taxonomy export (keep in sync with taxonomy_spec.py)

**Before modifying splits or data tracking**:
- `.gitattributes` — LFS tracking patterns
- `.gitignore` — Negation rules for tracked data paths
- `data/splits/split_manifest.json` — SHA-256 checksums (update after regenerating splits)

**Before modifying EDA**:
- `docs/eda.md` — Methodology documentation (keep in sync with code)
- `configs/analysis/eda.yaml` — EDA parameters (GenTel thresholds, plot settings)
- `notebooks/01_exploratory_data_analysis.ipynb` — Interactive notebook (reflects latest findings)

**Before modifying embeddings**:
- `src/loato_bench/embeddings/base.py` — EmbeddingModel ABC (don't break interface)
- `src/loato_bench/embeddings/cache.py` — Caching logic (.npz format)
- `configs/embeddings/*.yaml` — Model-specific configs (dim, batch_size, device, etc.)

**Before modifying classifiers**:
- `src/loato_bench/classifiers/base.py` — Classifier ABC (fit, predict, predict_proba)
- `configs/classifiers/*.yaml` — Hyperparams + sweep configs

## Documentation

- **README.md**: Project overview, setup, data tracking, usage, experiment matrix
- **CLAUDE.md**: This file (architecture, conventions, data tracking, gotchas)
- **docs/eda.md**: Comprehensive EDA guide (goals, methodology, findings, recommendations)
- **docs/taxonomy_spec_v1.0.md**: Taxonomy v1.0 specification (7 categories, boundary rules)
- **docs/README.md**: Documentation navigation
- Code docstrings: Google-style (Parameters, Returns, Raises sections)

## W&B Integration

All experiments log to Weights & Biases:
- Run naming: `{experiment}_{embedding}_{classifier}_{fold}`
- Project: `loato-bench`
- Tags: experiment type, model names
- Artifacts: models, reports, figures
- Summary metrics: F1, accuracy, precision, recall, AUC

EDA also logs to W&B:
- `uv run loato-bench analyze eda` creates run tagged `eda`, `sprint-1a`
- Logs: dataset statistics, quality reports, all figures, JSON reports

Set `WANDB_API_KEY` in `.env` (copy from `.env.example`).
