# CLAUDE.md — LOATO-Bench

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
uv run ruff check src/ tests/              # Lint
uv run ruff format src/ tests/             # Format
```

## Project Layout

```
src/loato_bench/
  cli.py              # Typer CLI — entrypoint is app_entry()
  data/               # Person A: loaders, harmonize, taxonomy, splits
    base.py           # UnifiedSample dataclass + DatasetLoader ABC
  embeddings/         # Person B: 5 models + cache
    base.py           # EmbeddingModel ABC (name, dim, encode)
  classifiers/        # Person B: LogReg, SVM, XGBoost, MLP
    base.py           # Classifier ABC (name, fit, predict, predict_proba)
  evaluation/         # Person A: LOATO protocol, metrics, transfer, stats
  analysis/           # Person B: visualization, SHAP, report gen
  tracking/           # Person B: W&B utils
  utils/
    config.py         # Pydantic models + YAML loader; PROJECT_ROOT, CONFIGS_DIR, DATA_DIR
    device.py         # get_device() → CPU/MPS/CUDA
    reproducibility.py # seed_everything()
configs/              # YAML configs for embeddings, classifiers, experiments, data
data/                 # (gitignored) raw/ → processed/ → embeddings/ → splits/
results/              # (gitignored) models, metrics, figures
tests/                # pytest suite
```

## Architecture Contracts (ABCs)

**UnifiedSample** (`data/base.py`): Every dataset loader produces these.
- Fields: text, label (0=benign, 1=injection), source, attack_category, original_category,
  language, is_indirect, metadata

**DatasetLoader** (`data/base.py`): `load() -> list[UnifiedSample]`

**EmbeddingModel** (`embeddings/base.py`): Properties `name`, `dim`. Method `encode(texts, batch_size) -> NDArray[float32]`

**Classifier** (`classifiers/base.py`): Property `name`. Methods `fit(X, y)`, `predict(X)`, `predict_proba(X)`

## Key Design Decisions

- All classifiers wrap sklearn pipelines with StandardScaler → classifier
- Embeddings cached as .npz in data/embeddings/{model_name}/
- Splits stored as JSON index files in data/splits/
- Configs are per-model/per-classifier YAMLs loaded via utils/config.py
- Taxonomy uses 3-tier mapping: source mapping → regex heuristics → LLM-assisted (GPT-4o-mini)
- LOATO test sets include held-out attack category + 20% benign sample

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
| crosslingual     | Train English only → test non-English                   | 20    |

## Metrics

Primary: Macro F1 (LOATO). Also: accuracy, precision, recall, AUC-ROC, AUC-PR.
Generalization gap: ΔF1 = Standard_F1 − LOATO_F1.
Stats: bootstrap 95% CIs (10K resamples), McNemar, Friedman + Nemenyi, Cohen's d.

## Sprint Status

- [x] Sprint 0: Scaffolding, ABCs, CLI, configs
- [x] Sprint 1A: Data pipeline (5 dataset loaders, harmonization) — 5/5 loaders done, EDA remains
- [x] Sprint 1B: Embedding pipeline (5 models, cache, W&B utils) — all implemented + tested
- [ ] Sprint 2A: Taxonomy mapping + split generation
- [ ] Sprint 2B: Classifier implementations + training pipeline + sweeps
- [ ] Sprint 3: Core experiments (Standard CV + LOATO)
- [ ] Sprint 4A: Transfer experiments (direct→indirect, cross-lingual, LLM baseline)
- [ ] Sprint 4B: Analysis & visualization (UMAP, heatmaps, SHAP, report)
- [ ] Sprint 5: Integration + write-up

## Conventions

- Python 3.12, ruff for linting/formatting (line-length=100)
- Package managed with uv (pyproject.toml, uv.lock)
- W&B run naming: {experiment}_{embedding}_{classifier}_{fold}
- Seed: 42 everywhere (via seed_everything)
- Hardware: Apple Silicon Mac (18GB), MPS preferred, CPU fallback
- Two developers: Person A (data/evaluation), Person B (ML/infrastructure)

## Known Gotchas

- Instructor-large needs PYTORCH_ENABLE_MPS_FALLBACK=1 or falls back to CPU
- SVM is slow with 4096-dim (E5-Mistral) — may need PCA to 256d
- GenTel-Bench categories are content harm, not injection technique — needs quality gate
- HackAPrompt is injection-only (no benign samples) — needs MinHash near-dedup, cap 5K
- llama-cpp-python needs `CMAKE_ARGS="-DGGML_METAL=on"` for Metal acceleration
