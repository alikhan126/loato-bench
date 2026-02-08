# Sprint 1B — Embedding Pipeline Implementation Plan

## Pre-conditions (verified)
- ruff check: All checks passed
- pytest: 93 passed in 3.26s
- All 5 embedding YAML configs exist and are well-formed
- `EmbeddingModel` ABC, `EmbeddingConfig` Pydantic model, `get_device()`, `seed_everything()` all ready
- All target files exist as docstring-only stubs

---

## Implementation Order (TDD — tests first, then implementation)

### Step 1: Cache Manager (`embeddings/cache.py`)

**Why first**: Every embedding model will use the cache. Foundation for the entire sprint.

**What to build**:
- `EmbeddingCache` class with methods:
  - `__init__(model_name: str, base_dir: Path | None = None)` — defaults to `DATA_DIR / "embeddings" / model_name`
  - `save(embeddings: NDArray[float32], sample_ids: list[str], model_version: str, text_hash: str)` — writes `.npz` + `.meta.json` sidecar
  - `load() -> tuple[NDArray[float32], list[str]] | None` — returns `(embeddings, sample_ids)` or None if cache miss
  - `is_valid(model_version: str, text_hash: str) -> bool` — checks `.meta.json` against expected version/hash
  - `clear()` — deletes cache files
- `.npz` contains: `embeddings` array (float32 N×D), `sample_ids` array (str)
- `.meta.json` contains: `model_name`, `model_version`, `text_hash`, `n_samples`, `dim`, `created_at`
- Static helper: `compute_text_hash(texts: list[str]) -> str` — SHA-256 of sorted, joined texts

**Tests** (`tests/test_cache.py` — new file, ~12 tests):
- Save and load round-trip preserves data exactly
- `is_valid` returns True for matching hash/version
- `is_valid` returns False when text hash changes
- `is_valid` returns False when model version changes
- `load()` returns None when no cache exists
- `clear()` removes files
- Creates directory if it doesn't exist
- `compute_text_hash` is deterministic
- `compute_text_hash` is order-independent (sorted)
- Large array round-trip (1000×384)

---

### Step 2: Sentence-Transformer Models (`embeddings/sentence_tf.py`)

**What to build**:
- `SentenceTransformerEmbedding` class implementing `EmbeddingModel`:
  - `__init__(config: EmbeddingConfig)` — loads `sentence_transformers.SentenceTransformer(config.hf_path, device=config.device)`
  - Supports optional `config.prefix` prepended to each text (needed for BGE-large)
  - `name` → from config
  - `dim` → from config
  - `encode(texts, batch_size)` → calls `model.encode(texts, batch_size, show_progress_bar=True, convert_to_numpy=True)`
- Two concrete instances via config:
  - **MiniLM**: `hf_path="sentence-transformers/all-MiniLM-L6-v2"`, dim=384, no prefix
  - **BGE-large**: `hf_path="BAAI/bge-large-en-v1.5"`, dim=1024, prefix=`"Represent this sentence for classification: "`

**Tests** (`tests/test_sentence_tf.py` — new file, ~10 tests):
- Contract: is subclass of `EmbeddingModel`
- `.name` returns config name
- `.dim` returns config dim
- `encode()` output shape is (N, dim) with float32 dtype
- Prefix is prepended when configured (mock `SentenceTransformer.encode` to verify input)
- No prefix when config.prefix is None
- Batch size is forwarded
- Works with a single text
- Works with empty list (returns shape (0, dim))

**Note**: Tests will mock `sentence_transformers.SentenceTransformer` to avoid downloading models in CI. One optional integration test marker for real model loading.

---

### Step 3: Instructor Model (`embeddings/instructor.py`)

**What to build**:
- `InstructorEmbedding` class implementing `EmbeddingModel`:
  - `__init__(config: EmbeddingConfig)` — loads `InstructorEmbedding.INSTRUCTOR(config.hf_path)`, moves to device
  - `encode(texts, batch_size)` — passes `[[config.instruction, text] for text in texts]` to `model.encode()`
  - Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` env var if device is CPU (for Apple Silicon compatibility)

**Tests** (`tests/test_instructor.py` — new file, ~8 tests):
- Contract: is subclass of `EmbeddingModel`
- `.name` and `.dim` correct
- `encode()` passes instruction-text pairs (mock to verify)
- Output shape (N, 768) with float32
- Falls back to CPU gracefully
- Single text and empty list edge cases

---

### Step 4: OpenAI Embedding (`embeddings/openai_embed.py`)

**What to build**:
- `OpenAIEmbedding` class implementing `EmbeddingModel`:
  - `__init__(config: EmbeddingConfig)` — creates `openai.OpenAI()` client (reads `OPENAI_API_KEY` from env)
  - `encode(texts, batch_size)` — batches texts, calls `client.embeddings.create(model=config.model_id, input=batch)`
  - Uses `@tenacity.retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5), retry=retry_if_exception_type(openai.RateLimitError))` on the API call
  - Collects all batch results into a single NDArray
  - Batch size default: 2048 (OpenAI's max for this model)

**Tests** (`tests/test_openai_embed.py` — new file, ~9 tests):
- Contract: is subclass of `EmbeddingModel`
- `.name` and `.dim` correct
- `encode()` calls OpenAI API with correct model and input (mock client)
- Batching works correctly (10 texts with batch_size=3 → 4 API calls)
- Results concatenated into single array with correct shape
- Retry decorator is applied (verify with tenacity introspection)
- Empty list edge case
- Single text works

---

### Step 5: E5-Mistral GGUF (`embeddings/e5_mistral.py`)

**What to build**:
- `E5MistralEmbedding` class implementing `EmbeddingModel`:
  - `__init__(config: EmbeddingConfig)` — downloads GGUF if not cached, loads via `llama_cpp.Llama(model_path, embedding=True, n_gpu_layers=-1)` for Metal
  - `encode(texts, batch_size)` — formats each text with `config.prompt_template.format(text=text)`, calls `model.embed(text)` per text (batch_size=1 for GGUF)
  - GGUF download: uses `huggingface_hub.hf_hub_download(config.gguf_repo, config.gguf_file)` — caches locally
  - Extracts embedding from llama-cpp output, stacks into NDArray

**Tests** (`tests/test_e5_mistral.py` — new file, ~8 tests):
- Contract: is subclass of `EmbeddingModel`
- `.name` and `.dim` correct
- `encode()` formats text with prompt template (mock Llama to verify)
- Output shape (N, 4096) with float32
- Downloads GGUF on first use (mock `hf_hub_download`)
- Single text works
- Empty list edge case

**Script** (`scripts/setup_e5_gguf.sh`):
- Pre-downloads the GGUF file for offline use
- `huggingface-cli download second-state/E5-Mistral-7B-Instruct-Embedding-GGUF e5-mistral-7b-instruct-Q4_K_M.gguf`

---

### Step 6: W&B Tracking Utils (`tracking/wandb_utils.py`)

**What to build**:
- `init_run(experiment: str, embedding: str, classifier: str, fold: int | str, config: dict | None = None) -> wandb.Run`
  - Name: `{experiment}_{embedding}_{classifier}_{fold}`
  - Group: `{experiment}_{embedding}_{classifier}`
  - Project: `"promptguard-lite"`
  - Tags: `[experiment, embedding, classifier]`
  - Passes config dict
- `log_metrics(run: wandb.Run, metrics: dict, prefix: str = "") -> None`
  - Prefixes all keys with `{prefix}/` if prefix provided
- `log_confusion_matrix(run: wandb.Run, y_true, y_pred, class_names: list[str]) -> None`
  - Logs a `wandb.plot.confusion_matrix`
- `finish_run(run: wandb.Run) -> None`
  - Calls `run.finish()`

**Tests** (`tests/test_wandb_utils.py` — new file, ~7 tests):
- `init_run` calls `wandb.init` with correct name, group, project, tags
- `log_metrics` calls `run.log` with prefixed keys
- `log_confusion_matrix` logs a plot
- `finish_run` calls `run.finish`
- All with mocked `wandb` module

---

### Step 7: CLI Wiring (update `cli.py`)

**What to build**:
- Wire `embed run` command to:
  1. Load parquet from `data/processed/unified_dataset.parquet`
  2. Instantiate the model from config
  3. Check cache validity
  4. If cache miss: encode all texts, save cache
  5. If cache hit: print "Cache hit, skipping"
  6. Print summary (N samples × D dimensions)
- Factory function: `get_embedding_model(name: str) -> EmbeddingModel` that reads config YAML and returns the right class

**Tests**: Update `tests/test_cli_data.py` or add `tests/test_cli_embed.py` for embed CLI help text.

---

### Step 8: `__init__.py` Exports

Update `src/promptguard/embeddings/__init__.py` to export:
- `EmbeddingModel`
- `EmbeddingCache`
- `SentenceTransformerEmbedding`
- `InstructorEmbedding`
- `OpenAIEmbedding`
- `E5MistralEmbedding`
- `get_embedding_model`

---

## File Change Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `src/promptguard/embeddings/cache.py` | Implement | ~90 |
| `src/promptguard/embeddings/sentence_tf.py` | Implement | ~55 |
| `src/promptguard/embeddings/instructor.py` | Implement | ~55 |
| `src/promptguard/embeddings/openai_embed.py` | Implement | ~75 |
| `src/promptguard/embeddings/e5_mistral.py` | Implement | ~70 |
| `src/promptguard/embeddings/__init__.py` | Update exports | ~15 |
| `src/promptguard/tracking/wandb_utils.py` | Implement | ~55 |
| `src/promptguard/cli.py` | Wire embed command | ~40 |
| `tests/test_cache.py` | New | ~120 |
| `tests/test_sentence_tf.py` | New | ~110 |
| `tests/test_instructor.py` | New | ~80 |
| `tests/test_openai_embed.py` | New | ~100 |
| `tests/test_e5_mistral.py` | New | ~90 |
| `tests/test_wandb_utils.py` | New | ~70 |
| `tests/test_cli_embed.py` | New | ~30 |
| `scripts/setup_e5_gguf.sh` | New | ~10 |

**Total**: ~1,065 new/modified lines across 16 files

## Testing Strategy

- All model tests mock the underlying library (no model downloads in CI)
- Each test file follows the existing pattern: contract tests + functional tests + edge cases
- Optional `@pytest.mark.integration` marker for tests that load real models (skipped by default)
- Target: ~55-60 new tests, bringing total from 93 to ~150+

## Execution Plan

I'll implement in order (Steps 1-8), running `ruff check` + `pytest` after each step to stay green. Each step: write tests → implement → verify.
