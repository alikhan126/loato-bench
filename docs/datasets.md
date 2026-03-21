# Dataset Documentation

Complete reference for all datasets used in LOATO-Bench. The benchmark combines
5 injection-focused datasets with 4 benign augmentation datasets to create a
balanced corpus for training and evaluating prompt injection classifiers.

---

## Summary

| | Injection | Benign | Total |
|---|-----------|--------|-------|
| **Raw (downloaded)** | 39,863 | 40,575 | 80,438 |
| **After harmonization** | 28,828 | 40,017 | 68,845 |

Harmonization removes ~12K duplicates (exact SHA-256 + MinHash near-dedup at
Jaccard 0.90) and applies NFC normalization + language detection.

---

## Injection Datasets (5 sources)

### 1. Deepset — `deepset/prompt-injections`

- **HuggingFace ID**: `deepset/prompt-injections`
- **License**: MIT
- **Total rows**: 662 (train: 546, test: 116)
- **Labels**: Mixed — ~400 benign, ~260 injection
- **Language**: English
- **Loader**: `src/loato_bench/data/deepset.py` → `DeepsetLoader`
- **Schema**: `text`, `label` (0/1)
- **Notes**: Small but high-quality, manually labeled. One of the few datasets
  with both benign and injection samples. All samples marked as direct injection.

### 2. HackAPrompt — `hackaprompt/hackaprompt-dataset`

- **HuggingFace ID**: `hackaprompt/hackaprompt-dataset`
- **License**: Apache 2.0
- **Total rows**: ~12K raw → 5,000 after filtering + cap
- **Labels**: Injection-only (label=1)
- **Language**: English
- **Loader**: `src/loato_bench/data/hackaprompt.py` → `HackaPromptLoader`
- **Schema**: `user_input`, `prompt`, `completion`, `model`, `correct`, `level`,
  `error`, `score`
- **Filtering**: Keeps only `correct=True` and `error=False` (successful
  injection attempts). Deduplicates on `user_input`.
- **Notes**: Competition dataset from the HackAPrompt challenge. Multi-level
  injection attempts against GPT-3.5, FlanT5, etc. No benign samples.

### 3. GenTel-Bench — `GenTelLab/gentelbench-v1`

- **HuggingFace ID**: `GenTelLab/gentelbench-v1`
- **License**: CC-BY-4.0
- **Total rows**: 177K → capped at 10K in loader, ~2K after harmonization
- **Labels**: Mixed benign/injection
- **Language**: English
- **Loader**: `src/loato_bench/data/gentel.py` → `GenTelLoader`
- **Schema**: `id`, `text`, `label`, `domain`, `subdomain`
- **Quality gate**: Categories are content-harm types, NOT injection techniques.
  Requires injection confidence scoring (threshold=0.4) via `analysis/quality.py`
  to filter for actual injection samples.
- **Status**: Currently failing due to a stale HuggingFace cache
  (column `combined_text` vs `text` schema mismatch). Can be fixed by clearing
  `~/.cache/huggingface/datasets/GenTelLab*`. Not blocking since dataset is
  balanced without it.

### 4. PINT/Gandalf — `lakera/gandalf_ignore_instructions`

- **HuggingFace ID**: `lakera/gandalf_ignore_instructions`
- **License**: Apache 2.0
- **Total rows**: 1,000
- **Labels**: Injection-only (label=1)
- **Language**: English
- **Loader**: `src/loato_bench/data/pint.py` → `PINTLoader`
- **Schema**: `text`, `similarity`
- **Notes**: Prompts from the Gandalf challenge (Lakera). All are injection
  attempts designed to extract a secret password. Includes similarity scores.

### 5. Open-Prompt-Injection — `guychuk/open-prompt-injection`

- **HuggingFace ID**: `guychuk/open-prompt-injection`
- **License**: MIT
- **Total rows**: ~33.6K rows → ~35K samples (each row produces 1 injection +
  1 deduplicated benign)
- **Labels**: Mixed — ~1.4K unique benign, ~33.6K injection
- **Language**: English
- **Loader**: `src/loato_bench/data/open_prompt.py` → `OpenPromptLoader`
- **Schema**: `attack_input`, `normal_input`, `attack_type`, `task_type`,
  `injected_task`, `instruction`
- **Extraction**: Each row yields two samples:
  - Injection (label=1) from `attack_input`, marked `is_indirect=True`
  - Benign (label=0) from `normal_input`, deduplicated across rows
- **Notes**: Largest injection source. Primarily indirect injection attacks
  (injected into task context). The `attack_type` field maps to taxonomy via
  Tier 1 source mapping.

---

## Benign Augmentation Datasets (4 sources)

Added to address severe class imbalance (originally 94.4% injection / 5.6%
benign). These provide diverse, realistic benign prompts representative of
normal LLM usage.

### 6. Dolly 15K — `databricks/databricks-dolly-15k`

- **HuggingFace ID**: `databricks/databricks-dolly-15k`
- **License**: CC-BY-SA-3.0 (commercial + research OK)
- **Total rows**: 15,011 → ~14,779 after dedup (cap: 15,000)
- **Labels**: Benign-only (label=0)
- **Language**: English
- **Loader**: `src/loato_bench/data/dolly.py` → `DollyLoader`
- **Schema**: `instruction`, `context`, `response`, `category`
- **Text field**: `instruction`
- **Quality**: Human-written by ~5,000 Databricks employees. Categories include
  brainstorming, classification, closed QA, generation, information extraction,
  open QA, summarization.
- **Why chosen**: Largest source of high-quality human-written instructions.
  Diverse task types cover realistic LLM usage patterns.

### 7. OpenAssistant OASST1 — `OpenAssistant/oasst1`

- **HuggingFace ID**: `OpenAssistant/oasst1`
- **License**: Apache 2.0
- **Total rows**: ~88K messages → ~8,000 after filtering (cap: 8,000)
- **Labels**: Benign-only (label=0)
- **Language**: 35 languages in dataset; we filter for English (`lang == "en"`)
- **Loader**: `src/loato_bench/data/oasst.py` → `OASSTLoader`
- **Schema**: `message_id`, `parent_id`, `text`, `role`, `lang`
- **Filtering**: `role == "prompter"` (user messages only) + `lang == "en"`
- **Text field**: `text`
- **Quality**: Real human prompts from 13,500+ volunteers. Highly diverse topics
  including coding, writing, reasoning, factual questions.
- **Why chosen**: Real user prompts (not synthetic), Apache 2.0 license,
  conversation-style queries that complement the instruction-style Dolly samples.

### 8. WildChat — `allenai/WildChat-nontoxic`

- **HuggingFace ID**: `allenai/WildChat-nontoxic`
- **License**: ODC-BY (permissive, commercial OK)
- **Total rows**: ~530K conversations → ~8,000 after filtering (cap: 8,000)
- **Labels**: Benign-only (label=0)
- **Language**: Multilingual; we filter for English (`language == "English"`)
- **Loader**: `src/loato_bench/data/wildchat.py` → `WildChatLoader`
- **Schema**: `conversation` (list of `{role, content}` dicts), `language`,
  `model`
- **Extraction**: First `role == "user"` turn from each conversation
- **Access**: Gated dataset — requires requesting access on HuggingFace. Token
  must be available via `huggingface_hub.get_token()`.
- **Quality**: Real user prompts from ChatGPT interactions. The most realistic
  representation of actual LLM usage — includes code-switching, ambiguous
  requests, multi-part questions.
- **Why chosen**: Gold standard for "what users actually type." Pre-filtered for
  non-toxic content. Complements Dolly (instructions) and OASST (conversations)
  with naturalistic chat prompts.

### 9. Alpaca (cleaned) — `yahma/alpaca-cleaned`

- **HuggingFace ID**: `yahma/alpaca-cleaned`
- **License**: CC-BY-NC-4.0 (research/non-commercial OK)
- **Total rows**: 51,760 → 8,000 after dedup + cap (cap: 8,000)
- **Labels**: Benign-only (label=0)
- **Language**: English
- **Loader**: `src/loato_bench/data/alpaca.py` → `AlpacaLoader`
- **Schema**: `instruction`, `input`, `output`
- **Text composition**: `instruction` alone when `input` is empty;
  `instruction + "\n" + input` when `input` is non-empty
- **Quality**: Synthetic (generated by GPT-3 text-davinci-003 via self-instruct).
  Cleaned version removes duplicates and fixes errors from the original Stanford
  Alpaca. Diverse instruction types.
- **Why chosen**: Fills remaining gap to balance. Synthetic diversity complements
  the human-written sources. Cleaned version is higher quality than original.

---

## Benign Source Diversity

The 4 benign sources were chosen to cover different prompt styles users send to
LLMs:

| Source | Prompt Style | Origin | Count |
|--------|-------------|--------|-------|
| Dolly | Task instructions (summarize, classify, brainstorm) | Human-written | ~14.8K |
| OASST | Conversational questions (how-to, explain, help me) | Human-written | ~8K |
| WildChat | Naturalistic chat (real ChatGPT sessions) | Human-written | ~8K |
| Alpaca | Diverse instructions (translate, write, generate) | Synthetic (GPT-3) | ~8K |

This diversity prevents the classifier from learning surface-level shortcuts
(e.g., "all benign samples are instructions" or "all benign samples are
questions") and forces it to learn the actual distinction between benign prompts
and injection attacks.

---

## Post-Harmonization Distribution

After deduplication and language detection:

### By Label
| Label | Count | Percentage |
|-------|-------|------------|
| Benign (0) | 40,017 | 58.1% |
| Injection (1) | 28,828 | 41.9% |
| **Total** | **68,845** | **100%** |

### By Source
| Source | Count |
|--------|-------|
| open_prompt_injection | 24,286 |
| dolly | 14,738 |
| alpaca | 7,994 |
| oasst | 7,974 |
| wildchat | 7,524 |
| hackaprompt | 4,671 |
| pint | 999 |
| deepset | 659 |

### Injection Categories (Taxonomy v1.0)
| Category | Count | LOATO Eligible |
|----------|-------|:--------------:|
| instruction_override (C1) | 18,108 | Yes |
| other (C7) | 8,355 | No |
| information_extraction (C4) | 896 | Yes |
| jailbreak_roleplay (C2) | 622 | Yes |
| obfuscation_encoding (C3) | 544 | Yes |
| social_engineering (C5) | 303 | Yes |

---

## Pipeline Flow

```
Download (9 loaders)
  → 80,438 raw samples (all_samples.pkl)

Harmonize (NFC normalize → SHA-256 dedup → MinHash near-dedup → lang detect)
  → 68,845 samples (unified_dataset.parquet)

Label (Tier 1+2 regex → carry forward old labels → uncertain → "other")
  → 68,845 samples, 100% categorized (labeled_v1.parquet)

Split (standard CV, LOATO, direct→indirect, cross-lingual)
  → 4 split index JSON files (data/splits/)
```

---

## Reproducing the Dataset

```bash
# Full pipeline from scratch
uv run loato-bench data download      # ~5 min (WildChat is 530K rows)
uv run loato-bench data harmonize     # ~2 min
uv run loato-bench data label         # Tier 1+2 + LLM for unlabeled
uv run loato-bench data split         # Generate all evaluation splits
```

**Requirements**:
- HuggingFace token with access to `allenai/WildChat-nontoxic` (gated dataset)
- OpenAI API key for Tier 3 LLM labeling (only ~1K samples need this)

---

*Last updated: 2026-03-19*
