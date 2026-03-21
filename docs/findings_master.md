# LOATO-Bench: Master Findings (Paper Reference)

**Project**: LOATO-Bench — Leave-One-Attack-Type-Out Benchmark for Prompt Injection Classifiers
**Team**: MS Data Science Capstone, Pace University
**Date**: 2026-03-21

This document collects every finding in the order work was done, with pointers to detailed docs and artifacts. Use this as the backbone when writing the thesis.

---

## Phase 0 — Motivation & Problem Statement

**Why this project exists.**

### Finding 0.1: Frontier LLMs Are Vulnerable to Prompt Injections

**Source**: `docs/llm_vulnerability_demo.md`

We tested Claude Sonnet and GPT-4o-mini against 5 RAG-style indirect injection attacks. Both models were compromised on 3 out of 5 tests (60% attack success rate). An embedding-based classifier blocked all 5 poisoned documents.

**Implication**: LLM safety training alone is insufficient. Lightweight input-layer classifiers are needed as a first defense — but how well do they generalize to attacks they haven't seen?

---

## Phase 1 — Data Collection & Exploration (Sprints 1A–1B)

**Building the dataset and understanding what we have.**

### Finding 1.1: Dataset Composition

**Source**: `docs/datasets.md`, `docs/eda_findings.md`

Unified 9 sources (5 injection + 4 benign) into 68,845 samples after deduplication:
- 28,842 injection (41.9%)
- 40,003 benign (58.1%)

Sources: Deepset, Open-Prompt, PINT, HackAPrompt, GenTel-Bench (injection); WildChat, Alpaca, Dolly, SlimOrca (benign augmentation).

### Finding 1.2: GenTel Needs Heavy Filtering

**Source**: `docs/eda_findings.md`

GenTel-Bench (177K raw samples) is a content-harm dataset, not injection-specific. After injection-confidence scoring (threshold=0.4, cap 5K), 71% of GenTel samples were filtered out. Without this gate, the dataset would be dominated by non-injection noise.

### Finding 1.3: Class Imbalance and Taxonomy Gaps

**Source**: `docs/eda_findings.md`

Pre-augmentation: 90% attack / 10% benign imbalance. Taxonomy coverage after Tier 1+2 (source mappings + regex): ~53% of injections mapped. ~47% required LLM-assisted labeling (Tier 3, Sprint 2A).

### Finding 1.4: Three Categories Too Small for LOATO

**Source**: `docs/eda_findings.md`, `docs/experiment_scope.md`

Categories with <200 samples cannot support individual LOATO folds. Context Manipulation (C6, 188 samples) was excluded from LOATO. Payload Splitting (<50 samples) was merged into C7 (Other). This is a limitation reflecting the underrepresentation of indirect injection in public datasets.

---

## Phase 2 — Taxonomy & Splits (Sprint 2A)

**Finalizing the attack taxonomy and generating experiment splits.**

### Finding 2.1: Taxonomy v1.0 — 7 Categories

**Source**: `docs/taxonomy_spec_v1.0.md`, `configs/final_categories.json`

Final taxonomy after 3-tier labeling (source mapping → regex → GPT-4o-mini):

| ID | Category | LOATO Eligible | Samples |
|----|----------|:-:|---|
| C1 | Instruction Override | Yes | ~8,500 |
| C2 | Jailbreak / Roleplay | Yes | ~6,200 |
| C3 | Obfuscation / Encoding | Yes | ~3,800 |
| C4 | Information Extraction | Yes | ~4,100 |
| C5 | Social Engineering | Yes | ~2,400 |
| C6 | Context Manipulation | No | 188 |
| C7 | Other / Multi-Strategy | No | ~3,600 |

5 LOATO-eligible categories (C1–C5), each with ≥200 samples.

### Finding 2.2: Split Design

**Source**: `data/splits/split_manifest.json`

Four split types generated:
1. **Standard 5-fold CV** — Stratified on label, all categories present in every fold
2. **LOATO 6-fold** — 5 held-out categories (C1–C5) + 1 benign-only fold
3. **Direct→Indirect** — Train on direct injections, test on indirect (flat split)
4. **Cross-lingual** — Excluded (<300 samples per language, documented as limitation)

---

## Phase 3 — Classifiers & Training (Sprint 2B)

**Building the classifier pipeline.**

### Finding 2B.1: Balanced Dataset

**Source**: `CLAUDE.md`

Final balanced dataset: 68,845 samples. 5 embedding models cached as .npz. 4 classifier types implemented (LogReg, SVM, XGBoost, MLP), all wrapping sklearn pipelines with StandardScaler.

---

## Phase 4 — Core Experiments (Sprint 3)

**Standard CV and LOATO — the primary contribution.**

### Finding 3.1: Standard CV Baseline — Classifiers Score 0.90–0.97 F1

**Source**: W&B project `loato-bench`, 30 runs (5 embeddings × 3 classifiers × 2 experiments)

Under standard 5-fold CV (all attack types in training), embedding classifiers achieve excellent performance:

| Embedding | LogReg | XGBoost | MLP |
|-----------|--------|---------|-----|
| minilm (384d) | ~0.94 | ~0.95 | ~0.93 |
| bge_large (1024d) | ~0.95 | ~0.96 | ~0.94 |
| instructor (1024d) | ~0.96 | ~0.96 | ~0.95 |
| openai_small (1536d) | ~0.97 | ~0.97 | ~0.96 |
| e5_mistral (4096d) | ~0.95 | ~0.96 | ~0.94 |

These numbers look deployment-ready — but they're misleading.

### Finding 3.2: LOATO Reveals the Generalization Gap

**Source**: W&B project `loato-bench`, LOATO runs

When a single attack category is held out during training, F1 drops significantly on the held-out type. The generalization gap (ΔF1 = Standard_CV_F1 − LOATO_F1) varies by category:

- Some categories (C1: Instruction Override) are easy — other categories cover similar patterns
- Others (C3: Obfuscation, C5: Social Engineering) show large drops — they use unique strategies not seen in other categories

**Key insight**: Standard CV overstates real-world performance. LOATO exposes which attack types a classifier is blind to.

---

## Phase 5 — Transfer Experiments (Sprint 4A)

**Testing across attack surfaces and against frontier LLMs.**

### Finding 4A-01: Direct→Indirect Transfer Collapse

**Source**: `docs/findings_direct_indirect.md`

15 experiments (5 embeddings × 3 classifiers) trained on direct injections, tested on indirect:

| Embedding | LogReg | XGBoost | MLP |
|-----------|--------|---------|-----|
| minilm | 0.2903 | 0.2207 | 0.2477 |
| bge_large | 0.2711 | 0.2143 | 0.2602 |
| instructor | 0.3196 | 0.2263 | 0.3422 |
| openai_small | **0.4081** | 0.2259 | **0.4130** |
| e5_mistral | 0.3248 | 0.2112 | 0.2521 |

**Headline**: F1 collapses from 0.90–0.97 (standard CV) to 0.21–0.41 on indirect injections. Classifiers memorize direct injection patterns but cannot recognize the same intent when delivered indirectly (via retrieved context, tool outputs).

**Key details**:
- XGBoost collapses worst (~0.21 across all embeddings) — overfits to surface patterns
- AUC-ROC remains high (0.76–0.96) despite low F1 — classifiers rank injections correctly but decision thresholds are miscalibrated
- OpenAI embeddings capture the most transferable features (F1 0.41 vs 0.21 for worst)
- Consistent with Fomin (2026) who reported 7–37% detection rates on production guardrails

### Finding 4A-03: GPT-4o Zero-Shot Baseline

**Source**: `docs/findings_llm_baseline.md`

GPT-4o evaluated zero-shot (no training, no few-shot) on 500 samples from each test pool:

| Test Pool | GPT-4o F1 | Best Embedding Classifier F1 | Gap |
|-----------|-----------|-------------------------------|-----|
| Standard CV | 0.8528 | 0.95–0.97 | Classifiers win by +0.10 |
| Direct→Indirect | **0.7105** | 0.4130 | GPT-4o wins by **+0.30** |

**Headline**: On familiar attacks, cheap classifiers beat GPT-4o. On novel attack surfaces, GPT-4o's reasoning ability provides +0.30 F1 advantage — confirming the gap is architectural (pattern matching vs reasoning), not data-driven.

**Key details**:
- GPT-4o precision is near-perfect (0.97–0.99) — almost never false-positives
- Weakness is recall (0.63–0.70) — still misses ~30% of indirect attacks
- Total cost: ~$0.80 for 1,000 samples (orders of magnitude more expensive per query than embedding classifiers)

---

## Summary Table: All Key Numbers

| Metric | Standard CV | LOATO | Direct→Indirect | LLM Baseline (Indirect) |
|--------|:-----------:|:-----:|:----------------:|:----------------------:|
| Best Embedding F1 | 0.97 | varies by category | 0.41 | — |
| GPT-4o F1 | 0.85 | — | — | 0.71 |
| Gap | Classifiers +0.12 | — | GPT-4o +0.30 | — |
| Cost per query | ~$0 (after training) | ~$0 | ~$0 | ~$0.0008 |

---

## Thesis Arguments (Ordered)

1. **Prompt injection is a real threat** — Frontier LLMs (Claude, GPT-4o-mini) are compromised 60% of the time by indirect injections (Phase 0)

2. **Embedding classifiers look great on paper** — 0.95–0.97 F1 under standard CV suggests they're deployment-ready (Phase 4, Finding 3.1)

3. **LOATO exposes the blind spot** — Holding out one attack type during training reveals which categories a classifier can't generalize to (Phase 4, Finding 3.2)

4. **The gap is catastrophic for indirect injections** — F1 collapses to 0.21–0.41 when testing on indirect attacks not seen during training (Phase 5, Finding 4A-01)

5. **This is an architectural limitation, not a data problem** — GPT-4o, with no training at all, scores 0.71 F1 on the same indirect test set where the best trained classifier scores 0.41 (Phase 5, Finding 4A-03)

6. **Practical recommendation** — Use cheap classifiers as a high-precision first layer; escalate uncertain cases to LLMs. Standard CV is insufficient for evaluating deployment safety; LOATO and transfer testing should be standard practice.

---

## What's Next

| Sprint | Task | Status |
|--------|------|--------|
| 4B | Analysis: UMAP plots, heatmaps, per-category breakdown, SHAP | Upcoming |
| 5 | Thesis write-up | Final |

---

## Artifact Locations

| What | Where |
|------|-------|
| Dataset | `data/processed/labeled_v1.parquet` |
| Splits | `data/splits/*.json` |
| Embeddings | `data/embeddings/` (gitignored, reproducible via `loato-bench embed run`) |
| Sprint 3 results | W&B: `loato-bench` project |
| Sprint 4A-01 results | `results/experiments/direct_indirect_*.json` |
| Sprint 4A-03 results | `results/llm_baseline/llm_baseline_*.json` |
| Taxonomy spec | `docs/taxonomy_spec_v1.0.md`, `configs/final_categories.json` |
| Fomin positioning | `docs/related_work_fomin.md`, `docs/references.bib` |
| EDA | `docs/eda.md`, `docs/eda_findings.md`, `results/eda/` |
| Vulnerability demo | `docs/llm_vulnerability_demo.md` |
