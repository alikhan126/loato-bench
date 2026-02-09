# What We Found: Data Exploration Results

**LOATO-Bench Project** | Pace University, MS Data Science Capstone
**Date**: February 2026

---

## The Big Picture

We're studying whether AI safety classifiers can detect **new types of attacks** they've never
seen before. Think of it like training a spam filter on phishing emails and then testing whether
it can also catch fake invoice scams — a totally different trick, but still spam.

To do this, we collected prompt injection samples from 5 public datasets and merged them into one
unified benchmark. Before we could run any experiments, we needed to understand what we actually
have. That's what this exploration (EDA) was all about.

---

## What We Started With

We pulled **51,659 raw samples** from five different sources. After cleaning (removing exact
duplicates and near-duplicates), we ended up with **37,773 samples** — a 27% reduction.

### A Note on Deduplication

Our initial deduplication was too aggressive (Jaccard threshold 0.85 with word 3-grams), which
dropped us to only 25K samples — a 50% loss. After reviewing community standards (Lee et al.
2022, BigCode, SlimPajama), we adjusted to **0.90 threshold with word 5-grams**, which is the
recommended approach for benchmark datasets of this size. This recovered ~12,000 legitimate
samples while still removing true near-duplicates.

### Final Dataset After Cleaning

| Source | Raw | After Cleaning | What It Contains |
|--------|-----|---------------|-----------------|
| Open-Prompt-Injection | 34,997 | 24,286 | Largest source, broad mix of attacks and safe prompts |
| GenTel-Bench | 10,000 | 7,158 | Originally about AI safety, not just injection attacks |
| HackAPrompt | 5,000 | 4,671 | Competition entries — all attacks, no safe examples |
| PINT / Gandalf | 1,000 | 999 | Focused on indirect injection through documents |
| Deepset | 662 | 659 | Small but clean dataset of injections and benign prompts |
| **Total** | **51,659** | **37,773** | |

---

## Key Findings

### 1. The Dataset is Heavily Skewed Toward Attacks

Out of 37,773 samples, **90% are attacks** (33,914) and only **10% are safe prompts** (3,859).
This isn't surprising — most public datasets focus on collecting attacks, since that's the harder
problem. But it means we'll need to be careful about how we measure success (using balanced
metrics like Macro F1 instead of raw accuracy).

> **Figure**: See `results/eda/figures/label_distribution.png`

### 2. GenTel Needs Heavy Filtering

GenTel-Bench was originally built to study AI content harms broadly — things like bias, toxicity,
and misinformation. That's different from prompt injection (where someone tries to hijack the AI's
behavior). We scored each GenTel sample on how likely it is to be an actual injection attack:

- **62% scored low** (below 0.3 out of 1.0) — these are probably just harmful content, not
  injection attempts
- The average score was only **0.25** — most samples aren't what we need
- After filtering to keep only confident injection samples: **7,158 dropped to 2,068** (we
  removed ~71%)

This is a good thing. Better to have 2,068 high-quality samples than 7,158 questionable ones.

### 3. We Can Now Categorize About Half the Attacks

We have a system that automatically labels each attack by *technique* (e.g., "ignore previous
instructions" or "pretend you're a different AI"). Right now, this system uses two approaches:

- **Tier 1**: Known mappings from the original dataset labels
- **Tier 2**: Pattern matching (looking for telltale phrases)

Together, these cover **53% of injection samples** (18,063 out of 33,914). Here's what we've
mapped so far:

| Attack Category | Samples | Description |
|----------------|---------|-------------|
| Instruction Override | 15,677 | "Ignore your previous instructions and do X" |
| Jailbreak / Roleplay | 2,149 | "Pretend you are DAN, an AI with no rules" |
| Social Engineering | 160 | Manipulating through emotion or urgency |
| Information Extraction | 48 | Trying to extract training data or system prompts |
| Obfuscation / Encoding | 17 | Hiding attacks in Base64, ROT13, etc. |
| Context Manipulation | 12 | Injecting instructions through documents or context |

> **Figure**: See `results/eda/figures/attack_category_distribution.png`

The remaining **15,851 unmapped samples** (47%) will need a smarter approach — an LLM-assisted
categorization step planned for the next sprint.

### 4. Three Attack Categories Are Too Small

For our leave-one-out experiments to be meaningful, each attack category needs at least 200
samples. Three categories fall far short:

- Information Extraction: only **48** samples
- Obfuscation / Encoding: only **17** samples
- Context Manipulation: only **12** samples

These are too small to test on their own. They'll need to be merged into larger, related
categories.

### 5. It's Almost All English

**98% of samples are in English** (36,996 out of 37,773). The remaining 2% spans 30 languages,
with German (269), Dutch (116), and Danish (64) being the most common non-English ones. This
means our cross-lingual experiments will be limited, but we still have enough for a basic
test.

> **Figure**: See `results/eda/figures/language_heatmap.png`

### 6. A Few Outliers, But Nothing Alarming

- **3 very long samples** (over 10,000 characters — one is 15,726 characters long)
- **27 samples** that are mostly uppercase letters (could be encoded or obfuscated attacks)

These are minor quirks, not dealbreakers.

> **Figure**: See `results/eda/figures/text_length_distribution.png`

---

## What This Means

### The Good News
- We have a **solid foundation**: 37K+ samples from diverse sources
- The deduplication pipeline is now well-calibrated (0.90 Jaccard / 5-grams, aligned with
  community standards)
- Open-Prompt-Injection and Deepset are clean, well-labeled sources
- The two largest attack categories (Instruction Override and Jailbreak) have plenty of samples
  for robust experiments
- Taxonomy coverage improved significantly — over half the attacks are already categorized

### The Concerns
- **GenTel quality**: Most of it isn't actually prompt injection — filtering is essential
- **Taxonomy gap**: We still need to categorize ~47% of attacks using an LLM
- **Class imbalance**: 90/10 split means we need balanced evaluation metrics
- **Small categories**: Three attack types don't have enough samples to stand alone

---

## Recommendations

### 1. Apply GenTel Filtering (High Priority)
Use the 0.4 confidence threshold identified in this analysis. This drops GenTel from 7,158 to
2,068 samples but dramatically improves data quality. Better to have fewer, trustworthy
samples.

### 2. Merge Small Categories
Combine the three tiny categories into related larger ones:
- **Information Extraction** (48) and **Context Manipulation** (12) could merge into
  Instruction Override (they're both about getting the AI to reveal or do things it shouldn't)
- **Obfuscation / Encoding** (17) could merge into a broader "evasion techniques" group or
  into Jailbreak / Roleplay

### 3. Run LLM-Assisted Categorization (Next Sprint)
The 15,851 unmapped samples need attention. The plan is to use GPT-4o-mini to read each
sample and assign an attack category. This should push coverage from ~53% to 90%+.

### 4. Use Macro F1 as the Primary Metric
Given the 90/10 class imbalance, accuracy alone would be misleading (a model that always
predicts "attack" would score 90%). Macro F1 treats both classes equally and is the right
choice here.

### 5. Keep Cross-Lingual Scope Modest
With only ~777 non-English samples across 30 languages, cross-lingual experiments should be
treated as exploratory, not definitive. Focus the main conclusions on English performance.

---

## Generated Outputs

All results are saved in `results/eda/`:

| File | What It Contains |
|------|-----------------|
| `figures/label_distribution.png` | How many attacks vs. safe prompts |
| `figures/source_breakdown.png` | Sample counts from each dataset |
| `figures/text_length_distribution.png` | How long the text samples are |
| `figures/language_heatmap.png` | Which languages appear and how often |
| `figures/attack_category_distribution.png` | Breakdown of mapped attack types |
| `gentel_quality_report.json` | GenTel filtering analysis and recommendations |
| `taxonomy_coverage.json` | How many attacks we could auto-categorize |
| `data_integrity.json` | Data quality warnings (outliers, etc.) |

---

## How to Reproduce

```bash
# Step 1: Download the raw data
uv run loato-bench data download

# Step 2: Clean and merge the data
uv run loato-bench data harmonize

# Step 3: Run the exploration
uv run loato-bench analyze eda

# Results appear in results/eda/
```

Or explore interactively:
```bash
uv run jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## What Comes Next

Now that we understand the data, here's the game plan — in order of priority.

### Step 1: Clean Up GenTel (Quick Win)

The EDA showed that 71% of GenTel samples are noise (general content harm, not actual injection
attacks). Applying the 0.4 confidence threshold trims GenTel from 7,158 down to 2,068
high-quality samples. This is a quick win that immediately improves data quality.

### Step 2: Merge the Tiny Attack Categories

Three categories are too small to use on their own (our experiments need at least 200 samples
per category):

| Category | Samples | Merge Into |
|----------|---------|-----------|
| Information Extraction | 48 | Instruction Override |
| Context Manipulation | 12 | Instruction Override |
| Obfuscation / Encoding | 17 | Jailbreak / Roleplay |

After merging, we'll have 3 solid categories to work with — and more will likely emerge from
step 3.

### Step 3: Let an LLM Categorize the Rest (The Big One)

Right now, 15,851 injection samples (47%) are still uncategorized. The automated rules got us
to 53%. The next step is to use GPT-4o-mini to read each unlabeled sample and assign it an
attack category. This should push coverage from ~53% to 90%+ and is the single most important
task before we can run experiments.

Estimated cost: ~$0.50 for all ~16K samples (very affordable).

### Step 4: Generate Experiment Splits

Once the taxonomy is complete, we generate the actual train/test splits:
- **LOATO splits**: For each attack category, train on everything *except* that category,
  then test on it. This is the core of the research.
- **Standard CV splits**: Normal 5-fold cross-validation as a baseline comparison.
- **Transfer splits**: English-only training, test on other languages.

### After That: Build and Run

| Sprint | What Happens | Why It Matters |
|--------|-------------|----------------|
| Sprint 2B | Build the 4 classifiers (Logistic Regression, SVM, XGBoost, Neural Net) | Need all 4 models before we can compare them |
| Sprint 3 | Run the main experiments (Standard CV + LOATO) | This is the core contribution of the thesis |
| Sprint 4A | Run transfer experiments (direct vs. indirect attacks, cross-lingual) | Secondary research questions |
| Sprint 4B | Visualize everything (heatmaps, UMAP plots, feature importance) | Make results interpretable |
| Sprint 5 | Write it all up | Final thesis deliverable |

### Things to Watch Out For

- **Class balance after filtering**: Removing ~5K GenTel samples changes the benign/attack
  ratio. We may need to rebalance afterward.
- **Surprise categories**: The LLM might find attack types we didn't anticipate. Be ready
  to adjust the taxonomy.
- **Recheck feasibility**: After all the filtering, merging, and LLM labeling, re-run the
  split feasibility check to make sure each category still has 200+ samples.

The critical path forward is **LLM-assisted categorization** — every downstream step
(splits, experiments, results, thesis) depends on having a complete and accurate taxonomy.

---

*This document summarizes findings from Sprint 1A of the LOATO-Bench project.*
