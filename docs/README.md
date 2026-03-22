# LOATO-Bench Documentation

**MS Data Science Capstone Project** (Pace University)

This directory contains comprehensive documentation for the LOATO-Bench (Leave-One-Attack-Type-Out Benchmark) project.

---

## 📚 Available Documentation

### [**llm_vulnerability_demo.md**](./llm_vulnerability_demo.md) — LLM Vulnerability Demo Findings
**Scenario 0: Do LLMs Actually Fall for Prompt Injections?**

Live test results from Claude Sonnet and GPT-4o-mini against direct and RAG-style indirect injection attacks. Both models compromised on 3/5 indirect tests; embedding classifier blocked all 5.

**Status**: Complete
**Last Updated**: February 19, 2026

### [**findings_direct_indirect.md**](./findings_direct_indirect.md) — Direct→Indirect Transfer Findings
**Sprint 4A: LOATO-4A-01 + 4A-02 Transfer Experiments**

Results from 20 experiments (5 embeddings × 4 classifiers) training on direct injections and testing on indirect. F1 collapses to 0.21–0.52, revealing a deployment-critical blind spot. SVM added via Nystroem kernel approximation + PCA(128). Includes comparison with Fomin (2026).

**Status**: Complete
**Last Updated**: March 21, 2026

### [**findings_llm_baseline.md**](./findings_llm_baseline.md) — LLM Zero-Shot Baseline Findings
**Sprint 4A: LOATO-4A-03 LLM Baseline**

GPT-4o zero-shot evaluation on 500 samples from Standard CV and Direct→Indirect test pools. GPT-4o scores 0.85 F1 on standard tests (below trained classifiers) but 0.71 F1 on indirect injections (+0.30 over best embedding classifier), confirming the generalization gap is architectural.

**Status**: Complete
**Last Updated**: March 21, 2026

### [**related_work_fomin.md**](./related_work_fomin.md) — Fomin (2026) Positioning
Positioning against Fomin's LODO paper (arXiv:2602.14161). Includes citation, Related Work paragraph draft, sharpened contributions list, and differentiators table.

**Status**: Complete
**Last Updated**: March 21, 2026

### [**EDA.md**](./eda.md) — Exploratory Data Analysis
**Sprint 1A: Data Quality Assessment & Taxonomy Validation**

Complete documentation of the EDA process:
- What we're checking and why
- Goals and objectives
- Methodology and pipeline
- Key findings (to be filled after execution)
- Recommendations for Sprint 2A
- Technical implementation details
- How to run the analysis

**Status**: ✅ Complete
**Last Updated**: February 8, 2026

---

## 📖 Quick Links

### For Researchers
- **Understand the data**: Read [EDA.md § What We're Checking](./eda.md#what-were-checking)
- **View findings**: See [EDA.md § Key Findings](./eda.md#key-findings)
- **Reproduce analysis**: Follow [EDA.md § How to Run](./eda.md#how-to-run)

### For Developers
- **Implementation details**: [EDA.md § Technical Implementation](./eda.md#technical-implementation)
- **API reference**: Check module docstrings in `src/loato_bench/analysis/`
- **Configuration**: Review `configs/analysis/eda.yaml`

### For Stakeholders
- **Executive summary**: [EDA.md § Overview](./eda.md#overview)
- **Recommendations**: [EDA.md § Recommendations](./eda.md#recommendations)
- **Visualizations**: See `results/eda/figures/`

---

## 🔍 Document Structure

```
docs/
├── README.md          # This file (navigation guide)
├── eda.md            # Complete EDA documentation
└── [future docs]     # Sprint 2A, 2B, 3, 4A, 4B docs
```

---

## 🚀 Getting Started

### 1. Read the EDA Documentation
```bash
# Open in your browser or editor
open docs/eda.md
```

### 2. Run the EDA Analysis
```bash
# CLI (recommended)
uv run loato-bench analyze eda

# Or use the Jupyter notebook
uv run jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### 3. Review the Results
```bash
# Check generated reports
ls results/eda/*.json

# View visualizations
open results/eda/figures/
```

---

## 📊 What's in the EDA?

### Data Quality Checks
✅ GenTel quality gate (injection vs content harm filtering)
✅ Text property analysis (length, outliers)
✅ Data integrity validation (missing values, duplicates)

### Taxonomy Validation
✅ Tier 1 + 2 mapping (source mappings + regex patterns)
✅ Category coverage analysis
✅ Merge recommendations for small categories

### Experiment Feasibility
✅ LOATO split viability (≥200 samples per category)
✅ Direct→Indirect transfer (≥500 indirect samples)
✅ Cross-lingual transfer (≥300 per language)
✅ Standard 5-fold CV feasibility

---

## 🎯 Project Context

### Research Question
Can embedding-based classifiers **generalize to novel attack types** they've never seen during training?

### Methodology
**LOATO Evaluation** (Leave-One-Attack-Type-Out)
- Train on K-1 attack categories
- Test on the held-out category
- Measure generalization gap: ΔF1 = Standard_CV_F1 - LOATO_F1

### Dataset
~20,000+ prompt injection samples from 5 public datasets:
- Deepset (1K)
- Open-Prompt (3K)
- PINT (500)
- HackAPrompt (5K)
- GenTel-Bench (177K → filtered to ~5K)

---

## 📅 Project Timeline

| Sprint | Phase | Status | Documentation |
|--------|-------|--------|---------------|
| **Sprint 0** | Scaffolding | ✅ Complete | README.md |
| **Sprint 1A** | Data + EDA | ✅ Complete | **eda.md** |
| **Sprint 1B** | Embeddings | ✅ Complete | - |
| **Sprint 2A** | Taxonomy + Splits | ✅ Complete | taxonomy_spec_v1.0.md |
| **Sprint 2B** | Classifiers | ✅ Complete | - |
| **Sprint 3** | Experiments | ✅ Complete | - |
| **Sprint 4A** | Transfer + SVM | ✅ Complete | **findings_direct_indirect.md**, related_work_fomin.md, findings_llm_baseline.md |
| **Sprint 4B** | Analysis | ⏳ In Progress | **findings_master.md** (§5–§6), 4B-01 tables + 4B-02 threshold + 4B-03 cost-performance complete |
| **Sprint 5** | Write-up | ⏳ Final | TBD |

---

## 🛠️ Tools & Technologies

### Analysis Stack
- **pandas** — Data manipulation
- **numpy** — Numerical operations
- **matplotlib** — Visualization
- **seaborn** — Statistical plots

### ML Stack (Sprint 2B+)
- **scikit-learn** — Classifiers (LogReg, SVM, MLP)
- **xgboost** — Gradient boosting
- **sentence-transformers** — Embeddings
- **llama-cpp-python** — E5-Mistral GGUF

### Infrastructure
- **W&B** — Experiment tracking
- **Jupyter** — Interactive analysis
- **pytest** — Testing (773 tests)
- **mypy** — Type checking
- **ruff** — Linting & formatting

---

## 📧 Contact

**Team**: MS Data Science Capstone (Pace University)
**Project**: LOATO-Bench
**Repository**: [Your repo URL]

For questions:
1. Check relevant documentation in `docs/`
2. Review code docstrings in `src/loato_bench/`
3. Open an issue on GitHub

---

## 📝 Document Conventions

### Status Indicators
- ✅ Complete and verified
- ⏳ In progress
- 📝 Placeholder (TBD)
- ⚠️ Needs attention
- 🔴 Blocked

### Naming Conventions
- `eda.md` — Exploratory Data Analysis
- `taxonomy.md` — Attack taxonomy system (Sprint 2A)
- `experiments.md` — Experiment protocols (Sprint 3)
- `results.md` — Final results & analysis (Sprint 4B)

---

**Last Updated**: February 8, 2026
**Document Version**: 1.0
