# Experiment Scope Decision (LOATO-2A-04)

## Feasibility Outcome: A — Strong Benchmark

**5 of 6 eligible categories have ≥200 samples** after applying all filters (manual overrides, GenTel quality gate, within-category near-dedup). The full experimental plan stands.

## Viable LOATO Categories

| ID | Category | Sample Count | Viable |
|----|----------|:------------:|:------:|
| C1 | Instruction Override | 19,161 | Yes |
| C2 | Jailbreak / Roleplay | 1,614 | Yes |
| C3 | Obfuscation / Encoding | 545 | Yes |
| C4 | Information Extraction | 771 | Yes |
| C5 | Social Engineering | 311 | Yes |
| C6 | Context Manipulation | 188 | No |
| C7 | Other / Multi-Strategy | 8,242 | n/a |

**LOATO folds**: 5 (C1–C5). Each fold holds out one category for testing and trains on the remaining four plus benign samples.

## C6 Exclusion Note

Context Manipulation (C6) has 188 samples — 12 short of the 200 threshold. All 188 were LLM-labeled (185 from HackAPrompt, 3 from Deepset). C6 samples are retained in the dataset and included in training folds but are not used as a held-out test category. This is a limitation noted in the thesis: indirect injection is underrepresented in public prompt-injection datasets, which itself is an interesting finding.

## Filters Applied

1. **Manual overrides** (2A-03): 1,065 uncertain samples relabeled via GPT-4o-mini second pass
2. **GenTel quality gate**: injection_score ≥ 0.4 (dropped 0 — all GenTel injection samples met threshold)
3. **GenTel cap**: 5,000 max (not triggered — only 2,021 GenTel injection samples)
4. **Within-category near-dedup**: Jaccard 0.90, word 5-grams (removed 0 — dataset was already globally deduplicated during harmonization)

## Experiment Matrix

With 5 LOATO folds × 5 embedding models × 4 classifiers = **100 LOATO runs** (plus 100 standard CV runs for comparison).

## Thesis Table 1 — Dataset Summary

| Statistic | Count |
|-----------|------:|
| Total samples | 32,683 |
| Injection samples | 30,849 |
| Benign samples | 1,834 |
| LOATO-eligible categories | 5 |
| LOATO-eligible injection samples | 22,402 |
| Non-eligible injection (C6 + C7 + uncategorized) | 8,447 |
