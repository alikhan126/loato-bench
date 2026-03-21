# LOATO-4B-01: Core Results & ΔF1 Analysis — Summary

Across 15 embedding-classifier combinations, the average Standard CV F1 is **0.9912** while the average LOATO F1 is **0.9568**, yielding a mean generalization gap of **ΔF1 = 0.0344**. The best Standard CV performance is OpenAI-Small (1536d) × MLP (F1 = 0.9974), while the largest generalization gap belongs to MiniLM (384d) × LogReg (ΔF1 = 0.0601). The smallest gap is BGE-Large (1024d) × MLP (ΔF1 = 0.0181).

Statistical testing shows **3/15** combinations exhibit a statistically significant drop from Standard CV to LOATO (p < 0.05).
Per-fold analysis reveals **obfuscation_encoding** as the hardest held-out category (mean F1 = 0.9186) and **instruction_override** as the easiest (mean F1 = 0.9866). This connects to the template homogeneity analysis from Sprint 2A, where categories with higher homogeneity (surface-level patterns) are easier to detect even when held out, while semantically diverse categories expose the generalization gap most starkly.
