# LOATO-4B-01: Core Results & ΔF1 Analysis — Summary

Across 20 embedding-classifier combinations, the average Standard CV F1 is **0.9637** while the average LOATO F1 is **0.9130**, yielding a mean generalization gap of **ΔF1 = 0.0507**. The best Standard CV performance is OpenAI-Small (1536d) × MLP (F1 = 0.9974), while the largest generalization gap belongs to Instructor (768d) × SVM (ΔF1 = 0.1496). The smallest gap is BGE-Large (1024d) × MLP (ΔF1 = 0.0181).

Statistical testing shows **3/20** combinations exhibit a statistically significant drop from Standard CV to LOATO (p < 0.05).
Per-fold analysis reveals **obfuscation_encoding** as the hardest held-out category (mean F1 = 0.8738) and **other** as the easiest (mean F1 = 0.9507). This connects to the template homogeneity analysis from Sprint 2A, where categories with higher homogeneity (surface-level patterns) are easier to detect even when held out, while semantically diverse categories expose the generalization gap most starkly.
