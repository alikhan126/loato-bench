"""Meta Prompt-Guard-86M baseline detector."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PromptGuardBaseline:
    """Thin wrapper for Meta's Llama-Prompt-Guard-2-86M model.

    This is NOT an embedding model — it's a complete classifier that outputs
    labels directly via the HuggingFace transformers pipeline.
    """

    def __init__(self, model_id: str = "meta-llama/Llama-Prompt-Guard-2-86M") -> None:
        """Initialize the Prompt-Guard baseline.

        Args:
            model_id: HuggingFace model identifier
        """
        from transformers import pipeline

        self.model_id = model_id
        self.pipe = pipeline("text-classification", model=model_id)

    def predict(self, texts: list[str]) -> NDArray[np.int64]:
        """Run inference and return binary predictions.

        Args:
            texts: List of text samples to classify

        Returns:
            Binary predictions (0=benign, 1=injection)
        """
        results = self.pipe(texts)
        # Prompt-Guard outputs labels like "INJECTION" or "BENIGN"
        predictions = [1 if r["label"].upper() == "INJECTION" else 0 for r in results]
        return np.array(predictions, dtype=np.int64)

    def predict_proba(self, texts: list[str]) -> NDArray[np.float32]:
        """Run inference and return probability distributions.

        Args:
            texts: List of text samples to classify

        Returns:
            Probability matrix of shape (n_samples, 2)
        """
        results = self.pipe(texts)
        probs = []
        for r in results:
            if r["label"].upper() == "INJECTION":
                probs.append([1 - r["score"], r["score"]])
            else:
                probs.append([r["score"], 1 - r["score"]])
        return np.array(probs, dtype=np.float32)
