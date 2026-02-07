"""Instructor-large embedding model."""

from __future__ import annotations

import os

import numpy as np
from InstructorEmbedding import INSTRUCTOR
from numpy.typing import NDArray

from promptguard.embeddings.base import EmbeddingModel
from promptguard.utils.config import EmbeddingConfig


class InstructorEmbeddingModel(EmbeddingModel):
    """Embedding model backed by Instructor-large.

    Passes instruction-text pairs to the model for task-aware embeddings.
    Sets PYTORCH_ENABLE_MPS_FALLBACK=1 for Apple Silicon compatibility.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        # Instructor-large has MPS compatibility issues; enable fallback
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self._model = INSTRUCTOR(config.hf_path)

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def dim(self) -> int:
        return self._config.dim

    def encode(self, texts: list[str], batch_size: int = 32) -> NDArray[np.float32]:
        instruction = self._config.instruction or ""
        pairs = [[instruction, text] for text in texts]
        result = self._model.encode(pairs, batch_size=batch_size, show_progress_bar=True)
        return result.astype(np.float32)
