"""Sentence-transformers models (MiniLM, BGE-large)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from loato_bench.embeddings.base import EmbeddingModel
from loato_bench.utils.config import EmbeddingConfig


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model backed by sentence-transformers.

    Supports optional prefix prepending (needed for BGE-large).
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = SentenceTransformer(config.hf_path, device=config.device)

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def dim(self) -> int:
        return self._config.dim

    def encode(self, texts: list[str], batch_size: int = 32) -> NDArray[np.float32]:
        if self._config.prefix:
            texts = [self._config.prefix + t for t in texts]
        result = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return result.astype(np.float32)
