"""Base class for embedding models."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class EmbeddingModel(ABC):
    """Abstract base class for embedding models.

    All embedding implementations (sentence-transformers, Instructor,
    OpenAI, E5-Mistral) must implement this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this model (e.g., 'minilm', 'bge_large')."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int = 32) -> NDArray[np.float32]:
        """Encode a list of texts into embedding vectors.

        Args:
            texts: List of input strings to embed.
            batch_size: Number of texts to process per batch.

        Returns:
            Array of shape (len(texts), self.dim) with float32 embeddings.
        """
        ...
