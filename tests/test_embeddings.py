"""Tests for embedding models — Sprint 1B."""

import numpy as np

from loato_bench.embeddings.base import EmbeddingModel


def test_embedding_model_is_abstract():
    """EmbeddingModel cannot be instantiated directly."""
    try:
        EmbeddingModel()  # type: ignore[abstract]
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


class DummyEmbedding(EmbeddingModel):
    """Concrete dummy for testing the interface."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def dim(self) -> int:
        return 8

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.random.randn(len(texts), self.dim).astype(np.float32)


def test_dummy_embedding_shape():
    """Concrete embedding should produce correct shape."""
    model = DummyEmbedding()
    texts = ["hello world", "test prompt", "another one"]
    out = model.encode(texts)
    assert out.shape == (3, 8)
    assert out.dtype == np.float32
