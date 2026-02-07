"""Tests for sentence-transformer embedding models — Sprint 1B."""

from unittest.mock import MagicMock, patch

import numpy as np

from promptguard.embeddings.base import EmbeddingModel
from promptguard.embeddings.sentence_tf import SentenceTransformerEmbedding
from promptguard.utils.config import EmbeddingConfig


def _minilm_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        name="minilm",
        hf_path="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        library="sentence-transformers",
        device="cpu",
        batch_size=64,
    )


def _bge_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        name="bge_large",
        hf_path="BAAI/bge-large-en-v1.5",
        dim=1024,
        library="sentence-transformers",
        device="cpu",
        batch_size=32,
        prefix="Represent this sentence for classification: ",
    )


class TestSentenceTransformerContract:
    """ABC compliance tests."""

    def test_is_subclass_of_embedding_model(self):
        assert issubclass(SentenceTransformerEmbedding, EmbeddingModel)

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_name_returns_config_name(self, mock_cls):
        model = SentenceTransformerEmbedding(_minilm_config())
        assert model.name == "minilm"

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_dim_returns_config_dim(self, mock_cls):
        model = SentenceTransformerEmbedding(_minilm_config())
        assert model.dim == 384


class TestSentenceTransformerEncode:
    """Tests for the encode method."""

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_output_shape_and_dtype(self, mock_cls):
        """encode() returns (N, dim) float32 array."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock_cls.return_value = mock_instance

        model = SentenceTransformerEmbedding(_minilm_config())
        out = model.encode(["a", "b", "c"])

        assert out.shape == (3, 384)
        assert out.dtype == np.float32

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_no_prefix_passes_texts_directly(self, mock_cls):
        """Without prefix, texts are passed as-is."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.zeros((2, 384), dtype=np.float32)
        mock_cls.return_value = mock_instance

        model = SentenceTransformerEmbedding(_minilm_config())
        model.encode(["hello", "world"])

        call_args = mock_instance.encode.call_args
        assert call_args[0][0] == ["hello", "world"]

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_prefix_is_prepended(self, mock_cls):
        """With prefix configured, each text gets the prefix."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.zeros((2, 1024), dtype=np.float32)
        mock_cls.return_value = mock_instance

        model = SentenceTransformerEmbedding(_bge_config())
        model.encode(["hello", "world"])

        call_args = mock_instance.encode.call_args
        texts = call_args[0][0]
        prefix = "Represent this sentence for classification: "
        assert texts == [prefix + "hello", prefix + "world"]

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_batch_size_forwarded(self, mock_cls):
        """batch_size is passed through to the underlying model."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_cls.return_value = mock_instance

        model = SentenceTransformerEmbedding(_minilm_config())
        model.encode(["text"], batch_size=16)

        call_args = mock_instance.encode.call_args
        assert call_args[1]["batch_size"] == 16

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_single_text(self, mock_cls):
        """Works with a single text."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_cls.return_value = mock_instance

        model = SentenceTransformerEmbedding(_minilm_config())
        out = model.encode(["single"])
        assert out.shape == (1, 384)

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_empty_list(self, mock_cls):
        """Returns (0, dim) array for empty input."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.zeros((0, 384), dtype=np.float32)
        mock_cls.return_value = mock_instance

        model = SentenceTransformerEmbedding(_minilm_config())
        out = model.encode([])
        assert out.shape == (0, 384)

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_loads_model_with_correct_hf_path(self, mock_cls):
        """Constructor loads the correct HuggingFace model."""
        SentenceTransformerEmbedding(_minilm_config())
        mock_cls.assert_called_once_with("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
