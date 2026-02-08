"""Tests for OpenAI text-embedding-3-small model — Sprint 1B."""

from unittest.mock import MagicMock, patch

import numpy as np

from loato_bench.embeddings.base import EmbeddingModel
from loato_bench.embeddings.openai_embed import OpenAIEmbedding
from loato_bench.utils.config import EmbeddingConfig


def _openai_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        name="openai_small",
        model_id="text-embedding-3-small",
        dim=1536,
        library="openai",
        batch_size=2048,
    )


def _mock_embedding_response(texts, dim=1536):
    """Create a mock OpenAI embeddings response."""
    mock_response = MagicMock()
    mock_response.data = []
    for i, _ in enumerate(texts):
        item = MagicMock()
        item.embedding = np.random.randn(dim).tolist()
        item.index = i
        mock_response.data.append(item)
    return mock_response


class TestOpenAIEmbeddingContract:
    """ABC compliance tests."""

    def test_is_subclass_of_embedding_model(self):
        assert issubclass(OpenAIEmbedding, EmbeddingModel)

    @patch("loato_bench.embeddings.openai_embed.openai.OpenAI")
    def test_name_returns_config_name(self, mock_cls):
        model = OpenAIEmbedding(_openai_config())
        assert model.name == "openai_small"

    @patch("loato_bench.embeddings.openai_embed.openai.OpenAI")
    def test_dim_returns_config_dim(self, mock_cls):
        model = OpenAIEmbedding(_openai_config())
        assert model.dim == 1536


class TestOpenAIEmbeddingEncode:
    """Tests for the encode method."""

    @patch("loato_bench.embeddings.openai_embed.openai.OpenAI")
    def test_output_shape_and_dtype(self, mock_cls):
        """encode() returns (N, 1536) float32."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        texts = ["a", "b", "c"]
        mock_client.embeddings.create.return_value = _mock_embedding_response(texts)

        model = OpenAIEmbedding(_openai_config())
        out = model.encode(texts)

        assert out.shape == (3, 1536)
        assert out.dtype == np.float32

    @patch("loato_bench.embeddings.openai_embed.openai.OpenAI")
    def test_calls_api_with_correct_model(self, mock_cls):
        """API call uses the model_id from config."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        texts = ["test"]
        mock_client.embeddings.create.return_value = _mock_embedding_response(texts)

        model = OpenAIEmbedding(_openai_config())
        model.encode(texts)

        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "text-embedding-3-small"

    @patch("loato_bench.embeddings.openai_embed.openai.OpenAI")
    def test_batching(self, mock_cls):
        """10 texts with batch_size=3 produces 4 API calls."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        def side_effect(**kwargs):
            return _mock_embedding_response(kwargs["input"])

        mock_client.embeddings.create.side_effect = side_effect

        model = OpenAIEmbedding(_openai_config())
        texts = [f"text_{i}" for i in range(10)]
        out = model.encode(texts, batch_size=3)

        assert mock_client.embeddings.create.call_count == 4  # ceil(10/3)
        assert out.shape == (10, 1536)

    @patch("loato_bench.embeddings.openai_embed.openai.OpenAI")
    def test_single_text(self, mock_cls):
        """Works with one text."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        texts = ["single"]
        mock_client.embeddings.create.return_value = _mock_embedding_response(texts)

        model = OpenAIEmbedding(_openai_config())
        out = model.encode(texts)
        assert out.shape == (1, 1536)

    @patch("loato_bench.embeddings.openai_embed.openai.OpenAI")
    def test_empty_list(self, mock_cls):
        """Returns (0, dim) for empty input."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        model = OpenAIEmbedding(_openai_config())
        out = model.encode([])
        assert out.shape == (0, 1536)
        # No API calls for empty input
        mock_client.embeddings.create.assert_not_called()
