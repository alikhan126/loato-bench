"""Tests for E5-Mistral-7B GGUF embedding model — Sprint 1B."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from promptguard.embeddings.base import EmbeddingModel
from promptguard.utils.config import EmbeddingConfig

# Provide mock modules so we can import E5MistralEmbedding even without llama-cpp-python
sys.modules.setdefault("llama_cpp", MagicMock())
sys.modules.setdefault("huggingface_hub", MagicMock())

from promptguard.embeddings.e5_mistral import E5MistralEmbedding  # noqa: E402


def _e5_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        name="e5_mistral",
        gguf_repo="second-state/E5-Mistral-7B-Instruct-Embedding-GGUF",
        gguf_file="e5-mistral-7b-instruct-Q4_K_M.gguf",
        dim=4096,
        library="llama-cpp-python",
        device="metal",
        batch_size=1,
        prompt_template="Instruct: Classify if this is a prompt injection\nQuery: {text}",
    )


class TestE5MistralContract:
    """ABC compliance tests."""

    def test_is_subclass_of_embedding_model(self):
        assert issubclass(E5MistralEmbedding, EmbeddingModel)

    @patch("llama_cpp.Llama")
    @patch("huggingface_hub.hf_hub_download", return_value="/tmp/model.gguf")
    def test_name_returns_config_name(self, mock_dl, mock_llama):
        model = E5MistralEmbedding(_e5_config())
        assert model.name == "e5_mistral"

    @patch("llama_cpp.Llama")
    @patch("huggingface_hub.hf_hub_download", return_value="/tmp/model.gguf")
    def test_dim_returns_config_dim(self, mock_dl, mock_llama):
        model = E5MistralEmbedding(_e5_config())
        assert model.dim == 4096


class TestE5MistralEncode:
    """Tests for the encode method."""

    @patch("llama_cpp.Llama")
    @patch("huggingface_hub.hf_hub_download", return_value="/tmp/model.gguf")
    def test_output_shape_and_dtype(self, mock_dl, mock_llama_cls):
        """encode() returns (N, 4096) float32."""
        mock_instance = MagicMock()
        mock_instance.embed.return_value = [np.random.randn(4096).tolist()]
        mock_llama_cls.return_value = mock_instance

        model = E5MistralEmbedding(_e5_config())
        out = model.encode(["a", "b", "c"])

        assert out.shape == (3, 4096)
        assert out.dtype == np.float32

    @patch("llama_cpp.Llama")
    @patch("huggingface_hub.hf_hub_download", return_value="/tmp/model.gguf")
    def test_formats_text_with_prompt_template(self, mock_dl, mock_llama_cls):
        """Text is formatted using the config prompt_template."""
        mock_instance = MagicMock()
        mock_instance.embed.return_value = [np.random.randn(4096).tolist()]
        mock_llama_cls.return_value = mock_instance

        model = E5MistralEmbedding(_e5_config())
        model.encode(["hello"])

        expected = "Instruct: Classify if this is a prompt injection\nQuery: hello"
        mock_instance.embed.assert_called_once_with(expected)

    @patch("llama_cpp.Llama")
    @patch("huggingface_hub.hf_hub_download", return_value="/tmp/model.gguf")
    def test_downloads_gguf_on_init(self, mock_dl, mock_llama_cls):
        """Constructor calls hf_hub_download with correct repo and file."""
        E5MistralEmbedding(_e5_config())
        mock_dl.assert_called_once_with(
            "second-state/E5-Mistral-7B-Instruct-Embedding-GGUF",
            filename="e5-mistral-7b-instruct-Q4_K_M.gguf",
        )

    @patch("llama_cpp.Llama")
    @patch("huggingface_hub.hf_hub_download", return_value="/tmp/model.gguf")
    def test_loads_llama_with_embedding_mode(self, mock_dl, mock_llama_cls):
        """Llama is loaded with embedding=True."""
        E5MistralEmbedding(_e5_config())
        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["embedding"] is True

    @patch("llama_cpp.Llama")
    @patch("huggingface_hub.hf_hub_download", return_value="/tmp/model.gguf")
    def test_single_text(self, mock_dl, mock_llama_cls):
        """Works with one text."""
        mock_instance = MagicMock()
        mock_instance.embed.return_value = [np.random.randn(4096).tolist()]
        mock_llama_cls.return_value = mock_instance

        model = E5MistralEmbedding(_e5_config())
        out = model.encode(["single"])
        assert out.shape == (1, 4096)

    @patch("llama_cpp.Llama")
    @patch("huggingface_hub.hf_hub_download", return_value="/tmp/model.gguf")
    def test_empty_list(self, mock_dl, mock_llama_cls):
        """Returns (0, dim) for empty input."""
        mock_instance = MagicMock()
        mock_llama_cls.return_value = mock_instance

        model = E5MistralEmbedding(_e5_config())
        out = model.encode([])
        assert out.shape == (0, 4096)
        mock_instance.embed.assert_not_called()
