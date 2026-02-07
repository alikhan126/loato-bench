"""Tests for embedding model factory — Sprint 1B."""

from unittest.mock import patch

from promptguard.embeddings import get_embedding_model
from promptguard.embeddings.instructor import InstructorEmbeddingModel
from promptguard.embeddings.openai_embed import OpenAIEmbedding
from promptguard.embeddings.sentence_tf import SentenceTransformerEmbedding


class TestGetEmbeddingModel:
    """Tests for the factory function."""

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_minilm_returns_sentence_transformer(self, mock_cls):
        model = get_embedding_model("minilm")
        assert isinstance(model, SentenceTransformerEmbedding)
        assert model.name == "minilm"
        assert model.dim == 384

    @patch("promptguard.embeddings.sentence_tf.SentenceTransformer")
    def test_bge_large_returns_sentence_transformer(self, mock_cls):
        model = get_embedding_model("bge_large")
        assert isinstance(model, SentenceTransformerEmbedding)
        assert model.name == "bge_large"
        assert model.dim == 1024

    @patch("promptguard.embeddings.instructor.INSTRUCTOR")
    def test_instructor_returns_instructor_model(self, mock_cls):
        model = get_embedding_model("instructor")
        assert isinstance(model, InstructorEmbeddingModel)
        assert model.name == "instructor"
        assert model.dim == 768

    @patch("promptguard.embeddings.openai_embed.openai.OpenAI")
    def test_openai_small_returns_openai_model(self, mock_cls):
        model = get_embedding_model("openai_small")
        assert isinstance(model, OpenAIEmbedding)
        assert model.name == "openai_small"
        assert model.dim == 1536

    @patch("llama_cpp.Llama")
    @patch("huggingface_hub.hf_hub_download", return_value="/tmp/model.gguf")
    def test_e5_mistral_returns_e5_model(self, mock_dl, mock_llama):
        import sys

        sys.modules.setdefault("llama_cpp", __import__("unittest.mock", fromlist=["MagicMock"]))
        from promptguard.embeddings.e5_mistral import E5MistralEmbedding

        model = get_embedding_model("e5_mistral")
        assert isinstance(model, E5MistralEmbedding)
        assert model.name == "e5_mistral"
        assert model.dim == 4096
