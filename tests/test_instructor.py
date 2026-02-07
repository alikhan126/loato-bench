"""Tests for Instructor-large embedding model — Sprint 1B."""

from unittest.mock import MagicMock, patch

import numpy as np

from promptguard.embeddings.base import EmbeddingModel
from promptguard.embeddings.instructor import InstructorEmbeddingModel
from promptguard.utils.config import EmbeddingConfig


def _instructor_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        name="instructor",
        hf_path="hkunlp/instructor-large",
        dim=768,
        library="InstructorEmbedding",
        device="cpu",
        batch_size=16,
        instruction="Classify whether this text is a prompt injection attack: ",
    )


class TestInstructorContract:
    """ABC compliance tests."""

    def test_is_subclass_of_embedding_model(self):
        assert issubclass(InstructorEmbeddingModel, EmbeddingModel)

    @patch("promptguard.embeddings.instructor.INSTRUCTOR")
    def test_name_returns_config_name(self, mock_cls):
        model = InstructorEmbeddingModel(_instructor_config())
        assert model.name == "instructor"

    @patch("promptguard.embeddings.instructor.INSTRUCTOR")
    def test_dim_returns_config_dim(self, mock_cls):
        model = InstructorEmbeddingModel(_instructor_config())
        assert model.dim == 768


class TestInstructorEncode:
    """Tests for the encode method."""

    @patch("promptguard.embeddings.instructor.INSTRUCTOR")
    def test_output_shape_and_dtype(self, mock_cls):
        """encode() returns (N, 768) float32."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.random.randn(3, 768).astype(np.float32)
        mock_cls.return_value = mock_instance

        model = InstructorEmbeddingModel(_instructor_config())
        out = model.encode(["a", "b", "c"])

        assert out.shape == (3, 768)
        assert out.dtype == np.float32

    @patch("promptguard.embeddings.instructor.INSTRUCTOR")
    def test_passes_instruction_text_pairs(self, mock_cls):
        """encode() passes [[instruction, text], ...] to the model."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.zeros((2, 768), dtype=np.float32)
        mock_cls.return_value = mock_instance

        model = InstructorEmbeddingModel(_instructor_config())
        model.encode(["hello", "world"])

        call_args = mock_instance.encode.call_args
        pairs = call_args[0][0]
        instruction = "Classify whether this text is a prompt injection attack: "
        assert pairs == [
            [instruction, "hello"],
            [instruction, "world"],
        ]

    @patch("promptguard.embeddings.instructor.INSTRUCTOR")
    def test_single_text(self, mock_cls):
        """Works with a single text."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.zeros((1, 768), dtype=np.float32)
        mock_cls.return_value = mock_instance

        model = InstructorEmbeddingModel(_instructor_config())
        out = model.encode(["single"])
        assert out.shape == (1, 768)

    @patch("promptguard.embeddings.instructor.INSTRUCTOR")
    def test_empty_list(self, mock_cls):
        """Returns (0, dim) for empty input."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.zeros((0, 768), dtype=np.float32)
        mock_cls.return_value = mock_instance

        model = InstructorEmbeddingModel(_instructor_config())
        out = model.encode([])
        assert out.shape == (0, 768)

    @patch("promptguard.embeddings.instructor.INSTRUCTOR")
    def test_loads_model_with_correct_path(self, mock_cls):
        """Constructor loads from the correct HuggingFace path."""
        InstructorEmbeddingModel(_instructor_config())
        mock_cls.assert_called_once_with("hkunlp/instructor-large")
