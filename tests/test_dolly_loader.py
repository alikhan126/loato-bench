"""Tests for Dolly dataset loader."""

from unittest.mock import MagicMock, patch

import pytest

from loato_bench.data.base import DatasetLoader, UnifiedSample
from loato_bench.data.dolly import DollyLoader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_fake_split(rows: list[dict]) -> MagicMock:
    """Create a mock HuggingFace Dataset split from a list of dicts."""
    split = MagicMock()
    split.__iter__ = lambda self: iter(rows)
    split.__len__ = lambda self: len(rows)
    return split


@pytest.fixture()
def raw_rows():
    """Sample rows mimicking Dolly HF schema."""
    return [
        {
            "instruction": "Explain what machine learning is.",
            "context": "",
            "response": "Machine learning is...",
            "category": "open_qa",
        },
        {
            "instruction": "Write a poem about spring.",
            "context": "",
            "response": "Spring is here...",
            "category": "creative_writing",
        },
        {
            "instruction": "Summarize the following text.",
            "context": "Some long text here.",
            "response": "Summary...",
            "category": "summarization",
        },
        {
            "instruction": "Explain what machine learning is.",  # Duplicate of row 0
            "context": "Different context",
            "response": "ML is...",
            "category": "open_qa",
        },
        {
            "instruction": "   ",  # Empty after strip — should be filtered
            "context": "",
            "response": "...",
            "category": "open_qa",
        },
    ]


@pytest.fixture()
def fake_hf_dataset(raw_rows):
    """Return a mock datasets.DatasetDict with a single 'train' split."""
    dataset_dict = MagicMock()
    dataset_dict.__getitem__ = lambda self, key: _make_fake_split(raw_rows)
    dataset_dict.keys = lambda: ["train"]
    return dataset_dict


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestDollyLoaderContract:
    """Verify DollyLoader satisfies the DatasetLoader ABC contract."""

    def test_is_subclass_of_dataset_loader(self):
        assert issubclass(DollyLoader, DatasetLoader)

    def test_has_load_method(self):
        assert callable(getattr(DollyLoader, "load", None))


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestDollyLoaderLoad:
    """Test DollyLoader.load() transforms and filters correctly."""

    @patch("loato_bench.data.dolly.load_dataset")
    def test_returns_list_of_unified_samples(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = DollyLoader().load()
        assert isinstance(samples, list)
        assert all(isinstance(s, UnifiedSample) for s in samples)

    @patch("loato_bench.data.dolly.load_dataset")
    def test_all_labels_are_benign(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = DollyLoader().load()
        for s in samples:
            assert s.label == 0

    @patch("loato_bench.data.dolly.load_dataset")
    def test_source_is_dolly(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = DollyLoader().load()
        for s in samples:
            assert s.source == "dolly"

    @patch("loato_bench.data.dolly.load_dataset")
    def test_attack_category_is_none(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = DollyLoader().load()
        for s in samples:
            assert s.attack_category is None

    @patch("loato_bench.data.dolly.load_dataset")
    def test_skips_empty_instructions(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = DollyLoader().load()
        for s in samples:
            assert s.text.strip()

    @patch("loato_bench.data.dolly.load_dataset")
    def test_deduplicates_on_instruction(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = DollyLoader().load()
        texts = [s.text for s in samples]
        assert texts.count("Explain what machine learning is.") == 1

    @patch("loato_bench.data.dolly.load_dataset")
    def test_correct_sample_count(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = DollyLoader().load()
        # 5 rows: 1 empty + 1 duplicate = 3 unique
        assert len(samples) == 3

    @patch("loato_bench.data.dolly.load_dataset")
    def test_metadata_includes_category(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = DollyLoader().load()
        for s in samples:
            assert "category" in s.metadata
            assert "has_context" in s.metadata

    @patch("loato_bench.data.dolly.load_dataset")
    def test_calls_load_dataset_with_correct_path(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        DollyLoader().load()
        mock_load.assert_called_once_with("databricks/databricks-dolly-15k")


# ---------------------------------------------------------------------------
# Max samples cap
# ---------------------------------------------------------------------------


class TestDollyLoaderCap:
    """Test the max_samples cap."""

    @patch("loato_bench.data.dolly.load_dataset")
    def test_respects_max_samples(self, mock_load):
        rows = [
            {
                "instruction": f"Question {i}",
                "context": "",
                "response": f"Answer {i}",
                "category": "open_qa",
            }
            for i in range(100)
        ]
        ds = MagicMock()
        ds.keys = lambda: ["train"]
        ds.__getitem__ = lambda self, key: _make_fake_split(rows)
        mock_load.return_value = ds

        samples = DollyLoader(max_samples=10).load()
        assert len(samples) == 10

    def test_default_max_is_15000(self):
        loader = DollyLoader()
        assert loader.max_samples == 15_000
