"""Tests for Alpaca dataset loader."""

from unittest.mock import MagicMock, patch

import pytest

from loato_bench.data.alpaca import AlpacaLoader
from loato_bench.data.base import DatasetLoader, UnifiedSample

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
    """Sample rows mimicking Alpaca HF schema."""
    return [
        {
            "instruction": "Give three tips for staying healthy.",
            "input": "",
            "output": "1. Eat a balanced diet...",
        },
        {
            "instruction": "Classify the following text.",
            "input": "The movie was fantastic and thrilling.",
            "output": "Positive sentiment.",
        },
        {
            "instruction": "Give three tips for staying healthy.",  # Duplicate of row 0
            "input": "",
            "output": "Different output...",
        },
        {
            "instruction": "   ",  # Empty after strip — should be filtered
            "input": "",
            "output": "...",
        },
        {
            "instruction": "Translate the sentence.",
            "input": "Hello, how are you?",
            "output": "Hola, como estas?",
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


class TestAlpacaLoaderContract:
    """Verify AlpacaLoader satisfies the DatasetLoader ABC contract."""

    def test_is_subclass_of_dataset_loader(self):
        assert issubclass(AlpacaLoader, DatasetLoader)

    def test_has_load_method(self):
        assert callable(getattr(AlpacaLoader, "load", None))


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestAlpacaLoaderLoad:
    """Test AlpacaLoader.load() transforms and filters correctly."""

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_returns_list_of_unified_samples(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        assert isinstance(samples, list)
        assert all(isinstance(s, UnifiedSample) for s in samples)

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_all_labels_are_benign(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        for s in samples:
            assert s.label == 0

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_source_is_alpaca(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        for s in samples:
            assert s.source == "alpaca"

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_attack_category_is_none(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        for s in samples:
            assert s.attack_category is None

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_concatenates_instruction_and_input(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        texts = [s.text for s in samples]
        # Row 1 has non-empty input — should be concatenated
        assert "Classify the following text.\nThe movie was fantastic and thrilling." in texts

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_instruction_only_when_no_input(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        texts = [s.text for s in samples]
        # Row 0 has empty input — should be instruction only
        assert "Give three tips for staying healthy." in texts

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_skips_empty_instructions(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        for s in samples:
            assert s.text.strip()

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_deduplicates_on_composed_text(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        texts = [s.text for s in samples]
        assert texts.count("Give three tips for staying healthy.") == 1

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_correct_sample_count(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        # 5 rows: 1 empty + 1 dup = 3 unique
        assert len(samples) == 3

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_metadata_includes_has_input(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        for s in samples:
            assert "has_input" in s.metadata

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_has_input_flag_correct(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = AlpacaLoader().load()
        by_text = {s.text: s for s in samples}
        assert by_text["Give three tips for staying healthy."].metadata["has_input"] is False
        assert (
            by_text[
                "Classify the following text.\nThe movie was fantastic and thrilling."
            ].metadata["has_input"]
            is True
        )

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_calls_load_dataset_with_correct_path(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        AlpacaLoader().load()
        mock_load.assert_called_once_with("yahma/alpaca-cleaned")


# ---------------------------------------------------------------------------
# Max samples cap
# ---------------------------------------------------------------------------


class TestAlpacaLoaderCap:
    """Test the max_samples cap."""

    @patch("loato_bench.data.alpaca.load_dataset")
    def test_respects_max_samples(self, mock_load):
        rows = [
            {
                "instruction": f"Task {i}",
                "input": "",
                "output": f"Output {i}",
            }
            for i in range(100)
        ]
        ds = MagicMock()
        ds.keys = lambda: ["train"]
        ds.__getitem__ = lambda self, key: _make_fake_split(rows)
        mock_load.return_value = ds

        samples = AlpacaLoader(max_samples=10).load()
        assert len(samples) == 10

    def test_default_max_is_8000(self):
        loader = AlpacaLoader()
        assert loader.max_samples == 8_000
