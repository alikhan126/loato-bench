"""Tests for OASST1 dataset loader."""

from unittest.mock import MagicMock, patch

import pytest

from loato_bench.data.base import DatasetLoader, UnifiedSample
from loato_bench.data.oasst import OASSTLoader

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
    """Sample rows mimicking OASST1 HF schema."""
    return [
        {
            "message_id": "msg-001",
            "parent_id": None,
            "text": "How do I sort a list in Python?",
            "role": "prompter",
            "lang": "en",
        },
        {
            "message_id": "msg-002",
            "parent_id": "msg-001",
            "text": "You can use the sorted() function.",
            "role": "assistant",  # Should be filtered out
            "lang": "en",
        },
        {
            "message_id": "msg-003",
            "parent_id": None,
            "text": "Was ist Photosynthese?",
            "role": "prompter",
            "lang": "de",  # Should be filtered out (not English)
        },
        {
            "message_id": "msg-004",
            "parent_id": None,
            "text": "What is the capital of France?",
            "role": "prompter",
            "lang": "en",
        },
        {
            "message_id": "msg-005",
            "parent_id": None,
            "text": "How do I sort a list in Python?",  # Duplicate of row 0
            "role": "prompter",
            "lang": "en",
        },
        {
            "message_id": "msg-006",
            "parent_id": None,
            "text": "   ",  # Empty after strip
            "role": "prompter",
            "lang": "en",
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


class TestOASSTLoaderContract:
    """Verify OASSTLoader satisfies the DatasetLoader ABC contract."""

    def test_is_subclass_of_dataset_loader(self):
        assert issubclass(OASSTLoader, DatasetLoader)

    def test_has_load_method(self):
        assert callable(getattr(OASSTLoader, "load", None))


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestOASSTLoaderLoad:
    """Test OASSTLoader.load() transforms and filters correctly."""

    @patch("loato_bench.data.oasst.load_dataset")
    def test_returns_list_of_unified_samples(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = OASSTLoader().load()
        assert isinstance(samples, list)
        assert all(isinstance(s, UnifiedSample) for s in samples)

    @patch("loato_bench.data.oasst.load_dataset")
    def test_all_labels_are_benign(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = OASSTLoader().load()
        for s in samples:
            assert s.label == 0

    @patch("loato_bench.data.oasst.load_dataset")
    def test_source_is_oasst(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = OASSTLoader().load()
        for s in samples:
            assert s.source == "oasst"

    @patch("loato_bench.data.oasst.load_dataset")
    def test_filters_out_assistant_messages(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = OASSTLoader().load()
        texts = [s.text for s in samples]
        assert "You can use the sorted() function." not in texts

    @patch("loato_bench.data.oasst.load_dataset")
    def test_filters_out_non_english(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = OASSTLoader().load()
        texts = [s.text for s in samples]
        assert "Was ist Photosynthese?" not in texts

    @patch("loato_bench.data.oasst.load_dataset")
    def test_deduplicates_on_text(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = OASSTLoader().load()
        texts = [s.text for s in samples]
        assert texts.count("How do I sort a list in Python?") == 1

    @patch("loato_bench.data.oasst.load_dataset")
    def test_skips_empty_text(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = OASSTLoader().load()
        for s in samples:
            assert s.text.strip()

    @patch("loato_bench.data.oasst.load_dataset")
    def test_correct_sample_count(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = OASSTLoader().load()
        # 6 rows: 1 assistant + 1 German + 1 dup + 1 empty = 2 unique English prompter
        assert len(samples) == 2

    @patch("loato_bench.data.oasst.load_dataset")
    def test_metadata_includes_message_id(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = OASSTLoader().load()
        for s in samples:
            assert "message_id" in s.metadata
            assert "parent_id" in s.metadata

    @patch("loato_bench.data.oasst.load_dataset")
    def test_calls_load_dataset_with_correct_path(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        OASSTLoader().load()
        mock_load.assert_called_once_with("OpenAssistant/oasst1")


# ---------------------------------------------------------------------------
# Max samples cap
# ---------------------------------------------------------------------------


class TestOASSTLoaderCap:
    """Test the max_samples cap."""

    @patch("loato_bench.data.oasst.load_dataset")
    def test_respects_max_samples(self, mock_load):
        rows = [
            {
                "message_id": f"msg-{i}",
                "parent_id": None,
                "text": f"Question {i}",
                "role": "prompter",
                "lang": "en",
            }
            for i in range(100)
        ]
        ds = MagicMock()
        ds.keys = lambda: ["train"]
        ds.__getitem__ = lambda self, key: _make_fake_split(rows)
        mock_load.return_value = ds

        samples = OASSTLoader(max_samples=10).load()
        assert len(samples) == 10

    def test_default_max_is_8000(self):
        loader = OASSTLoader()
        assert loader.max_samples == 8_000
