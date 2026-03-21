"""Tests for WildChat dataset loader."""

from unittest.mock import MagicMock, patch

import pytest

from loato_bench.data.base import DatasetLoader, UnifiedSample
from loato_bench.data.wildchat import WildChatLoader

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
    """Sample rows mimicking WildChat HF schema."""
    return [
        {
            "conversation": [
                {"role": "user", "content": "What is the speed of light?"},
                {"role": "assistant", "content": "About 3e8 m/s."},
            ],
            "language": "English",
            "model": "gpt-3.5-turbo",
        },
        {
            "conversation": [
                {"role": "user", "content": "Write me a haiku about rain."},
                {"role": "assistant", "content": "Drops fall from the sky..."},
            ],
            "language": "English",
            "model": "gpt-4",
        },
        {
            "conversation": [
                {"role": "user", "content": "Quelle heure est-il?"},
                {"role": "assistant", "content": "Il est midi."},
            ],
            "language": "French",  # Should be filtered out
            "model": "gpt-3.5-turbo",
        },
        {
            "conversation": [
                {"role": "user", "content": "What is the speed of light?"},  # Dup of row 0
                {"role": "assistant", "content": "299,792,458 m/s."},
            ],
            "language": "English",
            "model": "gpt-4",
        },
        {
            "conversation": [
                {"role": "assistant", "content": "How can I help you?"},  # No user turn
            ],
            "language": "English",
            "model": "gpt-3.5-turbo",
        },
        {
            "conversation": [
                {"role": "user", "content": "   "},  # Empty after strip
                {"role": "assistant", "content": "Could you rephrase?"},
            ],
            "language": "English",
            "model": "gpt-3.5-turbo",
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


class TestWildChatLoaderContract:
    """Verify WildChatLoader satisfies the DatasetLoader ABC contract."""

    def test_is_subclass_of_dataset_loader(self):
        assert issubclass(WildChatLoader, DatasetLoader)

    def test_has_load_method(self):
        assert callable(getattr(WildChatLoader, "load", None))


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestWildChatLoaderLoad:
    """Test WildChatLoader.load() transforms and filters correctly."""

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_returns_list_of_unified_samples(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = WildChatLoader().load()
        assert isinstance(samples, list)
        assert all(isinstance(s, UnifiedSample) for s in samples)

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_all_labels_are_benign(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = WildChatLoader().load()
        for s in samples:
            assert s.label == 0

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_source_is_wildchat(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = WildChatLoader().load()
        for s in samples:
            assert s.source == "wildchat"

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_extracts_first_user_turn(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = WildChatLoader().load()
        texts = [s.text for s in samples]
        assert "What is the speed of light?" in texts
        # Should NOT include assistant text
        assert "About 3e8 m/s." not in texts

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_filters_out_non_english(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = WildChatLoader().load()
        texts = [s.text for s in samples]
        assert "Quelle heure est-il?" not in texts

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_deduplicates_on_first_user_turn(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = WildChatLoader().load()
        texts = [s.text for s in samples]
        assert texts.count("What is the speed of light?") == 1

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_skips_conversations_with_no_user_turn(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = WildChatLoader().load()
        texts = [s.text for s in samples]
        assert "How can I help you?" not in texts

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_skips_empty_first_user_turn(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = WildChatLoader().load()
        for s in samples:
            assert s.text.strip()

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_correct_sample_count(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = WildChatLoader().load()
        # 6 rows: 1 French + 1 dup + 1 no-user-turn + 1 empty = 2 unique English
        assert len(samples) == 2

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_metadata_includes_model(self, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        samples = WildChatLoader().load()
        for s in samples:
            assert "model" in s.metadata

    @patch("loato_bench.data.wildchat.load_dataset")
    @patch("loato_bench.data.wildchat.get_token", return_value="fake-token")
    def test_calls_load_dataset_with_correct_path(self, mock_token, mock_load, fake_hf_dataset):
        mock_load.return_value = fake_hf_dataset
        WildChatLoader().load()
        mock_load.assert_called_once_with("allenai/WildChat-nontoxic", token="fake-token")


# ---------------------------------------------------------------------------
# Max samples cap
# ---------------------------------------------------------------------------


class TestWildChatLoaderCap:
    """Test the max_samples cap."""

    @patch("loato_bench.data.wildchat.load_dataset")
    def test_respects_max_samples(self, mock_load):
        rows = [
            {
                "conversation": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ],
                "language": "English",
                "model": "gpt-3.5-turbo",
            }
            for i in range(100)
        ]
        ds = MagicMock()
        ds.keys = lambda: ["train"]
        ds.__getitem__ = lambda self, key: _make_fake_split(rows)
        mock_load.return_value = ds

        samples = WildChatLoader(max_samples=10).load()
        assert len(samples) == 10

    def test_default_max_is_8000(self):
        loader = WildChatLoader()
        assert loader.max_samples == 8_000
