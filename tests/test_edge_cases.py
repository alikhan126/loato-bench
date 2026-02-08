"""Edge case tests for coverage gaps."""

from unittest.mock import patch

import pytest

from loato_bench.data.harmonize import detect_language
from loato_bench.embeddings import get_embedding_model
from loato_bench.embeddings.cache import EmbeddingCache


class TestDetectLanguageEdgeCases:
    """Cover the LangDetectException branch in detect_language."""

    def test_empty_string_returns_unknown(self):
        assert detect_language("") == "unknown"

    def test_whitespace_only_returns_unknown(self):
        assert detect_language("   ") == "unknown"

    @patch(
        "loato_bench.data.harmonize.detect",
        side_effect=__import__(
            "langdetect.lang_detect_exception", fromlist=["LangDetectException"]
        ).LangDetectException(0, ""),
    )
    def test_lang_detect_exception_returns_unknown(self, mock_detect):
        assert detect_language("some text here") == "unknown"


class TestEmbeddingCacheDefaultBaseDir:
    """Cover the default base_dir fallback in EmbeddingCache.__init__ (line 37)."""

    @patch("loato_bench.utils.config.DATA_DIR")
    def test_default_base_dir_uses_data_dir(self, mock_data_dir, tmp_path):
        mock_data_dir.__truediv__ = lambda self, key: tmp_path / key
        cache = EmbeddingCache("test_model")
        assert "test_model" in str(cache.cache_dir)


class TestGetEmbeddingModelUnknownLibrary:
    """Cover the ValueError branch in get_embedding_model (lines 33-35)."""

    @patch("loato_bench.embeddings.load_embedding_config")
    def test_unknown_library_raises_value_error(self, mock_config):
        mock_config.return_value.library = "nonexistent_lib"
        with pytest.raises(ValueError, match="Unknown embedding library"):
            get_embedding_model("fake_model")
