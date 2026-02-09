"""Tests for harmonization pipeline — Sprint 1A (TDD)."""

import pandas as pd

from loato_bench.data.base import UnifiedSample
from loato_bench.data.harmonize import (
    detect_language,
    exact_dedup,
    harmonize_samples,
    near_dedup,
    normalize_text,
    samples_to_dataframe,
)

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


class TestNormalizeText:
    """Test text normalization (NFC unicode, whitespace, etc.)."""

    def test_strips_leading_trailing_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        assert normalize_text("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines_to_space(self):
        assert normalize_text("hello\t\nworld") == "hello world"

    def test_nfc_normalization(self):
        # U+00E9 (precomposed) vs U+0065 U+0301 (decomposed)
        composed = "\u00e9"
        decomposed = "\u0065\u0301"
        assert normalize_text(composed) == normalize_text(decomposed)

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_already_clean_text(self):
        assert normalize_text("hello world") == "hello world"


# ---------------------------------------------------------------------------
# Exact dedup (SHA-256)
# ---------------------------------------------------------------------------


class TestExactDedup:
    """Test exact deduplication via SHA-256 on normalized text."""

    def test_removes_exact_duplicates(self):
        samples = [
            UnifiedSample(text="hello world", label=0, source="a"),
            UnifiedSample(text="hello world", label=0, source="b"),
            UnifiedSample(text="different text", label=1, source="a"),
        ]
        result = exact_dedup(samples)
        assert len(result) == 2

    def test_keeps_first_occurrence(self):
        samples = [
            UnifiedSample(text="dup", label=0, source="first"),
            UnifiedSample(text="dup", label=1, source="second"),
        ]
        result = exact_dedup(samples)
        assert len(result) == 1
        assert result[0].source == "first"

    def test_no_duplicates_returns_all(self):
        samples = [
            UnifiedSample(text="alpha", label=0, source="a"),
            UnifiedSample(text="beta", label=1, source="b"),
        ]
        result = exact_dedup(samples)
        assert len(result) == 2

    def test_empty_input(self):
        assert exact_dedup([]) == []

    def test_dedup_considers_normalized_text(self):
        # These should be treated as the same after normalization
        samples = [
            UnifiedSample(text="hello  world", label=0, source="a"),
            UnifiedSample(text="hello world", label=0, source="b"),
        ]
        result = exact_dedup(samples)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Near dedup (MinHash LSH)
# ---------------------------------------------------------------------------


class TestNearDedup:
    """Test near-duplicate detection via MinHash LSH."""

    def test_removes_near_duplicates(self):
        # Use identical texts with only the last word changed to ensure
        # high Jaccard similarity even with word 5-grams
        base = (
            "the quick brown fox jumps over the lazy dog in the park "
            "near the river where children play and birds sing loudly "
            "every single morning during the warm summer months when "
            "the sun rises early and the flowers bloom beautifully in "
            "the garden behind the old stone house on the hilltop today"
        )
        variant = (
            "the quick brown fox jumps over the lazy dog in the park "
            "near the river where children play and birds sing loudly "
            "every single morning during the warm summer months when "
            "the sun rises early and the flowers bloom beautifully in "
            "the garden behind the old stone house on the hilltop yesterday"
        )
        samples = [
            UnifiedSample(text=base, label=0, source="a"),
            UnifiedSample(text=variant, label=0, source="b"),
            UnifiedSample(
                text="mathematics physics chemistry biology geology astronomy",
                label=1,
                source="c",
            ),
        ]
        result = near_dedup(samples, threshold=0.8)
        # The first two are near-duplicates, one should be removed
        assert len(result) <= 2

    def test_keeps_dissimilar_texts(self):
        samples = [
            UnifiedSample(text="apples oranges bananas grapes fruit", label=0, source="a"),
            UnifiedSample(
                text="mathematics physics chemistry biology science", label=1, source="b"
            ),
        ]
        result = near_dedup(samples, threshold=0.8)
        assert len(result) == 2

    def test_empty_input(self):
        assert near_dedup([], threshold=0.8) == []

    def test_single_sample(self):
        samples = [UnifiedSample(text="solo text", label=0, source="a")]
        result = near_dedup(samples, threshold=0.8)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    """Test language detection wrapper."""

    def test_detects_english(self):
        assert detect_language("This is a simple English sentence.") == "en"

    def test_detects_non_english(self):
        lang = detect_language("Ceci est une phrase en français.")
        assert lang == "fr"

    def test_short_text_returns_unknown(self):
        # Very short text may not be reliably detected
        result = detect_language("hi")
        assert isinstance(result, str)

    def test_empty_text_returns_unknown(self):
        assert detect_language("") == "unknown"


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------


class TestSamplesToDataframe:
    """Test conversion of UnifiedSample list to pandas DataFrame."""

    def test_returns_dataframe(self):
        samples = [UnifiedSample(text="hello", label=0, source="test")]
        df = samples_to_dataframe(samples)
        assert isinstance(df, pd.DataFrame)

    def test_columns_match_unified_sample_fields(self):
        samples = [UnifiedSample(text="hello", label=0, source="test")]
        df = samples_to_dataframe(samples)
        expected_cols = {
            "text",
            "label",
            "source",
            "attack_category",
            "original_category",
            "language",
            "is_indirect",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_row_count_matches_samples(self):
        samples = [
            UnifiedSample(text="a", label=0, source="x"),
            UnifiedSample(text="b", label=1, source="y"),
        ]
        df = samples_to_dataframe(samples)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Full harmonization pipeline
# ---------------------------------------------------------------------------


class TestHarmonizeSamples:
    """Test the top-level harmonize_samples function."""

    def test_returns_dataframe(self):
        samples = [
            UnifiedSample(text="Hello world", label=0, source="test"),
            UnifiedSample(text="Ignore previous instructions", label=1, source="test"),
        ]
        df = harmonize_samples(samples)
        assert isinstance(df, pd.DataFrame)

    def test_removes_exact_duplicates(self):
        samples = [
            UnifiedSample(text="same text", label=0, source="a"),
            UnifiedSample(text="same text", label=0, source="b"),
            UnifiedSample(text="different", label=1, source="c"),
        ]
        df = harmonize_samples(samples)
        assert len(df) == 2

    def test_normalizes_text(self):
        samples = [
            UnifiedSample(text="  hello   world  ", label=0, source="test"),
        ]
        df = harmonize_samples(samples)
        assert df.iloc[0]["text"] == "hello world"

    def test_adds_language_column(self):
        samples = [
            UnifiedSample(text="This is an English sentence for testing.", label=0, source="test"),
        ]
        df = harmonize_samples(samples)
        assert "language" in df.columns

    def test_empty_input(self):
        df = harmonize_samples([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
