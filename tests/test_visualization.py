"""Tests for visualization module."""

from pathlib import Path

import pandas as pd
import pytest

from loato_bench.analysis import visualization

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "text": [
                "Short text",
                "This is a medium length text with some words",
                "This is a much longer text that has many more words and characters to test",
                "Bonjour comment allez-vous",
                "Hola como estas",
            ],
            "label": [0, 1, 0, 1, 0],
            "source": ["deepset", "hackaprompt", "deepset", "gentel", "pint"],
            "attack_category": [None, "instruction_override", None, "jailbreak", None],
            "language": ["en", "en", "en", "fr", "es"],
        }
    )


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """Create an empty DataFrame with correct schema."""
    return pd.DataFrame(columns=["text", "label", "source", "attack_category", "language"])


# ---------------------------------------------------------------------------
# Test safe_output_path
# ---------------------------------------------------------------------------


class TestSafeOutputPath:
    """Tests for safe_output_path function."""

    def test_validates_extension_whitelist(self, tmp_path: Path) -> None:
        """Test that only allowed extensions are accepted."""
        valid_path = tmp_path / "test.png"
        result = visualization.safe_output_path(valid_path)
        assert result.suffix == ".png"

        with pytest.raises(ValueError, match="Invalid file extension"):
            visualization.safe_output_path(tmp_path / "test.txt")

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        path = tmp_path / "subdir" / "test.png"
        result = visualization.safe_output_path(path)
        assert result.parent.exists()

    def test_rejects_path_outside_base_dir(self, tmp_path: Path) -> None:
        """Test path traversal prevention."""
        base_dir = tmp_path / "allowed"
        base_dir.mkdir()
        outside_path = tmp_path / "outside" / "test.png"

        with pytest.raises(ValueError, match="outside base directory"):
            visualization.safe_output_path(outside_path, base_dir=base_dir)

    def test_allows_path_within_base_dir(self, tmp_path: Path) -> None:
        """Test that paths within base_dir are allowed."""
        base_dir = tmp_path / "allowed"
        base_dir.mkdir()
        inside_path = base_dir / "test.png"

        result = visualization.safe_output_path(inside_path, base_dir=base_dir)
        assert str(result).startswith(str(base_dir.resolve()))


# ---------------------------------------------------------------------------
# Test plot_label_distribution
# ---------------------------------------------------------------------------


class TestPlotLabelDistribution:
    """Tests for plot_label_distribution function."""

    def test_creates_figure_file(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that function creates output file."""
        output_path = tmp_path / "label_dist.png"
        visualization.plot_label_distribution(sample_df, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_handles_imbalanced_data(self, tmp_path: Path) -> None:
        """Test with highly imbalanced dataset."""
        df = pd.DataFrame(
            {
                "text": ["test"] * 100,
                "label": [0] * 95 + [1] * 5,
                "source": ["test"] * 100,
            }
        )
        output_path = tmp_path / "imbalanced.png"
        visualization.plot_label_distribution(df, output_path)

        assert output_path.exists()

    def test_custom_figsize_and_dpi(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test custom figure size and DPI."""
        output_path = tmp_path / "custom.png"
        visualization.plot_label_distribution(sample_df, output_path, figsize=(8, 6), dpi=100)

        assert output_path.exists()


# ---------------------------------------------------------------------------
# Test plot_source_breakdown
# ---------------------------------------------------------------------------


class TestPlotSourceBreakdown:
    """Tests for plot_source_breakdown function."""

    def test_creates_figure_file(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that function creates output file."""
        output_path = tmp_path / "source_breakdown.png"
        visualization.plot_source_breakdown(sample_df, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_handles_many_sources(self, tmp_path: Path) -> None:
        """Test with many different sources."""
        df = pd.DataFrame(
            {
                "text": [f"text{i}" for i in range(20)],
                "label": [0, 1] * 10,
                "source": [f"source{i}" for i in range(20)],
            }
        )
        output_path = tmp_path / "many_sources.png"
        visualization.plot_source_breakdown(df, output_path)

        assert output_path.exists()

    def test_handles_single_source(self, tmp_path: Path) -> None:
        """Test with only one source."""
        df = pd.DataFrame(
            {
                "text": ["test1", "test2", "test3"],
                "label": [0, 1, 0],
                "source": ["single_source"] * 3,
            }
        )
        output_path = tmp_path / "single_source.png"
        visualization.plot_source_breakdown(df, output_path)

        assert output_path.exists()


# ---------------------------------------------------------------------------
# Test plot_text_length_distribution
# ---------------------------------------------------------------------------


class TestPlotTextLengthDistribution:
    """Tests for plot_text_length_distribution function."""

    def test_creates_figure_file(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that function creates output file."""
        output_path = tmp_path / "text_length.png"
        visualization.plot_text_length_distribution(sample_df, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_custom_bins(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test with custom bin edges."""
        output_path = tmp_path / "custom_bins.png"
        bins = [0, 20, 40, 60, 100]
        visualization.plot_text_length_distribution(sample_df, output_path, bins=bins)

        assert output_path.exists()

    def test_handles_very_long_texts(self, tmp_path: Path) -> None:
        """Test with very long text samples."""
        df = pd.DataFrame(
            {
                "text": ["a" * 10000, "b" * 5000, "c" * 1000],
                "label": [0, 1, 0],
                "source": ["test"] * 3,
            }
        )
        output_path = tmp_path / "long_texts.png"
        visualization.plot_text_length_distribution(df, output_path)

        assert output_path.exists()


# ---------------------------------------------------------------------------
# Test plot_language_heatmap
# ---------------------------------------------------------------------------


class TestPlotLanguageHeatmap:
    """Tests for plot_language_heatmap function."""

    def test_creates_figure_file(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that function creates output file."""
        output_path = tmp_path / "language_heatmap.png"
        visualization.plot_language_heatmap(sample_df, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_handles_many_languages(self, tmp_path: Path) -> None:
        """Test with many different languages."""
        languages = ["en", "fr", "es", "de", "it", "pt", "ru", "zh", "ja", "ko"]
        df = pd.DataFrame(
            {
                "text": [f"text{i}" for i in range(100)],
                "label": [0, 1] * 50,
                "language": [languages[i % len(languages)] for i in range(100)],
            }
        )
        output_path = tmp_path / "many_languages.png"
        visualization.plot_language_heatmap(df, output_path)

        assert output_path.exists()

    def test_handles_single_language(self, tmp_path: Path) -> None:
        """Test with only English samples."""
        df = pd.DataFrame(
            {
                "text": ["test1", "test2", "test3"],
                "label": [0, 1, 0],
                "language": ["en"] * 3,
            }
        )
        output_path = tmp_path / "single_language.png"
        visualization.plot_language_heatmap(df, output_path)

        assert output_path.exists()


# ---------------------------------------------------------------------------
# Test plot_attack_category_distribution
# ---------------------------------------------------------------------------


class TestPlotAttackCategoryDistribution:
    """Tests for plot_attack_category_distribution function."""

    def test_creates_figure_file(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that function creates output file."""
        output_path = tmp_path / "attack_category.png"
        visualization.plot_attack_category_distribution(sample_df, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_filters_to_injection_samples_only(self, tmp_path: Path) -> None:
        """Test that only injection samples (label=1) are included."""
        df = pd.DataFrame(
            {
                "text": ["test1", "test2", "test3"],
                "label": [0, 1, 1],
                "attack_category": ["should_ignore", "cat1", "cat2"],
            }
        )
        output_path = tmp_path / "injection_only.png"
        visualization.plot_attack_category_distribution(df, output_path)

        assert output_path.exists()

    def test_handles_no_attack_categories(self, tmp_path: Path) -> None:
        """Test with no attack categories (all null)."""
        df = pd.DataFrame(
            {
                "text": ["test1", "test2"],
                "label": [1, 1],
                "attack_category": [None, None],
            }
        )
        output_path = tmp_path / "no_categories.png"

        # Should handle gracefully without creating file
        visualization.plot_attack_category_distribution(df, output_path)
        # Function returns early, no file created

    def test_handles_many_categories(self, tmp_path: Path) -> None:
        """Test with many attack categories."""
        categories = [f"category_{i}" for i in range(15)]
        df = pd.DataFrame(
            {
                "text": [f"text{i}" for i in range(15)],
                "label": [1] * 15,
                "attack_category": categories,
            }
        )
        output_path = tmp_path / "many_categories.png"
        visualization.plot_attack_category_distribution(df, output_path)

        assert output_path.exists()


# ---------------------------------------------------------------------------
# Test create_eda_dashboard
# ---------------------------------------------------------------------------


class TestCreateEdaDashboard:
    """Tests for create_eda_dashboard function."""

    def test_creates_multiple_plots(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that dashboard creates all expected plots."""
        output_dir = tmp_path / "dashboard"
        saved_paths = visualization.create_eda_dashboard(sample_df, output_dir)

        # Should create at least 4 plots (label, source, text_length, language)
        assert len(saved_paths) >= 4

        # Verify all returned paths exist
        for path in saved_paths:
            assert path.exists()
            assert path.stat().st_size > 0

    def test_creates_output_directory(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_dashboard_dir"
        assert not output_dir.exists()

        visualization.create_eda_dashboard(sample_df, output_dir)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_handles_partial_failures(self, tmp_path: Path) -> None:
        """Test that dashboard continues even if some plots fail."""
        # DataFrame missing some expected columns
        df = pd.DataFrame(
            {
                "text": ["test1", "test2"],
                "label": [0, 1],
                "source": ["test"] * 2,
                # Missing 'language' column
            }
        )
        output_dir = tmp_path / "partial"

        # Should not raise exception, but may have fewer plots
        saved_paths = visualization.create_eda_dashboard(df, output_dir)

        # Should create at least label and source plots
        assert len(saved_paths) >= 2

    def test_custom_figsize_and_dpi(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test custom figure parameters."""
        output_dir = tmp_path / "custom_dashboard"
        saved_paths = visualization.create_eda_dashboard(
            sample_df, output_dir, figsize=(10, 6), dpi=100
        )

        assert len(saved_paths) > 0
        for path in saved_paths:
            assert path.exists()

    def test_returns_list_of_paths(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test that function returns list of Path objects."""
        output_dir = tmp_path / "dashboard"
        saved_paths = visualization.create_eda_dashboard(sample_df, output_dir)

        assert isinstance(saved_paths, list)
        assert all(isinstance(p, Path) for p in saved_paths)
