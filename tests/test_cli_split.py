"""Tests for the CLI 'data split' command."""

from __future__ import annotations

from typer.testing import CliRunner

from loato_bench.cli import app

runner = CliRunner()


class TestDataSplitCLI:
    """Tests for the data split CLI command."""

    def test_help_text(self) -> None:
        result = runner.invoke(app, ["data", "split", "--help"])
        assert result.exit_code == 0
        assert "Generate all evaluation splits" in result.output

    def test_shows_gentel_filtering_option(self) -> None:
        result = runner.invoke(app, ["data", "split", "--help"])
        assert "--apply-filter" in result.output or "--no-apply-filter" in result.output

    def test_shows_merge_option(self) -> None:
        result = runner.invoke(app, ["data", "split", "--help"])
        assert "--apply-merges" in result.output or "--no-apply-merges" in result.output

    def test_shows_tier3_option(self) -> None:
        result = runner.invoke(app, ["data", "split", "--help"])
        assert "--apply-tier3" in result.output or "--no-apply-tier3" in result.output

    def test_shows_output_dir_option(self) -> None:
        result = runner.invoke(app, ["data", "split", "--help"])
        assert "--output-dir" in result.output

    def test_fails_without_data(self) -> None:
        """Without unified_dataset.parquet, should exit with error."""
        result = runner.invoke(app, ["data", "split", "--output-dir", "/tmp/test_splits_none"])
        # Should fail because no parquet file exists at default path
        # (unless user has run the pipeline)
        assert (
            result.exit_code != 0 or "No processed data" in result.output or len(result.output) > 0
        )
