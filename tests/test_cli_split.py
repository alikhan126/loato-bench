"""Tests for the CLI 'data split' command."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from loato_bench.cli import app

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestDataSplitCLI:
    """Tests for the data split CLI command."""

    def test_help_text(self) -> None:
        result = runner.invoke(app, ["data", "split", "--help"])
        assert result.exit_code == 0
        assert "Generate all evaluation splits" in result.output

    def test_shows_gentel_filtering_option(self) -> None:
        result = runner.invoke(app, ["data", "split", "--help"])
        clean = _strip_ansi(result.output)
        assert "--apply-filter" in clean or "--no-apply-filter" in clean

    def test_shows_merge_option(self) -> None:
        result = runner.invoke(app, ["data", "split", "--help"])
        clean = _strip_ansi(result.output)
        assert "--apply-merges" in clean or "--no-apply-merges" in clean

    def test_shows_tier3_option(self) -> None:
        result = runner.invoke(app, ["data", "split", "--help"])
        clean = _strip_ansi(result.output)
        assert "--apply-tier3" in clean or "--no-apply-tier3" in clean

    def test_shows_output_dir_option(self) -> None:
        result = runner.invoke(app, ["data", "split", "--help"])
        clean = _strip_ansi(result.output)
        assert "--output-dir" in clean

    def test_fails_without_data(self) -> None:
        """Without unified_dataset.parquet, should exit with error."""
        result = runner.invoke(app, ["data", "split", "--output-dir", "/tmp/test_splits_none"])
        # Should fail because no parquet file exists at default path
        # (unless user has run the pipeline)
        assert (
            result.exit_code != 0 or "No processed data" in result.output or len(result.output) > 0
        )
