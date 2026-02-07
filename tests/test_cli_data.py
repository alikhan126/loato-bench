"""Tests for data CLI commands — Sprint 1A (TDD)."""

from typer.testing import CliRunner

from promptguard.cli import app

runner = CliRunner()


class TestDataAppHelp:
    """Verify data sub-commands exist and have help text."""

    def test_data_help(self):
        result = runner.invoke(app, ["data", "--help"])
        assert result.exit_code == 0
        assert "download" in result.output
        assert "harmonize" in result.output
        assert "split" in result.output

    def test_download_help(self):
        result = runner.invoke(app, ["data", "download", "--help"])
        assert result.exit_code == 0

    def test_harmonize_help(self):
        result = runner.invoke(app, ["data", "harmonize", "--help"])
        assert result.exit_code == 0
