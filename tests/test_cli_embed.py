"""Tests for embed CLI commands — Sprint 1B."""

from typer.testing import CliRunner

from promptguard.cli import app

runner = CliRunner()


class TestEmbedAppHelp:
    """Smoke tests that embed CLI is wired correctly."""

    def test_embed_help(self):
        result = runner.invoke(app, ["embed", "--help"])
        assert result.exit_code == 0
        assert "Compute embeddings" in result.output

    def test_embed_run_help(self):
        result = runner.invoke(app, ["embed", "run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--all" in result.output
