"""Tests for embed CLI commands — Sprint 1B."""

import re

from typer.testing import CliRunner

from loato_bench.cli import app

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class TestEmbedAppHelp:
    """Smoke tests that embed CLI is wired correctly."""

    def test_embed_help(self):
        result = runner.invoke(app, ["embed", "--help"])
        assert result.exit_code == 0
        assert "Compute embeddings" in result.output

    def test_embed_run_help(self):
        result = runner.invoke(app, ["embed", "run", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "--model" in output
        assert "--all" in output
