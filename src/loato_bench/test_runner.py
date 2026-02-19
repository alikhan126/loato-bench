"""Test runner script for running pytest with coverage."""

from __future__ import annotations

from pathlib import Path
import sys


def run_tests() -> None:
    """Run pytest with coverage and display results."""
    import subprocess

    project_root = Path(__file__).resolve().parents[2]

    # Run pytest with coverage
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--cov=loato_bench",
            "--cov-report=term-missing",
            "--cov-report=html",
        ],
        cwd=project_root,
        check=False,
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    run_tests()
