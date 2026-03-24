# Contributing to LOATO-Bench

Thanks for your interest in contributing! This guide covers how to get started.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/loato-bench.git
   cd loato-bench
   ```
3. **Install dependencies** (requires Python 3.12 and [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```
4. **Install pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```

## Development Workflow

1. Create a branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Make your changes
3. Run the quality checks:
   ```bash
   uv run ruff check src/ tests/          # Lint
   uv run ruff format src/ tests/          # Format
   uv run mypy src/loato_bench/            # Type check
   uv run pytest tests/ -v                 # Tests
   ```
4. Commit with a descriptive message (we use [Conventional Commits](https://www.conventionalcommits.org/)):
   ```
   feat(data): add new dataset loader for XYZ
   fix(evaluation): correct LOATO fold indexing
   docs: update README with new results
   ```
5. Push your branch and open a Pull Request against `main`

## Pull Request Process

- All PRs require approval from at least one code owner before merging
- Pre-commit hooks and CI must pass (ruff, mypy, pytest)
- Keep PRs focused — one feature or fix per PR
- Update documentation if your change affects usage

## Code Standards

- **Python 3.12**, type hints required (mypy strict)
- **ruff** for linting and formatting (line-length 100)
- **Google-style** docstrings
- **pytest** for tests, aim for 90%+ coverage on new code
- Seed = 42 for all randomness (`seed_everything()`)

## Architecture

Before contributing, review the ABCs in the codebase:

- `DatasetLoader` — `data/base.py`
- `EmbeddingModel` — `embeddings/base.py`
- `Classifier` — `classifiers/base.py`

New implementations should follow these interfaces.

## Reporting Issues

- Use GitHub Issues for bugs and feature requests
- Include reproduction steps, expected vs actual behavior
- For data issues, specify which dataset and sample IDs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
