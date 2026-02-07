"""Tests for configuration loading."""

from promptguard.utils.config import (
    CONFIGS_DIR,
    PROJECT_ROOT,
    load_classifier_config,
    load_embedding_config,
    load_experiment_config,
)


def test_project_root_exists():
    """PROJECT_ROOT should point to the promptguard package root."""
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_configs_dir_exists():
    """CONFIGS_DIR should exist."""
    assert CONFIGS_DIR.exists()


def test_load_embedding_config():
    """Should load MiniLM config from YAML."""
    cfg = load_embedding_config("minilm")
    assert cfg.name == "minilm"
    assert cfg.dim == 384
    assert cfg.library == "sentence-transformers"


def test_load_classifier_config():
    """Should load logreg config from YAML."""
    cfg = load_classifier_config("logreg")
    assert cfg.name == "logreg"
    assert "C" in cfg.hyperparams


def test_load_experiment_config():
    """Should load LOATO experiment config from YAML."""
    cfg = load_experiment_config("loato")
    assert cfg.name == "loato"
    assert cfg.seed == 42
