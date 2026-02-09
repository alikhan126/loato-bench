"""Configuration loading and Pydantic models."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # loato-bench/
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class EmbeddingConfig(BaseModel):
    """Configuration for a single embedding model."""

    name: str
    dim: int
    library: str
    device: str = "cpu"
    batch_size: int = 32
    hf_path: str | None = None
    model_id: str | None = None
    gguf_repo: str | None = None
    gguf_file: str | None = None
    prefix: str | None = None
    instruction: str | None = None
    prompt_template: str | None = None


class ClassifierConfig(BaseModel):
    """Configuration for a single classifier."""

    name: str
    class_path: str = ""
    hyperparams: dict[str, Any] = {}
    sweep: dict[str, list[Any]] = {}


class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider: str  # "anthropic" or "openai"
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
    max_retries: int = 5


class ExperimentConfig(BaseModel):
    """Configuration for an experiment."""

    name: str
    description: str = ""
    split_file: str = ""
    n_folds: int | None = None
    stratify_by: list[str] = []
    benign_test_fraction: float | None = None
    seed: int = 42


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_embedding_config(name: str) -> EmbeddingConfig:
    """Load embedding config by model name (e.g., 'minilm')."""
    path = CONFIGS_DIR / "embeddings" / f"{name}.yaml"
    data = load_yaml(path)
    return EmbeddingConfig(**data["model"])


def load_classifier_config(name: str) -> ClassifierConfig:
    """Load classifier config by name (e.g., 'logreg')."""
    path = CONFIGS_DIR / "classifiers" / f"{name}.yaml"
    data = load_yaml(path)
    raw = data["classifier"]
    return ClassifierConfig(
        name=raw["name"],
        class_path=raw.get("class", ""),
        hyperparams=raw.get("hyperparams", {}),
        sweep=raw.get("sweep", {}),
    )


def load_llm_config() -> LLMConfig:
    """Load LLM provider config from configs/llm.yaml."""
    path = CONFIGS_DIR / "llm.yaml"
    data = load_yaml(path)
    return LLMConfig(**data["llm"])


def load_experiment_config(name: str) -> ExperimentConfig:
    """Load experiment config by name (e.g., 'loato')."""
    path = CONFIGS_DIR / "experiments" / f"{name}.yaml"
    data = load_yaml(path)
    return ExperimentConfig(**data["experiment"])
