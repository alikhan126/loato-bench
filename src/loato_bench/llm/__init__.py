"""LLM provider implementations and factory."""

from collections.abc import Callable

from loato_bench.llm.anthropic_llm import AnthropicLLM
from loato_bench.llm.base import LLMProvider
from loato_bench.llm.openai_llm import OpenAILLM
from loato_bench.utils.config import LLMConfig, load_llm_config

_PROVIDER_MAP: dict[str, Callable[[LLMConfig], LLMProvider]] = {
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
}


def get_llm_provider(config: LLMConfig | None = None) -> LLMProvider:
    """Factory: load config and return the right LLM provider instance.

    Args:
        config: LLM config. If None, loads from configs/llm.yaml.
    """
    if config is None:
        config = load_llm_config()

    cls = _PROVIDER_MAP.get(config.provider)
    if cls is None:
        msg = f"Unknown LLM provider: {config.provider!r} (expected one of {list(_PROVIDER_MAP)})"
        raise ValueError(msg)
    return cls(config)


__all__ = [
    "AnthropicLLM",
    "LLMProvider",
    "OpenAILLM",
    "get_llm_provider",
]
