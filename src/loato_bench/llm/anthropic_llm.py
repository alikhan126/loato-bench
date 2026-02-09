"""Anthropic LLM provider (Claude Sonnet, etc.)."""

from __future__ import annotations

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from loato_bench.llm.base import LLMProvider
from loato_bench.utils.config import LLMConfig


class AnthropicLLM(LLMProvider):
    """LLM provider using the Anthropic Messages API.

    Reads ANTHROPIC_API_KEY from the environment.
    Uses tenacity for automatic retries on rate-limit errors.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = anthropic.Anthropic()

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        temperature = temperature if temperature is not None else self._config.temperature
        max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens

        response = self._call_api(prompt, system, temperature, max_tokens)
        block = response.content[0]
        return block.text.strip() if hasattr(block, "text") else ""

    @retry(
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(anthropic.RateLimitError),
    )
    def _call_api(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> anthropic.types.Message:
        """Call the Anthropic Messages API with retry logic."""
        messages: list[anthropic.types.MessageParam] = [
            {"role": "user", "content": prompt},
        ]
        if system is not None:
            return self._client.messages.create(
                model=self._config.model,
                messages=messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._client.messages.create(
            model=self._config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
