"""OpenAI LLM provider (GPT-4o-mini, etc.)."""

from __future__ import annotations

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from loato_bench.llm.base import LLMProvider
from loato_bench.utils.config import LLMConfig


class OpenAILLM(LLMProvider):
    """LLM provider using the OpenAI Chat Completions API.

    Reads OPENAI_API_KEY from the environment.
    Uses tenacity for automatic retries on rate-limit errors.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = openai.OpenAI()

    @property
    def name(self) -> str:
        return "openai"

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

        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._call_api(messages, temperature, max_tokens)
        return (response.choices[0].message.content or "").strip()

    @retry(
        wait=wait_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(openai.RateLimitError),
    )
    def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> openai.types.chat.ChatCompletion:
        """Call the OpenAI Chat Completions API with retry logic."""
        return self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
