"""Base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM implementations (OpenAI, Anthropic) must implement this interface.
    Used by Tier 3 taxonomy mapping (Sprint 2A) and LLM baseline (Sprint 4A).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'openai', 'anthropic')."""
        ...

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The user prompt to send to the model.
            system: Optional system message. If None, no system message is sent.
            temperature: Sampling temperature override. If None, uses config default.
            max_tokens: Max tokens override. If None, uses config default.

        Returns:
            The model's text response (stripped of leading/trailing whitespace).
        """
        ...
