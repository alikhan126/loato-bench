"""Tests for LLM provider abstraction."""

from unittest.mock import MagicMock, patch

import pytest

from loato_bench.llm.base import LLMProvider
from loato_bench.utils.config import LLMConfig, load_llm_config

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestLLMConfig:
    """Tests for LLMConfig Pydantic model."""

    def test_load_llm_config_from_yaml(self):
        """Should load LLM config from configs/llm.yaml."""
        cfg = load_llm_config()
        assert cfg.provider in ("openai", "anthropic")
        assert isinstance(cfg.model, str)
        assert cfg.temperature >= 0.0
        assert cfg.max_tokens > 0
        assert cfg.max_retries > 0

    def test_llm_config_defaults(self):
        """Pydantic defaults should apply when fields are omitted."""
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini")
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 256
        assert cfg.max_retries == 5

    def test_llm_config_custom_values(self):
        """Custom values should override defaults."""
        cfg = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            temperature=0.5,
            max_tokens=1024,
            max_retries=3,
        )
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-sonnet-4-5-20250929"
        assert cfg.temperature == 0.5
        assert cfg.max_tokens == 1024
        assert cfg.max_retries == 3


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestLLMProviderABC:
    """Tests for the LLMProvider abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Should raise TypeError when trying to instantiate the ABC directly."""
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_name(self):
        """Subclass missing 'name' property should raise TypeError."""

        class Incomplete(LLMProvider):
            def complete(self, prompt: str, **kwargs: object) -> str:
                return ""

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_complete(self):
        """Subclass missing 'complete' method should raise TypeError."""

        class Incomplete(LLMProvider):
            @property
            def name(self) -> str:
                return "test"

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestGetLLMProvider:
    """Tests for the get_llm_provider factory function."""

    @patch("loato_bench.llm.openai_llm.openai.OpenAI")
    def test_factory_returns_openai_provider(self, mock_openai_cls: MagicMock):
        from loato_bench.llm import get_llm_provider
        from loato_bench.llm.openai_llm import OpenAILLM

        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        provider = get_llm_provider(config)
        assert isinstance(provider, OpenAILLM)
        assert provider.name == "openai"

    @patch("loato_bench.llm.anthropic_llm.anthropic.Anthropic")
    def test_factory_returns_anthropic_provider(self, mock_anthropic_cls: MagicMock):
        from loato_bench.llm import get_llm_provider
        from loato_bench.llm.anthropic_llm import AnthropicLLM

        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929")
        provider = get_llm_provider(config)
        assert isinstance(provider, AnthropicLLM)
        assert provider.name == "anthropic"

    def test_factory_raises_on_unknown_provider(self):
        from loato_bench.llm import get_llm_provider

        config = LLMConfig(provider="unknown", model="foo")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider(config)

    @patch("loato_bench.llm.anthropic_llm.anthropic.Anthropic")
    def test_factory_loads_default_config(self, mock_anthropic_cls: MagicMock):
        """When no config is passed, should load from configs/llm.yaml."""
        from loato_bench.llm import get_llm_provider

        provider = get_llm_provider()
        assert provider.name in ("openai", "anthropic")


# ---------------------------------------------------------------------------
# Provider implementation tests (mocked API calls)
# ---------------------------------------------------------------------------


class TestOpenAILLM:
    """Tests for the OpenAI provider with mocked API."""

    @patch("loato_bench.llm.openai_llm.openai.OpenAI")
    def test_complete_sends_user_message(self, mock_openai_cls: MagicMock):
        from loato_bench.llm.openai_llm import OpenAILLM

        mock_client = mock_openai_cls.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  category: jailbreak_roleplay  "
        mock_client.chat.completions.create.return_value = mock_response

        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        llm = OpenAILLM(config)
        result = llm.complete("Classify this text")

        assert result == "category: jailbreak_roleplay"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("loato_bench.llm.openai_llm.openai.OpenAI")
    def test_complete_with_system_message(self, mock_openai_cls: MagicMock):
        from loato_bench.llm.openai_llm import OpenAILLM

        mock_client = mock_openai_cls.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        mock_client.chat.completions.create.return_value = mock_response

        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        llm = OpenAILLM(config)
        llm.complete("Classify", system="You are a classifier")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @patch("loato_bench.llm.openai_llm.openai.OpenAI")
    def test_complete_uses_config_defaults(self, mock_openai_cls: MagicMock):
        from loato_bench.llm.openai_llm import OpenAILLM

        mock_client = mock_openai_cls.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create.return_value = mock_response

        config = LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.3, max_tokens=512)
        llm = OpenAILLM(config)
        llm.complete("test")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3
        assert call_kwargs.kwargs["max_tokens"] == 512

    @patch("loato_bench.llm.openai_llm.openai.OpenAI")
    def test_complete_overrides_temperature(self, mock_openai_cls: MagicMock):
        from loato_bench.llm.openai_llm import OpenAILLM

        mock_client = mock_openai_cls.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create.return_value = mock_response

        config = LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.0)
        llm = OpenAILLM(config)
        llm.complete("test", temperature=0.7, max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7
        assert call_kwargs.kwargs["max_tokens"] == 100


class TestAnthropicLLM:
    """Tests for the Anthropic provider with mocked API."""

    @patch("loato_bench.llm.anthropic_llm.anthropic.Anthropic")
    def test_complete_sends_user_message(self, mock_anthropic_cls: MagicMock):
        from loato_bench.llm.anthropic_llm import AnthropicLLM

        mock_client = mock_anthropic_cls.return_value
        mock_block = MagicMock()
        mock_block.text = "  instruction_override  "
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929")
        llm = AnthropicLLM(config)
        result = llm.complete("Classify this")

        assert result == "instruction_override"
        mock_client.messages.create.assert_called_once()

    @patch("loato_bench.llm.anthropic_llm.anthropic.Anthropic")
    def test_complete_with_system_message(self, mock_anthropic_cls: MagicMock):
        from loato_bench.llm.anthropic_llm import AnthropicLLM

        mock_client = mock_anthropic_cls.return_value
        mock_block = MagicMock()
        mock_block.text = "result"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929")
        llm = AnthropicLLM(config)
        llm.complete("Classify", system="You are a classifier")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are a classifier"

    @patch("loato_bench.llm.anthropic_llm.anthropic.Anthropic")
    def test_complete_without_system_omits_key(self, mock_anthropic_cls: MagicMock):
        from loato_bench.llm.anthropic_llm import AnthropicLLM

        mock_client = mock_anthropic_cls.return_value
        mock_block = MagicMock()
        mock_block.text = "result"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929")
        llm = AnthropicLLM(config)
        llm.complete("Classify")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs
