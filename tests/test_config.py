"""Tests for settings parsing from environment variables."""

import pytest

from llm_gateway.config import Settings

KEYS = {
    "GEMINI_API_KEY": "g-key",
    "GEMINI_MODEL": "gemini-test",
    "GROQ_API_KEY": "q-key",
    "GROQ_MODEL": "groq-test",
}


def test_both_providers_with_defaults() -> None:
    settings = Settings.from_env(KEYS)

    assert set(settings.providers) == {"gemini", "groq"}
    assert settings.providers["groq"].model == "groq-test"
    # Built-in free-tier defaults apply when no limits are set
    assert settings.providers["groq"].requests_per_minute == 30
    assert settings.providers["groq"].tokens_per_minute == 8000
    assert settings.providers["gemini"].tokens_per_minute is None
    # The first configured provider is the default, fallback is on, and so on
    assert settings.default_provider == "gemini"
    assert settings.enable_fallback is True
    assert (settings.max_attempts, settings.max_concurrency) == (4, 3)
    assert settings.timeout_seconds == 30.0


def test_template_placeholders_count_as_unset() -> None:
    # Someone copied .env.example and only filled in Groq
    env = {
        "GEMINI_API_KEY": "your-gemini-key-here",
        "GEMINI_MODEL": "replace-with-model-name",
        "GROQ_API_KEY": "q-key",
        "GROQ_MODEL": "groq-test",
    }

    settings = Settings.from_env(env)

    assert set(settings.providers) == {"groq"}
    assert settings.default_provider == "groq"


def test_overrides_and_zero_disables_a_limit() -> None:
    env = {
        **KEYS,
        "LLM_DEFAULT_PROVIDER": "Groq",
        "LLM_ENABLE_FALLBACK": "false",
        "LLM_MAX_RETRIES": "2",
        "LLM_MAX_CONCURRENCY": "5",
        "LLM_TIMEOUT_SECONDS": "12.5",
        "GROQ_REQUESTS_PER_MINUTE": "20",
        "GROQ_TOKENS_PER_MINUTE": "0",
    }

    settings = Settings.from_env(env)

    assert settings.default_provider == "groq"
    assert settings.enable_fallback is False
    assert (settings.max_attempts, settings.max_concurrency) == (2, 5)
    assert settings.timeout_seconds == 12.5
    assert settings.providers["groq"].requests_per_minute == 20
    # Zero turns the limiter off entirely
    assert settings.providers["groq"].tokens_per_minute is None


def test_no_providers_is_an_error() -> None:
    with pytest.raises(ValueError, match="No provider is configured"):
        Settings.from_env({})


def test_key_without_model_is_an_error() -> None:
    with pytest.raises(ValueError, match="GROQ_MODEL"):
        Settings.from_env({"GROQ_API_KEY": "q-key"})


def test_default_provider_must_be_configured() -> None:
    env = {"GROQ_API_KEY": "q-key", "GROQ_MODEL": "m", "LLM_DEFAULT_PROVIDER": "gemini"}

    with pytest.raises(ValueError, match="not configured"):
        Settings.from_env(env)
