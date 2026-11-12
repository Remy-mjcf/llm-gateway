"""Gateway settings, read from environment variables (your .env file)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self, cast, get_args

from llm_gateway.models import ProviderName

# Template values from .env.example mean "not filled in yet", so treat them as unset
_PLACEHOLDER_PREFIXES = ("your-", "replace-with")

# Free-tier defaults (requests/min, tokens/min). None means "no limit enforced".
# These are starting points: confirm yours in each provider's dashboard.
_DEFAULT_LIMITS: dict[ProviderName, tuple[float | None, float | None]] = {
    "gemini": (10.0, None),
    "groq": (30.0, 8000.0),
}


@dataclass(frozen=True)
class ProviderSettings:
    """Everything needed to build and throttle one provider."""

    api_key: str
    model: str
    requests_per_minute: float | None
    tokens_per_minute: float | None


@dataclass(frozen=True)
class Settings:
    """All gateway settings in one validated object."""

    providers: Mapping[ProviderName, ProviderSettings]
    groq_base_url: str
    # Total tries per provider, including the first (1 means no retries)
    max_attempts: int
    timeout_seconds: float
    max_concurrency: int
    default_provider: ProviderName
    enable_fallback: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Self:
        """Read and validate settings. Pass `env` in tests; defaults to os.environ."""
        source: Mapping[str, str] = os.environ if env is None else env

        providers: dict[ProviderName, ProviderSettings] = {}
        for name in get_args(ProviderName):
            provider_name = cast(ProviderName, name)
            prefix = provider_name.upper()
            api_key = _get(source, f"{prefix}_API_KEY")
            if api_key is None:
                continue  # No key means this provider is simply not configured
            model = _get(source, f"{prefix}_MODEL")
            if model is None:
                raise ValueError(f"{prefix}_MODEL is required when {prefix}_API_KEY is set")
            default_rpm, default_tpm = _DEFAULT_LIMITS[provider_name]
            providers[provider_name] = ProviderSettings(
                api_key=api_key,
                model=model,
                requests_per_minute=_get_limit(
                    source, f"{prefix}_REQUESTS_PER_MINUTE", default_rpm
                ),
                tokens_per_minute=_get_limit(source, f"{prefix}_TOKENS_PER_MINUTE", default_tpm),
            )

        if not providers:
            raise ValueError("No provider is configured: set GEMINI_API_KEY and/or GROQ_API_KEY")

        default = (_get(source, "LLM_DEFAULT_PROVIDER") or next(iter(providers))).lower()
        if default not in providers:
            raise ValueError(
                f"LLM_DEFAULT_PROVIDER={default!r} is not configured "
                f"(configured: {', '.join(providers)})"
            )

        return cls(
            providers=providers,
            groq_base_url=_get(source, "GROQ_BASE_URL") or "https://api.groq.com/openai/v1",
            max_attempts=int(_get(source, "LLM_MAX_RETRIES") or 4),
            timeout_seconds=float(_get(source, "LLM_TIMEOUT_SECONDS") or 30),
            max_concurrency=int(_get(source, "LLM_MAX_CONCURRENCY") or 3),
            default_provider=default,
            enable_fallback=(_get(source, "LLM_ENABLE_FALLBACK") or "true").lower()
            in ("1", "true", "yes", "on"),
        )


def _get(env: Mapping[str, str], key: str) -> str | None:
    """Read a setting, treating blank values and template placeholders as unset."""
    value = env.get(key, "").strip()
    if not value or value.startswith(_PLACEHOLDER_PREFIXES):
        return None
    return value


def _get_limit(env: Mapping[str, str], key: str, default: float | None) -> float | None:
    """Read a per-minute limit. Unset uses the default; 0 turns the limit off."""
    raw = _get(env, key)
    if raw is None:
        return default
    value = float(raw)
    return value if value > 0 else None
