"""API-key authentication for the HTTP API.

Callers send their key in an `X-API-Key` header. The accepted keys come from the
GATEWAY_API_KEYS environment variable (comma-separated, so a new key can be added
before the old one is removed). Auth is ON by default: a server with no keys
configured refuses to start, instead of silently serving the world.
"""

import hmac
import os
from collections.abc import Iterable, Mapping

# Keys shorter than this are too easy to guess; generate one with:
#   python -c "import secrets; print(secrets.token_urlsafe(32))"
MIN_KEY_LENGTH = 16

# Template values from .env.example count as "not set", like the provider settings
_PLACEHOLDER_MARKERS = ("your-", "replace-with")

_TRUE_VALUES = {"1", "true", "yes", "on"}


class ApiKeyAuth:
    """Checks a presented API key against the set of accepted keys."""

    def __init__(self, keys: Iterable[str]) -> None:
        cleaned = [key.strip() for key in keys if key.strip()]
        if not cleaned:
            raise ValueError("at least one API key is required")
        short = [key for key in cleaned if len(key) < MIN_KEY_LENGTH]
        if short:
            # Report how many, never the keys themselves
            raise ValueError(f"{len(short)} API key(s) shorter than {MIN_KEY_LENGTH} characters")
        self._keys = [key.encode() for key in cleaned]

    def verify(self, candidate: str | None) -> bool:
        """True if `candidate` matches any accepted key.

        Uses a constant-time comparison against EVERY key (no early exit), so response
        timing doesn't reveal which key matched or how much of a guess was correct.
        """
        if not candidate:
            return False
        presented = candidate.encode()
        matches = [hmac.compare_digest(presented, key) for key in self._keys]
        return any(matches)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ApiKeyAuth | None":
        """Build the auth check from environment variables.

        Returns None only when auth is switched off on purpose with
        GATEWAY_AUTH_DISABLED=true (local development). Missing keys otherwise raise,
        so a forgotten setting can never leave the API open by accident.
        """
        source = os.environ if env is None else env
        if source.get("GATEWAY_AUTH_DISABLED", "").strip().lower() in _TRUE_VALUES:
            return None

        raw = source.get("GATEWAY_API_KEYS", "")
        keys = [
            key.strip()
            for key in raw.split(",")
            if key.strip() and not any(m in key for m in _PLACEHOLDER_MARKERS)
        ]
        if not keys:
            raise ValueError(
                "No API keys configured. Set GATEWAY_API_KEYS (comma-separated), or set "
                "GATEWAY_AUTH_DISABLED=true to run without authentication (development only)."
            )
        return cls(keys)
