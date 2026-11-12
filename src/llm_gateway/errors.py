"""Error types for the gateway.

The key idea: every provider SDK raises its own exceptions, and the adapters translate
them into this small hierarchy. The retry layer then only needs one question per error:
"is this a RetryableError?", with no provider-specific knowledge.
"""

from collections.abc import Sequence


class GatewayError(Exception):
    """Base class for every error the gateway raises on purpose."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        # Which provider produced the error, useful for logs and fallback decisions
        self.provider = provider


class RetryableError(GatewayError):
    """A temporary failure: sending the same request again may succeed."""


class RateLimitedError(RetryableError):
    """The provider rejected the request for exceeding a quota (HTTP 429)."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        # Seconds the provider asked us to wait, when it said (the Retry-After header)
        self.retry_after = retry_after


class InvalidResponseError(RetryableError):
    """The model answered, but not in a usable shape (e.g. broken JSON).

    Retryable because model output varies between attempts. The retry layer's
    attempt limit stops this from looping forever.
    """


class NonRetryableError(GatewayError):
    """A failure that will repeat identically, so retrying only wastes quota."""


class InvalidRequestError(NonRetryableError):
    """The provider rejected the request itself (HTTP 400/404/422)."""


class AuthenticationError(NonRetryableError):
    """The API key was missing, wrong, or lacks permission (HTTP 401/403)."""


class AllProvidersFailedError(GatewayError):
    """Every provider the gateway tried (primary plus fallbacks) failed."""

    def __init__(self, failures: Sequence[tuple[str, GatewayError]]) -> None:
        summary = "; ".join(f"{name}: {type(error).__name__}: {error}" for name, error in failures)
        super().__init__(f"All providers failed ({summary})")
        # One (provider name, error) pair per provider tried, in the order tried
        self.failures = list(failures)
