"""The Gateway: one entry point that routes requests, retries, falls back, and batches."""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self

from llm_gateway.config import ProviderSettings, Settings
from llm_gateway.errors import (
    AllProvidersFailedError,
    AuthenticationError,
    GatewayError,
    InvalidRequestError,
    RetryableError,
)
from llm_gateway.models import (
    BatchResult,
    GenerateRequest,
    GenerateResponse,
    ProviderName,
    Usage,
)
from llm_gateway.providers.base import Provider
from llm_gateway.providers.gemini import GeminiProvider
from llm_gateway.providers.groq import GroqProvider
from llm_gateway.ratelimit import RateLimitedProvider, RateLimiter
from llm_gateway.retry import RetryingProvider, RetryPolicy

logger = logging.getLogger(__name__)


@dataclass
class ProviderStats:
    """Running totals for one provider since the gateway started."""

    calls: int = 0  # successful responses
    failures: int = 0  # requests that ended in an error after all retries
    usage: Usage = field(default_factory=Usage)


class Gateway:
    """Send one request shape to either provider and get one response shape back.

    Routing rules:
    - A request with `provider` set goes to that provider only (no fallback), because
      the caller asked for it specifically.
    - Otherwise it goes to the default provider, then to the others if fallback is on.
    - Fallback happens on errors that are the PROVIDER's problem: retryable failures that
      outlasted the retries, and bad or missing credentials. A request the provider
      rejected as invalid is NOT retried elsewhere, since it would most likely fail again.
    """

    def __init__(
        self,
        providers: Mapping[ProviderName, Provider],
        *,
        default_provider: ProviderName,
        enable_fallback: bool = True,
        max_concurrency: int = 3,
    ) -> None:
        if default_provider not in providers:
            raise ValueError(f"default provider {default_provider!r} is not configured")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._providers = dict(providers)
        self._default = default_provider
        self._enable_fallback = enable_fallback
        # One gateway-wide cap on simultaneous requests, shared by generate() and batch()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._stats: dict[ProviderName, ProviderStats] = {
            name: ProviderStats() for name in self._providers
        }

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build the real providers, each wrapped in rate limiting and retries."""
        policy = RetryPolicy(max_attempts=settings.max_attempts)
        providers: dict[ProviderName, Provider] = {}
        for name, provider_settings in settings.providers.items():
            raw = _build_provider(name, provider_settings, settings)
            providers[name] = _wrap(raw, provider_settings, policy)
        return cls(
            providers,
            default_provider=settings.default_provider,
            enable_fallback=settings.enable_fallback,
            max_concurrency=settings.max_concurrency,
        )

    @classmethod
    def from_env(cls) -> Self:
        """Build a gateway from environment variables (.env must already be loaded)."""
        return cls.from_settings(Settings.from_env())

    @property
    def provider_names(self) -> list[ProviderName]:
        """The providers this gateway can route to."""
        return list(self._providers)

    @property
    def default_provider(self) -> ProviderName:
        """The provider tried first when a request doesn't pin one."""
        return self._default

    @property
    def stats(self) -> dict[ProviderName, ProviderStats]:
        """Per-provider call, failure, and token totals since startup."""
        return dict(self._stats)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Answer one request, retrying and falling back as configured."""
        order = self._route(request)
        # Wait here if the gateway is already running its maximum number of requests
        async with self._semaphore:
            return await self._try_in_order(order, request)

    async def batch(self, requests: Sequence[GenerateRequest]) -> list[BatchResult]:
        """Run many requests concurrently and return one result per request, in order.

        A failing request never affects the others: its slot holds an error description
        instead of a response. Concurrency is capped by the gateway-wide semaphore.
        """

        async def run_one(index: int, request: GenerateRequest) -> BatchResult:
            try:
                response = await self.generate(request)
            except Exception as exc:
                # Catch everything (not just GatewayError) so one bug can't lose the
                # other results, but log it loudly so it is not hidden
                logger.warning("batch item %d failed: %s: %s", index, type(exc).__name__, exc)
                return BatchResult(
                    index=index, error_type=type(exc).__name__, error_message=str(exc)
                )
            return BatchResult(index=index, response=response)

        return list(await asyncio.gather(*(run_one(i, r) for i, r in enumerate(requests))))

    def _route(self, request: GenerateRequest) -> list[ProviderName]:
        """Decide which providers to try, in order."""
        if request.provider is not None:
            if request.provider not in self._providers:
                raise InvalidRequestError(f"provider {request.provider!r} is not configured")
            return [request.provider]  # pinned: no fallback
        order = [self._default]
        if self._enable_fallback:
            order += [name for name in self._providers if name != self._default]
        return order

    async def _try_in_order(
        self, order: list[ProviderName], request: GenerateRequest
    ) -> GenerateResponse:
        """Try each provider until one answers."""
        failures: list[tuple[ProviderName, GatewayError]] = []
        for name in order:
            try:
                response = await self._providers[name].generate(request)
            except (RetryableError, AuthenticationError) as exc:
                # The provider's problem, so another provider may succeed
                failures.append((name, exc))
                self._stats[name].failures += 1
                logger.warning("%s failed (%s: %s)", name, type(exc).__name__, exc)
                continue
            except GatewayError:
                # The request itself is the problem (e.g. invalid), so stop here
                self._stats[name].failures += 1
                raise
            stats = self._stats[name]
            stats.calls += 1
            stats.usage = stats.usage + response.usage
            logger.info(
                "%s ok: %d attempt(s), %.0f ms, %d tokens",
                name,
                response.attempts,
                response.latency_ms,
                response.usage.total_tokens,
            )
            return response.model_copy(update={"failed_providers": [n for n, _ in failures]})

        if len(failures) == 1:
            # Only one provider was tried, so surface its real error unchanged
            raise failures[0][1]
        raise AllProvidersFailedError(failures)


def _build_provider(name: ProviderName, provider: ProviderSettings, settings: Settings) -> Provider:
    """Create the raw adapter for one provider."""
    if name == "gemini":
        return GeminiProvider(
            api_key=provider.api_key, model=provider.model, timeout=settings.timeout_seconds
        )
    return GroqProvider(
        api_key=provider.api_key,
        model=provider.model,
        base_url=settings.groq_base_url,
        timeout=settings.timeout_seconds,
    )


def _wrap(raw: Provider, provider: ProviderSettings, policy: RetryPolicy) -> Provider:
    """Stack the layers: retries on the outside, rate limiting inside, adapter at the core."""
    requests = (
        RateLimiter.per_minute(provider.requests_per_minute)
        if provider.requests_per_minute
        else None
    )
    tokens = (
        RateLimiter.per_minute(provider.tokens_per_minute) if provider.tokens_per_minute else None
    )
    limited = RateLimitedProvider(raw, requests=requests, tokens=tokens)
    return RetryingProvider(limited, policy)
