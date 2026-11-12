"""Retry layer: exponential backoff with jitter that honors provider retry hints.

Only RetryableError is retried. Anything else (bad request, bad key, a bug in our own
code) fails immediately, because repeating it would only waste free-tier quota.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from llm_gateway.errors import RateLimitedError, RetryableError
from llm_gateway.models import GenerateRequest, GenerateResponse
from llm_gateway.providers.base import Provider

logger = logging.getLogger(__name__)

# Signatures of the two things tests replace: the clock and the randomness
Sleep = Callable[[float], Awaitable[None]]
Rng = Callable[[], float]


@dataclass(frozen=True)
class RetryPolicy:
    """How patient the gateway is with a failing provider."""

    # Total tries including the first one (4 means 1 try + up to 3 retries)
    max_attempts: int = 4
    # Backoff before the 2nd try; doubles each time after that
    base_delay: float = 1.0
    # Ceiling for any single wait. A provider asking for longer than this is not waited on.
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        # Fail loudly on nonsense settings instead of misbehaving later
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must not be negative")


def compute_delay(
    policy: RetryPolicy, failed_attempt: int, error: RetryableError, rng: Rng
) -> float | None:
    """Seconds to wait before the next try, or None to give up right now.

    `failed_attempt` is the 1-based number of the try that just failed.
    """
    # Exponential backoff: base, 2x base, 4x base, ... capped at max_delay
    backoff = min(policy.base_delay * 2.0 ** (failed_attempt - 1), policy.max_delay)
    # "Equal jitter": wait between half and all of the backoff. Randomness keeps many
    # concurrent callers from retrying in lockstep, and the half floor means a rate
    # limit is never answered with a near-zero wait.
    delay = backoff / 2 + rng() * backoff / 2

    if isinstance(error, RateLimitedError) and error.retry_after is not None:
        # The provider said how long to wait. If that is longer than we are willing to
        # sleep (e.g. a daily quota), give up now so the caller can fall back instead.
        if error.retry_after > policy.max_delay:
            return None
        delay = max(delay, error.retry_after)
    return delay


async def with_retries[T](
    call: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
    *,
    sleep: Sleep = asyncio.sleep,
    rng: Rng = random.random,
) -> T:
    """Run `call`, retrying retryable failures according to the policy.

    On giving up, the LAST error is re-raised unchanged, so callers see the real cause.
    """
    policy = policy or RetryPolicy()
    attempt = 1
    while True:
        try:
            return await call()
        except RetryableError as exc:
            delay = compute_delay(policy, attempt, exc, rng)
            if delay is None:
                logger.warning(
                    "%s: not retrying %s, provider asked for a wait longer than %.0fs",
                    exc.provider or "provider",
                    type(exc).__name__,
                    policy.max_delay,
                )
                raise
            if attempt >= policy.max_attempts:
                logger.warning(
                    "%s: giving up after %d attempts (%s)",
                    exc.provider or "provider",
                    attempt,
                    type(exc).__name__,
                )
                raise
            logger.warning(
                "%s: attempt %d/%d failed (%s), retrying in %.1fs",
                exc.provider or "provider",
                attempt,
                policy.max_attempts,
                type(exc).__name__,
                delay,
            )
            await sleep(delay)
            attempt += 1


class RetryingProvider:
    """Wraps any Provider so its generate() calls are retried per the policy.

    It is itself a Provider, so it can be dropped in anywhere a plain one is expected.
    """

    def __init__(
        self,
        inner: Provider,
        policy: RetryPolicy | None = None,
        *,
        sleep: Sleep = asyncio.sleep,
        rng: Rng = random.random,
    ) -> None:
        self._inner = inner
        self._policy = policy or RetryPolicy()
        self._sleep = sleep
        self._rng = rng
        # Mirror the wrapped provider's identity so routing and logs stay accurate
        self.name = inner.name
        self.model = inner.model

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Call the wrapped provider with retries and record how many tries it took."""
        attempts = 0

        async def attempt() -> GenerateResponse:
            # Count every try, including the ones that fail
            nonlocal attempts
            attempts += 1
            return await self._inner.generate(request)

        response = await with_retries(attempt, self._policy, sleep=self._sleep, rng=self._rng)
        return response.model_copy(update={"attempts": attempts})
