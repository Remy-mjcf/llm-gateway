"""Rate limiting: keep the gateway under each provider's free-tier quota.

A semaphore (see client.py) limits how many calls run AT ONCE. These limiters limit how
many calls happen PER MINUTE, which is what free tiers actually meter.
"""

import asyncio
import json
import math
import time
from collections.abc import Callable
from typing import Self

from llm_gateway.models import GenerateRequest, GenerateResponse
from llm_gateway.providers.base import Provider
from llm_gateway.retry import Sleep

# Assumed answer length when the caller sets no max_output_tokens. Measured: a short
# structured answer from a reasoning model used roughly 250 completion tokens.
DEFAULT_OUTPUT_TOKENS = 300


class RateLimiter:
    """Spaces calls out so the average rate never exceeds `units_per_second`.

    "Units" are whatever the cost says: 1 per request, or the estimated token count.
    """

    def __init__(
        self,
        units_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if units_per_second <= 0:
            raise ValueError("units_per_second must be positive")
        self._rate = units_per_second
        # Clock and sleep are injectable so tests can run instantly
        self._clock = clock
        self._sleep = sleep
        # The earliest moment the NEXT caller may start
        self._next_free = 0.0
        # Makes "read next_free, then update it" atomic between concurrent tasks
        self._lock = asyncio.Lock()

    @classmethod
    def per_minute(
        cls,
        units_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> Self:
        """Build a limiter from a per-minute quota, the unit providers publish."""
        return cls(units_per_minute / 60.0, clock=clock, sleep=sleep)

    async def acquire(self, cost: float = 1.0) -> None:
        """Wait until it is this caller's turn."""
        async with self._lock:
            now = self._clock()
            start = max(now, self._next_free)
            # Reserve this caller's slot: the next caller must wait cost/rate seconds
            self._next_free = start + cost / self._rate
        wait = start - now
        # Sleep OUTSIDE the lock so other callers can reserve their own slots meanwhile
        if wait > 0:
            await self._sleep(wait)


def estimate_tokens(request: GenerateRequest) -> int:
    """Rough token estimate for a request, used to budget tokens per minute.

    Uses the common ~4 characters per token rule of thumb plus an assumed answer length.
    It is only an estimate: the true usage is known after the response arrives.
    """
    chars = len(request.system or "")
    for message in request.messages:
        chars += len(message.content or "")
        for call in message.tool_calls:
            chars += len(json.dumps(call.arguments))
        for result in message.tool_results:
            chars += len(json.dumps(result.content))
    # Tool definitions and the response schema are sent to the model as input too
    for tool in request.tools:
        chars += len(tool.name) + len(tool.description) + len(json.dumps(tool.parameters))
    if request.response_schema is not None:
        chars += len(json.dumps(request.response_schema))
    answer = request.max_output_tokens or DEFAULT_OUTPUT_TOKENS
    return math.ceil(chars / 4) + answer


class RateLimitedProvider:
    """Wraps a Provider so each call first waits for a rate-limit slot.

    Wrap this INSIDE the retry layer, so that every retry attempt also passes through
    the limiter. Retries spend real quota, so they must not skip the queue.
    """

    def __init__(
        self,
        inner: Provider,
        *,
        requests: RateLimiter | None = None,
        tokens: RateLimiter | None = None,
    ) -> None:
        self._inner = inner
        self._requests = requests
        self._tokens = tokens
        # Mirror the wrapped provider's identity so routing and logs stay accurate
        self.name = inner.name
        self.model = inner.model

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Wait for any configured limits, then call the wrapped provider."""
        if self._requests is not None:
            await self._requests.acquire()
        if self._tokens is not None:
            await self._tokens.acquire(cost=estimate_tokens(request))
        return await self._inner.generate(request)
