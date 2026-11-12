"""Tests for the rate limiter, token estimator, and rate-limited provider wrapper."""

import asyncio

import pytest

from llm_gateway.errors import RetryableError
from llm_gateway.models import (
    GenerateRequest,
    GenerateResponse,
    Message,
    ToolSpec,
    Usage,
)
from llm_gateway.ratelimit import (
    DEFAULT_OUTPUT_TOKENS,
    RateLimitedProvider,
    RateLimiter,
    estimate_tokens,
)
from llm_gateway.retry import RetryingProvider, RetryPolicy


class FakeTime:
    """A controllable clock: sleeping records the wait and moves time forward."""

    def __init__(self) -> None:
        self.now = 100.0
        self.waits: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


def make_limiter(units_per_second: float, fake: FakeTime) -> RateLimiter:
    """Build a limiter wired to the fake clock."""
    return RateLimiter(units_per_second, clock=fake.monotonic, sleep=fake.sleep)


async def test_requests_are_spaced_by_the_rate() -> None:
    fake = FakeTime()
    limiter = make_limiter(2.0, fake)  # 2 per second -> one every 0.5s

    for _ in range(3):
        await limiter.acquire()

    # The first call goes straight through; the next two each wait half a second
    assert fake.waits == [0.5, 0.5]


async def test_cost_scales_the_spacing() -> None:
    fake = FakeTime()
    limiter = make_limiter(100.0, fake)

    await limiter.acquire(cost=250)  # reserves 2.5s of capacity
    await limiter.acquire(cost=1)

    # The second caller must wait for the expensive first call's slot to pass
    assert fake.waits == [pytest.approx(2.5)]


async def test_idle_time_is_not_banked() -> None:
    fake = FakeTime()
    limiter = make_limiter(1.0, fake)

    await limiter.acquire()
    fake.now += 60  # a long quiet period
    await limiter.acquire()
    await limiter.acquire()

    # After idling, the first call is immediate, then normal spacing resumes (no burst)
    assert fake.waits == [1.0]


async def test_concurrent_callers_get_distinct_slots() -> None:
    # Real clock and real (tiny) sleeps: three simultaneous callers must not share a slot
    limiter = RateLimiter(50.0)  # one every 20ms
    started: list[float] = []

    async def caller() -> None:
        await limiter.acquire()
        started.append(asyncio.get_running_loop().time())

    await asyncio.gather(caller(), caller(), caller())

    gaps = [b - a for a, b in zip(started, started[1:], strict=False)]
    assert all(gap >= 0.015 for gap in gaps)


def test_per_minute_converts_to_per_second() -> None:
    fake = FakeTime()
    limiter = RateLimiter.per_minute(120, clock=fake.monotonic, sleep=fake.sleep)

    # 120 per minute is 2 per second
    assert limiter._rate == pytest.approx(2.0)  # noqa: SLF001


def test_limiter_rejects_nonpositive_rates() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_estimate_tokens_counts_input_and_assumed_output() -> None:
    request = GenerateRequest(messages=[Message.user("x" * 400)])

    # 400 chars is about 100 tokens, plus the default answer allowance
    assert estimate_tokens(request) == 100 + DEFAULT_OUTPUT_TOKENS


def test_estimate_tokens_uses_max_output_tokens_when_given() -> None:
    request = GenerateRequest(messages=[Message.user("x" * 400)], max_output_tokens=50)

    assert estimate_tokens(request) == 100 + 50


def test_estimate_tokens_includes_schemas_and_tools() -> None:
    plain = GenerateRequest(messages=[Message.user("hi")])
    with_schema = GenerateRequest(
        messages=[Message.user("hi")], response_schema={"type": "object", "x": "y" * 400}
    )
    with_tool = GenerateRequest(
        messages=[Message.user("hi")],
        tools=[ToolSpec(name="t", description="d" * 400, parameters={})],
    )

    # Schemas and tool definitions are input the model must read, so they raise the estimate
    assert estimate_tokens(with_schema) > estimate_tokens(plain) + 90
    assert estimate_tokens(with_tool) > estimate_tokens(plain) + 90


class RecordingLimiter:
    """Stands in for a RateLimiter and records every acquire() it receives."""

    def __init__(self) -> None:
        self.costs: list[float] = []

    async def acquire(self, cost: float = 1.0) -> None:
        self.costs.append(cost)


class FakeProvider:
    """Fails with a retryable error `failures` times, then succeeds."""

    name = "groq"
    model = "fake-model"

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = 0

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise RetryableError("temporary", provider="groq")
        return GenerateResponse(
            provider="groq",
            model=self.model,
            message=Message(role="assistant", content="hi"),
            usage=Usage(),
            latency_ms=1.0,
        )


async def test_provider_wrapper_acquires_request_and_token_slots() -> None:
    requests, tokens = RecordingLimiter(), RecordingLimiter()
    provider = RateLimitedProvider(
        FakeProvider(),
        requests=requests,  # type: ignore[arg-type]
        tokens=tokens,  # type: ignore[arg-type]
    )
    request = GenerateRequest(messages=[Message.user("x" * 400)])

    await provider.generate(request)

    assert requests.costs == [1.0]
    assert tokens.costs == [estimate_tokens(request)]


async def test_every_retry_attempt_passes_through_the_limiter() -> None:
    # The lesson from the async notebook (experiment 5): retries spend real quota too
    requests = RecordingLimiter()
    inner = FakeProvider(failures=2)
    limited = RateLimitedProvider(inner, requests=requests)  # type: ignore[arg-type]

    async def no_sleep(seconds: float) -> None:
        return None

    provider = RetryingProvider(limited, RetryPolicy(max_attempts=4), sleep=no_sleep)

    response = await provider.generate(GenerateRequest(messages=[Message.user("hi")]))

    # 3 attempts means 3 trips through the limiter, not 1
    assert response.attempts == 3
    assert requests.costs == [1.0, 1.0, 1.0]


async def test_wrapper_without_limiters_just_passes_through() -> None:
    inner = FakeProvider()
    provider = RateLimitedProvider(inner)

    await provider.generate(GenerateRequest(messages=[Message.user("hi")]))

    assert inner.calls == 1
    assert (provider.name, provider.model) == ("groq", "fake-model")
