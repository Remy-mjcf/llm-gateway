"""Tests for the retry layer. A fake clock records waits instead of sleeping."""

import logging

import pytest

from llm_gateway.errors import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitedError,
    RetryableError,
)
from llm_gateway.models import GenerateRequest, GenerateResponse, Message, Usage
from llm_gateway.retry import RetryingProvider, RetryPolicy, with_retries


class FakeClock:
    """Stands in for asyncio.sleep: records each requested wait and returns instantly."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)


class Flaky:
    """A callable that raises the queued errors in order, then returns "ok"."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def no_jitter_max() -> float:
    """rng stub returning 1.0: the delay equals the full backoff (the jitter ceiling)."""
    return 1.0


def no_jitter_min() -> float:
    """rng stub returning 0.0: the delay is half the backoff (the jitter floor)."""
    return 0.0


async def test_success_on_first_try_does_not_wait() -> None:
    clock = FakeClock()
    flaky = Flaky()

    result = await with_retries(flaky, sleep=clock.sleep)

    assert result == "ok"
    assert flaky.calls == 1
    assert clock.waits == []


async def test_retries_retryable_errors_until_success() -> None:
    clock = FakeClock()
    flaky = Flaky(RetryableError("503"), RetryableError("503"))

    result = await with_retries(flaky, sleep=clock.sleep, rng=no_jitter_max)

    assert result == "ok"
    assert flaky.calls == 3
    # Backoff doubles: 1s before the 2nd try, 2s before the 3rd
    assert clock.waits == [1.0, 2.0]


async def test_non_retryable_errors_fail_immediately() -> None:
    clock = FakeClock()
    flaky = Flaky(InvalidRequestError("bad schema"))

    with pytest.raises(InvalidRequestError):
        await with_retries(flaky, sleep=clock.sleep)

    # One call, no waiting: repeating a bad request would only burn quota
    assert flaky.calls == 1
    assert clock.waits == []


async def test_authentication_errors_are_not_retried() -> None:
    flaky = Flaky(AuthenticationError("bad key"))

    with pytest.raises(AuthenticationError):
        await with_retries(flaky, sleep=FakeClock().sleep)

    assert flaky.calls == 1


async def test_unexpected_exceptions_propagate_without_retry() -> None:
    # A plain bug (not a gateway error) must surface instead of being retried
    flaky = Flaky(ValueError("bug"))

    with pytest.raises(ValueError):
        await with_retries(flaky, sleep=FakeClock().sleep)

    assert flaky.calls == 1


async def test_gives_up_after_max_attempts_and_raises_last_error() -> None:
    clock = FakeClock()
    last = RetryableError("third failure")
    flaky = Flaky(RetryableError("first"), RetryableError("second"), last)

    with pytest.raises(RetryableError) as caught:
        await with_retries(flaky, RetryPolicy(max_attempts=3), sleep=clock.sleep)

    # The original last error is re-raised unchanged, not wrapped
    assert caught.value is last
    assert flaky.calls == 3
    # Two waits between three attempts; no wait after the final failure
    assert len(clock.waits) == 2


async def test_backoff_is_capped_at_max_delay() -> None:
    clock = FakeClock()
    flaky = Flaky(*[RetryableError("x")] * 4)
    policy = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=3.0)

    await with_retries(flaky, policy, sleep=clock.sleep, rng=no_jitter_max)

    # 1, 2, then 4 and 8 are clipped to the 3s ceiling
    assert clock.waits == [1.0, 2.0, 3.0, 3.0]


async def test_jitter_range_is_half_to_full_backoff() -> None:
    low, high = FakeClock(), FakeClock()

    await with_retries(Flaky(RetryableError("x")), sleep=low.sleep, rng=no_jitter_min)
    await with_retries(Flaky(RetryableError("x")), sleep=high.sleep, rng=no_jitter_max)

    # Base delay is 1s, so the first wait is anywhere from 0.5s to 1.0s
    assert low.waits == [0.5]
    assert high.waits == [1.0]


async def test_retry_after_hint_is_honored_when_longer_than_backoff() -> None:
    clock = FakeClock()
    flaky = Flaky(RateLimitedError("429", retry_after=5.0))

    await with_retries(flaky, sleep=clock.sleep, rng=no_jitter_min)

    # Backoff alone would wait 0.5s, but the provider asked for 5s
    assert clock.waits == [5.0]


async def test_gives_up_immediately_when_retry_after_exceeds_max_delay() -> None:
    clock = FakeClock()
    flaky = Flaky(RateLimitedError("daily quota", retry_after=3600.0))

    with pytest.raises(RateLimitedError):
        await with_retries(flaky, RetryPolicy(max_delay=30.0), sleep=clock.sleep)

    # Waiting an hour is pointless: surface the error so a fallback can take over
    assert flaky.calls == 1
    assert clock.waits == []


async def test_retries_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    flaky = Flaky(RetryableError("503", provider="gemini"))

    with caplog.at_level(logging.WARNING, logger="llm_gateway.retry"):
        await with_retries(flaky, sleep=FakeClock().sleep)

    assert "gemini: attempt 1/4 failed" in caplog.text


class FakeProvider:
    """A minimal Provider that fails a set number of times, then succeeds."""

    name = "groq"
    model = "fake-model"

    def __init__(self, failures: int) -> None:
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
            latency_ms=12.0,
        )


async def test_retrying_provider_records_attempt_count() -> None:
    inner = FakeProvider(failures=2)
    provider = RetryingProvider(inner, sleep=FakeClock().sleep)

    response = await provider.generate(GenerateRequest(messages=[Message.user("hi")]))

    # Two failures plus the successful try
    assert response.attempts == 3
    assert inner.calls == 3
    # The wrapper looks like the provider it wraps
    assert (provider.name, provider.model) == ("groq", "fake-model")


async def test_retrying_provider_first_try_success_reports_one_attempt() -> None:
    provider = RetryingProvider(FakeProvider(failures=0), sleep=FakeClock().sleep)

    response = await provider.generate(GenerateRequest(messages=[Message.user("hi")]))

    assert response.attempts == 1


def test_policy_rejects_nonsense_settings() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay=-1)
