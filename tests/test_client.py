"""Tests for the Gateway: routing, fallback, batching, concurrency, and stats."""

import asyncio
from typing import Any

import pytest

from llm_gateway.client import Gateway
from llm_gateway.config import Settings
from llm_gateway.errors import (
    AllProvidersFailedError,
    AuthenticationError,
    InvalidRequestError,
    RateLimitedError,
    RetryableError,
)
from llm_gateway.models import (
    GenerateRequest,
    GenerateResponse,
    Message,
    ProviderName,
    Usage,
)
from llm_gateway.ratelimit import RateLimitedProvider
from llm_gateway.retry import RetryingProvider


class FakeProvider:
    """A scripted provider: each call pops the next outcome (a response text or an error)."""

    def __init__(
        self,
        name: ProviderName,
        outcomes: list[str | Exception] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.outcomes = outcomes or []
        self.delay = delay
        self.calls = 0
        # Used by the concurrency test to see how many calls overlap
        self.in_flight = 0
        self.peak = 0

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            outcome = self.outcomes.pop(0) if self.outcomes else "ok"
            if isinstance(outcome, Exception):
                raise outcome
            return GenerateResponse(
                provider=self.name,
                model=self.model,
                message=Message(role="assistant", content=outcome),
                usage=Usage(input_tokens=10, output_tokens=5, reasoning_tokens=1),
                latency_ms=1.0,
            )
        finally:
            self.in_flight -= 1


def request(provider: ProviderName | None = None, text: str = "hi") -> GenerateRequest:
    """A minimal request, optionally pinned to a provider."""
    return GenerateRequest(messages=[Message.user(text)], provider=provider)


def make_gateway(gemini: FakeProvider, groq: FakeProvider, **kwargs: Any) -> Gateway:
    """A gateway over two fake providers, defaulting to gemini first."""
    return Gateway({"gemini": gemini, "groq": groq}, default_provider="gemini", **kwargs)


async def test_default_provider_answers_first() -> None:
    gemini, groq = FakeProvider("gemini", ["from gemini"]), FakeProvider("groq")

    response = await make_gateway(gemini, groq).generate(request())

    assert (response.provider, response.text) == ("gemini", "from gemini")
    assert response.failed_providers == []
    assert groq.calls == 0


async def test_falls_back_when_primary_has_a_retryable_error() -> None:
    gemini = FakeProvider("gemini", [RetryableError("503", provider="gemini")])
    groq = FakeProvider("groq", ["from groq"])

    response = await make_gateway(gemini, groq).generate(request())

    assert (response.provider, response.text) == ("groq", "from groq")
    # The response records who failed, so callers and logs can see the fallback happened
    assert response.failed_providers == ["gemini"]


async def test_falls_back_on_rate_limit_after_retries_ran_out() -> None:
    gemini = FakeProvider("gemini", [RateLimitedError("daily quota", retry_after=3600)])
    groq = FakeProvider("groq")

    response = await make_gateway(gemini, groq).generate(request())

    assert response.provider == "groq"


async def test_falls_back_on_authentication_error() -> None:
    gemini = FakeProvider("gemini", [AuthenticationError("bad key", provider="gemini")])
    groq = FakeProvider("groq")

    response = await make_gateway(gemini, groq).generate(request())

    assert response.provider == "groq"


async def test_invalid_request_does_not_fall_back() -> None:
    gemini = FakeProvider("gemini", [InvalidRequestError("bad schema", provider="gemini")])
    groq = FakeProvider("groq")

    with pytest.raises(InvalidRequestError):
        await make_gateway(gemini, groq).generate(request())

    # A request the provider rejected would most likely fail again, so groq is never tried
    assert groq.calls == 0


async def test_pinned_provider_never_falls_back() -> None:
    error = RetryableError("503", provider="groq")
    gemini, groq = FakeProvider("gemini"), FakeProvider("groq", [error])

    with pytest.raises(RetryableError) as caught:
        await make_gateway(gemini, groq).generate(request(provider="groq"))

    # The caller asked for groq specifically, and sees groq's real error
    assert caught.value is error
    assert gemini.calls == 0


async def test_fallback_can_be_disabled() -> None:
    error = RetryableError("503", provider="gemini")
    gemini, groq = FakeProvider("gemini", [error]), FakeProvider("groq")

    with pytest.raises(RetryableError) as caught:
        await make_gateway(gemini, groq, enable_fallback=False).generate(request())

    assert caught.value is error
    assert groq.calls == 0


async def test_all_providers_failing_raises_a_combined_error() -> None:
    gemini = FakeProvider("gemini", [RetryableError("503", provider="gemini")])
    groq = FakeProvider("groq", [RateLimitedError("429", provider="groq")])

    with pytest.raises(AllProvidersFailedError) as caught:
        await make_gateway(gemini, groq).generate(request())

    # Both failures are kept, in the order tried
    assert [name for name, _ in caught.value.failures] == ["gemini", "groq"]
    assert "gemini" in str(caught.value) and "groq" in str(caught.value)


async def test_unconfigured_pinned_provider_is_an_invalid_request() -> None:
    gateway = Gateway({"groq": FakeProvider("groq")}, default_provider="groq")

    with pytest.raises(InvalidRequestError, match="not configured"):
        await gateway.generate(request(provider="gemini"))


def test_default_provider_must_exist() -> None:
    with pytest.raises(ValueError):
        Gateway({"groq": FakeProvider("groq")}, default_provider="gemini")


async def test_batch_returns_results_in_order_with_failures_isolated() -> None:
    # Each call pops the next scripted outcome, so item 1 fails permanently
    groq = FakeProvider("groq", ["a", InvalidRequestError("bad input"), "c"])
    gateway = Gateway({"groq": groq}, default_provider="groq")

    results = await gateway.batch([request(text="one"), request(text="two"), request(text="three")])

    assert [r.index for r in results] == [0, 1, 2]
    assert [r.ok for r in results] == [True, False, True]
    assert results[0].response is not None and results[0].response.text == "a"
    assert results[1].error_type == "InvalidRequestError"
    assert results[1].error_message == "bad input"


async def test_batch_captures_unexpected_exceptions_too() -> None:
    groq = FakeProvider("groq", [ValueError("a bug"), "fine"])
    gateway = Gateway({"groq": groq}, default_provider="groq")

    results = await gateway.batch([request(), request()])

    # A bug in one item must not lose the other results
    assert results[0].error_type == "ValueError"
    assert results[1].ok


async def test_batch_of_nothing_is_empty() -> None:
    gateway = Gateway({"groq": FakeProvider("groq")}, default_provider="groq")

    assert await gateway.batch([]) == []


async def test_concurrency_is_capped_across_a_batch() -> None:
    groq = FakeProvider("groq", delay=0.02)
    gateway = Gateway({"groq": groq}, default_provider="groq", max_concurrency=2)

    results = await gateway.batch([request() for _ in range(8)])

    assert all(r.ok for r in results)
    # Eight requests were submitted at once, but never more than two ran together
    assert groq.peak == 2


async def test_stats_track_calls_failures_and_tokens() -> None:
    gemini = FakeProvider("gemini", [RetryableError("503", provider="gemini")])
    groq = FakeProvider("groq")
    gateway = make_gateway(gemini, groq)

    await gateway.generate(request())  # gemini fails, groq answers
    await gateway.generate(request())  # gemini answers

    stats = gateway.stats
    assert (stats["gemini"].calls, stats["gemini"].failures) == (1, 1)
    assert (stats["groq"].calls, stats["groq"].failures) == (1, 0)
    # Each fake response used 10 + 5 + 1 = 16 tokens
    assert stats["groq"].usage.total_tokens == 16


def test_from_settings_stacks_retries_over_rate_limits() -> None:
    settings = Settings.from_env(
        {
            "GEMINI_API_KEY": "g-key",
            "GEMINI_MODEL": "gemini-test",
            "GROQ_API_KEY": "q-key",
            "GROQ_MODEL": "groq-test",
            "LLM_DEFAULT_PROVIDER": "groq",
        }
    )

    gateway = Gateway.from_settings(settings)

    providers = gateway._providers  # noqa: SLF001
    assert set(providers) == {"gemini", "groq"}
    # Outermost layer retries; inside it, calls wait for a rate-limit slot
    for provider in providers.values():
        assert isinstance(provider, RetryingProvider)
        assert isinstance(provider._inner, RateLimitedProvider)  # noqa: SLF001
    assert gateway._default == "groq"  # noqa: SLF001
