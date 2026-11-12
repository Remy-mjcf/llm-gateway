"""Tests for the HTTP API.

The real Gateway runs over FAKE providers, so the whole stack (routing, fallback,
batching, error mapping) is exercised with no network and no API keys.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import MAX_BATCH_SIZE, create_app
from llm_gateway.client import Gateway
from llm_gateway.errors import (
    AuthenticationError,
    GatewayError,
    InvalidRequestError,
    InvalidResponseError,
    RateLimitedError,
    RetryableError,
)
from llm_gateway.models import GenerateRequest, GenerateResponse, Message, Usage


class FakeProvider:
    """Scripted provider: each call pops the next outcome (reply text or an error)."""

    def __init__(self, name: str, outcomes: list[str | Exception] | None = None) -> None:
        self.name = name
        self.model = f"{name}-model"
        self.outcomes = outcomes or []

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        return GenerateResponse(
            provider=self.name,  # type: ignore[arg-type]
            model=self.model,
            message=Message(role="assistant", content=outcome),
            usage=Usage(input_tokens=10, output_tokens=5),
            latency_ms=1.0,
        )


def make_client(
    groq: list[str | Exception] | None = None,
    gemini: list[str | Exception] | None = None,
    *,
    fallback: bool = True,
) -> TestClient:
    """A test client over a two-provider gateway. Use as a context manager."""
    gateway = Gateway(
        {"gemini": FakeProvider("gemini", gemini), "groq": FakeProvider("groq", groq)},  # type: ignore[dict-item]
        default_provider="groq",
        enable_fallback=fallback,
    )
    return TestClient(create_app(gateway))


@pytest.fixture
def client() -> Iterator[TestClient]:
    # The `with` block keeps one event loop alive for every request in the test
    with make_client() as test_client:
        yield test_client


BODY = {"messages": [{"role": "user", "content": "hi"}]}


def test_health_lists_providers_without_calling_them(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "providers": ["gemini", "groq"],
        "default_provider": "groq",
    }


def test_generate_returns_the_normalized_response(client: TestClient) -> None:
    response = client.post("/v1/generate", json=BODY)

    assert response.status_code == 200
    data = response.json()
    assert (data["provider"], data["message"]["content"]) == ("groq", "ok")
    assert data["usage"] == {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0}
    assert data["attempts"] == 1
    assert data["failed_providers"] == []
    # Provider-internal state must never leak into HTTP responses
    assert "provider_state" not in data["message"]


def test_provider_can_be_pinned_in_the_request(client: TestClient) -> None:
    response = client.post("/v1/generate", json={**BODY, "provider": "gemini"})

    assert response.json()["provider"] == "gemini"


def test_fallback_is_visible_in_the_response() -> None:
    with make_client(groq=[RetryableError("503", provider="groq")]) as client:
        data = client.post("/v1/generate", json=BODY).json()

    assert data["provider"] == "gemini"
    assert data["failed_providers"] == ["groq"]


@pytest.mark.parametrize(
    "bad_body",
    [
        {"messages": []},  # at least one message is required
        {},  # messages is required
        {**BODY, "temperature": 5},  # out of range
        {**BODY, "provider": "openai"},  # not a supported provider
        {  # tools and a response schema can't be combined
            **BODY,
            "tools": [{"name": "t", "description": "d", "parameters": {}}],
            "response_schema": {"type": "object"},
        },
    ],
)
def test_invalid_bodies_are_rejected_with_422(
    client: TestClient, bad_body: dict[str, object]
) -> None:
    assert client.post("/v1/generate", json=bad_body).status_code == 422


def test_pinning_an_unconfigured_provider_is_a_400() -> None:
    gateway = Gateway({"groq": FakeProvider("groq")}, default_provider="groq")  # type: ignore[dict-item]
    with TestClient(create_app(gateway)) as client:
        response = client.post("/v1/generate", json={**BODY, "provider": "gemini"})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "InvalidRequestError"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (InvalidRequestError("bad schema", provider="groq"), 400),
        (AuthenticationError("bad key", provider="groq"), 502),
        (InvalidResponseError("broken json", provider="groq"), 502),
        (RetryableError("overloaded", provider="groq"), 503),
        (RateLimitedError("slow down", provider="groq"), 429),
    ],
)
def test_gateway_errors_map_to_http_statuses(error: GatewayError, status: int) -> None:
    # Pin the provider so the error surfaces directly instead of triggering a fallback
    with make_client(groq=[error]) as client:
        response = client.post("/v1/generate", json={**BODY, "provider": "groq"})

    assert response.status_code == status
    body = response.json()["error"]
    assert body["type"] == type(error).__name__
    assert body["provider"] == "groq"


def test_rate_limit_error_sets_retry_after_header_rounded_up() -> None:
    error = RateLimitedError("slow down", provider="groq", retry_after=2.4)
    with make_client(groq=[error]) as client:
        response = client.post("/v1/generate", json={**BODY, "provider": "groq"})

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "3"


def test_all_providers_failing_is_a_503_listing_every_failure() -> None:
    with make_client(
        groq=[RetryableError("503", provider="groq")],
        gemini=[RateLimitedError("429", provider="gemini")],
    ) as client:
        response = client.post("/v1/generate", json=BODY)

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "AllProvidersFailedError"
    assert [f["provider"] for f in error["failures"]] == ["groq", "gemini"]
    assert [f["type"] for f in error["failures"]] == ["RetryableError", "RateLimitedError"]


def test_batch_reports_per_item_results_in_order() -> None:
    with make_client(groq=["a", InvalidRequestError("bad input"), "c"]) as client:
        response = client.post("/v1/batch", json={"requests": [BODY, BODY, BODY]})

    # A failing item never turns the whole batch into an error response
    assert response.status_code == 200
    data = response.json()
    assert (data["succeeded"], data["failed"]) == (2, 1)
    assert [r["index"] for r in data["results"]] == [0, 1, 2]
    assert data["results"][0]["response"]["message"]["content"] == "a"
    assert data["results"][1]["response"] is None
    assert data["results"][1]["error_type"] == "InvalidRequestError"


def test_batch_size_limits(client: TestClient) -> None:
    assert client.post("/v1/batch", json={"requests": []}).status_code == 422

    too_many = {"requests": [BODY] * (MAX_BATCH_SIZE + 1)}
    assert client.post("/v1/batch", json=too_many).status_code == 422

    at_limit = {"requests": [BODY] * MAX_BATCH_SIZE}
    assert client.post("/v1/batch", json=at_limit).status_code == 200


def test_stats_endpoint_reflects_traffic(client: TestClient) -> None:
    client.post("/v1/generate", json=BODY)
    client.post("/v1/generate", json=BODY)

    data = client.get("/v1/stats").json()["providers"]

    assert data["groq"]["calls"] == 2
    assert data["groq"]["usage"]["input_tokens"] == 20
    assert data["gemini"]["calls"] == 0


def test_openapi_schema_documents_the_endpoints(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert {"/health", "/v1/generate", "/v1/batch", "/v1/stats"} <= set(schema["paths"])
    # The Pydantic models become the documented request/response schemas
    assert "GenerateRequest" in schema["components"]["schemas"]
    assert "GenerateResponse" in schema["components"]["schemas"]
