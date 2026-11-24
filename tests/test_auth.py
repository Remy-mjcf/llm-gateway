"""Tests for API-key authentication: the key checker, env parsing, and the HTTP behavior."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import create_app
from llm_gateway.auth import MIN_KEY_LENGTH, ApiKeyAuth
from llm_gateway.client import Gateway
from llm_gateway.models import GenerateRequest, GenerateResponse, Message, Usage

# Realistic-length fake keys (the checker rejects short ones)
KEY = "test-key-0123456789abcdef"
NEW_KEY = "test-key-fedcba9876543210"


# ------------------------------------------------------------------ ApiKeyAuth itself
def test_verify_accepts_only_configured_keys() -> None:
    auth = ApiKeyAuth([KEY, NEW_KEY])

    assert auth.verify(KEY)
    assert auth.verify(NEW_KEY)  # a second key lets you rotate without downtime
    assert not auth.verify("test-key-wrong-wrong-wrong")
    assert not auth.verify(KEY[:-1])  # a prefix of a real key is not enough
    assert not auth.verify("")
    assert not auth.verify(None)


def test_short_keys_are_rejected_without_echoing_them() -> None:
    with pytest.raises(ValueError, match=str(MIN_KEY_LENGTH)) as caught:
        ApiKeyAuth(["short"])

    assert "short" not in str(caught.value).replace("shorter", "")


def test_at_least_one_key_is_required() -> None:
    with pytest.raises(ValueError):
        ApiKeyAuth([" ", ""])


def test_from_env_parses_a_comma_separated_list() -> None:
    auth = ApiKeyAuth.from_env({"GATEWAY_API_KEYS": f" {KEY} , {NEW_KEY} ,"})

    assert auth is not None
    assert auth.verify(KEY) and auth.verify(NEW_KEY)


def test_from_env_fails_closed_when_no_keys_are_set() -> None:
    # A forgotten setting must stop the server, not leave it open
    with pytest.raises(ValueError, match="GATEWAY_API_KEYS"):
        ApiKeyAuth.from_env({})


def test_template_placeholders_count_as_unset() -> None:
    with pytest.raises(ValueError, match="GATEWAY_API_KEYS"):
        ApiKeyAuth.from_env({"GATEWAY_API_KEYS": "your-api-key-here-please"})


@pytest.mark.parametrize("flag", ["true", "TRUE", "1", "yes"])
def test_auth_can_be_disabled_explicitly(flag: str) -> None:
    assert ApiKeyAuth.from_env({"GATEWAY_AUTH_DISABLED": flag}) is None


def test_disabled_flag_must_be_a_clear_yes() -> None:
    # "false" (or anything unclear) must not switch authentication off
    with pytest.raises(ValueError):
        ApiKeyAuth.from_env({"GATEWAY_AUTH_DISABLED": "false"})


# ------------------------------------------------------------------ over HTTP
class CountingProvider:
    """Always answers, and counts calls so tests can prove rejected requests never reach it."""

    name = "groq"
    model = "groq-model"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.calls += 1
        return GenerateResponse(
            provider="groq",
            model=self.model,
            message=Message(role="assistant", content="ok"),
            usage=Usage(input_tokens=1, output_tokens=1),
            latency_ms=1.0,
        )


BODY = {"messages": [{"role": "user", "content": "hi"}]}
HEADERS = {"X-API-Key": KEY}


@pytest.fixture
def provider() -> CountingProvider:
    return CountingProvider()


@pytest.fixture
def client(provider: CountingProvider) -> Iterator[TestClient]:
    gateway = Gateway({"groq": provider}, default_provider="groq")  # type: ignore[dict-item]
    app = create_app(gateway, auth=ApiKeyAuth([KEY, NEW_KEY]))
    with TestClient(app) as test_client:
        yield test_client


def test_health_stays_open_for_container_healthchecks(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_missing_key_is_a_401_in_the_standard_error_shape(
    client: TestClient, provider: CountingProvider
) -> None:
    response = client.post("/v1/generate", json=BODY)

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "UnauthorizedError"
    assert response.headers["WWW-Authenticate"] == "ApiKey"
    # The request was stopped before it could spend any provider quota
    assert provider.calls == 0


def test_wrong_key_is_a_401(client: TestClient, provider: CountingProvider) -> None:
    response = client.post("/v1/generate", json=BODY, headers={"X-API-Key": "nope-nope-nope-nope"})

    assert response.status_code == 401
    assert provider.calls == 0


def test_correct_key_is_accepted(client: TestClient) -> None:
    response = client.post("/v1/generate", json=BODY, headers=HEADERS)

    assert response.status_code == 200
    assert response.json()["message"]["content"] == "ok"


def test_either_configured_key_works(client: TestClient) -> None:
    response = client.post("/v1/generate", json=BODY, headers={"X-API-Key": NEW_KEY})

    assert response.status_code == 200


def test_every_v1_endpoint_is_protected(client: TestClient) -> None:
    assert client.post("/v1/batch", json={"requests": [BODY]}).status_code == 401
    assert client.get("/v1/stats").status_code == 401

    assert client.post("/v1/batch", json={"requests": [BODY]}, headers=HEADERS).status_code == 200
    assert client.get("/v1/stats", headers=HEADERS).status_code == 200


def test_auth_is_checked_before_the_body_is_validated(client: TestClient) -> None:
    # An unauthenticated caller learns nothing about the request schema from 422 details
    assert client.post("/v1/generate", json={}).status_code == 401


def test_openapi_documents_the_api_key_header(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    schemes = schema["components"]["securitySchemes"]
    assert any(s["type"] == "apiKey" and s["name"] == "X-API-Key" for s in schemes.values())


# ------------------------------------------------------------------ startup from the environment
@pytest.fixture
def provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment with one (fake) provider configured and no auth settings."""
    # Never read a developer's real .env file during tests
    monkeypatch.setattr("llm_gateway.api.load_dotenv", lambda: False)
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key-for-tests")
    monkeypatch.setenv("GROQ_MODEL", "fake-model")
    for name in ("GEMINI_API_KEY", "GATEWAY_API_KEYS", "GATEWAY_AUTH_DISABLED"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("provider_env")
def test_server_refuses_to_start_without_keys() -> None:
    # The real startup path: no keys and no explicit opt-out means no server
    with pytest.raises(ValueError, match="GATEWAY_API_KEYS"), TestClient(create_app()):
        pass


@pytest.mark.usefixtures("provider_env")
def test_server_starts_with_keys_and_enforces_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_API_KEYS", KEY)

    with TestClient(create_app()) as test_client:
        assert test_client.get("/v1/stats").status_code == 401
        assert test_client.get("/v1/stats", headers=HEADERS).status_code == 200


@pytest.mark.usefixtures("provider_env")
def test_server_can_start_open_when_explicitly_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_AUTH_DISABLED", "true")

    with TestClient(create_app()) as test_client:
        assert test_client.get("/v1/stats").status_code == 200
