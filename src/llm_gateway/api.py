"""The HTTP API: FastAPI endpoints over the Gateway.

Run it with:  uvicorn llm_gateway.api:app --reload
Interactive docs (generated from the Pydantic models) appear at /docs.
"""

import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from llm_gateway.auth import ApiKeyAuth
from llm_gateway.client import Gateway
from llm_gateway.errors import (
    AllProvidersFailedError,
    AuthenticationError,
    GatewayError,
    InvalidRequestError,
    InvalidResponseError,
    RateLimitedError,
    RetryableError,
)
from llm_gateway.models import (
    BatchResult,
    GenerateRequest,
    GenerateResponse,
    ProviderName,
    Usage,
)

logger = logging.getLogger(__name__)

# Most requests one /v1/batch call may carry. On a free tier a large batch is slow
# (the rate limiter spaces requests out), so keep one HTTP call a reasonable size.
MAX_BATCH_SIZE = 50


# ---------------------------------------------------------------- API-only models
class BatchRequest(BaseModel):
    """Body of POST /v1/batch."""

    requests: list[GenerateRequest] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    """Result of POST /v1/batch: one entry per request, in the order submitted."""

    results: list[BatchResult]
    succeeded: int
    failed: int


class HealthResponse(BaseModel):
    """Result of GET /health."""

    status: str
    providers: list[ProviderName]
    default_provider: ProviderName


class ProviderStatsOut(BaseModel):
    """Running totals for one provider."""

    calls: int
    failures: int
    usage: Usage


class StatsResponse(BaseModel):
    """Result of GET /v1/stats."""

    providers: dict[ProviderName, ProviderStatsOut]


class ProviderFailure(BaseModel):
    """One provider's failure inside a combined error."""

    provider: str
    type: str
    message: str


class ErrorBody(BaseModel):
    """The `error` object in every error response."""

    type: str
    message: str
    provider: str | None = None
    # Filled in only when every provider failed
    failures: list[ProviderFailure] | None = None


class ErrorResponse(BaseModel):
    """Shape of every error response the gateway returns."""

    error: ErrorBody


# Documents the error responses in /docs
ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key header."},
    400: {"model": ErrorResponse, "description": "The request was rejected as invalid."},
    429: {"model": ErrorResponse, "description": "Rate limited; see the Retry-After header."},
    502: {"model": ErrorResponse, "description": "Upstream credentials or output were bad."},
    503: {"model": ErrorResponse, "description": "Providers are temporarily unavailable."},
}


# ---------------------------------------------------------------- error mapping
def status_for(exc: GatewayError) -> int:
    """Pick the HTTP status code for a gateway error."""
    if isinstance(exc, InvalidRequestError):
        return 400
    if isinstance(exc, RateLimitedError):
        return 429
    # Our upstream credentials or the model's output were bad: not the caller's fault
    if isinstance(exc, AuthenticationError | InvalidResponseError):
        return 502
    if isinstance(exc, RetryableError | AllProvidersFailedError):
        return 503
    return 500


def error_response(exc: GatewayError) -> JSONResponse:
    """Build the JSON error response (and Retry-After header) for a gateway error."""
    body = ErrorBody(type=type(exc).__name__, message=str(exc), provider=exc.provider)
    if isinstance(exc, AllProvidersFailedError):
        body.failures = [
            ProviderFailure(provider=name, type=type(error).__name__, message=str(error))
            for name, error in exc.failures
        ]
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitedError) and exc.retry_after is not None:
        # The Retry-After header is whole seconds, so round up
        headers["Retry-After"] = str(math.ceil(exc.retry_after))
    return JSONResponse(
        status_code=status_for(exc),
        content=ErrorResponse(error=body).model_dump(),
        headers=headers,
    )


# ---------------------------------------------------------------- authentication
class UnauthorizedError(Exception):
    """The caller sent no API key, or one that isn't accepted."""


# Declares the header to FastAPI so /docs shows an "Authorize" button for it
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,  # we raise our own error so it matches the gateway's error shape
    description="One of the keys listed in the GATEWAY_API_KEYS environment variable.",
)


async def require_api_key(
    request: Request, api_key: Annotated[str | None, Security(api_key_header)]
) -> None:
    """Dependency: reject the request with 401 unless it carries an accepted key."""
    auth = cast("ApiKeyAuth | None", request.app.state.auth)
    if auth is None:
        return  # authentication was switched off on purpose (development)
    if not auth.verify(api_key):
        raise UnauthorizedError


# Added to the /v1 routes. /health stays open so container healthchecks need no secret.
PROTECTED = [Depends(require_api_key)]


# ---------------------------------------------------------------- app wiring
def get_gateway(request: Request) -> Gateway:
    """Dependency: the Gateway this app serves (tests can override it)."""
    return cast(Gateway, request.app.state.gateway)


GatewayDep = Annotated[Gateway, Depends(get_gateway)]


def create_app(gateway: Gateway | None = None, *, auth: ApiKeyAuth | None = None) -> FastAPI:
    """Build the FastAPI app.

    Tests pass a ready-made `gateway` (and optionally `auth`); with no `auth`, the API is
    open. Otherwise both are built from the environment at startup, where auth is REQUIRED
    unless GATEWAY_AUTH_DISABLED=true.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Build the real gateway at startup, from environment variables. A local .env
        # file is loaded if present; real environment variables always win (Docker/prod).
        if getattr(app.state, "gateway", None) is None:
            load_dotenv()
            app.state.gateway = Gateway.from_env()
            if auth is None:
                # Fails startup (on purpose) when no keys are set and auth isn't disabled
                app.state.auth = ApiKeyAuth.from_env()
        if app.state.auth is None:
            logger.warning("API authentication is DISABLED; do not expose this server")
        yield

    app = FastAPI(
        title="llm-gateway",
        description="One API over multiple LLM providers, with retries, fallback, and batching.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.auth = auth
    if gateway is not None:
        app.state.gateway = gateway

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        logger.warning("%s %s -> %s: %s", request.method, request.url.path, type(exc).__name__, exc)
        return error_response(exc)

    @app.exception_handler(UnauthorizedError)
    async def handle_unauthorized(request: Request, exc: UnauthorizedError) -> JSONResponse:
        # Log the path only, never the presented key
        logger.warning("%s %s -> 401 missing or invalid API key", request.method, request.url.path)
        body = ErrorBody(type="UnauthorizedError", message="Missing or invalid API key.")
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(error=body).model_dump(),
            headers={"WWW-Authenticate": "ApiKey"},
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(gateway: GatewayDep) -> HealthResponse:
        """Liveness check. Makes no provider calls, so it uses no quota."""
        return HealthResponse(
            status="ok",
            providers=gateway.provider_names,
            default_provider=gateway.default_provider,
        )

    @app.post(
        "/v1/generate",
        response_model=GenerateResponse,
        responses=ERROR_RESPONSES,
        dependencies=PROTECTED,
        tags=["generate"],
    )
    async def generate(body: GenerateRequest, gateway: GatewayDep) -> GenerateResponse:
        """Answer one request: text, structured output (response_schema), or tool calls."""
        return await gateway.generate(body)

    @app.post(
        "/v1/batch",
        response_model=BatchResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key header."},
            422: {"description": "Empty batch or more than the maximum size."},
        },
        dependencies=PROTECTED,
        tags=["generate"],
    )
    async def batch(body: BatchRequest, gateway: GatewayDep) -> BatchResponse:
        """Run many requests concurrently. Failures are reported per item, never as a 5xx."""
        results = await gateway.batch(body.requests)
        succeeded = sum(1 for result in results if result.ok)
        return BatchResponse(results=results, succeeded=succeeded, failed=len(results) - succeeded)

    @app.get(
        "/v1/stats",
        response_model=StatsResponse,
        responses={
            401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key header."}
        },
        dependencies=PROTECTED,
        tags=["ops"],
    )
    async def stats(gateway: GatewayDep) -> StatsResponse:
        """Calls, failures, and token totals per provider since the server started."""
        return StatsResponse(
            providers={
                name: ProviderStatsOut(calls=s.calls, failures=s.failures, usage=s.usage)
                for name, s in gateway.stats.items()
            }
        )

    return app


# The app object uvicorn serves: `uvicorn llm_gateway.api:app`
app = create_app()
