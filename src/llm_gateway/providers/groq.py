"""Groq adapter, built on the OpenAI SDK pointed at Groq's OpenAI-compatible endpoint."""

import json
import time
from collections.abc import Mapping
from typing import Any, cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from llm_gateway.errors import (
    AuthenticationError,
    GatewayError,
    InvalidRequestError,
    InvalidResponseError,
    RateLimitedError,
    RetryableError,
)
from llm_gateway.models import (
    GenerateRequest,
    GenerateResponse,
    Message,
    ProviderName,
    ToolCall,
    Usage,
)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def make_strict(node: Any) -> Any:
    """Rewrite a JSON Schema so Groq's strict mode accepts it.

    Strict mode requires every object to list ALL its properties under `required`
    and to forbid extra keys. Pydantic doesn't emit that by default, and nested
    objects need it too, so this walks the whole schema recursively.
    """
    if isinstance(node, dict):
        # Rebuild each child first, so nested objects are fixed before their parent
        fixed = {key: make_strict(value) for key, value in node.items()}
        if fixed.get("type") == "object" and "properties" in fixed:
            fixed["additionalProperties"] = False
            fixed["required"] = list(fixed["properties"])
        return fixed
    if isinstance(node, list):
        return [make_strict(item) for item in node]
    return node


def to_openai_messages(request: GenerateRequest) -> list[dict[str, Any]]:
    """Convert gateway messages into OpenAI/Groq chat format."""
    out: list[dict[str, Any]] = []
    if request.system:
        out.append({"role": "system", "content": request.system})

    for message in request.messages:
        if message.role == "user":
            out.append({"role": "user", "content": message.content})
        elif message.role == "assistant":
            turn: dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                # Groq expects arguments as a JSON STRING, the reverse of what we store
                turn["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in message.tool_calls
                ]
            out.append(turn)
        else:
            # One "tool" message per result; each is tied to its call by id
            for result in message.tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": json.dumps(result.content),
                    }
                )
    return out


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Read the Retry-After header (in seconds) from a 429 response, if present."""
    raw = headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def map_error(exc: openai.OpenAIError) -> GatewayError:
    """Translate an OpenAI-SDK exception into the gateway's error hierarchy."""
    if isinstance(exc, openai.RateLimitError):
        return RateLimitedError(
            str(exc), provider="groq", retry_after=_retry_after_seconds(exc.response.headers)
        )
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return AuthenticationError(str(exc), provider="groq")
    if isinstance(
        exc, openai.BadRequestError | openai.NotFoundError | openai.UnprocessableEntityError
    ):
        return InvalidRequestError(str(exc), provider="groq")
    # Timeouts, dropped connections, and 5xx responses are temporary
    if isinstance(
        exc, openai.APITimeoutError | openai.APIConnectionError | openai.InternalServerError
    ):
        return RetryableError(str(exc), provider="groq")
    # Anything unrecognized: assume permanent so we never retry blindly
    return InvalidRequestError(str(exc), provider="groq")


class GroqProvider:
    """Calls Groq-hosted models and returns provider-neutral responses."""

    name: ProviderName = "groq"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        # `client` can be injected so tests supply a fake and never touch the network.
        # max_retries=0 turns off the SDK's own retries: our retry layer owns that job.
        self._client = client or AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Make one Groq call and normalize the result."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": cast(
                "list[ChatCompletionMessageParam]", to_openai_messages(request)
            ),
        }
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
            kwargs["tool_choice"] = "auto"
        if request.response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": make_strict(request.response_schema),
                },
            }
        if request.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.reasoning_effort is not None:
            # Not a standard OpenAI parameter, so it travels in extra_body
            kwargs["extra_body"] = {"reasoning_effort": request.reasoning_effort}

        # Time the call ourselves so providers are compared on the same clock
        started = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            raise map_error(exc) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        raw_message = completion.choices[0].message

        tool_calls: list[ToolCall] = []
        for raw_call in raw_message.tool_calls or []:
            try:
                arguments = json.loads(raw_call.function.arguments)
            except json.JSONDecodeError as exc:
                raise InvalidResponseError(
                    f"Tool call arguments were not valid JSON: {exc}", provider="groq"
                ) from exc
            tool_calls.append(
                ToolCall(id=raw_call.id, name=raw_call.function.name, arguments=arguments)
            )

        parsed: dict[str, Any] | None = None
        if request.response_schema is not None:
            try:
                parsed = json.loads(raw_message.content or "")
            except json.JSONDecodeError as exc:
                raise InvalidResponseError(
                    f"Structured output was not valid JSON: {exc}", provider="groq"
                ) from exc
            if not isinstance(parsed, dict):
                raise InvalidResponseError(
                    "Structured output was not a JSON object", provider="groq"
                )

        return GenerateResponse(
            provider="groq",
            model=self.model,
            message=Message(
                role="assistant", content=raw_message.content, tool_calls=tool_calls
            ),
            parsed=parsed,
            usage=_usage_from(completion.usage),
            latency_ms=latency_ms,
        )


def _usage_from(raw: Any) -> Usage:
    """Map Groq's usage block onto the gateway's Usage.

    Groq's completion_tokens INCLUDES hidden reasoning tokens, so subtract them to get
    the visible answer length. Missing fields count as zero.
    """
    if raw is None:
        return Usage()
    details = getattr(raw, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) or 0
    completion = raw.completion_tokens or 0
    return Usage(
        input_tokens=raw.prompt_tokens or 0,
        output_tokens=max(completion - reasoning, 0),
        reasoning_tokens=reasoning,
    )
