"""Gemini adapter, built on the native google-genai SDK."""

import json
import time
import uuid
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

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
    ReasoningEffort,
    ToolCall,
    Usage,
)

# Gateway reasoning effort -> Gemini 3 "thinking level" (see Google's Thinking docs)
_THINKING_LEVELS: dict[ReasoningEffort, types.ThinkingLevel] = {
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}


def _uses_thinking_level(model: str) -> bool:
    """Whether this model takes `thinking_level` (Gemini 3 and later).

    Gemini 2.x models use a token `thinking_budget` instead, and sending them a level
    would be rejected. For those models reasoning_effort is ignored for now.
    """
    return not model.startswith("gemini-2")


def to_gemini_contents(request: GenerateRequest) -> list[types.Content]:
    """Convert gateway messages into Gemini `Content` turns."""
    contents: list[types.Content] = []
    for message in request.messages:
        if message.role == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part(text=message.content or "")])
            )
        elif message.role == "assistant":
            if isinstance(message.provider_state, types.Content):
                # Echo Gemini's own turn back untouched. Newer models attach hidden
                # metadata (thought signatures) that they expect to see again.
                contents.append(message.provider_state)
            else:
                # The turn came from another provider (e.g. after a fallback), so
                # rebuild it. Caveat: Gemini 3 may reject rebuilt tool-call turns that
                # lack signatures, so a fallback in the MIDDLE of a tool loop can fail.
                parts: list[types.Part] = []
                if message.content:
                    parts.append(types.Part(text=message.content))
                for call in message.tool_calls:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(name=call.name, args=call.arguments)
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))
        else:
            # All results of one round go back together in a single user turn.
            # Gemini matches results to calls by function name, not by id.
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=result.name, response={"result": result.content}
                        )
                        for result in message.tool_results
                    ],
                )
            )
    return contents


def _find_retry_delay(node: Any) -> Any:
    """Search an error body (any nesting of dicts/lists) for a `retryDelay` value."""
    if isinstance(node, dict):
        if "retryDelay" in node:
            return node["retryDelay"]
        children: list[Any] = list(node.values())
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _find_retry_delay(child)
        if found is not None:
            return found
    return None


def _retry_delay_seconds(details: Any) -> float | None:
    """Turn Gemini's retry hint (a string like "27s") into seconds, if present."""
    raw = _find_retry_delay(details)
    if isinstance(raw, str) and raw.endswith("s"):
        try:
            return float(raw[:-1])
        except ValueError:
            return None
    return None


def map_error(exc: genai_errors.APIError) -> GatewayError:
    """Translate a google-genai API error into the gateway's error hierarchy."""
    code = exc.code
    message = f"{exc.status or code}: {exc.message}"
    if code == 429:
        return RateLimitedError(
            message, provider="gemini", retry_after=_retry_delay_seconds(exc.details)
        )
    # Gemini reports a bad API key as 400 INVALID_ARGUMENT, so check the text too
    if code in (401, 403) or (code == 400 and "api key" in (exc.message or "").lower()):
        return AuthenticationError(message, provider="gemini")
    # Server-side trouble and request timeouts are temporary
    if code >= 500 or code == 408:
        return RetryableError(message, provider="gemini")
    # Remaining 4xx errors (bad model name, bad schema, ...) will repeat identically
    return InvalidRequestError(message, provider="gemini")


def _is_transport_error(exc: Exception) -> bool:
    """Whether an exception is a network-level failure (timeout, dropped connection).

    Checked by module name so it works however the SDK's HTTP library is packaged.
    """
    return type(exc).__module__.split(".")[0] in {"httpx", "httpx2", "httpcore"}


def _usage_from(raw: types.GenerateContentResponseUsageMetadata | None) -> Usage:
    """Map Gemini's usage block onto the gateway's Usage.

    The provider's total is the number that counts against quota, so it is trusted:
    visible output = total - input - reasoning. That stays correct whether or not
    `candidates_token_count` happens to include the thinking tokens.
    """
    if raw is None:
        return Usage()
    input_tokens = (raw.prompt_token_count or 0) + (raw.tool_use_prompt_token_count or 0)
    reasoning = raw.thoughts_token_count or 0
    if raw.total_token_count is not None:
        output = max(raw.total_token_count - input_tokens - reasoning, 0)
    else:
        output = raw.candidates_token_count or 0
    return Usage(input_tokens=input_tokens, output_tokens=output, reasoning_tokens=reasoning)


class GeminiProvider:
    """Calls Gemini models and returns provider-neutral responses."""

    name: ProviderName = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        client: genai.Client | None = None,
    ) -> None:
        self.model = model
        # `client` can be injected so tests supply a fake and never touch the network.
        # The SDK's timeout is in milliseconds. retry_options is left unset, which
        # leaves SDK-level retries off: our retry layer owns that job.
        self._client = client or genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000))
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Make one Gemini call and normalize the result."""
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
        )
        if request.tools:
            config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=tool.parameters,
                        )
                        for tool in request.tools
                    ]
                )
            ]
            # Return tool requests to us instead of letting the SDK run anything itself
            config.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=True)
        if request.response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = request.response_schema
        if request.reasoning_effort is not None and _uses_thinking_level(self.model):
            config.thinking_config = types.ThinkingConfig(
                thinking_level=_THINKING_LEVELS[request.reasoning_effort]
            )

        # Time the call ourselves so providers are compared on the same clock
        started = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=to_gemini_contents(request), config=config
            )
        except genai_errors.APIError as exc:
            raise map_error(exc) from exc
        except Exception as exc:
            if _is_transport_error(exc):
                raise RetryableError(f"Network error: {exc}", provider="gemini") from exc
            raise
        latency_ms = (time.perf_counter() - started) * 1000

        candidates = response.candidates or []
        if not candidates or candidates[0].content is None:
            # Usually a safety block: no candidate, or one with no content
            reason = candidates[0].finish_reason if candidates else None
            block = response.prompt_feedback.block_reason if response.prompt_feedback else None
            raise InvalidResponseError(
                f"Gemini returned no content (finish_reason={reason}, block_reason={block})",
                provider="gemini",
            )
        content = candidates[0].content
        parts = content.parts or []

        # Join visible text only; thought summaries (part.thought) are not the answer
        text = "".join(part.text for part in parts if part.text and not part.thought) or None

        tool_calls = [
            ToolCall(
                # Gemini often sends no call id, so the gateway invents one
                id=part.function_call.id or f"gemini-{uuid.uuid4().hex[:12]}",
                name=part.function_call.name or "",
                arguments=dict(part.function_call.args or {}),
            )
            for part in parts
            if part.function_call is not None
        ]

        parsed: dict[str, Any] | None = None
        if request.response_schema is not None:
            try:
                parsed = json.loads(text or "")
            except json.JSONDecodeError as exc:
                raise InvalidResponseError(
                    f"Structured output was not valid JSON: {exc}", provider="gemini"
                ) from exc
            if not isinstance(parsed, dict):
                raise InvalidResponseError(
                    "Structured output was not a JSON object", provider="gemini"
                )

        return GenerateResponse(
            provider="gemini",
            model=self.model,
            message=Message(
                role="assistant",
                content=text,
                tool_calls=tool_calls,
                # Keep Gemini's original turn so the next request can echo it back
                provider_state=content,
            ),
            parsed=parsed,
            usage=_usage_from(response.usage_metadata),
            latency_ms=latency_ms,
        )
