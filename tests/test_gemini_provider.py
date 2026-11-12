"""Tests for the Gemini adapter. The Gemini SDK call is mocked: no network or API key needed."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from llm_gateway.errors import (
    AuthenticationError,
    InvalidRequestError,
    InvalidResponseError,
    RateLimitedError,
    RetryableError,
)
from llm_gateway.models import GenerateRequest, Message, ToolCall, ToolResult, ToolSpec
from llm_gateway.providers.gemini import GeminiProvider


def make_usage(
    prompt: int = 47,
    candidates: int = 79,
    thoughts: int | None = None,
    total: int | None = 126,
) -> SimpleNamespace:
    """Build fake usage metadata with the field names the Gemini SDK uses."""
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        tool_use_prompt_token_count=None,
        total_token_count=total,
    )


def make_response(
    content: types.Content | None,
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """Build a fake generate_content response around REAL SDK content objects."""
    candidates = [] if content is None else [SimpleNamespace(content=content, finish_reason=None)]
    return SimpleNamespace(
        candidates=candidates,
        usage_metadata=usage or make_usage(),
        prompt_feedback=None,
    )


def text_content(text: str) -> types.Content:
    """A model turn containing only text."""
    return types.Content(role="model", parts=[types.Part(text=text)])


def make_provider(generate: AsyncMock, model: str = "gemini-3.5-flash-lite") -> GeminiProvider:
    """Build a provider whose SDK client is a fake wrapping the given mock."""
    fake_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    return GeminiProvider(api_key="test", model=model, client=cast(genai.Client, fake_client))


def api_error(cls: type[genai_errors.APIError], code: int, message: str, **extra: Any) -> Any:
    """Build an SDK error from the JSON body shape Google sends."""
    status = {
        400: "INVALID_ARGUMENT",
        404: "NOT_FOUND",
        429: "RESOURCE_EXHAUSTED",
        503: "UNAVAILABLE",
    }
    body = {"error": {"code": code, "message": message, "status": status[code], **extra}}
    return cls(code, body)


async def test_structured_output_sends_json_schema_and_parses() -> None:
    generate = AsyncMock(
        return_value=make_response(text_content('{"category": "billing", "needs_human": true}'))
    )
    provider = make_provider(generate)
    schema = {"type": "object", "properties": {"category": {"type": "string"}}}

    response = await provider.generate(
        GenerateRequest(messages=[Message.user("extract")], response_schema=schema, system="sys")
    )

    # The JSON text was decoded into a dict
    assert response.parsed == {"category": "billing", "needs_human": True}
    # The request asked Gemini for JSON conforming to the schema
    config = generate.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema
    assert config.system_instruction == "sys"


async def test_reasoning_effort_maps_to_thinking_level_on_gemini_3() -> None:
    generate = AsyncMock(return_value=make_response(text_content("ok")))

    await make_provider(generate).generate(
        GenerateRequest(messages=[Message.user("hi")], reasoning_effort="low")
    )

    config = generate.call_args.kwargs["config"]
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW


async def test_reasoning_effort_is_ignored_for_gemini_2_models() -> None:
    generate = AsyncMock(return_value=make_response(text_content("ok")))

    await make_provider(generate, model="gemini-2.5-flash-lite").generate(
        GenerateRequest(messages=[Message.user("hi")], reasoning_effort="low")
    )

    # Gemini 2.x uses token budgets instead, and rejects a thinking level
    assert generate.call_args.kwargs["config"].thinking_config is None


async def test_usage_matches_your_notebook_numbers() -> None:
    # 47 prompt + 79 answer tokens + no thinking = 126, as in the earlier notebook run
    generate = AsyncMock(return_value=make_response(text_content("hi"), make_usage()))

    response = await make_provider(generate).generate(
        GenerateRequest(messages=[Message.user("hi")])
    )

    assert (response.usage.input_tokens, response.usage.output_tokens) == (47, 79)
    assert response.usage.reasoning_tokens == 0
    assert response.usage.total_tokens == 126


async def test_usage_trusts_provider_total_even_if_candidates_include_thoughts() -> None:
    # Pretend candidates_token_count (50) already includes the 20 thinking tokens
    usage = make_usage(prompt=10, candidates=50, thoughts=20, total=60)
    generate = AsyncMock(return_value=make_response(text_content("hi"), usage))

    response = await make_provider(generate).generate(
        GenerateRequest(messages=[Message.user("hi")])
    )

    # Visible output is derived from the total, so the sum still equals the quota charge
    assert response.usage.output_tokens == 30
    assert response.usage.reasoning_tokens == 20
    assert response.usage.total_tokens == 60


async def test_thought_parts_are_excluded_from_text() -> None:
    content = types.Content(
        role="model",
        parts=[types.Part(text="let me think...", thought=True), types.Part(text="the answer")],
    )
    generate = AsyncMock(return_value=make_response(content))

    response = await make_provider(generate).generate(
        GenerateRequest(messages=[Message.user("hi")])
    )

    assert response.text == "the answer"


async def test_function_calls_get_generated_ids_and_keep_provider_state() -> None:
    content = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="get_order_status", args={"order_id": "A"})
            ),
            types.Part(
                function_call=types.FunctionCall(name="get_order_status", args={"order_id": "B"})
            ),
        ],
    )
    generate = AsyncMock(return_value=make_response(content))

    response = await make_provider(generate).generate(
        GenerateRequest(
            messages=[Message.user("compare")],
            tools=[ToolSpec(name="get_order_status", description="d", parameters={})],
        )
    )

    # Both parallel calls come back, each with a generated id and dict arguments
    assert [call.arguments for call in response.tool_calls] == [
        {"order_id": "A"},
        {"order_id": "B"},
    ]
    assert all(call.id.startswith("gemini-") for call in response.tool_calls)
    assert response.tool_calls[0].id != response.tool_calls[1].id
    assert response.text is None
    # Gemini's original turn is kept for echoing back, but never serialized
    assert response.message.provider_state is content
    assert "provider_state" not in response.message.model_dump_json()
    # Tool loops need the SDK to hand calls back instead of executing them
    config = generate.call_args.kwargs["config"]
    assert config.automatic_function_calling.disable is True


async def test_history_echoes_provider_state_and_sends_results_by_name() -> None:
    original_turn = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(name="get_order_status", args={"order_id": "A"})
            )
        ],
    )
    generate = AsyncMock(return_value=make_response(text_content("done")))
    history = [
        Message.user("status?"),
        Message(role="assistant", tool_calls=[], provider_state=original_turn),
        Message(
            role="tool",
            tool_results=[
                ToolResult(call_id="x", name="get_order_status", content={"status": "shipped"})
            ],
        ),
    ]

    await make_provider(generate).generate(GenerateRequest(messages=history))

    sent = generate.call_args.kwargs["contents"]
    # Gemini's own turn goes back exactly as received (same object)
    assert sent[1] is original_turn
    # The result is matched to the call by function name
    response_part = sent[2].parts[0].function_response
    assert response_part.name == "get_order_status"
    assert response_part.response == {"result": {"status": "shipped"}}


async def test_assistant_turn_from_another_provider_is_rebuilt() -> None:
    generate = AsyncMock(return_value=make_response(text_content("done")))
    history = [
        Message.user("status?"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="call_1", name="get_order_status", arguments={"order_id": "A"})
            ],
        ),
    ]

    await make_provider(generate).generate(GenerateRequest(messages=history))

    # With no provider_state (e.g. the turn came from Groq), the call is rebuilt as a model turn
    rebuilt = generate.call_args.kwargs["contents"][1]
    assert rebuilt.role == "model"
    assert rebuilt.parts[0].function_call.args == {"order_id": "A"}


async def test_rate_limit_reads_retry_delay_from_error_body() -> None:
    error = api_error(
        genai_errors.ClientError,
        429,
        "quota exceeded",
        details=[{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "27s"}],
    )
    provider = make_provider(AsyncMock(side_effect=error))

    with pytest.raises(RateLimitedError) as caught:
        await provider.generate(GenerateRequest(messages=[Message.user("hi")]))

    assert caught.value.retry_after == 27.0


@pytest.mark.parametrize(
    ("error_cls", "code", "message", "expected"),
    [
        (
            genai_errors.ClientError,
            400,
            "API key not valid. Please pass a valid API key.",
            AuthenticationError,
        ),
        (genai_errors.ClientError, 404, "models/nope is not found", InvalidRequestError),
        (genai_errors.ServerError, 503, "The model is overloaded", RetryableError),
    ],
)
async def test_error_codes_map_to_gateway_errors(
    error_cls: type[genai_errors.APIError], code: int, message: str, expected: type[Exception]
) -> None:
    provider = make_provider(AsyncMock(side_effect=api_error(error_cls, code, message)))

    with pytest.raises(expected):
        await provider.generate(GenerateRequest(messages=[Message.user("hi")]))


async def test_network_errors_are_retryable() -> None:
    provider = make_provider(AsyncMock(side_effect=httpx.ConnectError("connection dropped")))

    with pytest.raises(RetryableError):
        await provider.generate(GenerateRequest(messages=[Message.user("hi")]))


async def test_response_without_candidates_raises_invalid_response() -> None:
    provider = make_provider(AsyncMock(return_value=make_response(None)))

    with pytest.raises(InvalidResponseError):
        await provider.generate(GenerateRequest(messages=[Message.user("hi")]))


async def test_invalid_json_in_structured_output_raises_invalid_response() -> None:
    provider = make_provider(AsyncMock(return_value=make_response(text_content("not json"))))

    with pytest.raises(InvalidResponseError):
        await provider.generate(
            GenerateRequest(messages=[Message.user("x")], response_schema={"type": "object"})
        )
