"""Tests for the Groq adapter. The Groq SDK call is mocked, so no network or API key is needed."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from llm_gateway.errors import InvalidResponseError, RateLimitedError
from llm_gateway.models import GenerateRequest, Message, ToolCall, ToolResult, ToolSpec
from llm_gateway.providers.groq import GroqProvider, make_strict


class Ticket(BaseModel):
    """Schema used by the structured-output tests (same shape as the notebook)."""

    category: str
    needs_human: bool
    order_id: str | None


def make_completion(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    usage: tuple[int, int, int] = (307, 240, 187),
) -> SimpleNamespace:
    """Build a fake chat completion shaped like the OpenAI SDK's response object."""
    prompt, completion, reasoning = usage
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
    )


def make_provider(create: AsyncMock) -> GroqProvider:
    """Build a provider whose SDK client is a fake wrapping the given mock."""
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return GroqProvider(api_key="test", model="test-model", client=cast(AsyncOpenAI, fake_client))


async def test_structured_output_is_parsed_and_usage_normalized() -> None:
    create = AsyncMock(
        return_value=make_completion(
            content='{"category": "billing", "needs_human": true, "order_id": "A-10492"}'
        )
    )
    provider = make_provider(create)

    response = await provider.generate(
        GenerateRequest(
            messages=[Message.user("extract")], response_schema=Ticket.model_json_schema()
        )
    )

    # The JSON string was decoded into a dict
    assert response.parsed == {"category": "billing", "needs_human": True, "order_id": "A-10492"}
    # 240 completion tokens minus 187 reasoning = 53 visible tokens
    assert response.usage.input_tokens == 307
    assert response.usage.output_tokens == 53
    assert response.usage.reasoning_tokens == 187
    # The schema sent to Groq was made strict (all keys required, no extras)
    sent = create.call_args.kwargs["response_format"]["json_schema"]
    assert sent["strict"] is True
    assert sent["schema"]["additionalProperties"] is False


async def test_tool_call_arguments_are_parsed_from_json_string() -> None:
    raw_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="get_order_status", arguments='{"order_id": "A-10492"}'),
    )
    provider = make_provider(AsyncMock(return_value=make_completion(tool_calls=[raw_call])))

    response = await provider.generate(
        GenerateRequest(
            messages=[Message.user("status?")],
            tools=[ToolSpec(name="get_order_status", description="d", parameters={})],
        )
    )

    # Groq sends arguments as a string; the gateway exposes them as a dict
    assert response.tool_calls == [
        ToolCall(id="call_1", name="get_order_status", arguments={"order_id": "A-10492"})
    ]
    assert response.text is None


async def test_tool_results_are_sent_back_matched_by_call_id() -> None:
    create = AsyncMock(return_value=make_completion(content="done"))
    provider = make_provider(create)
    history = [
        Message.user("status?"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="call_1", name="get_order_status", arguments={"order_id": "A"})
            ],
        ),
        Message(
            role="tool",
            tool_results=[
                ToolResult(call_id="call_1", name="get_order_status", content={"status": "shipped"})
            ],
        ),
    ]

    await provider.generate(GenerateRequest(messages=history))

    sent = create.call_args.kwargs["messages"]
    # The assistant turn carries the call, with arguments re-encoded as a JSON string
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == '{"order_id": "A"}'
    # The tool message points back at the call it answers
    assert sent[2] == {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "shipped"}'}


async def test_rate_limit_maps_to_retryable_error_with_retry_after() -> None:
    response = httpx.Response(
        429,
        headers={"retry-after": "2"},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    create = AsyncMock(side_effect=openai.RateLimitError("slow down", response=response, body=None))
    provider = make_provider(create)

    with pytest.raises(RateLimitedError) as caught:
        await provider.generate(GenerateRequest(messages=[Message.user("hi")]))

    # The retry layer will read this to know how long to wait
    assert caught.value.retry_after == 2.0


async def test_invalid_json_in_structured_output_raises_invalid_response() -> None:
    provider = make_provider(AsyncMock(return_value=make_completion(content="not json")))

    with pytest.raises(InvalidResponseError):
        await provider.generate(
            GenerateRequest(messages=[Message.user("x")], response_schema={"type": "object"})
        )


def test_make_strict_fixes_nested_objects() -> None:
    schema = {
        "type": "object",
        "properties": {"inner": {"type": "object", "properties": {"a": {"type": "string"}}}},
    }

    strict = make_strict(schema)

    # Both the outer and the nested object must forbid extras and require every key
    assert strict["additionalProperties"] is False
    assert strict["properties"]["inner"]["required"] == ["a"]
    assert strict["properties"]["inner"]["additionalProperties"] is False
