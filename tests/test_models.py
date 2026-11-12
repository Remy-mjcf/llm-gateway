import pytest
from pydantic import ValidationError

from llm_gateway.models import GenerateRequest, Message, ToolSpec, Usage


def test_usage_total_and_addition() -> None:
    # Adding two usages must sum every field, including reasoning tokens
    combined = Usage(input_tokens=1, output_tokens=2, reasoning_tokens=3) + Usage(input_tokens=10)
    assert combined.total_tokens == 16


def test_request_rejects_tools_with_schema() -> None:
    # Groq cannot combine tools with structured output, so the gateway forbids it everywhere
    with pytest.raises(ValidationError):
        GenerateRequest(
            messages=[Message.user("hi")],
            tools=[ToolSpec(name="t", description="d", parameters={})],
            response_schema={"type": "object"},
        )


def test_request_requires_at_least_one_message() -> None:
    # An empty conversation is never valid, so fail at construction, not at the provider
    with pytest.raises(ValidationError):
        GenerateRequest(messages=[])