"""Shared data types for the gateway.

Every provider adapter translates its SDK's format into these models, so the rest of
the codebase (retries, API layer, tests) never touches provider-specific shapes.
"""

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

# Names of the supported providers. A Literal makes a typo a type-checking error.
ProviderName = Literal["gemini", "groq"]

# Shared vocabulary for how much a reasoning model should think before answering.
# Each adapter maps this onto its own provider's setting.
ReasoningEffort = Literal["low", "medium", "high"]


class Usage(BaseModel):
    """Token accounting in one shape, however each provider reports it."""

    input_tokens: int = 0
    # Visible answer tokens only; hidden reasoning is tracked separately below.
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """All tokens that count against a provider's quota for this call."""
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

    def __add__(self, other: "Usage") -> "Usage":
        """Sum two usages, e.g. to total every round of a tool-calling loop."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class ToolSpec(BaseModel):
    """Describes a tool the model is allowed to call."""

    name: str
    description: str
    # JSON Schema for the tool's arguments (the same dict both providers accept).
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """A request from the model to run one tool."""

    # Groq supplies an id; Gemini does not, so the Gemini adapter generates one.
    id: str
    name: str
    # Already parsed into a dict (Groq sends a JSON string; the adapter decodes it).
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """The output of running a tool, sent back to the model."""

    # Groq matches results to calls by id, Gemini by name, so we carry both.
    call_id: str
    name: str
    content: dict[str, Any]


class Message(BaseModel):
    """One turn of conversation in the provider-neutral format."""

    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    # Set on assistant turns where the model asked to run tools.
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Set on "tool" turns; a list because a model may request several calls at once.
    tool_results: list[ToolResult] = Field(default_factory=list)
    # The provider's own version of this turn. Gemini requires its original turn
    # (including hidden metadata) to be echoed back, so the adapter stashes it here.
    # exclude=True keeps it out of JSON so it never leaks through the API.
    provider_state: Any = Field(default=None, exclude=True)

    @classmethod
    def user(cls, text: str) -> Self:
        """Shortcut for the most common message: plain text from the user."""
        return cls(role="user", content=text)


class GenerateRequest(BaseModel):
    """Everything needed to ask a model for a response."""

    # The conversation so far; at least one message is required.
    messages: list[Message] = Field(min_length=1)
    system: str | None = None
    # None means "use the gateway's default provider".
    provider: ProviderName | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    # JSON Schema the answer must follow (structured output). None means free text.
    response_schema: dict[str, Any] | None = None
    reasoning_effort: ReasoningEffort | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)

    @model_validator(mode="after")
    def _tools_and_schema_are_exclusive(self) -> Self:
        """Reject requests that combine tools with structured output.

        Groq cannot do both in one request, so the gateway forbids it everywhere.
        That keeps behavior identical across providers: callers use tools first,
        then make a separate structured-output call if they need one.
        """
        if self.tools and self.response_schema is not None:
            raise ValueError("Use either tools or response_schema in one request, not both")
        return self


class GenerateResponse(BaseModel):
    """What the gateway returns, identical in shape for every provider."""

    provider: ProviderName
    model: str
    # The assistant's turn. Append it to the history to continue a conversation.
    message: Message
    # The validated JSON object when response_schema was used, otherwise None.
    parsed: dict[str, Any] | None = None
    usage: Usage
    # Wall-clock time of the successful attempt, measured by the gateway so providers
    # compare fairly. Time spent on failed attempts and backoff waits is not included.
    latency_ms: float
    # How many tries it took (1 means the first try worked). Set by the retry layer.
    attempts: int = 1
    # Providers that failed before this one answered (empty unless the gateway fell back)
    failed_providers: list[ProviderName] = Field(default_factory=list)

    @property
    def text(self) -> str | None:
        """The model's text answer, or None if it asked for tool calls instead."""
        return self.message.content

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Tool calls the model requested; empty when it answered in text."""
        return self.message.tool_calls


class BatchResult(BaseModel):
    """The outcome of one request in a batch: either a response or an error description."""

    # Position of the request in the submitted list, so callers can match results up
    index: int
    response: GenerateResponse | None = None
    # Set instead of `response` when the request failed
    error_type: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        """True when the request succeeded."""
        return self.response is not None
