"""Live smoke test: one real call through GeminiProvider (uses a little free-tier quota)."""

import asyncio
import os

from dotenv import load_dotenv

from llm_gateway.errors import GatewayError, RateLimitedError
from llm_gateway.models import GenerateRequest, Message
from llm_gateway.providers.gemini import GeminiProvider


async def main() -> None:
    # Read GEMINI_API_KEY and GEMINI_MODEL from the .env file
    load_dotenv(override=True)

    # Build the adapter exactly the way the gateway will in production
    provider = GeminiProvider(
        api_key=os.environ["GEMINI_API_KEY"],
        model=os.environ["GEMINI_MODEL"],
    )

    # Same prompt as the Groq smoke test, so the two providers are directly comparable.
    # reasoning_effort exercises the thinking_level mapping (Gemini 3 models only).
    request = GenerateRequest(
        messages=[Message.user("Name three uses for a paperclip.")],
        reasoning_effort="low",
        max_output_tokens=300,
    )

    try:
        response = await provider.generate(request)
    except RateLimitedError as exc:
        # A 429 still proves the plumbing works; show what the retry layer will see
        print(f"Rate limited. retry_after={exc.retry_after}")
        return
    except GatewayError as exc:
        # Any other mapped failure: show the gateway error type and the SDK message
        print(f"{type(exc).__name__}: {exc}")
        return

    # Normalized fields: text, token usage in the gateway's shape, and our own timing
    print(response.text)
    print(response.usage)
    print(f"total_tokens={response.usage.total_tokens}, latency={response.latency_ms:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
