"""Live smoke test: one real call through GroqProvider (uses a little free-tier quota)."""

import asyncio
import os

from dotenv import load_dotenv

from llm_gateway.errors import GatewayError, RateLimitedError
from llm_gateway.models import GenerateRequest, Message
from llm_gateway.providers.groq import GroqProvider


async def main() -> None:
    # Read GROQ_API_KEY, GROQ_BASE_URL, and GROQ_MODEL from the .env file
    load_dotenv(override=True)

    # Build the adapter exactly the way the gateway will in production
    provider = GroqProvider(
        api_key=os.environ["GROQ_API_KEY"],
        model=os.environ["GROQ_MODEL"],
        base_url=os.environ["GROQ_BASE_URL"],
    )

    # A small request that also exercises the two settings we want to verify:
    # reasoning_effort (sent via extra_body) and max_output_tokens (max_completion_tokens)
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
