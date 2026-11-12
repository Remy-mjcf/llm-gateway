"""Live smoke test for the whole Gateway (uses roughly 6 requests of free-tier quota).

Run it normally:                  python scripts/smoke_gateway.py
Run it with a broken default key: python scripts/smoke_gateway.py --break-default
The second form shows fallback working: the default provider fails and the other answers.
"""

import argparse
import asyncio
import logging
from dataclasses import replace

from dotenv import load_dotenv

from llm_gateway import Gateway, GenerateRequest, Message, Settings


async def main(break_default: bool) -> None:
    # Read keys, models, and limits from the .env file
    load_dotenv(override=True)

    # Show the gateway's own log lines (retries, fallbacks, per-call summaries)
    logging.basicConfig(level=logging.WARNING, format="  log: %(message)s")
    logging.getLogger("llm_gateway").setLevel(logging.INFO)

    settings = Settings.from_env()
    if break_default:
        # Sabotage the default provider's key on purpose to trigger a fallback
        broken = replace(settings.providers[settings.default_provider], api_key="invalid-key")
        settings = replace(
            settings, providers={**settings.providers, settings.default_provider: broken}
        )
        print(f"(default provider {settings.default_provider!r} now has an invalid key)\n")

    gateway = Gateway.from_settings(settings)

    # 1. One request through default routing (and fallback, if the default fails)
    print("== single request ==")
    response = await gateway.generate(
        GenerateRequest(
            messages=[Message.user("Name three uses for a paperclip, briefly.")],
            reasoning_effort="low",
            max_output_tokens=300,
        )
    )
    print(response.text)
    print(
        f"-> answered by {response.provider}, attempts={response.attempts}, "
        f"failed_providers={response.failed_providers}\n"
    )

    # 2. A small batch, run concurrently under the gateway's concurrency and rate limits
    print("== batch of 4 ==")
    topics = ["a spoon", "a rubber band", "a coffee mug", "a shoelace"]
    batch = await gateway.batch(
        [
            GenerateRequest(
                messages=[Message.user(f"One creative use for {topic}, in one sentence.")],
                reasoning_effort="low",
                max_output_tokens=200,
            )
            for topic in topics
        ]
    )
    for topic, result in zip(topics, batch, strict=True):
        if result.response is not None:
            r = result.response
            print(
                f"{topic:14} ok   via {r.provider:6} attempts={r.attempts} "
                f"{r.latency_ms:6.0f} ms  {r.usage.total_tokens:4} tokens"
            )
        else:
            print(f"{topic:14} FAIL {result.error_type}: {result.error_message}")

    # 3. Totals the gateway tracked on its own
    print("\n== stats ==")
    for name, stats in gateway.stats.items():
        print(
            f"{name:6} calls={stats.calls} failures={stats.failures} "
            f"tokens={stats.usage.total_tokens}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--break-default", action="store_true", help="invalidate the default key")
    asyncio.run(main(parser.parse_args().break_default))
