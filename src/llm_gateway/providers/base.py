"""The interface every provider adapter implements.

Adapters differ wildly inside (different SDKs, message formats, error types), but the
rest of the gateway only ever sees this one contract.
"""

from typing import Protocol

from llm_gateway.models import GenerateRequest, GenerateResponse, ProviderName


class Provider(Protocol):
    """Anything that can turn a GenerateRequest into a GenerateResponse."""

    # Which provider this is, used for routing, logging, and fallback
    name: ProviderName
    # The model ID this adapter calls (comes from the environment, never hardcoded)
    model: str

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Make exactly one provider call.

        Contract for implementers:
        - Do NOT retry here. The retry layer wraps this method, so retrying inside
          would multiply attempts and hide failures.
        - Raise only GatewayError subclasses (see errors.py), never raw SDK exceptions.
        - Fill in `latency_ms` by timing the call yourself.
        """
        ...
