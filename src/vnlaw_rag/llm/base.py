"""Provider-neutral LLM interface used by the graph's `generate` node.

The graph only needs one capability: "given a system prompt and a user prompt, return an
object that validates against this Pydantic schema". Gemini implements it today
(`llm/gemini.py`); another provider (e.g. Claude) can be added without touching the graph.
Tests use a fake implementation, so CI never calls a paid or rate-limited API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredResult(Generic[T]):
    value: T
    model: str  # the model that actually answered (a fallback may have been used)
    attempts: int = 1


class LLMError(RuntimeError):
    """The LLM call failed after retries, or returned output that does not fit the schema."""


class Unavailable(LLMError):
    """A retryable failure (429, 5xx, timeout) persisted through all attempts: worth trying
    another model."""


class RateLimited(Unavailable):
    """Quota or rate limit still exceeded after all retries (HTTP 429)."""


class LLMClient(Protocol):
    def generate_structured(
        self, *, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]: ...
