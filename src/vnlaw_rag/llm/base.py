"""Provider-neutral LLM interface used by the graph's `generate` node.

The graph only needs one capability: "given a system prompt and a user prompt, return an
object that validates against this Pydantic schema", optionally streaming the answer text as
it is produced. Gemini implements it today (`llm/gemini.py`); another provider (e.g. Claude)
can be added without touching the graph. Tests use a fake implementation, so CI never calls a
rate-limited API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

OnText = Callable[[str], None]  # receives each new piece of the answer text
OnEvent = Callable[[dict[str, Any]], None]  # receives progress events such as a model switch


@dataclass
class StructuredResult(Generic[T]):
    value: T
    model: str  # the model that actually answered (a fallback may have been used)
    attempts: int = 1


class LLMError(RuntimeError):
    """The LLM call failed, or returned output that does not fit the schema."""


class Unavailable(LLMError):
    """Retryable failures (timeout, 5xx, 429) used up the attempts or the time budget."""


class RateLimited(Unavailable):
    """The last failure was a rate or quota limit (HTTP 429)."""


class LLMClient(Protocol):
    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        on_text: OnText | None = None,
        on_event: OnEvent | None = None,
    ) -> StructuredResult[T]: ...
