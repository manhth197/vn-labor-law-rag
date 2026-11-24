"""Gemini API client (official `google-genai` SDK, Interactions API).

Checked on 2026-11-24 against the official docs (ai.google.dev, now served from
aistudio.google.com) and the installed SDK (google-genai 2.33.0):
- calls go through `client.interactions.create(...)`, structured output through
  `response_format={"type": "text", "mime_type": "application/json", "schema": ...}`;
- Gemini 4 and 3.6+ ignore `temperature`/`top_p`, so they are not sent;
- `store=False`: the request is not kept for later retrieval;
- the free tier answers HTTP 429 when a per-minute or per-day quota is exceeded.

Retries: the SDK's own retries are turned off and this client retries 429, 5xx, timeouts
and connection errors with exponential backoff, so there is one retry policy to reason about.
After the last attempt a 429 becomes `RateLimited` and a persistent 5xx/timeout `Unavailable`;
then, if configured, a fallback model is tried once (it has its own free-tier quota and load;
the smoke test on 2026-11-24 met three 503s in a row on gemini-4-hydrogen). The answering
model is always reported.
The API key is passed explicitly and is never logged.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from vnlaw_rag.llm.base import LLMError, RateLimited, StructuredResult, T, Unavailable

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRYABLE_ERROR_NAMES = {"APIConnectionError", "APITimeoutError"}
MAX_BACKOFF_SECONDS = 60.0


def status_of(exc: BaseException) -> int | None:
    """HTTP status of an SDK error. The Interactions path raises errors with `status_code`,
    the older `google.genai.errors.APIError` uses `code`; read either."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_retryable(exc: BaseException) -> bool:
    return status_of(exc) in RETRYABLE_STATUS or type(exc).__name__ in RETRYABLE_ERROR_NAMES


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        thinking_level: str = "low",
        max_output_tokens: int = 2048,
        timeout_seconds: float = 60.0,
        max_attempts: int = 4,
        backoff_seconds: float = 5.0,
        fallback_model: str | None = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=int(timeout_seconds * 1000),  # milliseconds
                    retry_options=types.HttpRetryOptions(attempts=1),  # retries are ours
                ),
            )
        self._client = client
        self.model = model
        self.fallback_model = fallback_model
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep

    def generate_structured(
        self, *, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]:
        try:
            return self._generate(self.model, system, prompt, schema)
        except Unavailable as exc:  # rate limited or overloaded after all retries
            if not self.fallback_model:
                raise
            log.warning("%s; trying fallback %s", exc, self.fallback_model)
            return self._generate(self.fallback_model, system, prompt, schema)

    def _generate(
        self, model: str, system: str, prompt: str, schema: type[T]
    ) -> StructuredResult[T]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                interaction = self._client.interactions.create(
                    model=model,
                    system_instruction=system,
                    input=prompt,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": schema.model_json_schema(),
                    },
                    generation_config={
                        "thinking_level": self.thinking_level,
                        "max_output_tokens": self.max_output_tokens,
                    },
                    store=False,
                )
            except Exception as exc:  # SDK error classes differ by API path; see status_of
                if not is_retryable(exc) or attempt == self.max_attempts:
                    if status_of(exc) == 429:
                        raise RateLimited(
                            f"{model}: rate limited after {attempt} attempts"
                        ) from exc
                    if is_retryable(exc):
                        raise Unavailable(
                            f"{model}: {type(exc).__name__} (status {status_of(exc)}) "
                            f"after {attempt} attempts"
                        ) from exc
                    raise LLMError(
                        f"{model}: {type(exc).__name__} (status {status_of(exc)})"
                    ) from exc
                delay = min(self.backoff_seconds * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
                log.warning(
                    "%s: %s (status %s), retry %d/%d in %.0f s",
                    model,
                    type(exc).__name__,
                    status_of(exc),
                    attempt,
                    self.max_attempts - 1,
                    delay,
                )
                self._sleep(delay)
                continue
            text = getattr(interaction, "output_text", None) or ""
            try:
                return StructuredResult(schema.model_validate_json(text), model, attempt)
            except ValidationError as exc:
                raise LLMError(f"{model}: output does not match {schema.__name__}") from exc
        raise LLMError("unreachable")  # pragma: no cover


def client_from_settings(settings, use_fallback: bool = True) -> GeminiClient:
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    if not key:
        raise LLMError("GEMINI_API_KEY is not set (environment or .env)")
    return GeminiClient(
        api_key=key,
        model=settings.llm_model,
        thinking_level=settings.llm_thinking_level,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        backoff_seconds=settings.llm_backoff_seconds,
        fallback_model=(settings.llm_fallback_model or None) if use_fallback else None,
    )
