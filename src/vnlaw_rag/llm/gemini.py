"""Gemini API client (official `google-genai` SDK, Interactions API).

Checked on 2026-11-24 against the official docs (ai.google.dev, now served from
aistudio.google.com) and the installed SDK (google-genai 2.33.0):
- calls go through `client.interactions.create(...)`, structured output through
  `response_format={"type": "text", "mime_type": "application/json", "schema": ...}`;
- `stream=True` yields `step.start` / `step.delta` (text deltas) / `interaction.completed` /
  `error` events; only text of `model_output` steps is the answer;
- Gemini 4 and 3.6+ ignore `temperature`/`top_p`, so they are not sent;
- `store=False`: the request is not kept for later retrieval.

Latency policy (docs/decisions/008): every call has its own timeout; a timeout, 5xx or 429
switches to the next model at once (no waiting), later attempts back off briefly, and the
whole request has a time budget. When the budget or the attempts run out, `Unavailable` /
`RateLimited` is raised and the API shows a friendly Vietnamese message. The answering model
is always reported. The SDK's own retries are off, so this is the only retry policy.
The API key is passed explicitly and is never logged.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from pydantic import ValidationError

from vnlaw_rag.llm.base import (
    LLMError,
    OnEvent,
    OnText,
    RateLimited,
    StructuredResult,
    T,
    Unavailable,
)

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRYABLE_ERROR_NAMES = {"APIConnectionError", "APITimeoutError", "CallTimeout"}
MIN_CALL_SECONDS = 3.0  # do not start a call with less time than this left in the budget


class CallTimeout(TimeoutError):
    """A single call exceeded its own time limit (streaming calls are checked as they go)."""


class StreamError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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
        models: Sequence[str],
        thinking_level: str = "low",
        max_output_tokens: int = 2048,
        call_timeout_seconds: float = 25.0,
        deadline_seconds: float = 60.0,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not models:
            raise ValueError("at least one model is required")
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=int(call_timeout_seconds * 1000),  # milliseconds
                    retry_options=types.HttpRetryOptions(attempts=1),  # retries are ours
                ),
            )
        self._client = client
        self.models = list(models)
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.call_timeout_seconds = call_timeout_seconds
        self.deadline_seconds = deadline_seconds
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._clock = clock

    @property
    def model(self) -> str:
        return self.models[0]

    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        on_text: OnText | None = None,
        on_event: OnEvent | None = None,
    ) -> StructuredResult[T]:
        deadline = self._clock() + self.deadline_seconds
        model_index, last_exc = 0, None
        for attempt in range(1, self.max_attempts + 1):
            remaining = deadline - self._clock()
            if remaining < MIN_CALL_SECONDS:
                break
            model = self.models[model_index]
            timeout = min(self.call_timeout_seconds, remaining)
            try:
                text = self._call(model, system, prompt, schema, timeout, on_text)
            except Exception as exc:  # SDK error classes differ by API path; see status_of
                if not is_retryable(exc):
                    raise LLMError(
                        f"{model}: {type(exc).__name__} (status {status_of(exc)})"
                    ) from exc
                last_exc = exc
                log.warning("%s: %s (status %s)", model, type(exc).__name__, status_of(exc))
                if model_index + 1 < len(self.models):
                    model_index += 1  # switch at once: another model has its own load/quota
                    if on_event:
                        on_event(
                            {"stage": "fallback", "from": model, "to": self.models[model_index]}
                        )
                    continue
                delay = min(self.backoff_seconds * 2 ** (attempt - 1), deadline - self._clock())
                if delay > 0:
                    self._sleep(delay)
                continue
            try:
                return StructuredResult(schema.model_validate_json(text), model, attempt)
            except ValidationError as exc:
                raise LLMError(f"{model}: output does not match {schema.__name__}") from exc

        message = f"no answer within {self.deadline_seconds:.0f} s / {self.max_attempts} attempts"
        if last_exc is not None and status_of(last_exc) == 429:
            raise RateLimited(message) from last_exc
        raise Unavailable(message) from last_exc

    # -- one call ----------------------------------------------------------------------
    def _request(self, model: str, system: str, prompt: str, schema: type[T]) -> dict:
        return {
            "model": model,
            "system_instruction": system,
            "input": prompt,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema.model_json_schema(),
            },
            "generation_config": {
                "thinking_level": self.thinking_level,
                "max_output_tokens": self.max_output_tokens,
            },
            "store": False,
        }

    def _call(
        self,
        model: str,
        system: str,
        prompt: str,
        schema: type[T],
        timeout: float,
        on_text: OnText | None,
    ) -> str:
        """One call with a hard time limit. The HTTP timeout only bounds each read, and closing
        a stream from another thread does not always interrupt a blocked socket read (the
        2026-11-24 benchmark saw a 25 s call last 71 s). So the call runs in a daemon thread
        and the caller waits at most `timeout`; on expiry the stream is closed, late tokens
        are dropped, and the abandoned thread ends on its own."""
        request = self._request(model, system, prompt, schema)
        cancelled = threading.Event()
        holder: dict[str, Any] = {}

        def guarded(text: str) -> None:
            if not cancelled.is_set() and on_text is not None:
                on_text(text)

        def work() -> None:
            try:
                if on_text is None:
                    interaction = self._client.interactions.create(**request, timeout=timeout)
                    holder["text"] = getattr(interaction, "output_text", None) or ""
                    return
                events = self._client.interactions.create(**request, stream=True, timeout=timeout)
                holder["events"] = events
                holder["text"] = self._consume_stream(events, guarded)
            except BaseException as exc:  # handed to the caller below
                holder["error"] = exc

        worker = threading.Thread(target=work, name=f"gemini-{model}", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            cancelled.set()
            if "events" in holder:
                _close(holder["events"])
            raise CallTimeout(f"{model}: no complete answer within {timeout:.0f} s")
        if "error" in holder:
            raise holder["error"]
        return holder["text"]

    def _consume_stream(self, events: Iterable[Any], on_text: OnText) -> str:
        """Collect the JSON text of model_output steps and forward the growing `answer` field
        to `on_text`. The time limit is enforced by the caller (`_call`)."""
        output_steps: set[int] = set()
        buffer = ""
        streamed = AnswerStreamer(on_text)
        try:
            for event in events:
                kind = getattr(event, "event_type", None)
                if kind == "error":
                    error = getattr(event, "error", None)
                    raise StreamError(str(error), status_of(error) if error else None)
                if kind == "step.start":
                    step = getattr(event, "step", None)
                    if getattr(step, "type", None) == "model_output":
                        output_steps.add(event.index)
                elif kind == "step.delta" and event.index in output_steps:
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) == "text":
                        buffer += delta.text
                        streamed.feed(buffer)
        finally:
            _close(events)
        return buffer


def _close(events: Any) -> None:
    close = getattr(events, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # closing a broken stream must not hide the real error
            log.debug("closing the stream failed", exc_info=True)


class AnswerStreamer:
    """Turns a growing JSON document into the growing text of its "answer" field."""

    def __init__(self, on_text: OnText) -> None:
        self.on_text = on_text
        self.sent = 0

    def feed(self, partial_json: str) -> None:
        text = answer_prefix(partial_json)
        if len(text) > self.sent:
            self.on_text(text[self.sent :])
            self.sent = len(text)


def answer_prefix(partial_json: str) -> str:
    """The decoded part of the "answer" string value received so far ("" if not started)."""
    key = partial_json.find('"answer"')
    if key < 0:
        return ""
    colon = partial_json.find(":", key + 8)
    quote = partial_json.find('"', colon + 1) if colon >= 0 else -1
    if quote < 0:
        return ""
    i = quote + 1
    out: list[str] = []
    while i < len(partial_json):
        ch = partial_json[i]
        if ch == '"':
            break
        if ch == "\\":
            escape = (
                partial_json[i : i + 6]
                if partial_json[i + 1 : i + 2] == "u"
                else (partial_json[i : i + 2])
            )
            if len(escape) < (6 if escape[1:2] == "u" else 2):
                break  # escape sequence not complete yet
            try:
                out.append(json.loads(f'"{escape}"'))
            except json.JSONDecodeError:
                break
            i += len(escape)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def client_from_settings(settings, *, for_eval: bool = False) -> GeminiClient:
    """The app client (main + fallback model, short time budget) or, `for_eval`, a client that
    only uses the main model and waits longer, so eval numbers belong to one model."""
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    if not key:
        raise LLMError("GEMINI_API_KEY is not set (environment or .env)")
    models = [settings.llm_model]
    if settings.llm_fallback_model and not for_eval:
        models.append(settings.llm_fallback_model)
    return GeminiClient(
        api_key=key,
        models=models,
        thinking_level=settings.llm_thinking_level,
        max_output_tokens=settings.llm_max_output_tokens,
        call_timeout_seconds=settings.llm_timeout_seconds,
        deadline_seconds=settings.llm_eval_deadline_seconds
        if for_eval
        else settings.llm_deadline_seconds,
        max_attempts=settings.llm_max_attempts,
        backoff_seconds=settings.llm_backoff_seconds,
    )
