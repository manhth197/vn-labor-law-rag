"""GeminiClient with a fake SDK client: request shape, retries, fallback, parsing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vnlaw_rag.graph.prompts import GeneratedAnswer
from vnlaw_rag.llm.base import LLMError, RateLimited
from vnlaw_rag.llm.gemini import GeminiClient, is_retryable, status_of

GOOD = (
    '{"answer": "Không quá 60 ngày [BLLD2019-D25].", "citations": ["BLLD2019-D25"], '
    '"sufficient": true}'
)


class FakeStatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeInteractions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # per call: an exception to raise or a text to return
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(output_text=outcome)


def make(outcomes, **kw):
    interactions = FakeInteractions(outcomes)
    sleeps = []
    client = GeminiClient(
        api_key="unused", model="main-model", client=SimpleNamespace(interactions=interactions),
        sleep=sleeps.append, backoff_seconds=1.0, max_attempts=3, **kw,
    )  # fmt: skip
    return client, interactions, sleeps


def test_request_uses_structured_output_and_no_storage():
    client, interactions, _ = make([GOOD])
    result = client.generate_structured(system="sys", prompt="q", schema=GeneratedAnswer)
    assert result.value.citations == ["BLLD2019-D25"]
    assert (result.model, result.attempts) == ("main-model", 1)
    (call,) = interactions.calls
    assert call["model"] == "main-model"
    assert call["system_instruction"] == "sys" and call["input"] == "q"
    assert call["store"] is False
    assert call["response_format"]["mime_type"] == "application/json"
    assert call["response_format"]["schema"]["required"] == ["answer", "citations", "sufficient"]
    assert "temperature" not in call["generation_config"]


def test_retries_429_and_5xx_with_exponential_backoff():
    client, interactions, sleeps = make([FakeStatusError(429), FakeStatusError(503), GOOD])
    result = client.generate_structured(system="s", prompt="q", schema=GeneratedAnswer)
    assert result.attempts == 3
    assert sleeps == [1.0, 2.0]


def test_rate_limit_after_all_attempts_then_fallback_model():
    client, interactions, _ = make(
        [FakeStatusError(429)] * 3 + [GOOD], fallback_model="fallback-model"
    )
    result = client.generate_structured(system="s", prompt="q", schema=GeneratedAnswer)
    assert result.model == "fallback-model"
    assert [c["model"] for c in interactions.calls] == ["main-model"] * 3 + ["fallback-model"]


def test_rate_limit_without_fallback_raises():
    client, _, _ = make([FakeStatusError(429)] * 3)
    with pytest.raises(RateLimited):
        client.generate_structured(system="s", prompt="q", schema=GeneratedAnswer)


def test_client_errors_are_not_retried():
    client, interactions, sleeps = make([FakeStatusError(400)])
    with pytest.raises(LLMError, match="status 400"):
        client.generate_structured(system="s", prompt="q", schema=GeneratedAnswer)
    assert len(interactions.calls) == 1 and sleeps == []


def test_output_that_does_not_fit_the_schema_is_an_error():
    client, _, _ = make(['{"answer": "x"}'])
    with pytest.raises(LLMError, match="does not match"):
        client.generate_structured(system="s", prompt="q", schema=GeneratedAnswer)


def test_error_classification_reads_either_attribute():
    assert status_of(FakeStatusError(429)) == 429
    assert status_of(SimpleNamespace(code=503)) == 503
    assert is_retryable(type("APITimeoutError", (Exception,), {})())
    assert not is_retryable(FakeStatusError(404))


def test_overload_after_all_attempts_also_falls_back():
    client, interactions, _ = make([FakeStatusError(503)] * 3 + [GOOD], fallback_model="fb")
    result = client.generate_structured(system="s", prompt="q", schema=GeneratedAnswer)
    assert result.model == "fb"
    assert len(interactions.calls) == 4
