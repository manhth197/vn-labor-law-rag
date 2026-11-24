"""GeminiClient with a fake SDK client: request shape, fallback, time budget, streaming."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vnlaw_rag.graph.prompts import GeneratedAnswer
from vnlaw_rag.llm.base import LLMError, RateLimited, Unavailable
from vnlaw_rag.llm.gemini import GeminiClient, answer_prefix, is_retryable, status_of

GOOD = (
    '{"answer": "Không quá 60 ngày [BLLD2019-D25].", "citations": ["BLLD2019-D25"], '
    '"sufficient": true}'
)


class FakeStatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeInteractions:
    """Per call: an exception to raise, a text to return, a list of stream events, or a
    callable(clock) for timing scenarios."""

    def __init__(self, outcomes, clock):
        self.outcomes = list(outcomes)
        self.calls = []
        self.clock = clock

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if callable(outcome) and not isinstance(outcome, type):
            outcome = outcome(self.clock)
        if isinstance(outcome, Exception):
            raise outcome
        if kwargs.get("stream"):
            return outcome if hasattr(outcome, "close") else iter(outcome)
        return SimpleNamespace(output_text=outcome)


def make(outcomes, models=("main", "fb"), **kw):
    clock = FakeClock()
    interactions = FakeInteractions(outcomes, clock)
    params = dict(call_timeout_seconds=25, deadline_seconds=60, max_attempts=3, backoff_seconds=2)
    params.update(kw)
    client = GeminiClient(
        api_key="unused", models=list(models), client=SimpleNamespace(interactions=interactions),
        sleep=clock.sleep, clock=clock, **params,
    )  # fmt: skip
    return client, interactions, clock


def ask(client, **kw):
    return client.generate_structured(system="sys", prompt="q", schema=GeneratedAnswer, **kw)


def test_request_uses_structured_output_timeout_and_no_storage():
    client, interactions, _ = make([GOOD])
    result = ask(client)
    assert result.value.citations == ["BLLD2019-D25"]
    assert (result.model, result.attempts) == ("main", 1)
    (call,) = interactions.calls
    assert call["model"] == "main" and call["timeout"] == 25
    assert call["system_instruction"] == "sys" and call["input"] == "q"
    assert call["store"] is False
    assert call["response_format"]["mime_type"] == "application/json"
    assert call["response_format"]["schema"]["required"] == ["answer", "citations", "sufficient"]
    assert "temperature" not in call["generation_config"]


@pytest.mark.parametrize("failure", [FakeStatusError(503), FakeStatusError(429)])
def test_overload_or_rate_limit_switches_model_at_once(failure):
    events = []
    client, interactions, clock = make([failure, GOOD])
    result = ask(client, on_event=events.append)
    assert result.model == "fb" and result.attempts == 2
    assert [c["model"] for c in interactions.calls] == ["main", "fb"]
    assert clock.now == 0  # no waiting before the switch
    assert events == [{"stage": "fallback", "from": "main", "to": "fb"}]


def test_models_take_turns_without_waiting():
    client, interactions, clock = make([FakeStatusError(503)] * 2 + [GOOD])
    result = ask(client)
    assert result.model == "main" and result.attempts == 3
    assert [c["model"] for c in interactions.calls] == ["main", "fb", "main"]
    assert clock.now == 0


def test_single_model_backs_off_between_attempts():
    client, interactions, clock = make([FakeStatusError(503)] * 2 + [GOOD], models=("main",))
    result = ask(client)
    assert result.attempts == 3
    assert clock.now == 2 + 4  # 2 s, then 4 s


def test_stream_error_without_status_is_retried_but_4xx_is_not():
    from vnlaw_rag.llm.gemini import StreamError

    assert is_retryable(StreamError("busy"))
    assert not is_retryable(StreamError("bad request", 400))


def test_time_budget_exhausted_raises_unavailable():
    def slow_timeout(clock):
        clock.now += 25  # the call used its whole timeout, then the SDK timed out
        return type("APITimeoutError", (Exception,), {})()

    client, interactions, _ = make([slow_timeout, slow_timeout, slow_timeout], deadline_seconds=60)
    with pytest.raises(Unavailable):
        ask(client)
    # 25 s + 25 s used; the third call (back on the main model) gets the remaining 10 s
    assert [c["timeout"] for c in interactions.calls] == [25, 25, 10]


def test_rate_limit_on_every_model_raises_rate_limited():
    client, _, _ = make([FakeStatusError(429)] * 3)
    with pytest.raises(RateLimited):
        ask(client)


def test_client_errors_are_not_retried():
    client, interactions, clock = make([FakeStatusError(400)])
    with pytest.raises(LLMError, match="status 400"):
        ask(client)
    assert len(interactions.calls) == 1 and clock.now == 0


def test_output_that_does_not_fit_the_schema_is_an_error():
    client, _, _ = make(['{"answer": "x"}'])
    with pytest.raises(LLMError, match="does not match"):
        ask(client)


def stream_events(json_text, pieces=4):
    size = max(1, len(json_text) // pieces)
    events = [
        SimpleNamespace(event_type="interaction.created"),
        SimpleNamespace(event_type="step.start", index=0, step=SimpleNamespace(type="thought")),
        SimpleNamespace(event_type="step.delta", index=0,
                        delta=SimpleNamespace(type="text", text="(thinking)")),
        SimpleNamespace(event_type="step.start", index=1,
                        step=SimpleNamespace(type="model_output")),
    ]  # fmt: skip
    for i in range(0, len(json_text), size):
        events.append(
            SimpleNamespace(event_type="step.delta", index=1,
                            delta=SimpleNamespace(type="text", text=json_text[i : i + size]))
        )  # fmt: skip
    events.append(SimpleNamespace(event_type="interaction.completed"))
    return events


def test_streaming_forwards_only_the_answer_text_of_model_output():
    pieces = []
    client, interactions, _ = make([stream_events(GOOD, pieces=9)])
    result = ask(client, on_text=pieces.append)
    assert interactions.calls[0]["stream"] is True
    assert "".join(pieces) == "Không quá 60 ngày [BLLD2019-D25]."
    assert len(pieces) > 1
    assert result.value.sufficient is True


def test_stream_error_event_falls_back():
    error = SimpleNamespace(event_type="error", error=SimpleNamespace(code=503, message="busy"))
    client, interactions, _ = make([[error], stream_events(GOOD)])
    result = ask(client, on_text=lambda _: None)
    assert result.model == "fb"


def test_answer_prefix_decodes_partial_json():
    assert answer_prefix('{"answ') == ""
    assert answer_prefix('{"answer": "Kh') == "Kh"
    assert answer_prefix('{"answer": "a\\nb') == "a\nb"
    assert answer_prefix('{"answer": "a\\') == "a"  # incomplete escape waits
    assert answer_prefix('{"answer": "\\u00e0x') == "àx"
    assert answer_prefix('{"answer": "\\u00e') == ""
    assert answer_prefix('{"answer": "done", "citations": [') == "done"


def test_error_classification_reads_either_attribute():
    assert status_of(FakeStatusError(429)) == 429
    assert status_of(SimpleNamespace(code=503)) == 503
    assert is_retryable(type("APITimeoutError", (Exception,), {})())
    assert not is_retryable(FakeStatusError(404))


class StallingStream:
    """A stream that sends one event and then hangs until it is closed."""

    def __init__(self):
        import threading

        self.closed = threading.Event()

    def __iter__(self):
        yield SimpleNamespace(event_type="interaction.created")
        self.closed.wait(5)  # a real stalled connection; close() releases it
        if self.closed.is_set():
            raise ConnectionError("stream closed")

    def close(self):
        self.closed.set()


def test_watchdog_closes_a_stalled_stream_and_falls_back():
    import time

    stalled = StallingStream()
    client = GeminiClient(
        api_key="unused", models=["main", "fb"], call_timeout_seconds=0.3,
        deadline_seconds=10, client=SimpleNamespace(
            interactions=FakeInteractions([lambda _: stalled, stream_events(GOOD)], FakeClock())
        ),
    )  # fmt: skip
    started = time.monotonic()
    result = client.generate_structured(
        system="s", prompt="q", schema=GeneratedAnswer, on_text=lambda _: None
    )
    assert result.model == "fb"
    assert stalled.closed.is_set()
    assert time.monotonic() - started < 3  # not the 5 s the stalled stream would take


def test_blocking_call_is_cut_at_its_time_limit():
    import threading
    import time

    release = threading.Event()

    def hang(_clock):
        release.wait(5)  # a request that never answers in time
        return GOOD

    client = GeminiClient(
        api_key="unused", models=["main", "fb"], call_timeout_seconds=0.3, deadline_seconds=10,
        client=SimpleNamespace(interactions=FakeInteractions([hang, GOOD], FakeClock())),
    )  # fmt: skip
    started = time.monotonic()
    result = client.generate_structured(system="s", prompt="q", schema=GeneratedAnswer)
    release.set()
    assert result.model == "fb"
    assert time.monotonic() - started < 2


def test_stream_error_text_codes_map_to_http_status():
    from vnlaw_rag.llm.gemini import stream_error_status

    assert stream_error_status(SimpleNamespace(code="service_unavailable")) == 503
    assert stream_error_status(SimpleNamespace(code="RESOURCE_EXHAUSTED")) == 429
    assert stream_error_status(SimpleNamespace(code=503)) == 503
    assert stream_error_status(SimpleNamespace(code="something_else")) is None


def test_quota_error_in_a_stream_becomes_rate_limited():
    quota = SimpleNamespace(event_type="error", error=SimpleNamespace(code="resource_exhausted"))
    client, _, _ = make([[quota]] * 3)
    with pytest.raises(RateLimited):
        ask(client, on_text=lambda _: None)
