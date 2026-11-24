# 008. LLM latency policy, streaming answers and the measured model choice

Date: 2026-11-24 · Status: accepted (coordinator) · Updates the model choice of 007

## Context
In the M3 smoke test an answer took 3 to 160 s, mostly three 503s from gemini-4-hydrogen plus backoff. A demo UI cannot sit silent that long. Requirements: a per-call timeout, an immediate switch to another model, a total budget with a friendly error, progress events and token streaming, and a model chosen by success rate first, then p50/p95, then quality.

## Decision
- **Hard 25 s per call:** the call runs in a daemon thread and the caller waits at most 25 s (closing the stream from a timer was not enough: a 25 s call once lasted 71 s). Late tokens of an abandoned call are dropped.
- **Models take turns without waiting:** a timeout, 5xx, 429 or mid-stream error event switches at once to the other model (main, fallback, main); one model alone backs off 2 s, 4 s. **60 s budget** per answer, then a Vietnamese error. Eval uses one model with a 180 s budget.
- **Streaming:** SSE `progress` (searching, found N articles, generating, switching model), `token` (the `answer` field decoded from the partial JSON), then `result` or `error`.
- **Model:** `LLM_MODEL=gemini-3.5-flash-lite`, `LLM_FALLBACK_MODEL=gemini-3.8-flash`, from the measurement below. gemini-4-hydrogen is not the default: it never answered within 25 s.

## Measurement (2026-11-24 06:53 UTC, git c51d5c4, `bench-llm`)
5 reviewed questions per model, one streaming attempt each, 25 s limit, 4 s pause, sequential:

| Model | OK | Failures | TTFT p50 | Total p50 | Total p95 |
|---|---|---|---|---|---|
| gemini-3.5-flash-lite | 4/5 | 1 mid-stream error event (1.4 s) | 0.98 s | 1.68 s | 2.0 s |
| gemini-4-hydrogen | 0/5 | 5 timeouts | - | - | - |
| gemini-3.8-flash | 0/5 | 5 timeouts | - | - | - |
| gemini-3.1-flash-lite (fallback check, 3 q.) | 0/3 | 1 error event, 2 timeouts | - | - | - |
| gemini-2.5-flash-lite (fallback check, 3 q.) | 0/3 | 404: not served by the Interactions API | - | - | - |

Earlier that day gemini-4-hydrogen also returned 3 x 503 in the smoke test and 3 x 503 in a benchmark cut short by low memory. With 5 questions per model, p95 is the maximum. Raw samples: `data/eval/llm_latency.json`, `data/eval/llm_latency_fallback.json`.

## Consequences
- In the browser test (Edge, 3 questions) answers took 2.3 s and 1.4 s; the out-of-scope question stopped at `grade` in 0.1 s without an LLM call.
- The flash-lite model is weaker than Gemini 4; answer quality is measured in the M5 answer eval. The fallback (3.8-flash) gives better answers when it responds in time, but under the 25 s limit it mostly costs time: revisit when it is faster.
- Free-tier behavior changes from hour to hour; re-run `bench-llm` before relying on these numbers.
