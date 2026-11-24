# 008. LLM latency policy and streaming answers

Date: 2026-11-24 · Status: policy accepted; default model choice pending a clean measurement

## Context
In the M3 smoke test an answer took 3 to 160 s; most of the 160 s were three 503 responses from gemini-4-hydrogen with backoff in between. A demo UI cannot sit silent that long. The coordinator asked for a per-call timeout, an immediate switch to the fallback model, a total time budget with a friendly error, progress events and token streaming, and a model choice based on measured reliability and speed.

## Decision
- **Hard per-call limit (25 s):** each call runs in a daemon thread and the caller waits at most 25 s. A first version closed the stream from a timer, but the benchmark showed a 25 s call lasting 71 s (closing does not always interrupt a blocked socket read), so the wait itself is bounded now. Late tokens of an abandoned call are dropped.
- **Immediate fallback:** a timeout, 5xx or 429 switches to the next model at once; only repeated failures on the last model back off (2 s, 4 s). **Budget 60 s** per answer, then `Unavailable`/`RateLimited`, shown as a Vietnamese message. Eval uses the main model only with a 180 s budget.
- **Streaming:** `GET /api/ask/stream` sends SSE `progress` events from the graph nodes (searching, found N articles, generating, switching model), `token` events with the growing `answer` field decoded from the partial JSON of the Interactions stream, then `result` (validated citations, warnings, model, timings) or `error`. The UI shows progress, the streamed text, then the final answer with citation cards, warnings, model and time.

## Measurements so far (not enough to choose)
| When (2026-11-24) | Model | Calls | OK | Failures | Notes |
|---|---|---|---|---|---|
| M3 smoke, ~12:10 | gemini-4-hydrogen | 6 attempts | 3 | 3 × 503 | successes took 3.4 to 48 s |
| benchmark, ~13:05 | gemini-4-hydrogen | 3 | 0 | 3 × 503 (31-37 s) | run stopped early |
| benchmark, ~13:05 | gemini-3.8-flash | 3 | 0 | 1 × 503, 2 × timeout (72 s) | timeouts exposed the watchdog flaw |

The benchmark was stopped by Claude Code because the machine ran critically low on memory (it loaded bge-m3 while other services ran), so its timings are not trustworthy; the 503s come from Google's side. **No p50/p95 yet.** The benchmark no longer loads the embedding model (prompts are rebuilt from the recorded vector ranking), so it can be re-run with little RAM: `python -m vnlaw_rag.cli bench-llm` (10 requests). Until then the default stays gemini-4-hydrogen with gemini-3.8-flash as fallback; adding `gemini-3.5-flash-lite` (stable, positioned as the fastest) to the comparison is suggested.

## Consequences
- Worst case an answer ends after about 60 s with a clear message instead of hanging; with the free tier's 503s the fallback path is the normal path at busy times.
- The model choice will be revisited with numbers; the decision is one config value (`LLM_MODEL`, `LLM_FALLBACK_MODEL`).
