# 007. Generation with Gemini (free tier), verified citations and code-level warnings

Date: 2026-11-24 · Status: accepted (provider chosen by the owner: Google Gemini API, free tier)

## Context
The `generate` node must answer in Vietnamese from the retrieved chunks only and cite every claim. A legal assistant fails badly if it cites an article it was never given, or forgets that a cited article is about to change. The free tier limits requests per minute/day, returns 429 when exceeded, and may use the data sent to improve Google's products. The public rate-limit page does not list free-tier numbers per model (they are shown per account in AI Studio).

## Decision
- **SDK and model** (checked 2026-11-24 on ai.google.dev / aistudio.google.com and in the installed `google-genai` 2.33.0): `client.interactions.create` with structured output (`response_format` = JSON schema of `{answer, citations[chunk_id], sufficient}`), `store=False`, no `temperature` (ignored by Gemini 4 and 3.6+). `LLM_MODEL=gemini-4-hydrogen`, the strongest model marked free of charge; `LLM_FALLBACK_MODEL=gemini-3.8-flash` (stable). Not `gemini-2.5-flash`: deprecated, re-routed from 01/12/2026.
- **One retry policy:** SDK retries off; ours retries 429/5xx/timeouts with exponential backoff (5, 10, 20 s), then falls back to the second model once, and reports which model answered.
- **Citations are checked in code:** every id in `citations` or inline `[ID]` must belong to the retrieved chunks; others are dropped from the list and the text and logged. An answer with no valid citation, or `sufficient=false`, becomes the fixed "không tìm thấy" answer.
- **Warnings are code, not prompt:** `amended_by` / `upcoming_change` of the cited chunks are returned as `warnings` (with dates); the prompt also asks the model to mention them.
- **Not found is two-layer:** `grade` stops before the LLM when the best cosine similarity is under 0.55 (docs/EVAL.md: answerable top-1 min 0.537, median 0.729; out-of-scope max 0.674); between the layers the model's `sufficient` flag decides.
- `LLMClient` is a one-method protocol, so Claude or another provider can be added without touching the graph. Tests use a fake LLM; CI never calls the API.

## Alternatives
- Free-text answers with citation regexes only: no machine-checkable `sufficient` signal. LangChain's Gemini wrapper: one more layer between us and a fast-moving SDK whose Interactions API it may not cover.

## Consequences
- Smoke test (3 questions, 6 HTTP calls): correct cited answers for probation and Tết (with the 01/03/2027 warning), "not found" for a tax question; three 503s from gemini-4-hydrogen before success, hence fallback also on overload. Latency 3-160 s: the free tier is for a demo, not production.
