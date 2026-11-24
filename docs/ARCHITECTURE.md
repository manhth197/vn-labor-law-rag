# Vietnamese Labor Law Assistant — Architecture (draft v0.1)

## Goal
A question-answering assistant for Vietnamese labor law. It answers in Vietnamese, grounded only in the official legal text, and cites every claim to the article (Điều) and clause (Khoản). It is a portfolio project that demonstrates RAG engineering end to end: structured ingestion, hybrid retrieval on pgvector, LangGraph orchestration, Claude generation with citations, measurable evaluation, Docker and CI.

Not legal advice: the UI shows a disclaimer.

## Corpus (v1)
- Bộ luật Lao động 2019 (Law No. 45/2019/QH14), effective 2021-01-01.
- Nghị định 145/2020/NĐ-CP (guidance on working conditions and labor relations).

Source: official government portals (vbpl.vn, chinhphu.vn, quochoi.vn). Raw files go to `data/raw/` with `data/raw/manifest.json` (url, retrieved_at, sha256). Vietnamese normative legal documents are not protected by copyright (Law on Intellectual Property, Art. 15), so the text may live in the repo; always link the source. Files over 5 MB are not committed; a fetch script recreates them.

## Ingestion pipeline
1. **Fetch** official text → `data/raw/`.
2. **Parse** into a structure: Document → Chương → Mục → Điều → Khoản → Điểm. Output `data/processed/articles.jsonl` (one record per Điều, clauses nested).
3. **Chunk**: one chunk per Điều. If a Điều is longer than about 1,200 tokens, split by Khoản and prefix each part with the Điều heading. Metadata: `doc_id`, `doc_short` (e.g. `BLLD2019`, `ND145`), `chapter`, `article_no`, `article_title`, `clause_no`, `source_url`.
4. **Embed** with `BAAI/bge-m3` (sentence-transformers, normalized vectors, dim 1024, CPU).
5. **Store** in PostgreSQL 16 + pgvector (HNSW, cosine), plus Postgres full-text search (`simple` config + `unaccent`) for keyword matching.

## Query flow (LangGraph)
`route → (article_lookup | retrieve) → grade → generate | not_found → format`

- **route**: a regex catches explicit references ("Điều 35", "khoản 2 Điều 113 BLLĐ") and fetches them directly; clearly out-of-scope questions get a polite refusal.
- **retrieve**: hybrid search. Vector top 20 + keyword top 20, fused with Reciprocal Rank Fusion, keep the top 6–8 chunks. A cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) is a later experiment, kept only if the eval shows a gain.
- **grade**: if the best score is under a threshold, answer "không tìm thấy quy định phù hợp" instead of guessing.
- **generate**: Claude through the official `anthropic` Python SDK (not `langchain-anthropic`), so the app can use citation / search-result blocks, effort and fallbacks. The model id comes from config: default `claude-opus-5-5`; the owner may switch to `claude-haiku-5-5` to cut cost. The system prompt says: answer in Vietnamese, use only the provided sources, cite every claim, say clearly when the sources are not enough.
- LangChain provides text-splitting utilities, the PGVector store (`langchain-postgres`), the HuggingFace embeddings wrapper and retriever interfaces. LangGraph provides the state graph.

## API (FastAPI)
- `POST /api/ask` `{question, top_k?}` → `{answer, citations[], sources[], timings_ms, model}`
- `GET /api/ask/stream?question=...` (Server-Sent Events)
- `GET /api/articles/{doc_short}/{article_no}`
- `GET /health`
- A static chat UI at `/` (vanilla HTML/JS, Vietnamese). Citations appear as expandable cards with the source text and a link to the official page.

## Evaluation
- Golden set `data/eval/golden.jsonl`: 60–80 Vietnamese questions with expected article references and a short reference answer. Drafted with LLM help from the articles, then a sample of at least 20 is reviewed by a human (the owner). Mix: direct lookups, paraphrased questions, multi-article questions, numeric facts (annual leave days, retirement age, probation length), and unanswerable or out-of-scope questions.
- **Retrieval metrics** (no LLM, free): Hit@k, MRR@10, broken down by question type. Re-run after every retrieval change.
- **Answer metrics**: citation accuracy (cited articles vs. expected), correct refusal on unanswerable questions, and LLM-as-judge scores for faithfulness and correctness. The judge costs money, so it runs manually with the owner's approval and its cost is recorded.
- Results go to `docs/EVAL.md` with date, git sha and config. No invented numbers.

## Repo layout
```
src/vnlaw_rag/
  config.py            # pydantic-settings, reads .env
  cli.py               # Typer: fetch | parse | index | eval | serve
  ingest/  fetch.py parse.py chunk.py
  index/   embed.py store.py
  retrieval/ hybrid.py rerank.py
  graph/   state.py nodes.py build.py
  llm/     claude.py
  api/     main.py routes.py static/
  eval/    retrieval.py answers.py
tests/
data/  raw/ processed/ eval/
docs/  ARCHITECTURE.md EVAL.md decisions/
docker-compose.yml  Dockerfile  .env.example  pyproject.toml
```

## Ops
- `docker-compose.yml`: `db` (`pgvector/pgvector:pg16`) and `app` (`python:3.11-slim`, CPU-only PyTorch), plus a volume for the Hugging Face model cache.
- CI (GitHub Actions): ruff, pytest. Unit tests use a fake embedder and a mocked LLM; integration tests use a pgvector service container. CI never calls paid APIs.
- Deployment: a later phase (options: an AWS EC2 host shared with the CRM project, or Hugging Face Spaces + managed Postgres).

## Milestones
| # | Milestone | Done when |
|---|---|---|
| M1 | Corpus & parsing | Both documents fetched with manifest; parser yields every Điều with correct numbering (tests on real samples); chunks JSONL with metadata |
| M2 | Index & hybrid retrieval | pgvector + full-text index built by one command; hybrid search with RRF; golden set drafted; retrieval eval script reports Hit@5 / MRR@10 |
| M3 | Graph + generation | LangGraph flow with routing, grading and not-found path; Claude answers with citations; unit tests with mocked LLM |
| M4 | API + UI | FastAPI endpoints incl. SSE streaming; chat UI with citation cards and disclaimer |
| M5 | Eval, Docker, CI | Answer eval (approved run) in docs/EVAL.md; `docker compose up` runs everything; CI green |
| M6 | Polish | README with architecture diagram, demo GIF, decision notes; optional reranker experiment |
