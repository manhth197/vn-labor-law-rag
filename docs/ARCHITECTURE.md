# Vietnamese Labor Law Assistant — Architecture (v1, as built)

## Goal
A question-answering assistant for Vietnamese labor law. It answers in Vietnamese, grounded only in the official legal text, and cites every claim to the article (Điều) and clause (Khoản). It is a portfolio project that demonstrates RAG engineering end to end: structured ingestion, hybrid retrieval on pgvector, LangGraph orchestration, LLM generation (Gemini) with verified citations, measurable evaluation, Docker and CI.

Not legal advice: the UI shows a disclaimer with the corpus notice built from the amendment registry, currently "Theo VBHN 125/VBHN-VPQH (2025); một số quy định sẽ thay đổi từ 01/03/2027.", and cited articles show their `amended_by` / `upcoming_change` notes.

## Corpus (v1)
- Bộ luật Lao động 2019 (Law No. 45/2019/QH14), effective 2021-01-01.
- Nghị định 145/2020/NĐ-CP (guidance on working conditions and labor relations).

Source: the Official Gazette (Công báo) PDFs on congbaocdn.chinhphu.vn, which have a text layer; citations link to the document page on vanban.chinhphu.vn. (vbpl.vn is behind an anti-bot layer and the signed PDFs on vanban.chinhphu.vn are image scans; see `docs/decisions/001-corpus-source-and-pdf-extraction.md`.) Raw files go to `data/raw/` with `data/raw/manifest.json` (url, retrieved_at, sha256). Vietnamese normative legal documents are not protected by copyright (Law on Intellectual Property, Art. 15), so the text may live in the repo; always link the source. PDFs are not committed; `fetch` recreates them.

**Versions and amendments:** the Labor Code is its consolidated text 125/VBHN-VPQH (27/08/2025), i.e. the law in force including clause 8a of Điều 154 (Luật 71/2025/QH15, in force 01/01/2026); Nghị định 145/2020/NĐ-CP is the original text. `data/amendments.json` records every known amending document (status `in_force | passed_pending_text | draft`, effective date, affected articles, official source and access date). The law passed on 21/11/2026 (effective 01/03/2027) is in it with articles inferred from its topics until its text is published; the draft decree for Nghị định 145 is recorded but never used. See `docs/decisions/006-consolidated-text-and-amendment-registry.md`.

## Ingestion pipeline
1. **Fetch** official PDFs → `data/raw/` (+ manifest).
2. **Extract** text lines with a bold flag, x position and footnote information (superscript markers removed from the text; 12 pt footnote lines flagged) (PyMuPDF, optional `[ingest]` extra, AGPL, offline only) → `data/interim/<DOC>.lines.jsonl` (committed, so later steps never need PyMuPDF).
3. **Parse** into a structure: Document → Chương → Mục → Điều → Khoản → Điểm, using bold headings, indentation, sequential numbering (including inserted clauses such as "8a") and quote tracking; footnotes are kept out of the body and attached as notes to the clause they annotate (`docs/decisions/002-layout-aware-legal-parser.md`). Output `data/processed/articles.jsonl` (one record per Điều, clauses nested) and `appendices.jsonl` (Phụ lục).
4. **Chunk**: one chunk per Điều. If a Điều is longer than 1,200 bge-m3 tokens, split by Khoản (greedy packing of consecutive clauses; by Điểm if one clause is still too long) and prefix each part with a context header, the Điều heading and its lead-in (`docs/decisions/003-article-level-chunking.md`). Metadata: `chunk_id`, `doc_short` (e.g. `BLLD2019`, `ND145`), `doc_number`, `chapter_no`, `chapter_title`, `section_no`, `article_no`, `article_title`, `clause_nos`, `source_url`, and from the amendment registry `amended_by` (changes already in the text) and `upcoming_change` (announced changes with their effective date). Output `data/processed/chunks.jsonl` and `build_info.json`.
5. **Embed** with `BAAI/bge-m3` (sentence-transformers, normalized vectors, dim 1024, CPU).
6. **Store** in PostgreSQL 16 + pgvector with our own SQL (`index/store.py`): one `chunks` table, `vector(1024)` with an HNSW cosine index, and a generated `tsvector` column (`simple` config + `unaccent`) with a GIN index for keyword matching. `python -m vnlaw_rag.cli index` rebuilds it in one command (embeddings are cached in `data/cache/`).

## Query flow (LangGraph)
`route → (lookup | retrieve) → grade → (generate | not_found) → finalize`, a LangGraph `StateGraph` (`graph/build.py`). Nodes emit progress events through LangGraph's stream writer; each node's time is recorded in `timings_ms`.

- **route**: a regex catches explicit references ("Điều 35", "khoản 2 Điều 113 BLLĐ", unaccented "dieu 21 nghi dinh 145"); `lookup` then fetches those articles directly. Out-of-scope questions are not detected here: they go through `retrieve` and are stopped by `grade` or by the model's `sufficient` flag.
- **retrieve**: `HybridRetriever` with three modes. Default `vector`: bge-m3 cosine search, keep the top 6 chunks (`LLM_MAX_SOURCES`). `hybrid`: vector top 20 + keyword top 20 fused with Reciprocal Rank Fusion (k = 60), then the top 6. The keyword query OR-s adjacent syllable pairs as phrases (`thu <-> viec | ...`), because Vietnamese words are mostly two syllables and a natural question never contains all the words of its answer; terms are weighted by BM25-style IDF computed in SQL. Hybrid did not beat vector-only on the draft golden set, so vector is the default (`docs/decisions/005-retrieval-mode-vector-default.md`, `docs/EVAL.md`). A cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) is the next experiment, kept only if the eval shows a gain: the answer eval shows that the remaining wrong refusals are retrieval misses on paraphrased questions.
- **grade**: if the best cosine similarity is under `GRADE_MIN_SCORE` (0.50, below every answerable question's top-1 score in `docs/EVAL.md`; it was 0.55 until the answer eval, see `docs/decisions/009-answer-evaluation.md`), answer "không tìm thấy quy định phù hợp" without calling the LLM.
- **generate**: Google Gemini API (free tier, chosen by the owner) through the official `google-genai` SDK (Interactions API), behind a one-method `LLMClient` interface so another provider can be added. Model from config (`LLM_MODEL=gemini-3.5-flash-lite`, fallback `gemini-3.8-flash`, chosen by measurement in `docs/decisions/008-llm-latency-policy-and-streaming.md`). Structured output `{answer, citations[chunk_id], sufficient}`; citations are verified in code against the retrieved chunks; amendment warnings of the cited chunks are attached by code. The system prompt says: answer in Vietnamese, use only the provided sources, cite every claim with Điều/khoản and the chunk id, mention amendment notes with their dates, say clearly when the sources are not enough. In the same node, an answer without a valid citation, or with `sufficient=false`, becomes the fixed "không tìm thấy" answer. See `docs/decisions/007-gemini-generation-and-citation-checks.md`.
- **finalize**: attaches the corpus notice built from the amendment registry.
- LangChain provides the HuggingFace embeddings wrapper (`langchain-huggingface`), `Document` and the retriever interface: `HybridRetriever(BaseRetriever)` wraps our pgvector SQL and plugs into the LangGraph `retrieve` node. `langchain-postgres` was evaluated and not used (shared-state bug in its hybrid search, AND-only keyword query, Windows event-loop issue; see `docs/decisions/004-own-pgvector-sql-instead-of-langchain-postgres.md`). LangGraph provides the state graph.

## API (FastAPI)
- `POST /api/ask` `{question, top_k?}` → `{answer, citations[], warnings[], notice, status, sources[], timings_ms, model}`
- `GET /api/ask/stream?question=...` (Server-Sent Events: `progress`, `token`, `result` | `error`)
- `GET /api/articles/{doc_short}/{article_no}`
- `GET /health`
- A static chat UI at `/` (vanilla HTML/JS, Vietnamese). Citations appear as expandable cards with the source text and a link to the official page.

## MCP server (stdio)
- `python -m vnlaw_rag.cli mcp` (`mcp_server.py`, optional extra `[mcp]`): tools `search_labor_law` (no LLM call), `get_labor_law_article` and `ask_labor_law`. The tools call the same retriever, graph and article index as the API, opened lazily on the first call that needs them. See `docs/decisions/010-mcp-server.md`.

## Evaluation
- Golden set `data/eval/golden.jsonl`: 77 Vietnamese questions (numeric 23, paraphrase 19, lookup 15, multi 10, unanswerable 10) with expected article references, evidence quotes (checked by a test to occur in the expected articles) and a short reference answer. Drafted by the coding agent; the owner reviewed a stratified sample of 25 through an Excel workbook (`review-export` / `review-import`) and accepted 22.
- **Retrieval metrics** (no LLM, free): Hit@k, MRR@10, broken down by question type. Re-run after every retrieval change.
- **Answer metrics** (`eval-answers`, Gemini free tier, run with the owner's approval): deterministic metrics first (article-level citation precision and recall, refusal correctness, numeric exact match of a key figure); a reference-based LLM judge (1-5 correctness and faithfulness) is secondary because it is the generator model. Sequential requests with stop rules; results in resumable JSONL; `--ids` re-runs selected questions under the current configuration, `--report-only` re-renders without API calls (`docs/decisions/009-answer-evaluation.md`).
- **LLM latency** (`bench-llm`): success rate, time to first token and total time per model, used to choose the default and fallback models (`docs/decisions/008-llm-latency-policy-and-streaming.md`).
- Results go to `docs/EVAL.md` with date, git sha and config. No invented numbers.

## Repo layout
```
src/vnlaw_rag/
  config.py            # pydantic-settings, reads .env
  cli.py               # Typer: fetch parse index search ask serve eval eval-answers bench-llm review-export review-import
  ingest/  sources.py fetch.py pdf_text.py parse.py chunk.py models.py pipeline.py amendments.py
  index/   embed.py store.py build.py
  retrieval/ hybrid.py keyword.py factory.py
  graph/   state.py nodes.py build.py prompts.py references.py
  llm/     base.py gemini.py
  api/     main.py static/ (index.html app.js style.css)
  mcp_server.py        # MCP tools over stdio
  eval/    golden.py retrieval.py report.py review.py latency.py answers.py
tests/                 # fake LLM and fake embeddings; DB integration tests against pgvector
scripts/screenshots.py # Playwright with the system Edge
data/  raw/ interim/ processed/ eval/ amendments.json
docs/  ARCHITECTURE.md EVAL.md decisions/ screenshots/
docker-compose.yml  Dockerfile  .github/workflows/ci.yml  .env.example  pyproject.toml
```

## Ops
- `docker-compose.yml`: `db` (`pgvector/pgvector:pg16`, 127.0.0.1:5442) and `app` (`Dockerfile`: `python:3.11-slim`, CPU-only PyTorch, non-root user, healthcheck), a volume for the Hugging Face model cache and a bind mount for the embedding cache. The index is built with `docker compose run --rm app python -m vnlaw_rag.cli index`.
- CI (GitHub Actions, `.github/workflows/ci.yml`): ruff and pytest on Python 3.10/3.11 with a pgvector service container for the integration tests. Unit tests use a fake embedder and a fake LLM. CI never calls paid or rate-limited APIs.
- Deployment: a later phase (options: an AWS EC2 host shared with the CRM project, or Hugging Face Spaces + managed Postgres).

## Milestones
| # | Milestone | Done when | Status |
|---|---|---|---|
| M1 | Corpus & parsing | Both documents fetched with manifest; parser yields every Điều with correct numbering (tests on real samples); chunks JSONL with metadata | done |
| M2 | Index & hybrid retrieval | pgvector + full-text index built by one command; hybrid search with RRF; golden set drafted; retrieval eval script reports Hit@5 / MRR@10 | done (vector-only won; hybrid kept as an option) |
| M3 | Graph + generation | LangGraph flow with routing, grading and not-found path; Gemini answers with verified citations; unit tests with a fake LLM | done |
| M4 | API + UI | FastAPI endpoints incl. SSE streaming; chat UI with citation cards and disclaimer | done (browser-tested in Edge) |
| M5 | Eval, Docker, CI | Answer eval (approved run) in docs/EVAL.md; `docker compose up` runs everything; CI green | done (CI green on GitHub) |
| M6 | Polish | README with architecture diagram, demo GIF, decision notes; optional reranker experiment | done, plus an MCP server (decision 010); screenshots instead of a demo GIF; the reranker experiment is deferred (README, Future work) |
