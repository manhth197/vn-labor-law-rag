# Vietnamese Labor Law Assistant — Architecture (draft v0.1)

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
`route → (article_lookup | retrieve) → grade → generate | not_found → format`

- **route**: a regex catches explicit references ("Điều 35", "khoản 2 Điều 113 BLLĐ") and fetches them directly; clearly out-of-scope questions get a polite refusal.
- **retrieve**: `HybridRetriever` with three modes. Default `vector`: bge-m3 cosine search, keep the top 8 chunks. `hybrid`: vector top 20 + keyword top 20 fused with Reciprocal Rank Fusion (k = 60). The keyword query OR-s adjacent syllable pairs as phrases (`thu <-> viec | ...`), because Vietnamese words are mostly two syllables and a natural question never contains all the words of its answer; terms are weighted by BM25-style IDF computed in SQL. Hybrid did not beat vector-only on the draft golden set, so vector is the default (`docs/decisions/005-retrieval-mode-vector-default.md`, `docs/EVAL.md`). A cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) is a later experiment, kept only if the eval shows a gain.
- **grade**: if the best cosine similarity is under `GRADE_MIN_SCORE` (0.55, from the score separation in `docs/EVAL.md`), answer "không tìm thấy quy định phù hợp" without calling the LLM.
- **generate**: Google Gemini API (free tier, chosen by the owner) through the official `google-genai` SDK (Interactions API), behind a one-method `LLMClient` interface so another provider can be added. Model from config (`LLM_MODEL=gemini-3.5-flash-lite`, fallback `gemini-3.8-flash`, chosen by measurement in `docs/decisions/008-llm-latency-policy-and-streaming.md`). Structured output `{answer, citations[chunk_id], sufficient}`; citations are verified in code against the retrieved chunks; amendment warnings of the cited chunks are attached by code. The system prompt says: answer in Vietnamese, use only the provided sources, cite every claim with Điều/khoản and the chunk id, mention amendment notes with their dates, say clearly when the sources are not enough. See `docs/decisions/007-gemini-generation-and-citation-checks.md`.
- LangChain provides the HuggingFace embeddings wrapper (`langchain-huggingface`), `Document` and the retriever interface: `HybridRetriever(BaseRetriever)` wraps our pgvector SQL and plugs into the LangGraph `retrieve` node. `langchain-postgres` was evaluated and not used (shared-state bug in its hybrid search, AND-only keyword query, Windows event-loop issue; see `docs/decisions/004-own-pgvector-sql-instead-of-langchain-postgres.md`). LangGraph provides the state graph.

## API (FastAPI)
- `POST /api/ask` `{question, top_k?}` → `{answer, citations[], warnings[], notice, status, sources[], timings_ms, model}`
- `GET /api/ask/stream?question=...` (Server-Sent Events: `progress`, `token`, `result` | `error`)
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
  ingest/  sources.py fetch.py pdf_text.py parse.py chunk.py models.py pipeline.py
  index/   embed.py store.py
  retrieval/ hybrid.py rerank.py
  graph/   state.py nodes.py build.py prompts.py references.py
  llm/     base.py gemini.py
  api/     main.py static/ (index.html app.js style.css)
  eval/    retrieval.py answers.py
tests/
data/  raw/ interim/ processed/ eval/
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
| M3 | Graph + generation | LangGraph flow with routing, grading and not-found path; Gemini answers with verified citations; unit tests with a fake LLM |
| M4 | API + UI | FastAPI endpoints incl. SSE streaming; chat UI with citation cards and disclaimer |
| M5 | Eval, Docker, CI | Answer eval (approved run) in docs/EVAL.md; `docker compose up` runs everything; CI green |
| M6 | Polish | README with architecture diagram, demo GIF, decision notes; optional reranker experiment |
