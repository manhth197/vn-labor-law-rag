# 004. Own pgvector SQL instead of langchain-postgres

Date: 2026-11-24 · Status: accepted (approved by the coordinator; replaces the earlier plan to use `langchain-postgres`)

## Context
The plan was to store chunks with `langchain-postgres` and use its built-in hybrid search. Reading the installed source of **langchain-postgres 0.0.19** (`langchain_postgres/v2/`) before wiring it in showed three problems:
1. **State leak across queries.** `async_vectorstore.py` lines 799–805 (`asimilarity_search`) and 840–846 (`asimilarity_search_with_score`): if `hybrid_search_config.fts_query` is empty, the method writes the current question into it. The config object is the one stored on the vector store and shared by every call, so from the second question on the keyword leg searches for the **first** question's words. In a server this silently breaks hybrid search.
2. **AND semantics.** Line 726 builds the keyword query with `plainto_tsquery`, which requires every word of the question to be in the chunk. With the `simple` config (Postgres has no Vietnamese stop words), a natural question such as "Làm thêm giờ ban đêm được trả lương thế nào?" matches almost nothing, so the keyword leg contributes little.
3. **Event loop on Windows.** `engine.py` lines 103–110 (line 106) run async SQLAlchemy on a private loop from `asyncio.new_event_loop()`. On Windows that is a Proactor loop, which psycopg's async mode does not support; it would need asyncpg or a global loop-policy change.

## Decision
Write the storage layer directly (`index/store.py`, about 150 lines): one `chunks` table with `vector(1024)` + HNSW (`vector_cosine_ops`), a generated `tsvector` column using a `simple` + `unaccent` text search config + GIN index, sync psycopg 3 with a `psycopg_pool` pool (for FastAPI). Every value goes through query parameters. Keyword query building and RRF are our own code (`retrieval/keyword.py`, `retrieval/hybrid.py`) and unit-tested. LangChain is still used where it fits: `HuggingFaceEmbeddings` (langchain-huggingface), `Document`, and `HybridRetriever(BaseRetriever)`, which plugs into the LangGraph `retrieve` node.

## Alternatives
- **B: keep `PGVectorStore` for the table and vector search, write only the keyword leg and RRF.** Rejected: we would still have to work around the shared-config bug, run two database stacks (async SQLAlchemy on a background thread plus sync psycopg for our keyword SQL), add asyncpg for Windows, and adopt the library's schema conventions, all to avoid about 20 lines of SQL.
- **Legacy `PGVector` class:** fixed JSONB schema, no full-text column.

## Consequences
- Full control of the schema and both queries; each retrieval leg can be evaluated on its own (see `docs/EVAL.md`).
- We maintain the SQL ourselves; integration tests run against a real pgvector database (separate `vnlaw_test` DB, skipped when unreachable).
- The bug report for issue 1 is drafted in `docs/upstream/langchain-postgres-fts-query-leak.md`; filing it is the owner's decision.
