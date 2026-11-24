# Draft issue for langchain-postgres (not filed)

> Draft only. Filing it is an outward-facing action for the repository owner to decide.
> Found by reading the source; not yet reproduced against a running database.

**Title:** `PGVectorStore` hybrid search reuses the first query's `fts_query` for all later queries

**Version:** langchain-postgres 0.0.19

**Where:** `langchain_postgres/v2/async_vectorstore.py`, `asimilarity_search` (lines 799–805) and `asimilarity_search_with_score` (lines 840–846):

```python
hybrid_search_config = kwargs.get("hybrid_search_config", self.hybrid_search_config)
if hybrid_search_config and not hybrid_search_config.fts_query:
    hybrid_search_config.fts_query = query
    kwargs["hybrid_search_config"] = hybrid_search_config
```

**Problem:** when no per-call config is passed, `hybrid_search_config` is the instance's own config object. The first call stores its query text in `fts_query`; on every later call `fts_query` is no longer empty, so it is not updated and the keyword (full-text) leg keeps searching for the first query.

**Expected:** each call uses its own query for the full-text leg, without mutating shared state.

**Suggested fix:** copy the config per call, e.g. `hybrid_search_config = dataclasses.replace(hybrid_search_config, fts_query=query)` when `fts_query` is empty, instead of assigning to the shared instance.

**Reproduction sketch:** create a `PGVectorStore` with a `HybridSearchConfig` (no `fts_query`), call `similarity_search("query A")`, then `similarity_search("query B")`, and log the SQL or inspect `store.hybrid_search_config.fts_query`: it is still "query A".
