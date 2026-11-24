"""Wire the retriever to the real store and embedding model."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.embeddings import Embeddings

from vnlaw_rag.config import Settings
from vnlaw_rag.index.embed import load_embeddings
from vnlaw_rag.index.store import PgStore
from vnlaw_rag.retrieval.hybrid import HybridRetriever


@contextmanager
def open_retriever(
    settings: Settings, embeddings: Embeddings | None = None, **overrides
) -> Iterator[HybridRetriever]:
    """Retriever configured from settings; keyword arguments override single fields."""
    store = PgStore.open(settings.database_url)
    try:
        params = dict(
            store=store,
            embeddings=embeddings
            or load_embeddings(settings.embedding_model, settings.embedding_batch_size),
            mode=settings.retrieval_mode,
            keyword_mode=settings.keyword_mode,
            keyword_scoring=settings.keyword_scoring,
            keyword_stopwords=settings.keyword_stopwords,
            vector_k=settings.retrieval_vector_k,
            keyword_k=settings.retrieval_keyword_k,
            top_k=settings.retrieval_top_k,
            rrf_k=settings.rrf_k,
        )
        params.update(overrides)
        yield HybridRetriever(**params)
    finally:
        store.close()
