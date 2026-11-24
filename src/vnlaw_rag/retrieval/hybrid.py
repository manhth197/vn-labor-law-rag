"""Hybrid retriever: vector search + keyword search, fused with Reciprocal Rank Fusion.

RRF (Cormack et al., 2009) scores a document by sum(1 / (k + rank)) over the ranked lists it
appears in. It uses ranks only, so the two legs need no score calibration (cosine similarity
and ts_rank_cd live on different scales). k = 60 is the value from the paper.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from vnlaw_rag.index.store import KeywordScoring, SearchHit
from vnlaw_rag.retrieval.keyword import KeywordMode, query_terms

RetrievalMode = Literal["hybrid", "vector", "keyword"]


class ChunkStore(Protocol):
    def vector_search(self, embedding: Sequence[float], k: int) -> list[SearchHit]: ...
    def keyword_search(
        self, terms: Sequence[str], k: int, scoring: KeywordScoring = ...
    ) -> list[SearchHit]: ...
    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]: ...


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Ties break by id so results are deterministic."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float  # RRF score in hybrid mode, else the leg's own score
    vector_rank: int | None = None
    vector_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    row: dict[str, Any] = field(default_factory=dict)

    def to_document(self) -> Document:
        metadata = {k: v for k, v in self.row.items() if k != "content"}
        metadata.update(
            score=self.score,
            vector_rank=self.vector_rank,
            vector_score=self.vector_score,
            keyword_rank=self.keyword_rank,
            keyword_score=self.keyword_score,
        )
        return Document(
            page_content=self.row.get("content", ""), metadata=metadata, id=self.chunk_id
        )


class HybridRetriever(BaseRetriever):
    """LangChain retriever over the pgvector store (plugs into the LangGraph `retrieve` node)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Any  # ChunkStore; typed loosely because pydantic cannot validate a Protocol
    embeddings: Embeddings
    mode: RetrievalMode = "hybrid"
    keyword_mode: KeywordMode = "bigram"
    keyword_scoring: KeywordScoring = "idf"
    keyword_stopwords: bool = True
    vector_k: int = 20
    keyword_k: int = 20
    top_k: int = 8
    rrf_k: int = 60

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.top_k
        vector_hits: list[SearchHit] = []
        keyword_hits: list[SearchHit] = []
        if self.mode in ("hybrid", "vector"):
            vector_hits = self.store.vector_search(
                self.embeddings.embed_query(query), self.vector_k
            )
        if self.mode in ("hybrid", "keyword"):
            terms = query_terms(query, self.keyword_mode, self.keyword_stopwords)
            keyword_hits = self.store.keyword_search(terms, self.keyword_k, self.keyword_scoring)

        by_vector = {h.chunk_id: h for h in vector_hits}
        by_keyword = {h.chunk_id: h for h in keyword_hits}
        if self.mode == "hybrid":
            fused = reciprocal_rank_fusion(
                [[h.chunk_id for h in vector_hits], [h.chunk_id for h in keyword_hits]],
                k=self.rrf_k,
            )
        else:
            hits = vector_hits if self.mode == "vector" else keyword_hits
            fused = [(h.chunk_id, h.score) for h in hits]
        fused = fused[:top_k]

        rows = self.store.get_chunks([cid for cid, _ in fused])
        results = []
        for chunk_id, score in fused:
            v, kw = by_vector.get(chunk_id), by_keyword.get(chunk_id)
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=score,
                    vector_rank=v.rank if v else None,
                    vector_score=v.score if v else None,
                    keyword_rank=kw.rank if kw else None,
                    keyword_score=kw.score if kw else None,
                    row=rows.get(chunk_id, {}),
                )
            )
        return results

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [r.to_document() for r in self.search(query)]
