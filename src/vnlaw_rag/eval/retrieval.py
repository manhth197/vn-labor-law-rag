"""Retrieval evaluation: Hit@k, MRR@10 and Recall@10 against the golden set (no LLM, free).

A retrieved chunk is relevant when it belongs to an expected unit (same document and article,
or same appendix). Matching is at article level because a chunk usually holds a whole
article. Unanswerable questions have no relevant units; they are excluded from the ranking
metrics and used instead to compare top similarity scores (input for M3's grade threshold).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings

from vnlaw_rag.eval.golden import GoldenItem, UnitKey
from vnlaw_rag.retrieval.hybrid import HybridRetriever, RetrievedChunk

EVAL_DEPTH = 10


class CachedEmbeddings(Embeddings):
    """Embed each question once even when several retrieval modes are evaluated."""

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner
        self.cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = self.inner.embed_query(text)
        return self.cache[text]


def chunk_unit(row: dict[str, Any]) -> UnitKey:
    if row.get("unit") == "appendix":
        return (row["doc_short"], "appendix", str(row["appendix_no"]))
    return (row["doc_short"], "article", str(row["article_no"]))


@dataclass
class QuestionResult:
    id: str
    type: str
    first_relevant_rank: int | None  # 1-based, within EVAL_DEPTH
    recall_at_10: float | None
    top1_vector_score: float | None
    ranked_units: list[str] = field(default_factory=list)

    def hit(self, k: int) -> bool:
        return self.first_relevant_rank is not None and self.first_relevant_rank <= k

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_relevant_rank if self.first_relevant_rank else 0.0


def score_question(item: GoldenItem, retrieved: Sequence[RetrievedChunk]) -> QuestionResult:
    units = [chunk_unit(r.row) for r in retrieved[:EVAL_DEPTH]]
    expected = {ref.key() for ref in item.expected}
    first = next((i for i, u in enumerate(units, 1) if u in expected), None)
    recall = len(expected & set(units)) / len(expected) if expected else None
    vector_scores = [r.vector_score for r in retrieved if r.vector_score is not None]
    return QuestionResult(
        id=item.id,
        type=item.type,
        first_relevant_rank=first if item.answerable else None,
        recall_at_10=recall,
        top1_vector_score=max(vector_scores) if vector_scores else None,
        ranked_units=["/".join(u) for u in units],
    )


def evaluate(retriever: HybridRetriever, items: Sequence[GoldenItem]) -> list[QuestionResult]:
    return [score_question(i, retriever.search(i.question, top_k=EVAL_DEPTH)) for i in items]


def summarize(results: Sequence[QuestionResult]) -> dict[str, dict[str, float]]:
    """Metrics overall ("all", answerable only) and per question type."""
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        if r.type == "unanswerable":
            continue
        groups["all"].append(r)
        groups[r.type].append(r)
    summary = {}
    for name, rs in groups.items():
        summary[name] = {
            "n": len(rs),
            "hit@1": sum(r.hit(1) for r in rs) / len(rs),
            "hit@5": sum(r.hit(5) for r in rs) / len(rs),
            "mrr@10": sum(r.reciprocal_rank for r in rs) / len(rs),
            "recall@10": statistics.fmean(r.recall_at_10 for r in rs),
        }
    return summary


def score_separation(results: Sequence[QuestionResult]) -> dict[str, dict[str, float]]:
    """Top-1 cosine similarity for answerable vs unanswerable questions."""
    out = {}
    for label, keep in (("answerable", True), ("unanswerable", False)):
        scores = [
            r.top1_vector_score
            for r in results
            if r.top1_vector_score is not None and (r.type != "unanswerable") == keep
        ]
        if scores:
            out[label] = {
                "n": len(scores),
                "min": min(scores),
                "median": statistics.median(scores),
                "max": max(scores),
            }
    return out
