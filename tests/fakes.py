"""Fakes shared by the graph and API tests: real chunks as DB rows, a retriever, an LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tests.conftest import REPO_ROOT, parsed, word_count
from vnlaw_rag.graph.prompts import GeneratedAnswer
from vnlaw_rag.ingest.amendments import annotate_chunks, load_registry
from vnlaw_rag.ingest.chunk import build_chunks
from vnlaw_rag.llm.base import StructuredResult
from vnlaw_rag.retrieval.hybrid import RetrievedChunk

NOTICE = "Theo VBHN 125/VBHN-VPQH (2025); một số quy định sẽ thay đổi từ 01/03/2027."


def build_rows() -> dict[str, dict[str, Any]]:
    """Real chunks with real amendment annotations, as the database would return them."""
    docs = [parsed("BLLD2019"), parsed("ND145")]
    chunks = build_chunks(
        [a for d in docs for a in d.articles], [a for d in docs for a in d.appendices],
        word_count, max_tokens=1200,
    )  # fmt: skip
    annotate_chunks(chunks, load_registry(REPO_ROOT / "data" / "amendments.json"))
    out = {}
    for c in chunks:
        row = c.model_dump(exclude={"text", "n_tokens"})
        row["content"] = c.text
        out[c.chunk_id] = row
    return out


class FakeStore:
    def __init__(self, rows):
        self.rows = rows

    def get_article_chunks(self, doc_short, article_no):
        return [
            r for r in self.rows.values()
            if r["doc_short"] == doc_short and r["article_no"] == article_no
        ]  # fmt: skip


@dataclass
class FakeRetriever:
    rows: dict[str, dict[str, Any]]
    ranked: list[tuple[str, float]]  # (chunk_id, cosine similarity) to return for any query
    queries: list[str] = field(default_factory=list)

    @property
    def store(self):
        return FakeStore(self.rows)

    def search(self, query, top_k=None):
        self.queries.append(query)
        return [
            RetrievedChunk(chunk_id=cid, score=s, vector_rank=i + 1, vector_score=s,
                           row=self.rows[cid])
            for i, (cid, s) in enumerate(self.ranked[:top_k])
        ]  # fmt: skip


class FakeLLM:
    def __init__(self, answer: GeneratedAnswer):
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    def generate_structured(self, *, system, prompt, schema, on_text=None, on_event=None):
        assert schema is GeneratedAnswer
        self.calls.append({"system": system, "prompt": prompt})
        if on_text:  # stream the answer in two pieces, like a real model would
            half = len(self.answer.answer) // 2
            on_text(self.answer.answer[:half])
            on_text(self.answer.answer[half:])
        return StructuredResult(self.answer, model="fake-model")
