"""Graph nodes: route, lookup, retrieve, grade, generate, not_found, finalize.

Two safeguards are code, not prompt: citations are filtered to the retrieved chunk ids (a
model cannot cite what it was not given), and amendment warnings are attached from the cited
chunks' metadata whether or not the model mentions them. See docs/decisions/007.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from vnlaw_rag.graph.prompts import SYSTEM_PROMPT, GeneratedAnswer, build_prompt
from vnlaw_rag.graph.references import find_references
from vnlaw_rag.graph.state import QAState
from vnlaw_rag.llm.base import LLMClient

log = logging.getLogger(__name__)

NOT_FOUND_ANSWER = (
    "Không tìm thấy quy định phù hợp trong Bộ luật Lao động 2019 (văn bản hợp nhất "
    "125/VBHN-VPQH) và Nghị định 145/2020/NĐ-CP cho câu hỏi này."
)
_CITATION = re.compile(r"\[([A-Z][A-Z0-9]*-[A-Za-z0-9-]+)\]")


def timed(name: str) -> Callable:
    def decorate(fn: Callable[..., dict]) -> Callable[..., dict]:
        @wraps(fn)
        def wrapper(self, state: QAState) -> dict:
            started = time.perf_counter()
            update = fn(self, state)
            elapsed = round((time.perf_counter() - started) * 1000)
            return {**update, "timings_ms": {name: elapsed}}

        return wrapper

    return decorate


class QANodes:
    def __init__(
        self,
        *,
        retriever: Any,  # HybridRetriever (search) whose `store` has get_article_chunks
        llm: LLMClient,
        min_score: float,
        max_sources: int,
        notice: str,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.min_score = min_score
        self.max_sources = max_sources
        self.notice = notice

    # -- routing and retrieval -----------------------------------------------------------
    @timed("route")
    def route(self, state: QAState) -> dict:
        refs = find_references(state["question"])
        route = "lookup" if refs else "search"
        return {"route": route, "refs": [r.__dict__ for r in refs]}

    @timed("lookup")
    def lookup(self, state: QAState) -> dict:
        sources: list[dict[str, Any]] = []
        for ref in state["refs"]:
            for row in self.retriever.store.get_article_chunks(ref["doc_short"], ref["article_no"]):
                sources.append({**row, "score": None, "vector_score": None})
        return {"sources": sources[: self.max_sources]}

    @timed("retrieve")
    def retrieve(self, state: QAState) -> dict:
        results = self.retriever.search(state["question"], top_k=self.max_sources)
        sources = [{**r.row, "score": r.score, "vector_score": r.vector_score} for r in results]
        return {"sources": sources}

    @timed("grade")
    def grade(self, state: QAState) -> dict:
        sources = state.get("sources", [])
        if state["route"] == "lookup":
            return {"status": "answered" if sources else "not_found", "top_score": None}
        scores = [s["vector_score"] for s in sources if s.get("vector_score") is not None]
        top = max(scores) if scores else None
        relevant = top is not None and top >= self.min_score
        return {"status": "answered" if relevant else "not_found", "top_score": top}

    # -- answering -----------------------------------------------------------------------
    @timed("generate")
    def generate(self, state: QAState) -> dict:
        sources = state["sources"]
        result = self.llm.generate_structured(
            system=SYSTEM_PROMPT,
            prompt=build_prompt(state["question"], sources),
            schema=GeneratedAnswer,
        )
        generated = result.value
        allowed = {s["chunk_id"] for s in sources}
        proposed = list(dict.fromkeys([*generated.citations, *_CITATION.findall(generated.answer)]))
        citations = [c for c in proposed if c in allowed]
        dropped = [c for c in proposed if c not in allowed]
        if dropped:
            log.warning("dropped citations not among the retrieved chunks: %s", dropped)
        answer = _CITATION.sub(lambda m: "" if m[1] in dropped else m[0], generated.answer)

        if not generated.sufficient or not citations:
            if generated.sufficient:
                log.warning("answer without a valid citation rejected")
            return {
                "status": "not_found",
                "answer": NOT_FOUND_ANSWER,
                "citations": [],
                "dropped_citations": dropped,
                "warnings": [],
                "model": result.model,
            }
        cited = [s for s in sources if s["chunk_id"] in citations]
        return {
            "status": "answered",
            "answer": answer.strip(),
            "citations": citations,
            "dropped_citations": dropped,
            "warnings": amendment_warnings(cited),
            "model": result.model,
        }

    @timed("not_found")
    def not_found(self, state: QAState) -> dict:
        return {
            "status": "not_found",
            "answer": NOT_FOUND_ANSWER,
            "citations": [],
            "dropped_citations": [],
            "warnings": [],
            "model": None,
        }

    @timed("finalize")
    def finalize(self, state: QAState) -> dict:
        return {"notice": self.notice}

    # -- edges ---------------------------------------------------------------------------
    @staticmethod
    def after_route(state: QAState) -> str:
        return state["route"]

    @staticmethod
    def after_grade(state: QAState) -> str:
        return "generate" if state["status"] == "answered" else "not_found"


def amendment_warnings(cited: list[dict[str, Any]]) -> list[str]:
    """One line per amendment note of the cited articles, deduplicated, in citation order."""
    warnings: list[str] = []
    for source in cited:
        where = (
            f"Điều {source['article_no']} {source['doc_title']}"
            if source.get("article_no") is not None
            else source.get("doc_title", "")
        )
        for note in [*source.get("amended_by", []), *source.get("upcoming_change", [])]:
            line = f"{where}: {note}"
            if line not in warnings:
                warnings.append(line)
    return warnings
