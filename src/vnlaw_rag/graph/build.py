"""Assemble the LangGraph state graph and wire it to the real store, retriever and LLM.

    START -> route -> lookup ----\\
                  \\-> retrieve -> grade -> generate --\\
                                       \\-> not_found -> finalize -> END
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langgraph.graph import END, START, StateGraph

from vnlaw_rag.config import Settings
from vnlaw_rag.graph.nodes import QANodes
from vnlaw_rag.graph.state import QAState
from vnlaw_rag.ingest.amendments import corpus_notice, load_registry
from vnlaw_rag.ingest.sources import DOCUMENTS
from vnlaw_rag.llm.base import LLMClient


def build_graph(nodes: QANodes) -> Any:
    graph = StateGraph(QAState)
    graph.add_node("route", nodes.route)
    graph.add_node("lookup", nodes.lookup)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("generate", nodes.generate)
    graph.add_node("not_found", nodes.not_found)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route", nodes.after_route, {"lookup": "lookup", "search": "retrieve"}
    )
    graph.add_edge("lookup", "grade")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", nodes.after_grade, {"generate": "generate", "not_found": "not_found"}
    )
    graph.add_edge("generate", "finalize")
    graph.add_edge("not_found", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def notice_for(settings: Settings) -> str:
    registry = load_registry(settings.data_dir / "amendments.json")
    return corpus_notice(registry, [d.text_version for d in DOCUMENTS])


def make_nodes(settings: Settings, retriever: Any, llm: LLMClient) -> QANodes:
    return QANodes(
        retriever=retriever,
        llm=llm,
        min_score=settings.grade_min_score,
        max_sources=settings.llm_max_sources,
        notice=notice_for(settings),
    )


@contextmanager
def open_qa(settings: Settings, llm: LLMClient | None = None) -> Iterator[Any]:
    """Compiled graph over the real pgvector store, bge-m3 and Gemini (unless `llm` is given)."""
    from vnlaw_rag.llm.gemini import client_from_settings
    from vnlaw_rag.retrieval.factory import open_retriever

    with open_retriever(settings) as retriever:
        yield build_graph(make_nodes(settings, retriever, llm or client_from_settings(settings)))
