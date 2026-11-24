"""State passed between the LangGraph nodes."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer: each node adds its own timing instead of replacing the whole dict."""
    return {**(left or {}), **(right or {})}


class QAState(TypedDict, total=False):
    question: str
    route: str  # "lookup" (explicit "Điều N") | "search"
    refs: list[dict[str, Any]]  # explicit references found by the router
    sources: list[dict[str, Any]]  # chunks given to the model: DB row + retrieval scores
    top_score: float | None  # best cosine similarity among the sources (search route)
    status: str  # "answered" | "not_found"
    answer: str
    citations: list[str]  # chunk ids, all of them present in `sources`
    dropped_citations: list[str]  # ids the model produced that were not retrieved
    warnings: list[str]  # amendment notes of the cited articles, with effective dates
    model: str | None
    notice: str
    timings_ms: Annotated[dict[str, int], merge_dicts]
