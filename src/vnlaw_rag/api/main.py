"""FastAPI app: question answering (JSON and Server-Sent Events), article lookup, health, UI.

One process loads bge-m3 and opens one psycopg pool at startup (lifespan); requests share
them. Endpoints are sync functions: FastAPI runs them in its thread pool, which fits the sync
graph, the sync store and the blocking LLM call.

    POST /api/ask                         -> JSON answer
    GET  /api/ask/stream?question=...     -> SSE: progress, token, result | error
    GET  /api/articles/{doc_short}/{n}    -> article text, structure and amendment notes
    GET  /health                          -> status, model, index size, corpus notice
    GET  /                                -> chat UI (static/)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vnlaw_rag.config import Settings, get_settings
from vnlaw_rag.ingest.amendments import annotate_chunks, load_registry
from vnlaw_rag.ingest.models import Article, Chunk
from vnlaw_rag.llm.base import LLMError, RateLimited, Unavailable

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
MAX_QUESTION_CHARS = 500

ERROR_MESSAGES = {
    RateLimited: "Đã chạm giới hạn miễn phí của dịch vụ trả lời (Gemini) trong lúc này. "
    "Vui lòng thử lại sau ít phút.",
    Unavailable: "Dịch vụ trả lời đang quá tải hoặc phản hồi quá chậm. Vui lòng thử lại sau.",
    LLMError: "Không tạo được câu trả lời cho câu hỏi này. Vui lòng thử lại.",
}
SOURCE_FIELDS = (
    "chunk_id", "doc_short", "doc_title", "unit", "chapter_no", "chapter_title", "article_no",
    "article_title", "clause_nos", "appendix_no", "source_url", "content", "vector_score",
    "amended_by", "upcoming_change",
)  # fmt: skip


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int | None = Field(default=None, ge=1, le=10)


def friendly_error(exc: LLMError) -> str:
    for cls in (RateLimited, Unavailable, LLMError):
        if isinstance(exc, cls):
            return ERROR_MESSAGES[cls]
    return ERROR_MESSAGES[LLMError]


def public_result(state: dict[str, Any], elapsed_ms: int) -> dict[str, Any]:
    """The answer as returned to clients (no internal fields such as prompts)."""
    return {
        "question": state.get("question"),
        "status": state.get("status", "not_found"),
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "warnings": state.get("warnings", []),
        "notice": state.get("notice"),
        "model": state.get("model"),
        "route": state.get("route"),
        "sources": [{k: s.get(k) for k in SOURCE_FIELDS} for s in state.get("sources", [])],
        "timings_ms": {**state.get("timings_ms", {}), "total": elapsed_ms},
    }


def load_articles(settings: Settings) -> dict[tuple[str, int], dict[str, Any]]:
    """Articles for the lookup endpoint, with their amendment notes (same registry and
    messages as the chunks, applied to a whole-article pseudo chunk)."""
    registry = load_registry(settings.data_dir / "amendments.json")
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with (settings.processed_dir / "articles.jsonl").open(encoding="utf-8") as fh:
        for row in fh:
            article = Article.model_validate_json(row)
            probe = Chunk(
                chunk_id="", doc_short=article.doc_short, doc_number=article.doc_number,
                doc_title="", unit="article", article_no=article.article_no,
                source_url=article.source_url, text="", n_tokens=0,
            )  # fmt: skip
            annotate_chunks([probe], registry)
            out[(article.doc_short, article.article_no)] = {
                **article.model_dump(),
                "heading": article.heading,
                "text": article.body_text(),
                "amended_by": probe.amended_by,
                "upcoming_change": probe.upcoming_change,
            }
    return out


def create_app(
    settings: Settings | None = None,
    qa: Any = None,
    articles: dict[tuple[str, int], dict[str, Any]] | None = None,
    health: dict[str, Any] | None = None,
) -> FastAPI:
    """Build the app. Tests pass a ready `qa` graph (fake retriever and LLM); otherwise the
    real graph is built at startup."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.articles = articles if articles is not None else load_articles(settings)
        if qa is not None:
            app.state.qa = qa
            app.state.health = health or {}
            yield
            return
        from vnlaw_rag.graph.build import build_graph, make_nodes
        from vnlaw_rag.llm.gemini import client_from_settings
        from vnlaw_rag.retrieval.factory import open_retriever

        with open_retriever(settings) as retriever:
            nodes = make_nodes(settings, retriever, client_from_settings(settings))
            app.state.qa = build_graph(nodes)
            info = retriever.store.index_info()
            app.state.health = {
                "model": settings.llm_model,
                "fallback_model": settings.llm_fallback_model,
                "retrieval_mode": settings.retrieval_mode,
                "chunks": info.get("chunks"),
                "indexed_at": info.get("indexed_at"),
                "notice": nodes.notice,
            }
            yield

    app = FastAPI(title="Trợ lý tra cứu pháp luật lao động", lifespan=lifespan)

    @app.get("/health")
    def health_check(request: Request) -> dict[str, Any]:
        return {"status": "ok", **request.app.state.health}

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            state = request.app.state.qa.invoke(
                {"question": body.question.strip(), "top_k": body.top_k}
            )
        except LLMError as exc:
            log.warning("ask failed: %s", exc)
            raise HTTPException(status_code=503, detail=friendly_error(exc)) from exc
        return public_result(state, round((time.perf_counter() - started) * 1000))

    @app.get("/api/ask/stream")
    def ask_stream(
        request: Request,
        question: str = Query(min_length=1, max_length=MAX_QUESTION_CHARS),
        top_k: int | None = Query(default=None, ge=1, le=10),
    ) -> StreamingResponse:
        graph = request.app.state.qa
        events = stream_events(graph, {"question": question.strip(), "top_k": top_k})
        return StreamingResponse(
            events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/articles/{doc_short}/{article_no}")
    def get_article(doc_short: str, article_no: int, request: Request) -> dict[str, Any]:
        article = request.app.state.articles.get((doc_short.upper(), article_no))
        if article is None:
            raise HTTPException(status_code=404, detail="Không có Điều này trong dữ liệu.")
        return article

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_events(graph: Any, inputs: dict[str, Any]) -> Iterator[str]:
    """Run the graph and translate its custom events (progress, token) and final state into
    SSE messages. Errors become a friendly `error` event instead of a broken stream."""
    started = time.perf_counter()
    final: dict[str, Any] | None = None
    try:
        for mode, payload in graph.stream(inputs, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield sse(payload.get("type", "progress"), payload)
            else:
                final = payload
    except LLMError as exc:
        log.warning("stream failed: %s", exc)
        yield sse("error", {"message": friendly_error(exc)})
        return
    except Exception:  # never leak a traceback into the UI
        log.exception("stream failed")
        yield sse("error", {"message": ERROR_MESSAGES[LLMError]})
        return
    elapsed = round((time.perf_counter() - started) * 1000)
    yield sse("result", public_result(final or {}, elapsed))
