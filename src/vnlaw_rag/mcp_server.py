"""MCP server (stdio): Vietnamese labor-law search, article lookup and answers as tools.

    python -m vnlaw_rag.cli mcp

Client configuration (Claude Code, Claude Desktop, IDEs), run from the repository root:

    {"command": ".venv/Scripts/python", "args": ["-m", "vnlaw_rag.cli", "mcp"]}

Tools:
    search_labor_law(query, top_k)          retrieval only: passages with scores, no LLM call
    get_labor_law_article(doc, article_no)  one article with its amendment notes
    ask_labor_law(question)                 the full answer graph (calls Gemini)

The same retriever, graph and article index as the HTTP API are opened on the first tool call
that needs them, so the server starts at once and listing tools loads nothing. stdout carries
the protocol, so nothing here prints: logs go to stderr.
"""

from __future__ import annotations

import logging
import threading
from contextlib import ExitStack, asynccontextmanager
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from vnlaw_rag.api.main import MAX_QUESTION_CHARS, friendly_error, load_articles
from vnlaw_rag.config import Settings, get_settings
from vnlaw_rag.llm.base import LLMError

log = logging.getLogger(__name__)
DOC_NAMES = {"BLLD2019": "Bộ luật Lao động 2019", "ND145": "Nghị định 145/2020/NĐ-CP"}
INSTRUCTIONS = (
    "Tra cứu Bộ luật Lao động 2019 (văn bản hợp nhất 125/VBHN-VPQH) và Nghị định 145/2020/NĐ-CP. "
    "Use search_labor_law to find the relevant articles (no LLM call), get_labor_law_article to "
    "read one article, ask_labor_law for a cited answer in Vietnamese. Quote article numbers "
    "and mention the amendment notes. Not legal advice."
)


class Passage(BaseModel):
    chunk_id: str
    reference: str = Field(
        description="Human-readable citation, e.g. 'Điều 25 Bộ luật Lao động 2019'"
    )
    title: str | None
    score: float | None = Field(description="Cosine similarity to the query (vector mode)")
    text: str
    source_url: str
    amended_by: list[str]
    upcoming_change: list[str]


class ArticleText(BaseModel):
    reference: str
    heading: str
    text: str
    source_url: str
    amended_by: list[str]
    upcoming_change: list[str]


class Answer(BaseModel):
    status: str = Field(description="'answered' or 'not_found'")
    answer: str
    citations: list[str] = Field(description="Chunk ids of the passages the answer relies on")
    warnings: list[str] = Field(description="Amendment notes of the cited articles")
    notice: str | None = Field(description="Corpus version notice; not legal advice")
    model: str | None
    sources: list[Passage]


def reference(row: dict[str, Any]) -> str:
    doc = DOC_NAMES.get(row["doc_short"], row["doc_short"])
    if row.get("article_no") is None:
        return f"Phụ lục {row.get('appendix_no')} {doc}"
    clauses = row.get("clause_nos") or []
    suffix = f" (khoản {', '.join(clauses)})" if clauses else ""
    return f"Điều {row['article_no']}{suffix} {doc}"


def passage(row: dict[str, Any], score: float | None) -> Passage:
    return Passage(
        chunk_id=row["chunk_id"],
        reference=reference(row),
        title=row.get("article_title"),
        score=round(score, 3) if score is not None else None,
        text=row["content"],
        source_url=row["source_url"],
        amended_by=row.get("amended_by") or [],
        upcoming_change=row.get("upcoming_change") or [],
    )


class Backend:
    """Retriever, answer graph and article index: opened lazily, or injected by tests."""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: Any = None,
        qa: Any = None,
        articles: dict[tuple[str, int], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._retriever, self._qa, self._articles = retriever, qa, articles
        self._lock = threading.Lock()  # sync tools run on worker threads
        self._stack = ExitStack()

    def retriever(self) -> Any:
        with self._lock:
            if self._retriever is None:
                from vnlaw_rag.retrieval.factory import open_retriever

                self._retriever = self._stack.enter_context(open_retriever(self.settings))
            return self._retriever

    def qa(self) -> Any:
        retriever = self.retriever()
        with self._lock:
            if self._qa is None:
                from vnlaw_rag.graph.build import build_graph, make_nodes
                from vnlaw_rag.llm.gemini import client_from_settings

                llm = client_from_settings(self.settings)
                self._qa = build_graph(make_nodes(self.settings, retriever, llm))
            return self._qa

    def articles(self) -> dict[tuple[str, int], dict[str, Any]]:
        with self._lock:
            if self._articles is None:
                self._articles = load_articles(self.settings)
            return self._articles

    def close(self) -> None:
        self._stack.close()


def build_server(backend: Backend | None = None) -> MCPServer:
    backend = backend or Backend()

    @asynccontextmanager
    async def lifespan(_server: MCPServer):
        try:
            yield {}
        finally:
            backend.close()

    mcp = MCPServer("vn-labor-law", instructions=INSTRUCTIONS, lifespan=lifespan)

    @mcp.tool()
    def search_labor_law(
        query: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)],
        top_k: Annotated[int, Field(ge=1, le=10)] = 6,
    ) -> list[Passage]:
        """Find the articles of Vietnamese labor law (Bộ luật Lao động 2019, Nghị định 145/2020)
        most relevant to a question in Vietnamese. Returns passages with their article
        reference, similarity score, official link and amendment notes. No LLM call."""
        results = backend.retriever().search(query, top_k=top_k)
        return [passage(r.row, r.vector_score) for r in results]

    @mcp.tool()
    def get_labor_law_article(
        doc: Literal["BLLD2019", "ND145"], article_no: Annotated[int, Field(ge=1, le=300)]
    ) -> ArticleText:
        """Full text of one article: doc 'BLLD2019' (Bộ luật Lao động 2019, consolidated
        text) or 'ND145' (Nghị định 145/2020/NĐ-CP), with its amendment notes."""
        article = backend.articles().get((doc, article_no))
        if article is None:
            raise ToolError(f"{DOC_NAMES[doc]} không có Điều {article_no}.")
        return ArticleText(
            reference=f"Điều {article_no} {DOC_NAMES[doc]}",
            heading=article["heading"],
            text=article["text"],
            source_url=article["source_url"],
            amended_by=article["amended_by"],
            upcoming_change=article["upcoming_change"],
        )

    @mcp.tool()
    def ask_labor_law(
        question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)],
    ) -> Answer:
        """Answer a question about Vietnamese labor law in Vietnamese, citing articles and
        clauses. Calls the Gemini API (free tier) once or twice. Returns status 'not_found'
        when the corpus has no relevant provision. Not legal advice."""
        try:
            state = backend.qa().invoke({"question": question})
        except LLMError as exc:
            log.warning("ask_labor_law failed: %s", exc)
            raise ToolError(friendly_error(exc)) from exc
        cited = set(state.get("citations", []))
        return Answer(
            status=state.get("status", "not_found"),
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
            warnings=state.get("warnings", []),
            notice=state.get("notice"),
            model=state.get("model"),
            sources=[
                passage(s, s.get("vector_score"))
                for s in state.get("sources", [])
                if s["chunk_id"] in cited
            ],
        )

    return mcp
