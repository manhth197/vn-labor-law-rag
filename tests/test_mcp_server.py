"""MCP server: tools listed and called through an in-memory MCP client, with fakes (no
bge-m3, no database, no Gemini)."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from tests.fakes import NOTICE, FakeLLM, FakeRetriever  # noqa: E402
from vnlaw_rag.api.main import load_articles  # noqa: E402
from vnlaw_rag.config import Settings  # noqa: E402
from vnlaw_rag.graph.build import build_graph  # noqa: E402
from vnlaw_rag.graph.nodes import QANodes  # noqa: E402
from vnlaw_rag.graph.prompts import GeneratedAnswer  # noqa: E402
from vnlaw_rag.llm.base import RateLimited  # noqa: E402
from vnlaw_rag.mcp_server import Backend, build_server  # noqa: E402

HOLIDAY = GeneratedAnswer(
    answer="Tết Âm lịch được nghỉ 05 ngày (điểm b khoản 1 Điều 112) [BLLD2019-D112] [FAKE-ID].",
    citations=["BLLD2019-D112", "FAKE-ID"],
    sufficient=True,
)


class LimitedLLM:
    def generate_structured(self, **_):
        raise RateLimited("429 from the fake")


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def articles():
    return load_articles(Settings())


def server_for(rows, articles, llm=None, ranked=(("BLLD2019-D112", 0.78), ("BLLD2019-D113", 0.7))):
    retriever = FakeRetriever(rows, list(ranked))
    nodes = QANodes(retriever=retriever, llm=llm or FakeLLM(HOLIDAY), min_score=0.5,
                    max_sources=6, notice=NOTICE)  # fmt: skip
    backend = Backend(Settings(), retriever=retriever, qa=build_graph(nodes), articles=articles)
    return build_server(backend), retriever


@pytest.mark.anyio
async def test_lists_the_three_tools_with_input_schemas(rows, articles):
    server, _ = server_for(rows, articles)
    async with Client(server, raise_exceptions=True) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"search_labor_law", "get_labor_law_article", "ask_labor_law"}
    search = tools["search_labor_law"].input_schema
    assert search["required"] == ["query"]
    assert search["properties"]["top_k"]["maximum"] == 10
    assert tools["get_labor_law_article"].input_schema["properties"]["doc"]["enum"] == [
        "BLLD2019", "ND145",
    ]  # fmt: skip
    assert "Not legal advice" in tools["ask_labor_law"].description


@pytest.mark.anyio
async def test_search_returns_passages_without_calling_the_llm(rows, articles):
    llm = FakeLLM(HOLIDAY)
    server, retriever = server_for(rows, articles, llm=llm)
    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool("search_labor_law", {"query": "nghỉ Tết", "top_k": 1})
    assert not result.is_error
    hits = result.structured_content["result"]
    assert [h["chunk_id"] for h in hits] == ["BLLD2019-D112"]
    assert hits[0]["reference"].startswith("Điều 112") and hits[0]["score"] == 0.78
    assert any("01/03/2027" in note for note in hits[0]["upcoming_change"])
    assert retriever.queries == ["nghỉ Tết"] and llm.calls == []


@pytest.mark.anyio
async def test_article_lookup_and_unknown_article(rows, articles):
    server, _ = server_for(rows, articles)
    async with Client(server, raise_exceptions=True) as client:
        ok = await client.call_tool("get_labor_law_article", {"doc": "BLLD2019", "article_no": 25})
        missing = await client.call_tool(
            "get_labor_law_article", {"doc": "ND145", "article_no": 200}
        )
    assert ok.structured_content["heading"].startswith("Điều 25. Thời gian thử việc")
    assert "180 ngày" in ok.structured_content["text"]
    assert missing.is_error and "không có Điều 200" in missing.content[0].text


@pytest.mark.anyio
async def test_ask_returns_verified_citations_warnings_and_notice(rows, articles):
    server, _ = server_for(rows, articles)
    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool("ask_labor_law", {"question": "Nghỉ Tết mấy ngày?"})
    answer = result.structured_content
    assert answer["status"] == "answered" and answer["citations"] == ["BLLD2019-D112"]
    assert "FAKE-ID" not in answer["answer"]  # dropped: not among the retrieved chunks
    assert [s["chunk_id"] for s in answer["sources"]] == ["BLLD2019-D112"]
    assert any("01/03/2027" in w for w in answer["warnings"])
    assert answer["notice"] == NOTICE and answer["model"] == "fake-model"


@pytest.mark.anyio
async def test_llm_failures_become_a_friendly_tool_error(rows, articles):
    server, _ = server_for(rows, articles, llm=LimitedLLM())
    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool("ask_labor_law", {"question": "Nghỉ Tết mấy ngày?"})
        empty = await client.call_tool("search_labor_law", {"query": ""})
    assert result.is_error and "giới hạn miễn phí" in result.content[0].text
    assert empty.is_error  # argument validation (min_length=1)
