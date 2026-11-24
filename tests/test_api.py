"""FastAPI endpoints with the real graph code, a fake retriever and a fake LLM."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.fakes import NOTICE, FakeLLM, FakeRetriever
from vnlaw_rag.api.main import create_app, load_articles
from vnlaw_rag.config import Settings
from vnlaw_rag.graph.build import build_graph
from vnlaw_rag.graph.nodes import QANodes
from vnlaw_rag.graph.prompts import GeneratedAnswer
from vnlaw_rag.llm.base import Unavailable

HOLIDAY = GeneratedAnswer(
    answer="Tết Âm lịch được nghỉ 05 ngày (điểm b khoản 1 Điều 112) [BLLD2019-D112].",
    citations=["BLLD2019-D112"],
    sufficient=True,
)


class FailingLLM:
    def generate_structured(self, **_):
        raise Unavailable("no answer within 60 s")


@pytest.fixture(scope="module")
def articles():
    return load_articles(Settings())


def client_for(rows, articles, llm, ranked=(("BLLD2019-D112", 0.78),)):
    nodes = QANodes(retriever=FakeRetriever(rows, list(ranked)), llm=llm, min_score=0.55,
                    max_sources=6, notice=NOTICE)  # fmt: skip
    app = create_app(
        Settings(), qa=build_graph(nodes), articles=articles,
        health={"model": "fake-model", "notice": NOTICE},
    )  # fmt: skip
    return TestClient(app)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health(rows, articles):
    with client_for(rows, articles, FakeLLM(HOLIDAY)) as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "model": "fake-model", "notice": NOTICE}


def test_ask_returns_answer_citations_warnings_and_timings(rows, articles):
    with client_for(rows, articles, FakeLLM(HOLIDAY)) as client:
        response = client.post("/api/ask", json={"question": "Nghỉ Tết mấy ngày?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["citations"] == ["BLLD2019-D112"]
    assert "01/03/2027" in body["warnings"][0]
    assert body["model"] == "fake-model" and body["notice"] == NOTICE
    assert body["sources"][0]["chunk_id"] == "BLLD2019-D112"
    assert "content" in body["sources"][0] and "prompt" not in body
    assert body["timings_ms"]["total"] >= 0 and "generate" in body["timings_ms"]


@pytest.mark.parametrize(
    "payload", [{"question": ""}, {"question": "x" * 501}, {"question": "a", "top_k": 50}]
)
def test_ask_validates_input(rows, articles, payload):
    with client_for(rows, articles, FakeLLM(HOLIDAY)) as client:
        assert client.post("/api/ask", json=payload).status_code == 422


def test_ask_turns_llm_failures_into_a_friendly_503(rows, articles):
    with client_for(rows, articles, FailingLLM()) as client:
        response = client.post("/api/ask", json={"question": "Nghỉ Tết mấy ngày?"})
    assert response.status_code == 503
    assert response.json()["detail"].startswith("Dịch vụ trả lời đang quá tải")


def test_stream_sends_progress_tokens_then_result(rows, articles):
    with client_for(rows, articles, FakeLLM(HOLIDAY)) as client:
        response = client.get("/api/ask/stream", params={"question": "Nghỉ Tết mấy ngày?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    kinds = [kind for kind, _ in events]
    assert kinds[:3] == ["progress", "progress", "progress"]
    assert [d["stage"] for k, d in events if k == "progress"] == [
        "searching",
        "found",
        "generating",
    ]
    assert "token" in kinds and kinds[-1] == "result"
    streamed = "".join(d["text"] for k, d in events if k == "token")
    assert streamed == HOLIDAY.answer
    result = events[-1][1]
    assert result["status"] == "answered" and result["warnings"]


def test_stream_reports_errors_as_an_event(rows, articles):
    with client_for(rows, articles, FailingLLM()) as client:
        response = client.get("/api/ask/stream", params={"question": "Nghỉ Tết mấy ngày?"})
    kind, data = parse_sse(response.text)[-1]
    assert kind == "error" and "quá tải" in data["message"]


def test_not_found_question_skips_the_llm(rows, articles):
    llm = FakeLLM(HOLIDAY)
    with client_for(rows, articles, llm, ranked=[("BLLD2019-D112", 0.40)]) as client:
        body = client.post("/api/ask", json={"question": "Lãi suất vay mua nhà?"}).json()
    assert body["status"] == "not_found" and body["model"] is None and llm.calls == []


def test_article_endpoint_includes_amendment_notes(rows, articles):
    with client_for(rows, articles, FakeLLM(HOLIDAY)) as client:
        art = client.get("/api/articles/blld2019/154").json()
        missing = client.get("/api/articles/BLLD2019/999")
    assert art["heading"].startswith("Điều 154.")
    assert [c["no"] for c in art["clauses"]][-2:] == [8, 9] and art["clauses"][8]["suffix"] == "a"
    assert "71/2025/QH15" in art["amended_by"][0]
    assert missing.status_code == 404


def test_ui_is_served(rows, articles):
    with client_for(rows, articles, FakeLLM(HOLIDAY)) as client:
        page = client.get("/")
        script = client.get("/static/app.js")
    assert page.status_code == 200 and "Không phải tư vấn pháp lý" in page.text
    assert script.status_code == 200 and ".innerHTML" not in script.text
