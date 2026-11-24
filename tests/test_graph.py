"""LangGraph flow with a fake retriever and a fake LLM (no database, no API)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.conftest import REPO_ROOT, parsed, word_count
from vnlaw_rag.graph.build import build_graph
from vnlaw_rag.graph.nodes import NOT_FOUND_ANSWER, QANodes
from vnlaw_rag.graph.prompts import GeneratedAnswer
from vnlaw_rag.graph.references import ArticleRef, find_references, fold
from vnlaw_rag.ingest.amendments import annotate_chunks, load_registry
from vnlaw_rag.ingest.chunk import build_chunks
from vnlaw_rag.llm.base import StructuredResult
from vnlaw_rag.retrieval.hybrid import RetrievedChunk

NOTICE = "Theo VBHN 125/VBHN-VPQH (2025); một số quy định sẽ thay đổi từ 01/03/2027."


@pytest.fixture(scope="module")
def rows() -> dict[str, dict[str, Any]]:
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

    def generate_structured(self, *, system, prompt, schema):
        assert schema is GeneratedAnswer
        self.calls.append({"system": system, "prompt": prompt})
        return StructuredResult(self.answer, model="fake-model")


def run(rows, ranked, answer, question="câu hỏi", min_score=0.55):
    llm = FakeLLM(answer)
    retriever = FakeRetriever(rows, ranked)
    nodes = QANodes(retriever=retriever, llm=llm, min_score=min_score, max_sources=6,
                    notice=NOTICE)  # fmt: skip
    return build_graph(nodes).invoke({"question": question}), llm, retriever


def test_answer_keeps_only_citations_that_were_retrieved(rows):
    answer = GeneratedAnswer(
        answer="Không quá 60 ngày (khoản 2 Điều 25 Bộ luật Lao động) [BLLD2019-D25]. "
        "Xem thêm [BLLD2019-D999].",
        citations=["BLLD2019-D25", "BLLD2019-D999", "ND145-D1"],
        sufficient=True,
    )
    out, llm, _ = run(rows, [("BLLD2019-D25", 0.74), ("BLLD2019-D24", 0.60)], answer)
    assert out["status"] == "answered"
    assert out["citations"] == ["BLLD2019-D25"]
    assert out["dropped_citations"] == ["BLLD2019-D999", "ND145-D1"]
    assert "[BLLD2019-D999]" not in out["answer"] and "[BLLD2019-D25]" in out["answer"]
    assert out["model"] == "fake-model"
    assert out["notice"] == NOTICE
    assert set(out["timings_ms"]) == {"route", "retrieve", "grade", "generate", "finalize"}
    prompt = llm.calls[0]["prompt"]
    assert "[BLLD2019-D25] Bộ luật Lao động 2019 - Điều 25. Thời gian thử việc" in prompt


def test_public_holiday_answer_warns_about_the_change_on_2027_03_01(rows):
    answer = GeneratedAnswer(
        answer="Tết Âm lịch được nghỉ 05 ngày (điểm b khoản 1 Điều 112) [BLLD2019-D112].",
        citations=["BLLD2019-D112"],
        sufficient=True,
    )
    out, llm, _ = run(rows, [("BLLD2019-D112", 0.78)], answer, "Nghỉ Tết mấy ngày?")
    assert out["status"] == "answered"
    (warning,) = out["warnings"]
    assert warning.startswith("Điều 112 Bộ luật Lao động 2019: Sẽ thay đổi từ 01/03/2027")
    assert "27/7" in warning and "24/11" in warning
    # the model was told about the change too
    assert "Lưu ý thay đổi: Sẽ thay đổi từ 01/03/2027" in llm.calls[0]["prompt"]


def test_amended_article_warning_names_the_amending_law(rows):
    answer = GeneratedAnswer(answer="… [BLLD2019-D154]", citations=[], sufficient=True)
    out, _, _ = run(rows, [("BLLD2019-D154", 0.70)], answer)
    (warning,) = out["warnings"]
    assert "Khoản 8a" in warning and "71/2025/QH15" in warning and "01/01/2026" in warning


def test_low_similarity_goes_to_not_found_without_calling_the_llm(rows):
    answer = GeneratedAnswer(answer="x", citations=["BLLD2019-D25"], sufficient=True)
    out, llm, _ = run(rows, [("BLLD2019-D25", 0.50)], answer, min_score=0.55)
    assert out["status"] == "not_found"
    assert out["answer"] == NOT_FOUND_ANSWER
    assert llm.calls == []
    assert out["top_score"] == pytest.approx(0.50)


def test_insufficient_or_uncited_answers_become_not_found(rows):
    for answer in (
        GeneratedAnswer(answer="Không đủ thông tin.", citations=[], sufficient=False),
        GeneratedAnswer(answer="Có, được phép.", citations=["BLLD2019-D999"], sufficient=True),
    ):
        out, _, _ = run(rows, [("BLLD2019-D25", 0.80)], answer)
        assert out["status"] == "not_found"
        assert out["answer"] == NOT_FOUND_ANSWER
        assert out["citations"] == []


def test_explicit_article_reference_is_looked_up_directly(rows):
    answer = GeneratedAnswer(answer="… [ND145-D21]", citations=["ND145-D21"], sufficient=True)
    out, llm, retriever = run(rows, [], answer, "Điều 21 Nghị định 145 quy định gì?")
    assert out["route"] == "lookup"
    assert [s["chunk_id"] for s in out["sources"]] == ["ND145-D21"]
    assert retriever.queries == []  # no vector search
    assert out["citations"] == ["ND145-D21"]
    assert "lookup" in out["timings_ms"]
    assert out["warnings"]  # labor-leasing license: upcoming change


def test_lookup_of_a_missing_article_is_not_found(rows):
    answer = GeneratedAnswer(answer="x", citations=[], sufficient=True)
    out, llm, _ = run(rows, [], answer, "Điều 999 BLLĐ")
    assert out["status"] == "not_found" and llm.calls == []


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Điều 35 quy định gì?", [ArticleRef("BLLD2019", 35)]),
        ("khoản 2 Điều 113 BLLĐ", [ArticleRef("BLLD2019", 113, "2")]),
        ("dieu 21 nghi dinh 145 noi gi", [ArticleRef("ND145", 21)]),
        (
            "Theo Nghị định 145/2020, Điều 7 và Điều 8",
            [ArticleRef("ND145", 7), ArticleRef("ND145", 8)],
        ),
        ("Điều kiện hưởng lương hưu là gì?", []),
        ("Bộ luật Lao động và Nghị định 145: Điều 54", [ArticleRef("BLLD2019", 54)]),
    ],
)
def test_find_references(question, expected):
    assert find_references(question) == expected


def test_fold_strips_vietnamese_diacritics():
    assert fold("Điều Khoản Nghị định") == "dieu khoan nghi dinh"
