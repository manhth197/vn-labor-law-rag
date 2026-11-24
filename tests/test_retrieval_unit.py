"""Keyword query building, RRF and the hybrid retriever with an in-memory fake store."""

from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from vnlaw_rag.index.store import SearchHit
from vnlaw_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from vnlaw_rag.retrieval.keyword import build_tsquery, query_terms, syllables


def test_syllables_lowercase_and_drop_punctuation():
    assert syllables("Thời gian thử việc (Điều 25)?") == [
        "thời",
        "gian",
        "thử",
        "việc",
        "điều",
        "25",
    ]


def test_bigram_query_uses_phrase_operator_and_or():
    assert build_tsquery("Thử việc bao lâu") == "thử <-> việc | việc <-> bao | bao <-> lâu"


def test_unigram_query_dedupes():
    assert build_tsquery("lao động, người lao động", mode="unigram") == "lao | động | người"


def test_query_cannot_inject_tsquery_operators():
    q = build_tsquery("a & b | !c <-> (d) : * '_'", mode="unigram")
    assert q == "a | b | c | d"
    assert build_tsquery("?!") == ""


def test_stopwords_drop_terms_with_function_words():
    question = "Tôi có được nghỉ thai sản bao nhiêu tháng không?"
    assert (
        query_terms(question, drop_stopwords=True)
        == [
            "nghỉ <-> thai",
            "thai <-> sản",
            "sản <-> bao",
        ][:2]
    )
    assert query_terms(question, mode="unigram", drop_stopwords=True) == [
        "nghỉ",
        "thai",
        "sản",
        "tháng",
    ]


def test_single_word_falls_back_to_unigram():
    assert build_tsquery("lương") == "lương"


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_tsquery("a b", mode="trigram")  # type: ignore[arg-type]


def test_rrf_rewards_agreement_and_is_deterministic():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    assert [doc for doc, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    # ties are broken by id
    assert [d for d, _ in reciprocal_rank_fusion([["x"], ["y"]])] == ["x", "y"]


class FakeStore:
    """vector_search ranks by a fixed list; keyword_search returns a fixed list."""

    def __init__(self, vector: list[str], keyword: list[str]):
        self.vector, self.keyword = vector, keyword
        self.tsqueries: list[str] = []

    def vector_search(self, embedding, k):
        return [SearchHit(cid, 1.0 - i / 10, i + 1) for i, cid in enumerate(self.vector[:k])]

    def keyword_search(self, terms, k, scoring="idf"):
        self.tsqueries.append(" | ".join(terms))
        return [SearchHit(cid, 0.5 - i / 10, i + 1) for i, cid in enumerate(self.keyword[:k])]

    def get_chunks(self, ids):
        return {
            cid: {"chunk_id": cid, "doc_short": "BLLD2019", "content": f"text {cid}"} for cid in ids
        }


def make_retriever(store: FakeStore, **kw) -> HybridRetriever:
    return HybridRetriever(store=store, embeddings=DeterministicFakeEmbedding(size=8), **kw)


def test_hybrid_mode_fuses_both_legs():
    store = FakeStore(vector=["a", "b", "c"], keyword=["c", "a"])
    results = make_retriever(store, top_k=3).search("thử việc")
    assert [r.chunk_id for r in results] == ["a", "c", "b"]
    a = results[0]
    assert (a.vector_rank, a.keyword_rank) == (1, 2)
    assert store.tsqueries == ["thử <-> việc"]


@pytest.mark.parametrize(
    ("mode", "expected"), [("vector", ["a", "b", "c"]), ("keyword", ["c", "a"])]
)
def test_single_leg_modes(mode, expected):
    store = FakeStore(vector=["a", "b", "c"], keyword=["c", "a"])
    assert [r.chunk_id for r in make_retriever(store, mode=mode).search("q")] == expected


def test_langchain_invoke_returns_documents_with_metadata():
    store = FakeStore(vector=["a"], keyword=[])
    (doc,) = make_retriever(store).invoke("câu hỏi")
    assert doc.id == "a"
    assert doc.page_content == "text a"
    assert doc.metadata["vector_rank"] == 1
    assert doc.metadata["keyword_rank"] is None
    assert "content" not in doc.metadata
