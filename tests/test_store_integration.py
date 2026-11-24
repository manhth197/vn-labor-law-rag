"""Integration tests against a real PostgreSQL + pgvector (skipped when none is reachable).

They use a separate database (`vnlaw_test`, created if missing) so they never touch the
real index. Set TEST_DATABASE_URL to point elsewhere (CI uses a service container).
"""

from __future__ import annotations

import os

import numpy as np
import psycopg
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from psycopg import sql

from tests.conftest import word_count
from vnlaw_rag.config import Settings
from vnlaw_rag.index.store import PgStore, init_database
from vnlaw_rag.ingest.chunk import build_chunks
from vnlaw_rag.retrieval.hybrid import HybridRetriever
from vnlaw_rag.retrieval.keyword import query_terms

pytestmark = pytest.mark.integration
DIM = 1024


def _test_url() -> str:
    if url := os.environ.get("TEST_DATABASE_URL"):
        return url
    base = Settings().database_url
    return base.rsplit("/", 1)[0] + "/vnlaw_test"


@pytest.fixture(scope="module")
def store():
    url = _test_url()
    admin = url.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (url.rsplit("/", 1)[1],)
            ).fetchone()
            if not exists:
                conn.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(url.rsplit("/", 1)[1])))  # fmt: skip
    except psycopg.OperationalError as exc:
        pytest.skip(f"no PostgreSQL reachable for integration tests: {exc}")
    init_database(url, DIM, reset=True)
    s = PgStore.open(url)
    yield s
    s.close()


@pytest.fixture(scope="module")
def loaded(store):
    from tests.conftest import parsed

    docs = [parsed("BLLD2019")]
    articles = [a for a in docs[0].articles if 20 <= a.article_no <= 40]
    chunks = build_chunks(articles, [], word_count, max_tokens=1200)
    embedder = DeterministicFakeEmbedding(size=DIM)
    vectors = np.asarray(embedder.embed_documents([c.text for c in chunks]), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store.replace_chunks(chunks, vectors, {"test": True})
    return chunks, vectors


def test_replace_is_a_full_rebuild(store, loaded):
    chunks, vectors = loaded
    assert store.count() == len(chunks)
    store.replace_chunks(chunks, vectors, {"test": True})
    assert store.count() == len(chunks)
    assert store.index_info()["test"] is True


def test_vector_search_finds_identical_vector_first(store, loaded):
    chunks, vectors = loaded
    hits = store.vector_search(vectors[5], k=3)
    assert hits[0].chunk_id == chunks[5].chunk_id
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert [h.rank for h in hits] == [1, 2, 3]


@pytest.mark.parametrize("scoring", ["idf", "ts_rank_cd"])
def test_keyword_search_is_accent_insensitive(store, loaded, scoring):
    accented = store.keyword_search(query_terms("thời gian thử việc"), k=5, scoring=scoring)
    plain = store.keyword_search(query_terms("thoi gian thu viec"), k=5, scoring=scoring)
    assert accented == plain
    assert "BLLD2019-D25" in [h.chunk_id for h in accented[:3]]


def test_idf_scoring_weights_rare_terms_above_common_ones(store, loaded):
    # "lao động" occurs in nearly every chunk, "thử việc" in a few: every chunk with the rare
    # term must outrank every chunk that only has the common one.
    rare = {h.chunk_id for h in store.keyword_search(["thử <-> việc"], k=100)}
    common = {h.chunk_id for h in store.keyword_search(["lao <-> động"], k=100)}
    assert 0 < len(rare) < len(common)
    hits = store.keyword_search(["lao <-> động", "thử <-> việc"], k=100, scoring="idf")
    with_rare = [h.score for h in hits if h.chunk_id in rare]
    common_only = [h.score for h in hits if h.chunk_id not in rare]
    assert min(with_rare) > max(common_only)


def test_keyword_search_with_no_terms_returns_nothing(store, loaded):
    assert store.keyword_search(query_terms("?!"), k=5) == []


def test_get_chunks_returns_metadata_and_content(store, loaded):
    row = store.get_chunks(["BLLD2019-D25"])["BLLD2019-D25"]
    assert row["article_title"] == "Thời gian thử việc"
    assert row["clause_nos"] == []
    assert row["amended_by"] == [] and row["upcoming_change"] == []
    assert "Không quá 60 ngày" in row["content"]


def test_hybrid_retriever_end_to_end(store, loaded):
    retriever = HybridRetriever(store=store, embeddings=DeterministicFakeEmbedding(size=DIM))
    docs = retriever.invoke("tiền lương thử việc")
    assert docs[0].metadata["article_no"] in (24, 25, 26, 27)
    assert docs[0].metadata["keyword_rank"] is not None
