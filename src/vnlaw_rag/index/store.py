"""PostgreSQL + pgvector storage for chunks: schema, bulk load, vector and keyword search.

Plain SQL through psycopg 3 (sync) and a psycopg_pool connection pool. Every value is passed
as a query parameter; the only composed SQL is the vector dimension, an int from config.
See docs/decisions/004-own-pgvector-sql-instead-of-langchain-postgres.md.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from vnlaw_rag.ingest.models import Chunk

# Full-text search config: the `simple` parser/dictionary (no stemming or stop words, which
# do not exist for Vietnamese in Postgres) with `unaccent` first, so "nghi phep" matches
# "nghỉ phép". Generated tsvector columns accept it because to_tsvector(regconfig, text) is
# immutable when the config is a constant.
TS_CONFIG = "vn_unaccent"

_PREPARE = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS unaccent",
    f"""
    DO $$ BEGIN
      IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = '{TS_CONFIG}') THEN
        CREATE TEXT SEARCH CONFIGURATION {TS_CONFIG} (COPY = simple);
        ALTER TEXT SEARCH CONFIGURATION {TS_CONFIG}
          ALTER MAPPING FOR hword, hword_part, word WITH unaccent, simple;
      END IF;
    END $$
    """,
]

_CHUNKS_TABLE = """
    CREATE TABLE IF NOT EXISTS chunks (
      chunk_id      text PRIMARY KEY,
      doc_short     text NOT NULL,
      doc_number    text NOT NULL,
      doc_title     text NOT NULL,
      unit          text NOT NULL,
      chapter_no    text,
      chapter_title text,
      section_no    int,
      section_title text,
      article_no    int,
      article_title text,
      clause_nos    text[] NOT NULL DEFAULT ARRAY[]::text[],  -- labels: '2', '8a'
      appendix_no   text,
      source_url    text NOT NULL,
      content       text NOT NULL,
      n_tokens      int NOT NULL,
      amended_by    text[] NOT NULL DEFAULT ARRAY[]::text[],  -- see ingest/amendments.py
      upcoming_change text[] NOT NULL DEFAULT ARRAY[]::text[],
      embedding     vector({dim}) NOT NULL,
      content_tsv   tsvector GENERATED ALWAYS AS
                    (to_tsvector('vn_unaccent'::regconfig, content)) STORED
    )
"""

_FINISH = [
    # HNSW with cosine distance; vectors are normalized, so cosine = inner product ranking.
    "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks "
    "USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin ON chunks USING gin (content_tsv)",
    "CREATE INDEX IF NOT EXISTS chunks_article ON chunks (doc_short, article_no)",
    "CREATE TABLE IF NOT EXISTS index_info (key text PRIMARY KEY, value jsonb NOT NULL)",
]

CHUNK_COLUMNS = (
    "chunk_id",
    "doc_short",
    "doc_number",
    "doc_title",
    "unit",
    "chapter_no",
    "chapter_title",
    "section_no",
    "section_title",
    "article_no",
    "article_title",
    "clause_nos",
    "appendix_no",
    "source_url",
    "n_tokens",
    "amended_by",
    "upcoming_change",
)


KeywordScoring = Literal["idf", "ts_rank_cd"]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float  # cosine similarity (vector) or ts_rank_cd (keyword); higher is better
    rank: int  # 1-based


def init_database(database_url: str, dim: int, reset: bool = False) -> None:
    """Create extensions, the text search config, tables and indexes (idempotent).
    `reset` drops the chunks table first: a full re-index then also picks up schema changes."""
    with psycopg.connect(database_url, autocommit=True) as conn:
        for statement in _PREPARE:
            conn.execute(statement)
        if reset:
            conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(sql.SQL(_CHUNKS_TABLE).format(dim=sql.Literal(int(dim))))
        for statement in _FINISH:
            conn.execute(statement)


class PgStore:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    @classmethod
    def open(cls, database_url: str, *, min_size: int = 1, max_size: int = 4) -> PgStore:
        # Autocommit: reads need no transaction, and `configure` must not leave one open
        # (the pool discards connections returned in INTRANS state). Writes use an explicit
        # `conn.transaction()` block.
        pool = ConnectionPool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"autocommit": True},
            configure=register_vector,
            open=True,
        )
        pool.wait()
        return cls(pool)

    def close(self) -> None:
        self.pool.close()

    # -- writes ------------------------------------------------------------------------
    def replace_chunks(
        self, chunks: Sequence[Chunk], embeddings: np.ndarray, info: dict[str, Any]
    ) -> int:
        """Replace the whole index in one transaction (348 rows: a full rebuild is simplest
        and keeps the table identical to chunks.jsonl)."""
        columns = sql.SQL(", ").join(
            sql.Identifier(c) for c in (*CHUNK_COLUMNS, "content", "embedding")
        )
        placeholders = sql.SQL(", ").join(sql.Placeholder() * (len(CHUNK_COLUMNS) + 2))
        insert = sql.SQL("INSERT INTO chunks ({}) VALUES ({})").format(columns, placeholders)
        rows = [
            (*(getattr(c, col) for col in CHUNK_COLUMNS), c.text, vec)
            for c, vec in zip(chunks, np.asarray(embeddings, dtype=np.float32), strict=True)
        ]
        with self.pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")
            cur.executemany(insert, rows)
            for key, value in info.items():
                cur.execute(
                    "INSERT INTO index_info (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (key, json.dumps(value, ensure_ascii=False)),
                )
        return len(rows)

    # -- reads -------------------------------------------------------------------------
    def count(self) -> int:
        with self.pool.connection() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def index_info(self) -> dict[str, Any]:
        with self.pool.connection() as conn:
            return dict(conn.execute("SELECT key, value FROM index_info").fetchall())

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        columns = sql.SQL(", ").join(sql.Identifier(c) for c in (*CHUNK_COLUMNS, "content"))
        query = sql.SQL("SELECT {} FROM chunks WHERE chunk_id = ANY(%s)").format(columns)
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(query, (list(chunk_ids),)).fetchall()
        return {r["chunk_id"]: r for r in rows}

    def get_article_chunks(self, doc_short: str, article_no: int) -> list[dict[str, Any]]:
        """All chunks of one article, in text order (for explicit "Điều N" questions)."""
        columns = sql.SQL(", ").join(sql.Identifier(c) for c in (*CHUNK_COLUMNS, "content"))
        query = sql.SQL(
            "SELECT {} FROM chunks WHERE doc_short = %s AND article_no = %s ORDER BY chunk_id"
        ).format(columns)
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(query, (doc_short, article_no)).fetchall()
        # chunk ids sort as text ("-K10" < "-K2"); restore clause order from the labels
        return sorted(rows, key=lambda r: _clause_order(r["clause_nos"]))

    def vector_search(self, embedding: Sequence[float], k: int) -> list[SearchHit]:
        query_vec = np.asarray(embedding, dtype=np.float32)
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT chunk_id, 1 - (embedding <=> %(q)s) AS score FROM chunks "
                "ORDER BY embedding <=> %(q)s, chunk_id LIMIT %(k)s",
                {"q": query_vec, "k": k},
            ).fetchall()
        return [SearchHit(cid, float(score), i) for i, (cid, score) in enumerate(rows, 1)]

    def keyword_search(
        self, terms: Sequence[str], k: int, scoring: KeywordScoring = "idf"
    ) -> list[SearchHit]:
        """Full-text search for chunks matching any of `terms` (tsquery fragments built by
        retrieval.keyword, e.g. "thu <-> viec").

        - "ts_rank_cd": Postgres cover density over the OR-ed query. It has no notion of how
          rare a term is, so common phrases ("người lao động") dominate.
        - "idf" (default): score = sum of BM25-style IDF, ln(1 + (N - df + 0.5) / (df + 0.5)),
          over the distinct terms a chunk contains (binary term frequency). Document
          frequencies come from the GIN index at query time; rare, specific terms count most.
        """
        if not terms:
            return []
        params = {"terms": list(terms), "k": k}
        if scoring == "ts_rank_cd":
            query = (
                "SELECT chunk_id, ts_rank_cd(content_tsv, q) AS score "
                "FROM chunks, to_tsquery('vn_unaccent'::regconfig, "
                "array_to_string(%(terms)s::text[], ' | ')) AS q "
                "WHERE content_tsv @@ q ORDER BY score DESC, chunk_id LIMIT %(k)s"
            )
        elif scoring == "idf":
            query = """
                WITH terms AS (
                  SELECT DISTINCT to_tsquery('vn_unaccent'::regconfig, t) AS q
                  FROM unnest(%(terms)s::text[]) AS t
                ),
                df AS (
                  SELECT q, (SELECT count(*) FROM chunks c WHERE c.content_tsv @@ terms.q) AS n
                  FROM terms
                ),
                weights AS (
                  SELECT q, ln(1 + ((SELECT count(*) FROM chunks) - n + 0.5) / (n + 0.5)) AS idf
                  FROM df WHERE n > 0
                )
                SELECT c.chunk_id, sum(w.idf) AS score
                FROM chunks c JOIN weights w ON c.content_tsv @@ w.q
                GROUP BY c.chunk_id
                ORDER BY score DESC, c.chunk_id
                LIMIT %(k)s
            """
        else:
            raise ValueError(f"unknown keyword scoring {scoring!r}")
        with self.pool.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [SearchHit(cid, float(score), i) for i, (cid, score) in enumerate(rows, 1)]


def _clause_order(labels: Sequence[str]) -> tuple[int, str]:
    if not labels:
        return (0, "")
    first = labels[0]
    digits = "".join(ch for ch in first if ch.isdigit())
    return (int(digits or 0), first[len(digits) :])
