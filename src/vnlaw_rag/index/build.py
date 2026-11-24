"""Build the pgvector index from data/processed/chunks.jsonl (one command, full rebuild).

Embedding 348 chunks with bge-m3 on CPU takes about 10 minutes, so vectors are cached in
data/cache/ keyed by the chunks file hash and the model name; re-indexing after a schema
change reuses them. The database is prepared first so a connection problem fails fast.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from vnlaw_rag.config import Settings
from vnlaw_rag.index.embed import load_embeddings
from vnlaw_rag.index.store import PgStore, init_database
from vnlaw_rag.ingest.models import Chunk

log = logging.getLogger(__name__)


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as fh:
        return [Chunk.model_validate_json(row) for row in fh if row.strip()]


def embedding_cache_path(settings: Settings, chunks_sha256: str) -> Path:
    model = re.sub(r"[^A-Za-z0-9]+", "-", settings.embedding_model).strip("-")
    return settings.data_dir / "cache" / f"embeddings-{model}-{chunks_sha256[:16]}.npy"


def build_index(settings: Settings, embeddings: Embeddings | None = None) -> dict:
    chunks_path = settings.processed_dir / "chunks.jsonl"
    chunks = load_chunks(chunks_path)
    chunks_sha256 = hashlib.sha256(chunks_path.read_bytes()).hexdigest()

    init_database(settings.database_url, settings.embedding_dim)
    store = PgStore.open(settings.database_url)
    try:
        vectors, embed_seconds = _embed(settings, chunks, chunks_sha256, embeddings)
        info = {
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "chunks_sha256": chunks_sha256,
            "chunks": len(chunks),
            "embed_seconds": embed_seconds,
            "indexed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        store.replace_chunks(chunks, vectors, info)
        info["rows_in_db"] = store.count()
    finally:
        store.close()
    return info


def _embed(
    settings: Settings, chunks: list[Chunk], chunks_sha256: str, embeddings: Embeddings | None
) -> tuple[np.ndarray, float | None]:
    cache = embedding_cache_path(settings, chunks_sha256)
    if embeddings is None and cache.exists():
        log.info("using cached embeddings %s", cache)
        return np.load(cache), None

    model = embeddings or load_embeddings(
        settings.embedding_model, settings.embedding_batch_size, show_progress=True
    )
    started = time.perf_counter()
    vectors = np.asarray(model.embed_documents([c.text for c in chunks]), dtype=np.float32)
    seconds = round(time.perf_counter() - started, 1)
    if vectors.shape != (len(chunks), settings.embedding_dim):
        raise ValueError(f"expected {len(chunks)}x{settings.embedding_dim}, got {vectors.shape}")
    log.info("embedded %d chunks in %.1f s", len(chunks), seconds)
    if embeddings is None:  # only cache the real model's vectors
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    return vectors, seconds
