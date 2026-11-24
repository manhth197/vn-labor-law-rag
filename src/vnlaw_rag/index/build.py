"""Build the pgvector index from data/processed/chunks.jsonl (one command, full rebuild).

Embedding 348 chunks with bge-m3 on CPU takes about 8 minutes, so vectors are cached in
data/cache/ by the sha256 of each chunk's text (per model and revision): after a corpus
update only the chunks whose text changed are embedded again. The database is prepared
first so a connection problem fails fast.
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


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_cache_path(settings: Settings) -> Path:
    model = re.sub(r"[^A-Za-z0-9]+", "-", settings.embedding_model).strip("-")
    return settings.data_dir / "cache" / f"embeddings-{model}-{settings.embedding_revision[:8]}.npz"


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path)
    return dict(zip(data["keys"].tolist(), data["vectors"], strict=True))


def save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache)
    np.savez(path, keys=np.array(keys), vectors=np.stack([cache[k] for k in keys]))


def build_index(settings: Settings, embeddings: Embeddings | None = None) -> dict:
    chunks_path = settings.processed_dir / "chunks.jsonl"
    chunks = load_chunks(chunks_path)

    init_database(settings.database_url, settings.embedding_dim, reset=True)  # full rebuild
    store = PgStore.open(settings.database_url)
    try:
        vectors, stats = _embed(settings, chunks, embeddings)
        info = {
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.embedding_revision,
            "embedding_dim": settings.embedding_dim,
            "chunks_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
            "chunks": len(chunks),
            **stats,
            "indexed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        store.replace_chunks(chunks, vectors, info)
        info["rows_in_db"] = store.count()
    finally:
        store.close()
    return info


def _embed(
    settings: Settings, chunks: list[Chunk], embeddings: Embeddings | None
) -> tuple[np.ndarray, dict]:
    """Vectors for `chunks`, computing only the texts missing from the cache. A model passed in
    (tests) bypasses the cache so fake vectors never mix with real ones."""
    use_cache = embeddings is None
    path = embedding_cache_path(settings)
    cache = load_cache(path) if use_cache else {}
    keys = [text_key(c.text) for c in chunks]
    missing = [i for i, k in enumerate(keys) if k not in cache]

    seconds = 0.0
    if missing:
        model = embeddings or load_embeddings(settings, show_progress=True)
        started = time.perf_counter()
        new = np.asarray(model.embed_documents([chunks[i].text for i in missing]), dtype=np.float32)
        seconds = round(time.perf_counter() - started, 1)
        if new.shape != (len(missing), settings.embedding_dim):
            raise ValueError(f"expected {len(missing)}x{settings.embedding_dim}, got {new.shape}")
        cache.update({keys[i]: vec for i, vec in zip(missing, new, strict=True)})
        log.info("embedded %d of %d chunks in %.1f s", len(missing), len(chunks), seconds)
        if use_cache:
            save_cache(path, cache)
    vectors = np.stack([cache[k] for k in keys]).astype(np.float32)
    stats = {
        "embeddings_computed": len(missing),
        "embeddings_from_cache": len(chunks) - len(missing),
        "embed_seconds": seconds,
    }
    return vectors, stats
