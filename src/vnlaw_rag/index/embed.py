"""Embedding model behind LangChain's `Embeddings` interface.

bge-m3 runs locally on CPU through sentence-transformers (wrapped by langchain-huggingface).
Vectors are L2-normalized, so cosine similarity equals the dot product. bge-m3 needs no query
instruction prefix (unlike the English bge v1.5 models). Tests swap in a fake `Embeddings`.

The model revision is pinned (EMBEDDING_REVISION) for reproducibility. The pinned commit is
the Hub's safetensors conversion of bge-m3: it loads the same weights as the main branch
(identical vectors, checked) from one safetensors file, online or offline, so only one weight
format needs to be cached.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

from vnlaw_rag.config import Settings

log = logging.getLogger(__name__)


def load_embeddings(settings: Settings, show_progress: bool = False) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    log.info(
        "loading embedding model %s@%s on CPU",
        settings.embedding_model,
        settings.embedding_revision[:8],
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu", "revision": settings.embedding_revision},
        encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embedding_batch_size},
        query_encode_kwargs={"normalize_embeddings": True},
        show_progress=show_progress,
    )
