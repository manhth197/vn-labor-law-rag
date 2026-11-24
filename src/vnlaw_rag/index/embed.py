"""Embedding model behind LangChain's `Embeddings` interface.

bge-m3 runs locally on CPU through sentence-transformers (wrapped by langchain-huggingface).
Vectors are L2-normalized, so cosine similarity equals the dot product. bge-m3 needs no query
instruction prefix (unlike the English bge v1.5 models). Tests swap in a fake `Embeddings`.
"""

from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings

log = logging.getLogger(__name__)


def load_embeddings(
    model_name: str, batch_size: int = 8, show_progress: bool = False
) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    log.info("loading embedding model %s on CPU", model_name)
    encode = {"normalize_embeddings": True, "batch_size": batch_size}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs=encode,
        query_encode_kwargs={"normalize_embeddings": True},
        show_progress=show_progress,
    )
