"""Application settings, read from environment variables and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Data layout
    data_dir: Path = REPO_ROOT / "data"

    # Ingestion
    http_user_agent: str = "vnlaw-rag/0.1 (educational research project)"
    http_delay_seconds: float = 1.0
    chunk_max_tokens: int = 1200
    tokenizer_name: str = "BAAI/bge-m3"

    # Index and retrieval
    database_url: str = "postgresql://vnlaw:vnlaw@127.0.0.1:5442/vnlaw"
    embedding_model: str = "BAAI/bge-m3"
    # Commit of the Hub's safetensors conversion of bge-m3 (same weights as main). Pinned for
    # reproducibility; see index/embed.py.
    embedding_revision: str = "9a0624b896d81da7492a910ffa53731274b6cf3d"
    embedding_dim: int = 1024
    embedding_batch_size: int = 8
    # Default mode chosen from docs/EVAL.md: vector-only scored best on the draft golden set;
    # hybrid stays available (RETRIEVAL_MODE=hybrid). See docs/decisions/005.
    retrieval_mode: Literal["vector", "hybrid", "keyword"] = "vector"
    keyword_mode: Literal["bigram", "unigram"] = "bigram"
    keyword_scoring: Literal["idf", "ts_rank_cd"] = "idf"
    keyword_stopwords: bool = True
    retrieval_vector_k: int = 20
    retrieval_keyword_k: int = 20
    retrieval_top_k: int = 8
    rrf_k: int = 60

    # LLM (Google Gemini API, free tier). The key is a SecretStr so it never shows up in
    # logs or reprs; it is read from the environment or .env and never written anywhere.
    gemini_api_key: SecretStr | None = None
    # Chosen by measurement (docs/decisions/008, 2026-11-24): the only model that answered
    # within the 25 s call limit (4/5, p50 1.7 s). The fallback is another generation with
    # better answers but slower; on failure models take turns: lite, 3.8-flash, lite.
    llm_model: str = "gemini-3.5-flash-lite"
    llm_fallback_model: str | None = "gemini-3.8-flash"
    llm_max_sources: int = 6  # retrieved chunks given to the model
    llm_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    llm_max_output_tokens: int = 2048
    # Latency policy (llm/gemini.py, docs/decisions/008): per-call timeout; on timeout, 5xx or
    # 429 switch to the fallback model at once; the whole answer has a time budget.
    llm_timeout_seconds: float = 25.0
    llm_deadline_seconds: float = 60.0
    llm_eval_deadline_seconds: float = 180.0  # eval: main model only, may wait longer
    llm_max_attempts: int = 3
    llm_backoff_seconds: float = 2.0
    # The `grade` node answers "not found" when the best cosine similarity is below this.
    # From docs/EVAL.md: answerable top-1 min 0.537 / median 0.729; out-of-scope max 0.674.
    grade_min_score: float = 0.55

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def manifest_path(self) -> Path:
        return self.raw_dir / "manifest.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
