"""Application settings, read from environment variables and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
