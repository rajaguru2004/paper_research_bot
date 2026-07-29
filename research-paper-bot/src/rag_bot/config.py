"""Central configuration.

Every tunable lives here so the notebook, the CLI scripts and the Streamlit app all
share one source of truth. Values come from environment variables / `.env`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings, overridable via environment or `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM server (OpenAI-compatible: LM Studio / Ollama / vLLM) -------------
    lmstudio_base_url: str = Field(default="http://10.42.80.38:1234/v1")
    lmstudio_api_key: str = Field(default="lm-studio")  # ignored by LM Studio, required by SDK
    llm_model: str = Field(default="liquid/lfm2.5-1.2b")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=768)
    llm_timeout: float = Field(default=180.0)

    # --- Embeddings -----------------------------------------------------------
    # Key into rag_bot.embeddings.registry.EMBEDDING_MODELS
    embed_model: str = Field(default="minilm")
    lmstudio_embed_model: str = Field(default="text-embedding-nomic-embed-text-v1.5")
    embed_batch_size: int = Field(default=32)
    embed_device: str = Field(default="auto")  # auto | cpu | cuda

    # --- Chunking -------------------------------------------------------------
    chunker: str = Field(default="recursive")  # fixed | recursive | semantic
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=150)

    # --- Retrieval ------------------------------------------------------------
    retrieval_strategy: str = Field(default="rerank")
    top_k: int = Field(default=3)  # passages cited with every answer
    fetch_k: int = Field(default=20)  # candidates pulled before reranking / MMR
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    hybrid_rrf_k: int = Field(default=60)

    # --- CRAG -----------------------------------------------------------------
    crag_enabled: bool = Field(default=False)
    crag_relevance_threshold: float = Field(default=0.5)
    crag_web_results: int = Field(default=3)

    # --- Paths ----------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    chroma_dir: Path = Field(default=PROJECT_ROOT / "chroma")
    cache_dir: Path = Field(default=PROJECT_ROOT / ".cache")
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")

    # --- Misc -----------------------------------------------------------------
    random_seed: int = Field(default=42)
    log_level: str = Field(default="INFO")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def ensure_dirs(self) -> None:
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.chroma_dir,
            self.cache_dir,
            self.reports_dir / "figures",
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
