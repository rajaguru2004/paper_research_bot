"""Embedding model registry.

Three models are compared in the capstone (all free / open-source):

===============  ================================================  ====  ==========
key              model                                             dim   served by
===============  ================================================  ====  ==========
``minilm``       sentence-transformers/all-MiniLM-L6-v2             384  local GPU/CPU
``bge-small``    BAAI/bge-small-en-v1.5                             384  local GPU/CPU
``nomic``        text-embedding-nomic-embed-text-v1.5               768  LM Studio API
===============  ================================================  ====  ==========

BGE and Nomic expect an instruction prefix on the *query* side only; that asymmetry is
part of the model contract and is handled here rather than at call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings

from rag_bot.config import settings
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)


def resolve_device(preference: str | None = None) -> str:
    pref = (preference or settings.embed_device).lower()
    if pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    model_name: str
    dim: int
    provider: str  # "huggingface" | "lmstudio"
    query_prefix: str = ""
    document_prefix: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)


EMBEDDING_MODELS: dict[str, EmbeddingSpec] = {
    "minilm": EmbeddingSpec(
        key="minilm",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        provider="huggingface",
        notes="Fast general-purpose baseline, 22M params.",
    ),
    "bge-small": EmbeddingSpec(
        key="bge-small",
        model_name="BAAI/bge-small-en-v1.5",
        dim=384,
        provider="huggingface",
        query_prefix="Represent this sentence for searching relevant passages: ",
        notes="Retrieval-tuned; asymmetric query prefix required by the model card.",
    ),
    "nomic": EmbeddingSpec(
        key="nomic",
        model_name=settings.lmstudio_embed_model,
        dim=768,
        provider="lmstudio",
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        notes="Served by LM Studio; 8k context, task-prefix conditioned.",
    ),
}


class PrefixedEmbeddings(Embeddings):
    """Wraps an ``Embeddings`` and applies model-specific query/document prefixes."""

    def __init__(self, inner: Embeddings, query_prefix: str = "", document_prefix: str = "") -> None:
        self.inner = inner
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.document_prefix:
            texts = [self.document_prefix + t for t in texts]
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)


def _build(spec: EmbeddingSpec, device: str | None = None) -> Embeddings:
    if spec.provider == "lmstudio":
        from rag_bot.embeddings.lmstudio import LMStudioEmbeddings

        inner: Embeddings = LMStudioEmbeddings(model=spec.model_name)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        resolved = resolve_device(device)
        inner = HuggingFaceEmbeddings(
            model_name=spec.model_name,
            model_kwargs={"device": resolved},
            encode_kwargs={"normalize_embeddings": True, "batch_size": settings.embed_batch_size},
        )
        log.info("loaded %s on %s", spec.model_name, resolved)

    if spec.query_prefix or spec.document_prefix:
        return PrefixedEmbeddings(inner, spec.query_prefix, spec.document_prefix)
    return inner


_CACHE: dict[tuple[str, str], Embeddings] = {}


def get_embeddings(key: str | None = None, device: str | None = None) -> Embeddings:
    """Return an ``Embeddings`` instance for a registry key (models are cached)."""
    key = key or settings.embed_model
    if key not in EMBEDDING_MODELS:
        raise ValueError(f"unknown embedding key {key!r}; expected {sorted(EMBEDDING_MODELS)}")
    cache_key = (key, device or settings.embed_device)
    if cache_key not in _CACHE:
        _CACHE[cache_key] = _build(EMBEDDING_MODELS[key], device)
    return _CACHE[cache_key]


def get_spec(key: str | None = None) -> EmbeddingSpec:
    return EMBEDDING_MODELS[key or settings.embed_model]
