"""ChromaDB persistence.

One collection per (embedding model, chunking strategy) pair so experiments can be
compared side by side without rebuilding, and so vectors of different dimensionality
never land in the same index.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_bot.config import settings
from rag_bot.embeddings.cache import cached
from rag_bot.embeddings.registry import get_embeddings, get_spec
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)

INDEX_BATCH = 256  # Chroma has a max-batch limit; also keeps progress visible


def collection_name(embed_key: str, chunker: str) -> str:
    return f"papers__{embed_key.replace('-', '_')}__{chunker}"


@dataclass
class IndexReport:
    collection: str
    embed_key: str
    chunker: str
    n_chunks: int
    dim: int
    build_seconds: float
    cache_hits: int = 0
    cache_misses: int = 0


def get_store(
    embed_key: str | None = None,
    chunker: str | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Open (or create) the persistent Chroma collection for this configuration."""
    embed_key = embed_key or settings.embed_model
    chunker = chunker or settings.chunker
    settings.ensure_dirs()
    return Chroma(
        collection_name=collection_name(embed_key, chunker),
        embedding_function=embeddings or get_embeddings(embed_key),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(
    chunks: Sequence[Document],
    embed_key: str | None = None,
    chunker: str | None = None,
    *,
    reset: bool = True,
    use_cache: bool = True,
) -> tuple[Chroma, IndexReport]:
    """Embed and persist chunks. Returns the store plus a timing/size report."""
    embed_key = embed_key or settings.embed_model
    chunker = chunker or settings.chunker
    spec = get_spec(embed_key)

    embeddings: Embeddings = get_embeddings(embed_key)
    cache_wrapper = cached(embeddings, spec.model_name) if use_cache else None
    if cache_wrapper is not None:
        embeddings = cache_wrapper

    store = get_store(embed_key, chunker, embeddings=embeddings)
    if reset:
        try:
            store.reset_collection()
        except Exception as exc:  # fresh directory: nothing to reset
            log.debug("reset skipped: %s", exc)

    start = time.perf_counter()
    for offset in range(0, len(chunks), INDEX_BATCH):
        batch = list(chunks[offset : offset + INDEX_BATCH])
        store.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
        log.info("indexed %d/%d", min(offset + INDEX_BATCH, len(chunks)), len(chunks))
    elapsed = time.perf_counter() - start

    report = IndexReport(
        collection=collection_name(embed_key, chunker),
        embed_key=embed_key,
        chunker=chunker,
        n_chunks=len(chunks),
        dim=spec.dim,
        build_seconds=elapsed,
        cache_hits=cache_wrapper.hits if cache_wrapper else 0,
        cache_misses=cache_wrapper.misses if cache_wrapper else 0,
    )
    log.info(
        "built %s: %d chunks, dim=%d, %.1fs (cache %d hit / %d miss)",
        report.collection,
        report.n_chunks,
        report.dim,
        report.build_seconds,
        report.cache_hits,
        report.cache_misses,
    )
    return store, report


def store_size(store: Chroma) -> int:
    try:
        return store._collection.count()
    except Exception:
        return 0
