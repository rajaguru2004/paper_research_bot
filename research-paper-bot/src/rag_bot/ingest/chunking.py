"""Chunking strategies.

Three strategies are compared in the capstone notebook:

* ``fixed``     — token-window splitting, the classic baseline
* ``recursive`` — structure-aware splitting on paragraph/sentence boundaries
* ``semantic``  — embedding-driven splitting at topic shifts

All of them preserve page-level metadata and add a stable ``chunk_id``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from rag_bot.config import settings
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)

CHUNKER_NAMES = ("fixed", "recursive", "semantic")


@dataclass(frozen=True)
class ChunkStats:
    strategy: str
    n_chunks: int
    mean_chars: float
    p95_chars: float
    min_chars: int
    max_chars: int


def _finalise(chunks: Sequence[Document], strategy: str) -> list[Document]:
    """Attach chunk-level metadata and drop empty fragments."""
    out: list[Document] = []
    per_page_counter: dict[tuple[str, int], int] = {}

    for chunk in chunks:
        text = chunk.page_content.strip()
        if len(text) < 30:  # splitter artefacts: page numbers, stray headers
            continue
        meta = dict(chunk.metadata)
        key = (str(meta.get("source", "?")), int(meta.get("page", 0)))
        index = per_page_counter.get(key, 0)
        per_page_counter[key] = index + 1

        meta["chunk_index"] = index
        meta["chunk_id"] = f"{key[0]}:p{key[1]}:c{index}"
        meta["chunk_strategy"] = strategy
        meta["n_chars"] = len(text)
        out.append(Document(page_content=text, metadata=meta))
    return out


def split_fixed(
    docs: Sequence[Document], chunk_size: int = 512, chunk_overlap: int = 64
) -> list[Document]:
    """Fixed token windows (~512 tokens) — the baseline from the brief."""
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return _finalise(splitter.split_documents(list(docs)), "fixed")


def split_recursive(
    docs: Sequence[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Recursive character splitting on natural boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return _finalise(splitter.split_documents(list(docs)), "recursive")


def split_semantic(
    docs: Sequence[Document],
    embeddings: Embeddings | None = None,
    breakpoint_threshold_amount: float = 90.0,
) -> list[Document]:
    """Semantic chunking: split where consecutive sentence embeddings diverge.

    Needs an embedding model, so it is the slowest strategy — measured in the notebook.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    if embeddings is None:
        from rag_bot.embeddings.registry import get_embeddings

        embeddings = get_embeddings("bge-small")

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    return _finalise(splitter.split_documents(list(docs)), "semantic")


CHUNKERS: dict[str, Callable[..., list[Document]]] = {
    "fixed": split_fixed,
    "recursive": split_recursive,
    "semantic": split_semantic,
}


def chunk_documents(docs: Sequence[Document], strategy: str | None = None, **kwargs) -> list[Document]:
    """Dispatch to a chunking strategy by name."""
    strategy = strategy or settings.chunker
    if strategy not in CHUNKERS:
        raise ValueError(f"unknown chunker {strategy!r}; expected one of {CHUNKER_NAMES}")
    chunks = CHUNKERS[strategy](docs, **kwargs)
    log.info("chunker=%s produced %d chunks from %d pages", strategy, len(chunks), len(docs))
    return chunks


def chunk_stats(chunks: Sequence[Document], strategy: str = "") -> ChunkStats:
    """Summary statistics used in the notebook comparison table."""
    import numpy as np

    lengths = np.array([len(c.page_content) for c in chunks]) if chunks else np.array([0])
    return ChunkStats(
        strategy=strategy or (chunks[0].metadata.get("chunk_strategy", "?") if chunks else "?"),
        n_chunks=len(chunks),
        mean_chars=float(lengths.mean()),
        p95_chars=float(np.percentile(lengths, 95)),
        min_chars=int(lengths.min()),
        max_chars=int(lengths.max()),
    )
