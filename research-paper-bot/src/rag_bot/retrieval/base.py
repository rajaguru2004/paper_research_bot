"""Shared retrieval plumbing.

Every strategy returns ``list[Document]`` whose metadata carries ``retrieval_score``
and ``rank`` so the citation layer can display *why* a passage was chosen, regardless
of which strategy produced it.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from rag_bot.config import settings


class RetrievalStrategy(abc.ABC):
    """Base class for all retrieval strategies."""

    name: str = "base"

    @abc.abstractmethod
    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Return the top-k documents for a query, best first."""

    def as_runnable(self, k: int | None = None) -> Runnable:
        """Adapt to an LCEL runnable so it can be composed into the RAG chain."""
        return RunnableLambda(lambda q: self.retrieve(q, k), name=f"retrieve:{self.name}")

    def __repr__(self) -> str:  # helpful in notebook output
        return f"<{type(self).__name__} name={self.name}>"


def annotate(docs: Sequence[Document], scores: Sequence[float], strategy: str) -> list[Document]:
    """Attach score/rank/strategy metadata without mutating the stored documents."""
    out: list[Document] = []
    for rank, (doc, score) in enumerate(zip(docs, scores, strict=True), start=1):
        meta = dict(doc.metadata)
        meta["retrieval_score"] = round(float(score), 4)
        meta["rank"] = rank
        meta["retrieval_strategy"] = strategy
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out


def doc_key(doc: Document) -> str:
    """Stable identity for de-duplication across strategies."""
    return str(doc.metadata.get("chunk_id") or hash(doc.page_content))


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    k: int,
    rrf_k: int | None = None,
    weights: Sequence[float] | None = None,
) -> tuple[list[Document], list[float]]:
    """Fuse several ranked lists with weighted Reciprocal Rank Fusion.

    RRF is used instead of raw score blending because BM25 scores and cosine
    similarities live on incompatible scales; only the ranks are comparable.
    """
    rrf_k = rrf_k or settings.hybrid_rrf_k
    weights = list(weights) if weights else [1.0] * len(ranked_lists)

    scores: dict[str, float] = {}
    best: dict[str, Document] = {}
    for docs, weight in zip(ranked_lists, weights, strict=True):
        for rank, doc in enumerate(docs, start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            best.setdefault(key, doc)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [best[key] for key, _ in ordered], [score for _, score in ordered]
