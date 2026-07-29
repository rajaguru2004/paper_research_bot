"""Hybrid retrieval: BM25 keyword search fused with dense search via RRF.

Keyword search catches exact identifiers dense embeddings blur together
("GQA", "8x7B", "rank r=8"); dense search catches paraphrases. Fusing both is the
single biggest retrieval win on a technical-paper corpus.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_bot.config import settings
from rag_bot.logging_utils import get_logger
from rag_bot.retrieval.base import RetrievalStrategy, annotate, reciprocal_rank_fusion
from rag_bot.retrieval.dense import DenseRetriever

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping hyphenated/dotted technical terms intact."""
    return _TOKEN.findall(text.lower())


class BM25Retriever(RetrievalStrategy):
    """Sparse lexical retrieval over the same chunk set."""

    name = "bm25"

    def __init__(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("BM25 needs a non-empty corpus")
        self.documents = list(documents)
        self._bm25 = BM25Okapi([tokenize(d.page_content) for d in self.documents])

    @classmethod
    def from_store(cls, store: Chroma) -> BM25Retriever:
        """Rebuild the lexical index from what is already persisted in Chroma."""
        payload = store.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(payload["documents"], payload["metadatas"], strict=True)
        ]
        log.debug("BM25 index built from %d stored chunks", len(docs))
        return cls(docs)

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or settings.top_k
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return annotate([self.documents[i] for i in order], [scores[i] for i in order], self.name)


class HybridRetriever(RetrievalStrategy):
    """Weighted RRF over BM25 and dense results."""

    name = "hybrid"

    def __init__(
        self,
        store: Chroma,
        bm25: BM25Retriever | None = None,
        fetch_k: int | None = None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> None:
        self.dense = DenseRetriever(store)
        self.sparse = bm25 or BM25Retriever.from_store(store)
        self.fetch_k = fetch_k or settings.fetch_k
        self.weights = (dense_weight, sparse_weight)

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or settings.top_k
        dense_hits = self.dense.retrieve(query, self.fetch_k)
        sparse_hits = self.sparse.retrieve(query, self.fetch_k)
        fused, scores = reciprocal_rank_fusion(
            [dense_hits, sparse_hits], k=k, weights=list(self.weights)
        )
        return annotate(fused, scores, self.name)
