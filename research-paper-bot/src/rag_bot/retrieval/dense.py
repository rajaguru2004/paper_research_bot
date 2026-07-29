"""Dense vector retrieval: plain cosine similarity and MMR."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_bot.config import settings
from rag_bot.retrieval.base import RetrievalStrategy, annotate


class DenseRetriever(RetrievalStrategy):
    """Baseline: cosine similarity over the whole collection."""

    name = "dense"

    def __init__(self, store: Chroma) -> None:
        self.store = store

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or settings.top_k
        pairs = self.store.similarity_search_with_relevance_scores(query, k=k)
        docs = [doc for doc, _ in pairs]
        scores = [score for _, score in pairs]
        return annotate(docs, scores, self.name)


class MMRRetriever(RetrievalStrategy):
    """Maximal Marginal Relevance: trades a little relevance for diversity.

    Useful here because papers repeat the same sentence across abstract, intro and
    conclusion — plain cosine happily returns three near-duplicates.
    """

    name = "mmr"

    def __init__(self, store: Chroma, fetch_k: int | None = None, lambda_mult: float = 0.5) -> None:
        self.store = store
        self.fetch_k = fetch_k or settings.fetch_k
        self.lambda_mult = lambda_mult

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or settings.top_k
        docs = self.store.max_marginal_relevance_search(
            query, k=k, fetch_k=self.fetch_k, lambda_mult=self.lambda_mult
        )
        # MMR does not expose scores; rank position is the only signal available.
        scores = [1.0 - i / max(len(docs), 1) for i in range(len(docs))]
        return annotate(docs, scores, self.name)
