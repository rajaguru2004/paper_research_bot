"""One entry point for every retrieval strategy."""

from __future__ import annotations

from langchain_chroma import Chroma

from rag_bot.config import settings
from rag_bot.retrieval.base import RetrievalStrategy
from rag_bot.retrieval.dense import DenseRetriever, MMRRetriever
from rag_bot.retrieval.hybrid import BM25Retriever, HybridRetriever
from rag_bot.retrieval.multiquery import MultiQueryRetriever
from rag_bot.retrieval.rerank import RerankRetriever

STRATEGIES = ("dense", "mmr", "bm25", "hybrid", "rerank", "multiquery")

DESCRIPTIONS: dict[str, str] = {
    "dense": "Cosine similarity over embeddings (baseline).",
    "mmr": "Maximal Marginal Relevance — relevance with diversity.",
    "bm25": "Sparse lexical keyword matching.",
    "hybrid": "BM25 + dense fused with weighted Reciprocal Rank Fusion.",
    "rerank": "Hybrid candidates reranked by a cross-encoder.",
    "multiquery": "LLM query expansion, results fused with RRF.",
}


def build_retriever(
    strategy: str | None = None, store: Chroma | None = None, **kwargs
) -> RetrievalStrategy:
    """Construct a retrieval strategy by name against a Chroma collection."""
    strategy = (strategy or settings.retrieval_strategy).lower()
    if store is None:
        from rag_bot.store.chroma_store import get_store

        store = get_store()

    match strategy:
        case "dense":
            return DenseRetriever(store)
        case "mmr":
            return MMRRetriever(store, **kwargs)
        case "bm25":
            return BM25Retriever.from_store(store)
        case "hybrid":
            return HybridRetriever(store, **kwargs)
        case "rerank":
            return RerankRetriever(store, **kwargs)
        case "multiquery":
            return MultiQueryRetriever(store, **kwargs)
        case _:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
