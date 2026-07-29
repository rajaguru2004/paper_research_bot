"""Cross-encoder reranking.

Bi-encoders score query and passage independently; a cross-encoder reads both together
and is far more precise. It is too slow to score the whole corpus, so it reranks the
top ``fetch_k`` candidates produced by a cheaper first stage — the standard
retrieve-then-rerank cascade.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_bot.config import settings
from rag_bot.embeddings.registry import resolve_device
from rag_bot.logging_utils import get_logger
from rag_bot.retrieval.base import RetrievalStrategy, annotate
from rag_bot.retrieval.hybrid import HybridRetriever

log = get_logger(__name__)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, device: str):
    from sentence_transformers import CrossEncoder

    log.info("loading reranker %s on %s", model_name, device)
    return CrossEncoder(model_name, device=device, max_length=512)


class RerankRetriever(RetrievalStrategy):
    """First stage retrieves ``fetch_k`` candidates, cross-encoder picks the top-k."""

    name = "rerank"

    def __init__(
        self,
        store: Chroma,
        first_stage: RetrievalStrategy | None = None,
        fetch_k: int | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.first_stage = first_stage or HybridRetriever(store)
        self.fetch_k = fetch_k or settings.fetch_k
        self.model_name = model_name or settings.reranker_model
        self.device = resolve_device(device)

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or settings.top_k
        candidates = self.first_stage.retrieve(query, self.fetch_k)
        if not candidates:
            return []

        model = _load_cross_encoder(self.model_name, self.device)
        scores = model.predict(
            [(query, doc.page_content) for doc in candidates],
            batch_size=16,
            show_progress_bar=False,
        )
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)[:k]
        return annotate(
            [candidates[i] for i in order], [float(scores[i]) for i in order], self.name
        )
