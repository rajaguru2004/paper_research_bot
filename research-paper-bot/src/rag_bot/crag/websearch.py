"""Free web search fallback for Corrective RAG (DuckDuckGo, no API key)."""

from __future__ import annotations

from langchain_core.documents import Document

from rag_bot.config import settings
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)


def web_search(query: str, max_results: int | None = None) -> list[Document]:
    """Return web results as Documents tagged ``source_type='web'``.

    Web hits are marked so citations can distinguish them from the paper corpus — a
    reader must always be able to tell grounded-in-paper from grounded-in-internet.
    """
    max_results = max_results or settings.crag_web_results
    try:
        from ddgs import DDGS

        with DDGS() as engine:
            hits = list(engine.text(query, max_results=max_results))
    except Exception as exc:
        log.warning("web search unavailable: %s", exc)
        return []

    documents: list[Document] = []
    for rank, hit in enumerate(hits, start=1):
        body = (hit.get("body") or "").strip()
        if not body:
            continue
        documents.append(
            Document(
                page_content=body,
                metadata={
                    "title": hit.get("title", "Web result"),
                    "url": hit.get("href", ""),
                    "source": "web",
                    "source_type": "web",
                    "page": None,
                    "rank": rank,
                    "retrieval_score": 1.0 / rank,
                    "retrieval_strategy": "web",
                },
            )
        )
    log.info("web search returned %d usable results", len(documents))
    return documents
