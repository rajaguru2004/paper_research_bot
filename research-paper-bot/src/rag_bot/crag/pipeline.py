"""Corrective RAG (CRAG) control flow.

    retrieve → grade each passage
        ├─ enough relevant   → answer from the papers
        ├─ partially relevant→ keep the good ones + top up from the web
        └─ none relevant     → rewrite the query, answer from the web

Implemented as explicit control flow rather than a graph framework: three branches are
easier to read, test and explain in a viva than a state machine.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_bot.config import settings
from rag_bot.crag.grader import DocumentGrader
from rag_bot.crag.websearch import web_search
from rag_bot.logging_utils import get_logger
from rag_bot.retrieval.base import RetrievalStrategy

log = get_logger(__name__)

REWRITE_FOR_WEB = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Turn the question into a concise web search query (max 12 words). "
            "Output only the query.",
        ),
        ("human", "{question}"),
    ]
)


class CorrectiveRAG:
    """Grades retrieved context and falls back to web search when it is weak."""

    def __init__(
        self,
        retriever: RetrievalStrategy,
        llm: BaseChatModel,
        threshold: float | None = None,
        fetch_multiplier: int = 2,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.grader = DocumentGrader(llm)
        self.threshold = threshold if threshold is not None else settings.crag_relevance_threshold
        self.fetch_multiplier = fetch_multiplier

    def rewrite_for_web(self, question: str) -> str:
        try:
            chain = REWRITE_FOR_WEB | self.llm | StrOutputParser()
            query = chain.invoke({"question": question}).strip().splitlines()[0].strip('"')
            return query if 5 <= len(query) <= 200 else question
        except Exception as exc:
            log.warning("web query rewrite failed: %s", exc)
            return question

    def retrieve(self, question: str, k: int) -> tuple[list[Document], bool, list[str]]:
        """Return ``(documents, used_web_search, verdicts)``."""
        candidates = self.retriever.retrieve(question, k * self.fetch_multiplier)
        if not candidates:
            web = web_search(self.rewrite_for_web(question), k)
            return web, bool(web), ["no local candidates"]

        graded = self.grader.grade_all(question, candidates)
        kept = [g.document for g in graded if g.keep]
        verdicts = [g.grade.value for g in graded]
        relevant_ratio = len(kept) / len(graded)
        log.info("CRAG: %d/%d passages kept (%.0f%%)", len(kept), len(graded), relevant_ratio * 100)

        if relevant_ratio >= self.threshold and len(kept) >= k:
            return kept[:k], False, verdicts

        # Weak or partial evidence → supplement with the open web.
        web = web_search(self.rewrite_for_web(question), settings.crag_web_results)
        merged = (kept + web)[:k] if kept else web[:k]
        return merged, bool(web), verdicts
