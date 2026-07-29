"""The RAG pipeline: retrieve → build context → generate → cite.

Composed with LCEL (``RunnableParallel | prompt | llm | parser``) so the chain is a
first-class ``Runnable``: streamable, batchable and inspectable.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from rag_bot.config import settings
from rag_bot.generation.llm import get_llm
from rag_bot.generation.prompts import (
    CONDENSE_QUESTION_PROMPT,
    CONVERSATIONAL_RAG_PROMPT,
    RAG_PROMPT,
    format_context,
)
from rag_bot.logging_utils import get_logger
from rag_bot.retrieval.base import RetrievalStrategy

log = get_logger(__name__)

NO_ANSWER = "I don't know based on the provided papers."


@dataclass
class Citation:
    """One supporting passage, as shown to the user."""

    index: int
    title: str
    page: int | None
    score: float
    snippet: str
    source_type: str = "paper"  # "paper" | "web"
    url: str | None = None
    chunk_id: str | None = None

    def format(self) -> str:
        where = self.url if self.source_type == "web" else f"p.{self.page}"
        return f"[{self.index}] {self.title} — {where} (score {self.score:.3f})"


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    strategy: str = ""
    latency_s: float = 0.0
    used_web_search: bool = False
    crag_verdicts: list[str] = field(default_factory=list)
    rewritten_query: str | None = None
    documents: list[Document] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        return self.answer.strip().lower().startswith("i don't know")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("documents")
        return payload

    def pretty(self) -> str:
        lines = [f"Q: {self.question}", "", self.answer, "", "Sources:"]
        lines += [f"  {c.format()}" for c in self.citations] or ["  (none)"]
        lines.append(f"\n[{self.strategy} · {self.latency_s:.1f}s]")
        return "\n".join(lines)


def build_citations(docs: Sequence[Document], snippet_chars: int = 320) -> list[Citation]:
    """Turn retrieved documents into display-ready citations."""
    citations: list[Citation] = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        page = meta.get("page")
        citations.append(
            Citation(
                index=index,
                title=str(meta.get("title", "Unknown")),
                page=int(page) if isinstance(page, int | float | str) and str(page).isdigit() else None,
                score=float(meta.get("retrieval_score", 0.0)),
                snippet=doc.page_content[:snippet_chars].strip(),
                source_type=str(meta.get("source_type", "paper")),
                url=meta.get("url"),
                chunk_id=meta.get("chunk_id"),
            )
        )
    return citations


class RAGPipeline:
    """Retrieval-augmented answering with grounded citations."""

    def __init__(
        self,
        retriever: RetrievalStrategy,
        llm: BaseChatModel | None = None,
        k: int | None = None,
        crag: bool | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or get_llm()
        self.k = k or settings.top_k
        self.crag_enabled = settings.crag_enabled if crag is None else crag
        self._crag = None
        if self.crag_enabled:
            from rag_bot.crag.pipeline import CorrectiveRAG

            self._crag = CorrectiveRAG(self.retriever, self.llm)

    # -- LCEL composition ---------------------------------------------------
    def _answer_chain(self, conversational: bool = False) -> Runnable:
        prompt = CONVERSATIONAL_RAG_PROMPT if conversational else RAG_PROMPT
        return (
            RunnablePassthrough.assign(
                context=RunnableLambda(lambda x: format_context(x["documents"]), name="format_context")
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def condense(self, question: str, history: Sequence[BaseMessage]) -> str:
        """Rewrite a follow-up question into a standalone one (chat-memory support)."""
        if not history:
            return question
        try:
            chain = CONDENSE_QUESTION_PROMPT | self.llm | StrOutputParser()
            rewritten = chain.invoke({"question": question, "history": list(history)}).strip()
            rewritten = rewritten.splitlines()[0].strip().strip('"') if rewritten else ""
            # A rewrite that collapses or explodes is worse than the original question.
            if 5 <= len(rewritten) <= 300:
                return rewritten
        except Exception as exc:
            log.warning("condense failed, using raw question: %s", exc)
        return question

    # -- public API ---------------------------------------------------------
    def answer(
        self,
        question: str,
        history: Sequence[BaseMessage] | None = None,
        k: int | None = None,
    ) -> RAGAnswer:
        started = time.perf_counter()
        k = k or self.k
        history = list(history or [])

        search_query = self.condense(question, history) if history else question
        used_web, verdicts = False, []

        if self._crag is not None:
            documents, used_web, verdicts = self._crag.retrieve(search_query, k)
        else:
            documents = self.retriever.retrieve(search_query, k)

        if not documents:
            return RAGAnswer(
                question=question,
                answer=NO_ANSWER,
                strategy=self.retriever.name,
                latency_s=time.perf_counter() - started,
                rewritten_query=search_query if search_query != question else None,
            )

        chain = self._answer_chain(conversational=bool(history))
        payload = {"question": question, "documents": documents}
        if history:
            payload["history"] = history

        try:
            text = chain.invoke(payload).strip()
        except Exception as exc:
            log.error("generation failed: %s", exc)
            text = f"Generation failed against the local LLM server: {exc}"

        return RAGAnswer(
            question=question,
            answer=text or NO_ANSWER,
            citations=build_citations(documents),
            strategy=self.retriever.name + ("+crag" if self._crag else ""),
            latency_s=time.perf_counter() - started,
            used_web_search=used_web,
            crag_verdicts=verdicts,
            rewritten_query=search_query if search_query != question else None,
            documents=documents,
        )

    def stream(self, question: str, history: Sequence[BaseMessage] | None = None, k: int | None = None):
        """Yield ``(token, citations)``: citations first, then the answer token by token."""
        history = list(history or [])
        search_query = self.condense(question, history) if history else question
        documents = (
            self._crag.retrieve(search_query, k or self.k)[0]
            if self._crag is not None
            else self.retriever.retrieve(search_query, k or self.k)
        )
        citations = build_citations(documents)
        chain = self._answer_chain(conversational=bool(history))
        payload = {"question": question, "documents": documents}
        if history:
            payload["history"] = history
        for token in chain.stream(payload):
            yield token, citations


def build_pipeline(
    strategy: str | None = None,
    embed_key: str | None = None,
    chunker: str | None = None,
    crag: bool | None = None,
    k: int | None = None,
) -> RAGPipeline:
    """Convenience constructor: open the store, build the retriever, wire the LLM."""
    from rag_bot.retrieval.factory import build_retriever
    from rag_bot.store.chroma_store import get_store

    store = get_store(embed_key, chunker)
    retriever = build_retriever(strategy, store)
    return RAGPipeline(retriever, k=k, crag=crag)
