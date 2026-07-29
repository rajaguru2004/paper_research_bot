"""Multi-query retrieval.

The LLM rewrites the question into several paraphrases; each is retrieved separately
and the ranked lists are fused. Improves recall when the user's phrasing differs from
the paper's vocabulary ("how does the model avoid forgetting?" vs "positional encoding").

The local 1.2B model is not a reliable rewriter, so the generation step is defensive:
malformed output degrades to the original query rather than poisoning retrieval.
"""

from __future__ import annotations

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rag_bot.config import settings
from rag_bot.logging_utils import get_logger
from rag_bot.retrieval.base import RetrievalStrategy, annotate, reciprocal_rank_fusion
from rag_bot.retrieval.dense import DenseRetriever

log = get_logger(__name__)

REWRITE_PROMPT = """Rewrite the research question below as {n} alternative search queries.
Each query must be one short line, no numbering, no explanation, no preamble.
Keep technical terms exactly as written.

Question: {question}

Queries:"""

_STRIP_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def generate_variants(llm: BaseChatModel, question: str, n: int = 3) -> list[str]:
    """Ask the LLM for query variants; always returns at least the original question."""
    variants: list[str] = [question]
    try:
        raw = llm.invoke(REWRITE_PROMPT.format(n=n, question=question)).content
        text = raw if isinstance(raw, str) else str(raw)
        for line in text.splitlines():
            candidate = _STRIP_PREFIX.sub("", line).strip().strip('"')
            # Guard against the model echoing the prompt or emitting prose.
            if 8 <= len(candidate) <= 200 and candidate.lower() != question.lower():
                variants.append(candidate)
            if len(variants) > n:
                break
    except Exception as exc:
        log.warning("query rewriting failed, falling back to the original: %s", exc)
    return variants[: n + 1]


class MultiQueryRetriever(RetrievalStrategy):
    """Fuse retrieval results across LLM-generated query variants."""

    name = "multiquery"

    def __init__(
        self,
        store: Chroma,
        llm: BaseChatModel | None = None,
        base: RetrievalStrategy | None = None,
        n_variants: int = 3,
        fetch_k: int | None = None,
    ) -> None:
        if llm is None:
            from rag_bot.generation.llm import get_llm

            llm = get_llm()
        self.llm = llm
        self.base = base or DenseRetriever(store)
        self.n_variants = n_variants
        self.fetch_k = fetch_k or settings.fetch_k
        self.last_variants: list[str] = []

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or settings.top_k
        self.last_variants = generate_variants(self.llm, query, self.n_variants)
        log.debug("variants: %s", self.last_variants)
        ranked = [self.base.retrieve(v, self.fetch_k) for v in self.last_variants]
        # The original query gets more weight than the machine paraphrases.
        weights = [1.5] + [1.0] * (len(ranked) - 1)
        fused, scores = reciprocal_rank_fusion(ranked, k=k, weights=weights)
        return annotate(fused, scores, self.name)
