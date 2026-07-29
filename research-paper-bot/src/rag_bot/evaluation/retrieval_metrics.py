"""Retrieval quality metrics.

Retrieval is graded before generation because it is the bottleneck: no prompt can
recover from context that never contained the answer.

A hit = a retrieved chunk whose ``title`` matches the gold paper. Page-level metrics
are reported separately and are stricter (the answer may legitimately live on a
neighbouring page, so page recall is informative rather than decisive).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document

from rag_bot.evaluation.dataset import ANSWERABLE, GoldQuestion
from rag_bot.logging_utils import get_logger
from rag_bot.retrieval.base import RetrievalStrategy

log = get_logger(__name__)


def _hits(docs: Sequence[Document], gold: GoldQuestion) -> list[bool]:
    return [doc.metadata.get("title") == gold.expected_title for doc in docs]


def hit_at_k(docs: Sequence[Document], gold: GoldQuestion) -> float:
    return 1.0 if any(_hits(docs, gold)) else 0.0


def reciprocal_rank(docs: Sequence[Document], gold: GoldQuestion) -> float:
    for rank, hit in enumerate(_hits(docs, gold), start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg(docs: Sequence[Document], gold: GoldQuestion) -> float:
    """Binary-relevance nDCG — rewards putting the right paper first."""
    gains = [1.0 if hit else 0.0 for hit in _hits(docs, gold)]
    if not any(gains):
        return 0.0
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(int(sum(gains)), len(gains))))
    return float(dcg / ideal) if ideal else 0.0


def page_hit(docs: Sequence[Document], gold: GoldQuestion, tolerance: int = 1) -> float:
    """Did any retrieved chunk land on (or next to) an expected page?"""
    if not gold.expected_pages:
        return float("nan")
    for doc in docs:
        if doc.metadata.get("title") != gold.expected_title:
            continue
        page = doc.metadata.get("page")
        if isinstance(page, int) and any(abs(page - p) <= tolerance for p in gold.expected_pages):
            return 1.0
    return 0.0


@dataclass
class RetrievalEval:
    label: str
    k: int
    hit_rate: float
    mrr: float
    ndcg: float
    page_hit_rate: float
    mean_latency_ms: float
    n_questions: int
    per_question: list[dict] = field(default_factory=list)

    def row(self) -> dict:
        return {
            "config": self.label,
            "k": self.k,
            f"hit@{self.k}": round(self.hit_rate, 3),
            "MRR": round(self.mrr, 3),
            "nDCG": round(self.ndcg, 3),
            "page_hit": round(self.page_hit_rate, 3),
            "latency_ms": round(self.mean_latency_ms, 1),
        }


def evaluate_retriever(
    retriever: RetrievalStrategy,
    k: int = 5,
    questions: Sequence[GoldQuestion] = ANSWERABLE,
    label: str | None = None,
) -> RetrievalEval:
    """Run the gold set through a retriever and compute ranking metrics."""
    label = label or retriever.name
    rows: list[dict] = []

    for gold in questions:
        started = time.perf_counter()
        docs = retriever.retrieve(gold.question, k)
        latency_ms = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "id": gold.id,
                "kind": gold.kind,
                "hit": hit_at_k(docs, gold),
                "rr": reciprocal_rank(docs, gold),
                "ndcg": ndcg(docs, gold),
                "page_hit": page_hit(docs, gold),
                "latency_ms": latency_ms,
                "top_title": docs[0].metadata.get("title") if docs else None,
                "expected_title": gold.expected_title,
            }
        )

    def mean(key: str) -> float:
        values = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(values)) if values else 0.0

    result = RetrievalEval(
        label=label,
        k=k,
        hit_rate=mean("hit"),
        mrr=mean("rr"),
        ndcg=mean("ndcg"),
        page_hit_rate=mean("page_hit"),
        mean_latency_ms=mean("latency_ms"),
        n_questions=len(rows),
        per_question=rows,
    )
    log.info(
        "%-28s hit@%d=%.2f MRR=%.2f nDCG=%.2f page=%.2f %.0fms",
        label,
        k,
        result.hit_rate,
        result.mrr,
        result.ndcg,
        result.page_hit_rate,
        result.mean_latency_ms,
    )
    return result
