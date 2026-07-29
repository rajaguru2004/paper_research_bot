"""Retrieval tests that need no vector DB and no network."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag_bot.evaluation.dataset import ANSWERABLE, GOLD_SET, UNANSWERABLE
from rag_bot.evaluation.retrieval_metrics import hit_at_k, ndcg, page_hit, reciprocal_rank
from rag_bot.retrieval.base import annotate, reciprocal_rank_fusion
from rag_bot.retrieval.hybrid import BM25Retriever, tokenize


def make_doc(title: str, page: int, text: str = "text about attention") -> Document:
    return Document(
        page_content=text,
        metadata={"title": title, "page": page, "chunk_id": f"{title}:{page}"},
    )


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Attention IS All") == ["attention", "is", "all"]

    def test_keeps_technical_terms_intact(self):
        assert "grouped-query" in tokenize("Grouped-Query attention")
        assert "v1.5" in tokenize("model v1.5 release")


class TestRRF:
    def test_document_in_both_lists_wins(self):
        shared, other_a, other_b = make_doc("S", 1), make_doc("A", 1), make_doc("B", 1)
        fused, scores = reciprocal_rank_fusion([[other_a, shared], [other_b, shared]], k=3)
        assert fused[0].metadata["title"] == "S"
        assert scores[0] > scores[1]

    def test_respects_weights(self):
        a, b = make_doc("A", 1), make_doc("B", 1)
        fused, _ = reciprocal_rank_fusion([[a], [b]], k=2, weights=[0.1, 10.0])
        assert fused[0].metadata["title"] == "B"

    def test_truncates_to_k(self):
        docs = [make_doc(f"T{i}", i) for i in range(10)]
        fused, _ = reciprocal_rank_fusion([docs], k=3)
        assert len(fused) == 3


class TestBM25:
    def test_finds_lexical_match(self):
        docs = [
            make_doc("A", 1, "grouped query attention reduces KV cache size"),
            make_doc("B", 2, "convolutional networks for image classification"),
        ]
        hits = BM25Retriever(docs).retrieve("grouped query attention", k=1)
        assert hits[0].metadata["title"] == "A"
        assert hits[0].metadata["rank"] == 1

    def test_empty_corpus_rejected(self):
        with pytest.raises(ValueError):
            BM25Retriever([])


def test_annotate_adds_score_rank_strategy():
    docs = annotate([make_doc("A", 1), make_doc("B", 2)], [0.9, 0.5], "dense")
    assert [d.metadata["rank"] for d in docs] == [1, 2]
    assert docs[0].metadata["retrieval_score"] == 0.9
    assert docs[0].metadata["retrieval_strategy"] == "dense"


class TestMetrics:
    gold = ANSWERABLE[0]  # expects "Attention Is All You Need", page 4

    def test_hit_and_rank(self):
        docs = [make_doc("Wrong", 1), make_doc(self.gold.expected_title, 4)]
        assert hit_at_k(docs, self.gold) == 1.0
        assert reciprocal_rank(docs, self.gold) == 0.5

    def test_miss_scores_zero(self):
        docs = [make_doc("Wrong", 1)]
        assert hit_at_k(docs, self.gold) == 0.0
        assert reciprocal_rank(docs, self.gold) == 0.0
        assert ndcg(docs, self.gold) == 0.0

    def test_ndcg_prefers_top_rank(self):
        top = [make_doc(self.gold.expected_title, 4), make_doc("Wrong", 1)]
        low = [make_doc("Wrong", 1), make_doc(self.gold.expected_title, 4)]
        assert ndcg(top, self.gold) > ndcg(low, self.gold)

    def test_page_hit_tolerates_neighbour_page(self):
        assert page_hit([make_doc(self.gold.expected_title, 5)], self.gold, tolerance=1) == 1.0
        assert page_hit([make_doc(self.gold.expected_title, 12)], self.gold, tolerance=1) == 0.0


class TestGoldSet:
    def test_ids_unique(self):
        assert len({q.id for q in GOLD_SET}) == len(GOLD_SET)

    def test_split_sizes(self):
        assert len(GOLD_SET) == 20
        assert len(ANSWERABLE) == 18
        assert len(UNANSWERABLE) == 2

    def test_expected_titles_exist_in_corpus(self):
        from rag_bot.ingest.corpus import PAPERS

        titles = {p.title for p in PAPERS}
        assert all(q.expected_title in titles for q in ANSWERABLE)
