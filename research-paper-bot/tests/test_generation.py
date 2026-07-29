"""Generation, citation and CRAG tests using a fake LLM — no server required."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_bot.crag.grader import DocumentGrader, Grade, parse_grade
from rag_bot.crag.pipeline import CorrectiveRAG
from rag_bot.generation.chain import NO_ANSWER, RAGPipeline, build_citations
from rag_bot.generation.prompts import format_context
from rag_bot.memory import ConversationalRAG, SessionStore
from rag_bot.retrieval.base import RetrievalStrategy


class StubRetriever(RetrievalStrategy):
    name = "stub"

    def __init__(self, docs: list[Document] | None = None) -> None:
        self.docs = docs if docs is not None else []
        self.queries: list[str] = []

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        self.queries.append(query)
        return self.docs[: k or 3]


def doc(title: str, page: int, text: str = "some passage text", **extra) -> Document:
    meta = {"title": title, "page": page, "retrieval_score": 0.9, "chunk_id": f"{title}:{page}"}
    meta.update(extra)
    return Document(page_content=text, metadata=meta)


class TestFormatContext:
    def test_numbers_and_labels_sources(self):
        rendered = format_context([doc("Paper A", 3), doc("Paper B", 7)])
        assert "[1]" in rendered and "[2]" in rendered
        assert "page 3" in rendered and "Paper B" in rendered

    def test_truncates_long_passages(self):
        rendered = format_context([doc("A", 1, "x" * 5000)], max_chars=100)
        assert rendered.endswith("...")

    def test_web_source_shows_url(self):
        rendered = format_context([doc("Blog", 0, "text", source_type="web", url="https://x.dev")])
        assert "https://x.dev" in rendered

    def test_empty_context_is_explicit(self):
        assert format_context([]) == "(no context retrieved)"


class TestCitations:
    def test_carries_title_page_score(self):
        citations = build_citations([doc("Attention Is All You Need", 4)])
        assert citations[0].index == 1
        assert citations[0].page == 4
        assert "Attention" in citations[0].format()
        assert "p.4" in citations[0].format()

    def test_web_citation_shows_url(self):
        citations = build_citations([doc("Web", 0, "t", source_type="web", url="https://a.io")])
        assert "https://a.io" in citations[0].format()

    def test_indices_are_sequential(self):
        citations = build_citations([doc("A", 1), doc("B", 2), doc("C", 3)])
        assert [c.index for c in citations] == [1, 2, 3]


class TestPipeline:
    def test_answers_with_citations(self):
        pipeline = RAGPipeline(
            StubRetriever([doc("Attention Is All You Need", 4)]),
            llm=FakeListChatModel(responses=["Attention scales by sqrt(d_k) [1]."]),
            crag=False,
        )
        result = pipeline.answer("Why scale attention?")
        assert "[1]" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0].title == "Attention Is All You Need"
        assert result.strategy == "stub"

    def test_refuses_when_nothing_retrieved(self):
        pipeline = RAGPipeline(
            StubRetriever([]), llm=FakeListChatModel(responses=["unused"]), crag=False
        )
        result = pipeline.answer("What is the price of tea?")
        assert result.answer == NO_ANSWER
        assert result.is_refusal
        assert result.citations == []

    def test_k_limits_citations(self):
        docs = [doc(f"P{i}", i) for i in range(1, 6)]
        pipeline = RAGPipeline(
            StubRetriever(docs), llm=FakeListChatModel(responses=["ok"]), crag=False, k=2
        )
        assert len(pipeline.answer("q").citations) == 2

    def test_generation_error_is_reported_not_raised(self):
        class BoomLLM(FakeListChatModel):
            def _generate(self, *args, **kwargs):
                raise RuntimeError("server down")

        pipeline = RAGPipeline(
            StubRetriever([doc("A", 1)]), llm=BoomLLM(responses=["x"]), crag=False
        )
        result = pipeline.answer("q")
        assert "Generation failed" in result.answer


class TestRefusalDetection:
    """A small model paraphrases the refusal instruction; detection must survive that."""

    @pytest.mark.parametrize(
        "answer",
        [
            "I don't know based on the provided papers.",
            "The provided context does not mention the winner of that tournament.",
            "The information provided does not specify the winner of the 2018 World Cup.",
            "The passages do not clearly state which team won.",
            "I cannot answer this from the given passages.",
            "There is no information about stock prices in these papers.",
            "This is not covered in the provided context.",
            "The context is insufficient to answer that.",
        ],
    )
    def test_paraphrased_refusals_are_detected(self, answer):
        pipeline = RAGPipeline(
            StubRetriever([doc("A", 1)]), llm=FakeListChatModel(responses=[answer]), crag=False
        )
        assert pipeline.answer("q").is_refusal

    @pytest.mark.parametrize(
        "answer",
        [
            "Attention is scaled by sqrt(d_k) to keep gradients stable [1].",
            # A mid-answer qualification is not a refusal — it must not trip the detector.
            "LoRA freezes the base weights and trains rank-decomposition matrices [1]. "
            "The paper does not mention inference-time overhead for r > 64.",
        ],
    )
    def test_real_answers_are_not_flagged(self, answer):
        pipeline = RAGPipeline(
            StubRetriever([doc("A", 1)]), llm=FakeListChatModel(responses=[answer]), crag=False
        )
        assert not pipeline.answer("q").is_refusal


class TestMemory:
    def test_history_grows_and_trims(self):
        store = SessionStore(max_turns=2)
        for i in range(5):
            store.append("s1", f"q{i}", f"a{i}")
        assert len(store.history("s1")) == 4  # 2 turns x 2 messages

    def test_sessions_are_isolated(self):
        store = SessionStore()
        store.append("alice", "q", "a")
        assert store.history("bob") == []
        assert set(store.sessions()) == {"alice", "bob"}

    def test_clear_removes_session(self):
        store = SessionStore()
        store.append("s", "q", "a")
        store.clear("s")
        assert store.history("s") == []

    def test_followup_is_condensed_before_retrieval(self):
        retriever = StubRetriever([doc("LoRA: Low-Rank Adaptation", 2)])
        pipeline = RAGPipeline(
            retriever,
            llm=FakeListChatModel(responses=["What are the limitations of LoRA?", "Answer [1]."]),
            crag=False,
        )
        bot = ConversationalRAG(pipeline, SessionStore())
        bot.ask("What is LoRA?")
        bot.ask("What about its limitations?")
        # Second turn must search with the rewritten, standalone query.
        assert retriever.queries[-1] != "What about its limitations?"


class TestCRAGGrading:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("yes", Grade.RELEVANT),
            ("Yes, it is", Grade.RELEVANT),
            ("no", Grade.IRRELEVANT),
            ("NO.", Grade.IRRELEVANT),
            ("maybe?", Grade.UNKNOWN),
            ("", Grade.UNKNOWN),
        ],
    )
    def test_parse_grade(self, raw, expected):
        assert parse_grade(raw) is expected

    def test_unknown_grade_keeps_the_document(self):
        """A confused grader must never silently discard context."""
        grader = DocumentGrader(FakeListChatModel(responses=["hmm"]))
        assert grader.grade("q", doc("A", 1)).keep is True

    def test_irrelevant_grade_drops_the_document(self):
        grader = DocumentGrader(FakeListChatModel(responses=["no"]))
        assert grader.grade("q", doc("A", 1)).keep is False

    def test_all_relevant_skips_web_search(self, monkeypatch):
        called = {"web": False}

        def fake_search(*args, **kwargs):
            called["web"] = True
            return []

        monkeypatch.setattr("rag_bot.crag.pipeline.web_search", fake_search)
        crag = CorrectiveRAG(
            StubRetriever([doc("A", 1), doc("B", 2)]),
            FakeListChatModel(responses=["yes"] * 10),
            threshold=0.5,
        )
        docs, used_web, verdicts = crag.retrieve("q", k=2)
        assert used_web is False and called["web"] is False
        assert len(docs) == 2 and set(verdicts) == {"relevant"}

    def test_irrelevant_context_triggers_web_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "rag_bot.crag.pipeline.web_search",
            lambda *a, **k: [doc("Web hit", 0, "text", source_type="web", url="https://x.io")],
        )
        crag = CorrectiveRAG(
            StubRetriever([doc("A", 1), doc("B", 2)]),
            FakeListChatModel(responses=["no"] * 10),
            threshold=0.5,
        )
        docs, used_web, _ = crag.retrieve("q", k=2)
        assert used_web is True
        assert docs[0].metadata["source_type"] == "web"
