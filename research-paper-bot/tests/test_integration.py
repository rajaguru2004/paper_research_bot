"""Integration tests. Skipped unless the local LLM server and a built index exist.

Run with: `pytest -m integration`
"""

from __future__ import annotations

import pytest

from rag_bot.config import settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_server() -> list[str]:
    from rag_bot.generation.llm import list_models

    try:
        return list_models(timeout=10)
    except Exception as exc:
        pytest.skip(f"LLM server unreachable at {settings.lmstudio_base_url}: {exc}")


@pytest.fixture(scope="module")
def store():
    from rag_bot.store.chroma_store import get_store, store_size

    store = get_store()
    if store_size(store) == 0:
        pytest.skip("no index built; run `make index` first")
    return store


def test_server_advertises_models(live_server):
    assert live_server, "server returned an empty model list"


def test_configured_chat_model_loads(live_server):
    from rag_bot.generation.llm import health_check

    ok, detail = health_check()
    assert ok, f"model {settings.llm_model} failed to load: {detail}"


def test_lmstudio_embeddings_have_stable_dimension(live_server):
    from rag_bot.embeddings.lmstudio import LMStudioEmbeddings

    vectors = LMStudioEmbeddings().embed_documents(["attention mechanism", "low rank adaptation"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0


def test_retrieval_finds_the_expected_paper(store):
    from rag_bot.retrieval.factory import build_retriever

    docs = build_retriever("hybrid", store).retrieve("What is scaled dot-product attention?", k=5)
    assert any(d.metadata["title"] == "Attention Is All You Need" for d in docs)


def test_end_to_end_answer_is_cited(live_server, store):
    from rag_bot.generation.chain import build_pipeline

    result = build_pipeline("hybrid").answer("What is scaled dot-product attention?")
    assert result.citations, "answer produced no citations"
    assert all(c.page is not None for c in result.citations)
    assert not result.is_refusal


def test_out_of_corpus_question_is_refused(live_server, store):
    from rag_bot.generation.chain import build_pipeline

    result = build_pipeline("hybrid").answer("Which team won the 2018 FIFA World Cup?")
    assert result.is_refusal, f"expected a refusal, got: {result.answer[:160]}"
