"""Ingestion tests. The metadata assertions guard the citation contract."""

from __future__ import annotations

import pytest

from rag_bot.ingest.chunking import chunk_documents, chunk_stats
from rag_bot.ingest.corpus import PAPERS, TITLE_BY_ID
from rag_bot.ingest.loader import MIN_CHARS_PER_PAGE, clean_text


class TestCleanText:
    def test_rejoins_hyphenated_line_breaks(self):
        assert "attention" in clean_text("atten-\ntion mechanism")

    def test_expands_ligatures(self):
        assert clean_text("ﬁne-tuning the classiﬁer").startswith("fine-tuning")

    def test_collapses_excess_blank_lines(self):
        assert "\n\n\n" not in clean_text("a\n\n\n\n\nb")

    def test_strips_control_characters(self):
        assert "\x07" not in clean_text("bad\x07char")


class TestCorpus:
    def test_ids_and_titles_unique(self):
        assert len({p.arxiv_id for p in PAPERS}) == len(PAPERS)
        assert len({p.title for p in PAPERS}) == len(PAPERS)

    def test_pdf_url_shape(self):
        assert PAPERS[0].pdf_url == "https://arxiv.org/pdf/1706.03762"
        assert PAPERS[0].filename == "1706_03762.pdf"

    def test_title_lookup(self):
        assert TITLE_BY_ID["1706.03762"] == "Attention Is All You Need"


@pytest.mark.parametrize("strategy", ["fixed", "recursive"])
class TestChunking:
    def test_preserves_citation_metadata(self, pages, strategy):
        """Every chunk must keep the title and page, or citations become impossible."""
        for chunk in chunk_documents(pages, strategy):
            assert chunk.metadata["title"]
            assert isinstance(chunk.metadata["page"], int)
            assert chunk.metadata["source"]

    def test_chunk_ids_unique(self, pages, strategy):
        ids = [c.metadata["chunk_id"] for c in chunk_documents(pages, strategy)]
        assert len(ids) == len(set(ids))

    def test_tags_strategy(self, pages, strategy):
        chunks = chunk_documents(pages, strategy)
        assert chunks and all(c.metadata["chunk_strategy"] == strategy for c in chunks)

    def test_no_tiny_fragments(self, pages, strategy):
        assert all(len(c.page_content) >= 30 for c in chunk_documents(pages, strategy))


def test_unknown_chunker_rejected(pages):
    with pytest.raises(ValueError, match="unknown chunker"):
        chunk_documents(pages, "nonsense")


def test_chunk_stats(pages):
    stats = chunk_stats(chunk_documents(pages, "recursive"), "recursive")
    assert stats.n_chunks > 0
    assert stats.min_chars <= stats.mean_chars <= stats.max_chars


def test_min_page_threshold_is_sane():
    assert 0 < MIN_CHARS_PER_PAGE < 1000
