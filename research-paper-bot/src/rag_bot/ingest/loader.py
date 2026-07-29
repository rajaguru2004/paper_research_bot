"""PDF loading: one LangChain Document per page, with citation metadata attached.

Metadata is the whole point of this module — `title` and `page` must survive all the
way to the final answer, so they are attached at load time and asserted in tests.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from rag_bot.config import settings
from rag_bot.ingest.corpus import SPEC_BY_FILENAME
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)

MIN_CHARS_PER_PAGE = 120  # below this a page is a cover/figure/scan — not useful text

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    """Normalise PDF-extracted text without destroying paragraph structure."""
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub(" ", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words split across lines
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


@dataclass
class LoadReport:
    """What happened during loading — surfaced in the notebook EDA section."""

    files: int = 0
    pages_total: int = 0
    pages_kept: int = 0
    pages_skipped: int = 0
    skipped_detail: list[tuple[str, int, int]] = field(default_factory=list)  # title, page, chars
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def skip_rate(self) -> float:
        return self.pages_skipped / self.pages_total if self.pages_total else 0.0


def _resolve_title(path: Path, reader: PdfReader) -> tuple[str, str | None]:
    """Prefer the curated title; fall back to PDF metadata, then the filename."""
    spec = SPEC_BY_FILENAME.get(path.name)
    if spec is not None:
        return spec.title, spec.arxiv_id
    meta_title = (reader.metadata or {}).get("/Title") if reader.metadata else None
    if meta_title and len(str(meta_title).strip()) > 8:
        return str(meta_title).strip(), None
    return path.stem.replace("_", " ").title(), None


def load_pdf(path: Path, report: LoadReport | None = None) -> list[Document]:
    """Load one PDF into per-page Documents carrying citation metadata."""
    reader = PdfReader(str(path))
    title, arxiv_id = _resolve_title(path, reader)
    docs: list[Document] = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # a single broken page must not kill the file
            log.warning("%s p.%d extract failed: %s", path.name, index, exc)
            raw = ""
        text = clean_text(raw)

        if report is not None:
            report.pages_total += 1
        if len(text) < MIN_CHARS_PER_PAGE:
            if report is not None:
                report.pages_skipped += 1
                report.skipped_detail.append((title, index, len(text)))
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "title": title,
                    "arxiv_id": arxiv_id or "",
                    "page": index,  # 1-indexed, matches what a reader sees
                    "total_pages": len(reader.pages),
                    "source": path.name,
                    "source_path": str(path),
                },
            )
        )
        if report is not None:
            report.pages_kept += 1

    log.debug("%s -> %d pages kept", path.name, len(docs))
    return docs


def load_corpus(raw_dir: Path | None = None) -> tuple[list[Document], LoadReport]:
    """Load every PDF in `raw_dir` into page-level Documents."""
    raw_dir = raw_dir or settings.raw_dir
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs in {raw_dir}. Run `make papers` (or drop your own PDFs there)."
        )

    report = LoadReport()
    documents: list[Document] = []
    for path in pdfs:
        try:
            documents.extend(load_pdf(path, report))
            report.files += 1
        except Exception as exc:
            log.error("failed to load %s: %s", path.name, exc)
            report.failures.append((path.name, str(exc)))

    log.info(
        "loaded %d files | %d pages kept, %d skipped (%.1f%%)",
        report.files,
        report.pages_kept,
        report.pages_skipped,
        report.skip_rate * 100,
    )
    return documents, report
