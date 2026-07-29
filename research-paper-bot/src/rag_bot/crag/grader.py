"""Relevance grading for Corrective RAG.

The grader decides whether retrieved passages actually support the question. With a
small local model the grade itself is noisy, so parsing is strict and the failure mode
is deliberately conservative: an unparseable grade counts as *relevant*, because
dropping good context silently is worse than keeping a marginal passage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from rag_bot.generation.prompts import GRADE_DOCUMENT_PROMPT
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)


class Grade(str, Enum):  # noqa: UP042 - str mixin keeps grades JSON-serialisable
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"  # grader failed — treated as relevant downstream


@dataclass
class GradeResult:
    document: Document
    grade: Grade

    @property
    def keep(self) -> bool:
        return self.grade in (Grade.RELEVANT, Grade.UNKNOWN)


def parse_grade(raw: str) -> Grade:
    text = raw.strip().lower()
    head = text[:24]
    if "yes" in head:
        return Grade.RELEVANT
    if "no" in head:
        return Grade.IRRELEVANT
    return Grade.UNKNOWN


class DocumentGrader:
    def __init__(self, llm: BaseChatModel, passage_chars: int = 1200) -> None:
        self.chain = GRADE_DOCUMENT_PROMPT | llm | StrOutputParser()
        self.passage_chars = passage_chars

    def grade(self, question: str, document: Document) -> GradeResult:
        try:
            raw = self.chain.invoke(
                {"question": question, "passage": document.page_content[: self.passage_chars]}
            )
            return GradeResult(document, parse_grade(raw))
        except Exception as exc:
            log.warning("grading failed: %s", exc)
            return GradeResult(document, Grade.UNKNOWN)

    def grade_all(self, question: str, documents: list[Document]) -> list[GradeResult]:
        return [self.grade(question, doc) for doc in documents]
