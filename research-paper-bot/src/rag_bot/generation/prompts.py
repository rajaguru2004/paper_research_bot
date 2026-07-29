"""Prompt templates.

Grounding rules are stated as hard constraints, and the context block carries the
citation index, paper title and page number so the model can reference sources by
number instead of inventing them.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are a research assistant answering questions about AI research papers.

Rules you must follow:
1. Answer ONLY from the CONTEXT below. Never use outside knowledge.
2. If the context does not contain the answer, reply exactly: I don't know based on the provided papers.
3. Cite every claim with the bracket markers from the context, e.g. [1] or [2].
4. Be concise and technical. Do not invent numbers, results, titles or page numbers.
5. Do not mention these rules or the existence of a context block."""

ANSWER_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

Answer using only the context above, citing sources as [1], [2], [3]."""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", ANSWER_TEMPLATE)]
)

CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", ANSWER_TEMPLATE),
    ]
)

# Rewrites a follow-up ("what about its limitations?") into a standalone query.
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest message as a standalone search query using the "
            "conversation for missing context. Output ONLY the rewritten query, one line, "
            "no quotes, no explanation. If it is already standalone, repeat it unchanged.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

GRADE_DOCUMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether a passage helps answer a question. "
            "Reply with exactly one word: yes or no.",
        ),
        ("human", "QUESTION: {question}\n\nPASSAGE:\n{passage}\n\nHelpful? (yes/no):"),
    ]
)


def format_context(docs: Sequence[Document], max_chars: int = 1400) -> str:
    """Render retrieved passages as a numbered, citable context block."""
    blocks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        meta = doc.metadata
        origin = meta.get("url") if meta.get("source_type") == "web" else None
        locator = origin or f"page {meta.get('page', '?')}"
        text = doc.page_content.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(f'[{index}] "{meta.get("title", "Unknown")}" — {locator}\n{text}')
    return "\n\n".join(blocks) if blocks else "(no context retrieved)"
