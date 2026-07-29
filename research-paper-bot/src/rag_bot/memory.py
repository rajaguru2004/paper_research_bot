"""Multi-user conversational memory (stretch goal 1).

Each ``session_id`` gets its own trimmed message history. Trimming matters: a 1.2B
model with a small context window degrades badly once the transcript grows, and the
retrieved passages must keep priority over old chatter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from rag_bot.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_MAX_TURNS = 6  # 6 turns = 12 messages kept in the prompt


@dataclass
class SessionStore:
    """Per-session chat histories, keyed by ``session_id``."""

    max_turns: int = DEFAULT_MAX_TURNS
    _sessions: dict[str, InMemoryChatMessageHistory] = field(default_factory=dict)

    def get(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
            log.debug("new session %s", session_id)
        return self._sessions[session_id]

    def history(self, session_id: str) -> list[BaseMessage]:
        """Recent messages, trimmed to the last ``max_turns`` exchanges."""
        messages = self.get(session_id).messages
        return list(messages[-2 * self.max_turns :])

    def append(self, session_id: str, question: str, answer: str) -> None:
        history = self.get(session_id)
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=answer))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def sessions(self) -> list[str]:
        return list(self._sessions)


class ConversationalRAG:
    """Wraps a ``RAGPipeline`` with per-session memory."""

    def __init__(self, pipeline, store: SessionStore | None = None) -> None:
        self.pipeline = pipeline
        self.store = store or SessionStore()

    def ask(self, question: str, session_id: str = "default", k: int | None = None):
        history = self.store.history(session_id)
        result = self.pipeline.answer(question, history=history, k=k)
        self.store.append(session_id, question, result.answer)
        return result
