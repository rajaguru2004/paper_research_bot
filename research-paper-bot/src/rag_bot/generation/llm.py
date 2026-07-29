"""Chat model wired to the local OpenAI-compatible server (LM Studio)."""

from __future__ import annotations

from functools import lru_cache

import httpx
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag_bot.config import settings
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=4)
def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Return a chat model pointed at the local server.

    LM Studio ignores the API key but the OpenAI SDK requires one to be present.
    """
    return ChatOpenAI(
        model=model or settings.llm_model,
        base_url=settings.lmstudio_base_url,
        api_key=SecretStr(settings.lmstudio_api_key),
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_completion_tokens=max_tokens or settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=2,
    )


def list_models(base_url: str | None = None, timeout: float = 15.0) -> list[str]:
    """Model ids advertised by the server — used for preflight checks and the UI."""
    url = f"{(base_url or settings.lmstudio_base_url).rstrip('/')}/models"
    with httpx.Client(timeout=timeout) as client:
        payload = client.get(url).raise_for_status().json()
    return [item["id"] for item in payload.get("data", [])]


def health_check(timeout: float = 120.0) -> tuple[bool, str]:
    """Verify the configured chat model actually loads and answers.

    Not every advertised model can be served — LM Studio may fail to load a model that
    does not fit in VRAM, and it only reports that at first completion.
    """
    try:
        reply = get_llm(max_tokens=8).invoke("Reply with the single word: ready")
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        return True, text.strip()[:80]
    except Exception as exc:
        return False, str(exc)[:300]
