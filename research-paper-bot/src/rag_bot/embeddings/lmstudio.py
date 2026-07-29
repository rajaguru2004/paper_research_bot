"""Embeddings served by LM Studio's OpenAI-compatible `/v1/embeddings` endpoint.

Deliberately *not* `langchain_openai.OpenAIEmbeddings`: that class applies OpenAI's
tiktoken-based context-length handling, which silently mis-splits input for non-OpenAI
models such as ``nomic-embed-text``. This thin subclass sends text through verbatim in
explicit batches instead.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.embeddings import Embeddings
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_bot.config import settings
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)


class LMStudioEmbeddings(Embeddings):
    """LangChain ``Embeddings`` backed by a local OpenAI-compatible server."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model or settings.lmstudio_embed_model
        self.base_url = base_url or settings.lmstudio_base_url
        self.batch_size = batch_size or settings.embed_batch_size
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or settings.lmstudio_api_key,
            timeout=timeout,
            max_retries=0,  # retries handled by tenacity below, with backoff
        )

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        # The server may return items out of order; `index` is authoritative.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            vectors.extend(self._embed_batch(batch))
            if total > self.batch_size:
                log.debug("embedded %d/%d", min(start + self.batch_size, total), total)
        if len(vectors) != total:
            raise RuntimeError(f"embedding count mismatch: got {len(vectors)}, want {total}")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text or " "])[0]
