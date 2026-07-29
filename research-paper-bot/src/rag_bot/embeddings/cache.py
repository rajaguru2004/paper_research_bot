"""On-disk embedding cache.

Re-running the experiment grid (3 embedding models x 3 chunkers x 5 retrieval
strategies) would otherwise re-embed the same text hundreds of times — expensive on a
4 GB GPU and slow over the LM Studio HTTP endpoint. Vectors are keyed by
``sha256(model | text)`` and stored in a single memory-mapped ``.npz``-style sidecar.
"""

from __future__ import annotations

import hashlib
import pickle
import threading
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from rag_bot.config import settings
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)


def _digest(model_id: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\x00{text}".encode()).hexdigest()


class CachedEmbeddings(Embeddings):
    """Decorator that memoises document embeddings on disk.

    Query embeddings are *not* cached: they are cheap, unbounded in variety, and
    caching them would grow the store without bound during interactive chat.
    """

    def __init__(self, inner: Embeddings, model_id: str, cache_dir: Path | None = None) -> None:
        self.inner = inner
        self.model_id = model_id
        self.path = (cache_dir or settings.cache_dir) / f"emb_{model_id.replace('/', '_')}.pkl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._store: dict[str, np.ndarray] = self._load()
        self.hits = 0
        self.misses = 0

    def _load(self) -> dict[str, np.ndarray]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("rb") as handle:
                data = pickle.load(handle)
            log.debug("embedding cache %s: %d entries", self.path.name, len(data))
            return data
        except Exception as exc:  # corrupt cache must never break a run
            log.warning("ignoring unreadable cache %s: %s", self.path.name, exc)
            return {}

    def flush(self) -> None:
        with self._lock, self.path.open("wb") as handle:
            pickle.dump(self._store, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [_digest(self.model_id, t) for t in texts]
        missing_positions = [i for i, k in enumerate(keys) if k not in self._store]
        self.hits += len(texts) - len(missing_positions)
        self.misses += len(missing_positions)

        if missing_positions:
            fresh = self.inner.embed_documents([texts[i] for i in missing_positions])
            for position, vector in zip(missing_positions, fresh, strict=True):
                self._store[keys[position]] = np.asarray(vector, dtype=np.float32)
            self.flush()

        return [self._store[k].tolist() for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._store)}


def cached(inner: Embeddings, model_id: str) -> CachedEmbeddings:
    return CachedEmbeddings(inner, model_id)
