"""Logging setup shared by scripts, notebook and app."""

from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np

from rag_bot.config import settings

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, with a compact single-line format."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, (level or settings.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Third-party libraries are chatty at INFO.
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "chromadb",
        "sentence_transformers",
        "openai",
        "primp",  # ddgs HTTP client — logs every search backend it tries
        "ddgs",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def set_seed(seed: int | None = None) -> int:
    """Seed every RNG we can reach, for reproducible runs."""
    seed = settings.random_seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch optional at import time
        pass
    return seed
