"""Download the paper corpus from arXiv (open access, no API key)."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_bot.config import settings
from rag_bot.ingest.corpus import PAPERS, PaperSpec
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)

_HEADERS = {"User-Agent": "research-paper-bot/0.1 (capstone project; contact: local user)"}
_MIN_PDF_BYTES = 20_000  # anything smaller is an error page, not a paper


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch(url: str) -> bytes:
    with httpx.Client(timeout=120.0, follow_redirects=True, headers=_HEADERS) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def download_paper(spec: PaperSpec, dest_dir: Path, *, force: bool = False) -> Path:
    """Download one paper; returns the local path. Skips existing valid files."""
    target = dest_dir / spec.filename
    if target.exists() and target.stat().st_size > _MIN_PDF_BYTES and not force:
        log.debug("cached: %s", spec.filename)
        return target

    payload = _fetch(spec.pdf_url)
    if len(payload) < _MIN_PDF_BYTES or not payload.startswith(b"%PDF"):
        raise RuntimeError(f"{spec.arxiv_id}: response is not a PDF ({len(payload)} bytes)")

    target.write_bytes(payload)
    log.info("downloaded %-14s %6.1f KB  %s", spec.arxiv_id, len(payload) / 1024, spec.title[:52])
    return target


def download_corpus(*, force: bool = False, delay: float = 1.0) -> list[Path]:
    """Download every paper in the corpus. Polite delay between requests."""
    settings.ensure_dirs()
    paths: list[Path] = []
    for i, spec in enumerate(PAPERS):
        try:
            paths.append(download_paper(spec, settings.raw_dir, force=force))
        except Exception as exc:  # keep going; a missing paper is not fatal
            log.error("failed %s: %s", spec.arxiv_id, exc)
            continue
        if i < len(PAPERS) - 1:
            time.sleep(delay)
    log.info("corpus ready: %d/%d papers in %s", len(paths), len(PAPERS), settings.raw_dir)
    return paths
