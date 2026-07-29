#!/usr/bin/env python
"""Load PDFs, chunk them, embed and persist into ChromaDB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_bot.config import settings
from rag_bot.embeddings.registry import EMBEDDING_MODELS
from rag_bot.ingest.chunking import CHUNKER_NAMES, chunk_documents, chunk_stats
from rag_bot.ingest.loader import load_corpus
from rag_bot.logging_utils import get_logger, set_seed, setup_logging
from rag_bot.store.chroma_store import build_index

log = get_logger("build_index")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed", default=settings.embed_model, choices=sorted(EMBEDDING_MODELS))
    parser.add_argument("--chunker", default=settings.chunker, choices=list(CHUNKER_NAMES))
    parser.add_argument("--no-cache", action="store_true", help="bypass the embedding cache")
    parser.add_argument("--keep", action="store_true", help="append instead of resetting")
    args = parser.parse_args()

    setup_logging()
    set_seed()

    documents, report = load_corpus()
    log.info(
        "corpus: %d files, %d pages kept, %d skipped",
        report.files,
        report.pages_kept,
        report.pages_skipped,
    )

    chunks = chunk_documents(documents, args.chunker)
    log.info("%s", chunk_stats(chunks, args.chunker))

    _, index_report = build_index(
        chunks,
        embed_key=args.embed,
        chunker=args.chunker,
        reset=not args.keep,
        use_cache=not args.no_cache,
    )
    print(
        f"\nCollection '{index_report.collection}': {index_report.n_chunks} chunks, "
        f"dim={index_report.dim}, built in {index_report.build_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
