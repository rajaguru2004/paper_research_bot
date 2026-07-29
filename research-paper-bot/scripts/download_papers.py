#!/usr/bin/env python
"""Download the arXiv paper corpus into data/raw/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_bot.ingest.download import download_corpus  # noqa: E402
from rag_bot.logging_utils import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args()

    setup_logging()
    paths = download_corpus(force=args.force, delay=args.delay)
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
