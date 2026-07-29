#!/usr/bin/env python
"""Ask the bot one question from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_bot.config import settings  # noqa: E402
from rag_bot.generation.chain import build_pipeline  # noqa: E402
from rag_bot.generation.llm import health_check  # noqa: E402
from rag_bot.logging_utils import setup_logging  # noqa: E402
from rag_bot.retrieval.factory import STRATEGIES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--strategy", default=settings.retrieval_strategy, choices=list(STRATEGIES))
    parser.add_argument("--embed", default=settings.embed_model)
    parser.add_argument("--chunker", default=settings.chunker)
    parser.add_argument("-k", type=int, default=settings.top_k)
    parser.add_argument("--crag", action="store_true", help="enable Corrective RAG")
    parser.add_argument("--snippets", action="store_true", help="print the supporting passages")
    args = parser.parse_args()

    setup_logging()
    ok, detail = health_check()
    if not ok:
        print(f"LLM server unreachable at {settings.lmstudio_base_url}: {detail}", file=sys.stderr)
        return 2

    pipeline = build_pipeline(args.strategy, args.embed, args.chunker, crag=args.crag, k=args.k)
    result = pipeline.answer(" ".join(args.question))
    print("\n" + result.pretty())
    if args.snippets:
        for citation in result.citations:
            print(f"\n--- [{citation.index}] {citation.title} p.{citation.page}\n{citation.snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
