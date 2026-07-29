#!/usr/bin/env python
"""Run the full experiment grid and write reports/experiments.md + figures.

Three experiments, in the order the capstone brief asks for them:

1. chunking strategy   (fixed | recursive | semantic)
2. embedding model     (minilm | bge-small | nomic)
3. retrieval strategy  (dense | mmr | bm25 | hybrid | rerank | multiquery)

Each stage fixes the winner of the previous one — a full cross-product would be
3x3x6 = 54 index builds, which buys little on a 15-paper corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rag_bot.config import settings  # noqa: E402
from rag_bot.embeddings.registry import EMBEDDING_MODELS, get_embeddings, get_spec  # noqa: E402
from rag_bot.evaluation.retrieval_metrics import evaluate_retriever  # noqa: E402
from rag_bot.ingest.chunking import chunk_documents, chunk_stats  # noqa: E402
from rag_bot.ingest.loader import load_corpus  # noqa: E402
from rag_bot.logging_utils import get_logger, set_seed, setup_logging  # noqa: E402
from rag_bot.retrieval.factory import DESCRIPTIONS, build_retriever  # noqa: E402
from rag_bot.store.chroma_store import build_index, get_store  # noqa: E402

log = get_logger("experiments")

CHUNKERS = ("fixed", "recursive", "semantic")
EMBEDDINGS = ("minilm", "bge-small", "nomic")
STRATEGIES = ("dense", "mmr", "bm25", "hybrid", "rerank", "multiquery")


def bar_chart(frame: pd.DataFrame, x: str, cols: list[str], title: str, path: Path) -> None:
    ax = frame.plot(x=x, y=cols, kind="bar", figsize=(8, 4.5), rot=20, width=0.78)
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    log.info("wrote %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", type=int, default=5, help="k used for retrieval metrics")
    parser.add_argument("--skip-semantic", action="store_true", help="semantic chunking is slow")
    parser.add_argument("--skip-multiquery", action="store_true", help="needs the LLM server")
    args = parser.parse_args()

    setup_logging()
    set_seed()
    settings.ensure_dirs()
    figures = settings.reports_dir / "figures"
    results: dict[str, list[dict]] = {}

    # ---------- Experiment 1: chunking -----------------------------------
    log.info("=== experiment 1: chunking ===")
    pages, load_report = load_corpus()
    chunk_rows = []
    for chunker in CHUNKERS:
        if chunker == "semantic" and args.skip_semantic:
            continue
        started = time.perf_counter()
        chunks = chunk_documents(pages, chunker)
        chunk_seconds = time.perf_counter() - started
        stats = chunk_stats(chunks, chunker)
        _, index_report = build_index(chunks, embed_key="bge-small", chunker=chunker)
        evaluation = evaluate_retriever(
            build_retriever("dense", get_store("bge-small", chunker)), k=args.k, label=chunker
        )
        chunk_rows.append(
            {
                **evaluation.row(),
                "chunks": stats.n_chunks,
                "mean_chars": round(stats.mean_chars),
                "chunk_s": round(chunk_seconds, 1),
                "index_s": round(index_report.build_seconds, 1),
            }
        )
    chunk_frame = pd.DataFrame(chunk_rows)
    best_chunker = str(chunk_frame.sort_values("MRR", ascending=False).iloc[0]["config"])
    results["chunking"] = chunk_rows
    log.info("best chunker: %s", best_chunker)

    # ---------- Experiment 2: embedding models ---------------------------
    log.info("=== experiment 2: embeddings ===")
    best_chunks = chunk_documents(pages, best_chunker)
    embed_rows = []
    for key in EMBEDDINGS:
        try:
            get_embeddings(key)  # fail fast if a model/server is unavailable
            _, index_report = build_index(best_chunks, embed_key=key, chunker=best_chunker)
            evaluation = evaluate_retriever(
                build_retriever("dense", get_store(key, best_chunker)), k=args.k, label=key
            )
            embed_rows.append(
                {
                    **evaluation.row(),
                    "dim": get_spec(key).dim,
                    "provider": get_spec(key).provider,
                    "index_s": round(index_report.build_seconds, 1),
                }
            )
        except Exception as exc:
            log.error("embedding %s failed: %s", key, exc)
            embed_rows.append({"config": key, "error": str(exc)[:120]})
    embed_frame = pd.DataFrame(embed_rows)
    usable = embed_frame[embed_frame.get("MRR").notna()] if "MRR" in embed_frame else embed_frame
    best_embed = str(usable.sort_values("MRR", ascending=False).iloc[0]["config"])
    results["embeddings"] = embed_rows
    log.info("best embedding: %s", best_embed)

    # ---------- Experiment 3: retrieval strategies -----------------------
    log.info("=== experiment 3: retrieval ===")
    store = get_store(best_embed, best_chunker)
    strategy_rows = []
    for strategy in STRATEGIES:
        if strategy == "multiquery" and args.skip_multiquery:
            continue
        try:
            evaluation = evaluate_retriever(
                build_retriever(strategy, store), k=args.k, label=strategy
            )
            strategy_rows.append({**evaluation.row(), "description": DESCRIPTIONS[strategy]})
        except Exception as exc:
            log.error("strategy %s failed: %s", strategy, exc)
            strategy_rows.append({"config": strategy, "error": str(exc)[:120]})
    strategy_frame = pd.DataFrame(strategy_rows)
    best_strategy = str(strategy_frame.sort_values("MRR", ascending=False).iloc[0]["config"])
    results["retrieval"] = strategy_rows
    log.info("best strategy: %s", best_strategy)

    # ---------- figures + report ----------------------------------------
    metric_cols = [f"hit@{args.k}", "MRR", "nDCG"]
    bar_chart(chunk_frame, "config", metric_cols, "Chunking strategy", figures / "chunking.png")
    bar_chart(usable, "config", metric_cols, "Embedding model", figures / "embeddings.png")
    bar_chart(
        strategy_frame.dropna(subset=["MRR"]),
        "config",
        metric_cols,
        "Retrieval strategy",
        figures / "retrieval.png",
    )

    report_path = settings.reports_dir / "experiments.md"
    report_path.write_text(
        "\n".join(
            [
                "# Experiment Results",
                "",
                f"Corpus: **{load_report.files} papers**, {load_report.pages_kept} pages "
                f"({load_report.pages_skipped} skipped as figure/cover pages). "
                f"Metrics computed at k={args.k} over the 18 answerable gold questions.",
                "",
                "## 1. Chunking strategy (embedding fixed: bge-small, retrieval: dense)",
                "",
                chunk_frame.to_markdown(index=False),
                "",
                f"**Selected: `{best_chunker}`**",
                "",
                "![chunking](figures/chunking.png)",
                "",
                f"## 2. Embedding model (chunker fixed: {best_chunker}, retrieval: dense)",
                "",
                embed_frame.to_markdown(index=False),
                "",
                f"**Selected: `{best_embed}`**",
                "",
                "![embeddings](figures/embeddings.png)",
                "",
                f"## 3. Retrieval strategy (embedding: {best_embed}, chunker: {best_chunker})",
                "",
                strategy_frame.to_markdown(index=False),
                "",
                f"**Selected: `{best_strategy}`**",
                "",
                "![retrieval](figures/retrieval.png)",
                "",
                "## Final configuration",
                "",
                f"- chunker: `{best_chunker}`",
                f"- embedding: `{best_embed}` (dim {get_spec(best_embed).dim})",
                f"- retrieval: `{best_strategy}`",
                f"- LLM: `{settings.llm_model}` via {settings.lmstudio_base_url}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (settings.reports_dir / "experiments.json").write_text(
        json.dumps(
            {
                "results": results,
                "best": {
                    "chunker": best_chunker,
                    "embedding": best_embed,
                    "strategy": best_strategy,
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {report_path}")
    print(f"best config: chunker={best_chunker} embed={best_embed} strategy={best_strategy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
