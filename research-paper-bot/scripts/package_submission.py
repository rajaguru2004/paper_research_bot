#!/usr/bin/env python
"""Build the submission zip: code, notebook, reports, deck outline.

Excludes the corpus PDFs, the Chroma index and caches — they are reproducible with
`make papers && make index`, and shipping ~130 MB of derived data helps nobody.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INCLUDE_FILES = (
    "README.md",
    "Makefile",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    "notebooks/capstone.ipynb",
)
INCLUDE_DIRS = ("src", "tests", "scripts", "app", "deck", "reports")
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def collect() -> list[Path]:
    paths = [ROOT / name for name in INCLUDE_FILES]
    for directory in INCLUDE_DIRS:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and not EXCLUDE_PARTS.intersection(path.parts):
                paths.append(path)
    return [p for p in paths if p.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "submission_research_paper_bot.zip"))
    args = parser.parse_args()

    notebook = ROOT / "notebooks" / "capstone.ipynb"
    if not notebook.exists():
        print("notebooks/capstone.ipynb is missing — run `make notebook` first", file=sys.stderr)
        return 1

    files = collect()
    out = Path(args.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))

    size_mb = out.stat().st_size / 1_000_000
    print(f"wrote {out} — {len(files)} files, {size_mb:.1f} MB")
    print("excluded: data/raw PDFs, chroma/, .cache/ (rebuild with `make papers && make index`)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
