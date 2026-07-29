# Research Paper Answer Bot

RAG chatbot over seminal GenAI research papers. **100% open-source / local — no paid API.**

- **LLM**: any OpenAI-compatible local server (LM Studio / Ollama / vLLM)
- **Embeddings**: HuggingFace sentence-transformers (local) + `nomic-embed-text` (served locally)
- **Vector DB**: ChromaDB (persistent, local)
- **Retrieval**: dense · MMR · hybrid BM25+dense (RRF) · cross-encoder rerank · multi-query
- **Stretch goals implemented**: conversational memory, Streamlit UI, Corrective RAG (CRAG) with free web search

Answers are grounded and always cite the **top-3 supporting passages with paper title and page number**.

---

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
cp .env.example .env          # point LMSTUDIO_BASE_URL at your server

make papers                   # download ~15 arXiv PDFs into data/raw
make index                    # chunk + embed + persist to Chroma
make ask Q="What is multi-head attention?"
make app                      # Streamlit UI on :8501
```

## Make targets

| Target | What it does |
|---|---|
| `make install` | create venv + install package with dev extras |
| `make papers` | download the paper corpus from arXiv |
| `make index` | build the Chroma index (default embedding + chunker) |
| `make experiments` | run embedding + retrieval comparisons → `reports/experiments.md` |
| `make ask Q="..."` | one-shot question with citations |
| `make app` | launch the Streamlit chat app |
| `make test` | unit tests (no network) |
| `make lint` | ruff + mypy |
| `make notebook` | execute `notebooks/capstone.ipynb` end-to-end |

## Configuration

All settings live in `.env` (see `.env.example`), read by `src/rag_bot/config.py`:

| Var | Default | Notes |
|---|---|---|
| `LMSTUDIO_BASE_URL` | `http://10.42.80.38:1234/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `liquid/lfm2.5-1.2b` | any chat model your server can load |
| `LMSTUDIO_EMBED_MODEL` | `text-embedding-nomic-embed-text-v1.5` | served embedding model |
| `EMBED_MODEL` | `bge-small` | default embedding key in the registry |
| `CHUNKER` | `recursive` | `fixed` \| `recursive` \| `semantic` |
| `RETRIEVAL_STRATEGY` | `rerank` | `dense` \| `mmr` \| `hybrid` \| `rerank` \| `multiquery` |
| `TOP_K` | `3` | passages cited per answer |

## Layout

```
src/rag_bot/
  config.py  logging_utils.py
  ingest/     download.py loader.py chunking.py
  embeddings/ lmstudio.py registry.py cache.py
  store/      chroma_store.py
  retrieval/  dense.py hybrid.py rerank.py multiquery.py factory.py
  generation/ llm.py prompts.py chain.py
  crag/       grader.py websearch.py pipeline.py
  memory.py
  evaluation/ dataset.py retrieval_metrics.py answer_eval.py
app/streamlit_app.py   notebooks/capstone.ipynb   scripts/   tests/
```
