# Research Paper Answer Bot

RAG chatbot over seminal GenAI research papers. **100% open-source / local — no paid API.**

- **LLM**: any OpenAI-compatible local server (LM Studio / Ollama / vLLM)
- **Embeddings**: HuggingFace sentence-transformers (local) + `nomic-embed-text` (served locally)
- **Vector DB**: ChromaDB (persistent, local)
- **Retrieval**: dense · MMR · hybrid BM25+dense (RRF) · cross-encoder rerank · multi-query
- **Stretch goals implemented**: conversational memory, Streamlit UI, Corrective RAG (CRAG) with free web search

Answers are grounded and always cite the **top-3 supporting passages with paper title and page number**.

## Results (k=3, 20-question gold set)

Selected configuration: **recursive** chunking · **all-MiniLM-L6-v2** embeddings · **cross-encoder rerank**
retrieval — chosen by a composite of MRR, nDCG and page_hit (see `reports/experiments_k3.md`).

| Retrieval strategy | hit@3 | MRR | nDCG | page_hit | score | latency |
|---|---|---|---|---|---|---|
| dense | 1.00 | 0.972 | 0.974 | 0.889 | 0.945 | 6 ms |
| mmr | 1.00 | 0.963 | 0.963 | 0.833 | 0.920 | 7 ms |
| bm25 | 0.944 | 0.917 | 0.923 | 0.833 | 0.891 | 5 ms |
| hybrid (RRF) | 1.00 | 0.963 | 0.968 | 0.889 | 0.940 | 13 ms |
| **rerank** | **1.00** | **0.972** | **0.979** | **0.889** | **0.947** | 593 ms |
| multiquery | 1.00 | 0.972 | 0.979 | 0.722 | 0.891 | 586 ms |

Answer level: refusal accuracy **1.00** on out-of-corpus questions, correct paper cited **1.00**,
LLM-as-a-judge faithfulness **3.8/5** (judge is the same small local model — comparative, not absolute).

> `hit@k` saturates on this corpus: 15 topically distinct papers means nearly every configuration
> finds the right paper. MRR and `page_hit` are what discriminate, so selection uses the composite.

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
| `make submission` | build the submission zip (code + notebook + reports + deck) |

## Configuration

All settings live in `.env` (see `.env.example`), read by `src/rag_bot/config.py`:

| Var | Default | Notes |
|---|---|---|
| `LMSTUDIO_BASE_URL` | `http://10.42.80.38:1234/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `liquid/lfm2.5-1.2b` | any chat model your server can load |
| `LMSTUDIO_EMBED_MODEL` | `text-embedding-nomic-embed-text-v1.5` | served embedding model |
| `EMBED_MODEL` | `minilm` | default embedding key in the registry |
| `CHUNKER` | `recursive` | `fixed` \| `recursive` \| `semantic` |
| `RETRIEVAL_STRATEGY` | `rerank` | `dense` \| `mmr` \| `hybrid` \| `rerank` \| `multiquery` |
| `TOP_K` | `3` | passages cited per answer |

## Layout

```
notebooks/capstone.ipynb   # graded deliverable: Steps 1-8, all experiments, executed end to end
scripts/make_notebook.py   # generates the notebook (edit here, not the .ipynb JSON)
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
app/streamlit_app.py   deck/   reports/   tests/
```

## Testing

```bash
make test        # 73 unit tests — no network, no server, fake LLM
make test-all    # adds 6 integration tests against the live LM Studio server + built index
make lint        # ruff + mypy, both clean
```

Integration tests skip themselves (rather than fail) when the server is unreachable or the index has
not been built.

## Notes on the local LLM server

`GET /v1/models` on the configured server advertises several chat models, but LM Studio only reports
a load failure at first completion — models that do not fit in the host's VRAM fail with
`"exited before becoming healthy"`. Everything LLM-dependent is therefore behind `LLM_MODEL`, and
degrades gracefully: query rewriting falls back to the original query, CRAG grading falls back to
"keep the passage", and generation errors are reported rather than raised.
