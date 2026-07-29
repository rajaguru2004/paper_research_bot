# Experiment Results

Corpus: **15 papers**, 512 pages (2 skipped as figure/cover pages). Metrics computed at k=5 over the 18 answerable gold questions.

## 1. Chunking strategy (embedding fixed: bge-small, retrieval: dense)

| config    |   k |   hit@5 |   MRR |   nDCG |   page_hit |   latency_ms |   chunks |   mean_chars |   chunk_s |   index_s |
|:----------|----:|--------:|------:|-------:|-----------:|-------------:|---------:|-------------:|----------:|----------:|
| fixed     |   5 |       1 | 0.956 |  0.954 |      0.833 |          9.4 |     1212 |         1485 |       5.7 |      21.4 |
| recursive |   5 |       1 | 0.935 |  0.959 |      0.889 |          7.7 |     2100 |          860 |       0.1 |       2.1 |
| semantic  |   5 |       1 | 0.903 |  0.913 |      0.778 |         11.2 |     2067 |          790 |      61.8 |      19.7 |

**Selected: `fixed`**

![chunking](figures/chunking.png)

## 2. Embedding model (chunker fixed: fixed, retrieval: dense)

| config    |   k |   hit@5 |   MRR |   nDCG |   page_hit |   latency_ms |   dim | provider    |   index_s |
|:----------|----:|--------:|------:|-------:|-----------:|-------------:|------:|:------------|----------:|
| minilm    |   5 |       1 | 0.944 |  0.964 |      0.889 |          5.7 |   384 | huggingface |       5.2 |
| bge-small |   5 |       1 | 0.956 |  0.954 |      0.833 |          7.9 |   384 | huggingface |       1.3 |
| nomic     |   5 |       1 | 0.972 |  0.979 |      0.889 |         13.6 |   768 | lmstudio    |      30.9 |

**Selected: `nomic`**

![embeddings](figures/embeddings.png)

## 3. Retrieval strategy (embedding: nomic, chunker: fixed)

| config     |   k |   hit@5 |   MRR |   nDCG |   page_hit |   latency_ms | description                                              |
|:-----------|----:|--------:|------:|-------:|-----------:|-------------:|:---------------------------------------------------------|
| dense      |   5 |       1 | 0.972 |  0.979 |      0.889 |         12.1 | Cosine similarity over embeddings (baseline).            |
| mmr        |   5 |       1 | 0.972 |  0.965 |      0.611 |         14.7 | Maximal Marginal Relevance — relevance with diversity.   |
| bm25       |   5 |       1 | 0.928 |  0.951 |      0.889 |          2.7 | Sparse lexical keyword matching.                         |
| hybrid     |   5 |       1 | 0.972 |  0.977 |      0.889 |         17.3 | BM25 + dense fused with weighted Reciprocal Rank Fusion. |
| rerank     |   5 |       1 | 0.958 |  0.97  |      0.889 |        719.4 | Hybrid candidates reranked by a cross-encoder.           |
| multiquery |   5 |       1 | 1     |  0.997 |      0.833 |        621.6 | LLM query expansion, results fused with RRF.             |

**Selected: `multiquery`**

![retrieval](figures/retrieval.png)

## Final configuration

- chunker: `fixed`
- embedding: `nomic` (dim 768)
- retrieval: `multiquery`
- LLM: `liquid/lfm2.5-1.2b` via http://10.42.80.38:1234/v1
