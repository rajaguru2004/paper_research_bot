# Experiment Results

Corpus: **15 papers**, 512 pages (2 skipped as figure/cover pages). Metrics computed at k=3 over the 18 answerable gold questions. Selection uses `score` = mean(MRR, nDCG, page_hit): MRR alone rewards ranking the right *paper* highly even when the wrong *page* is retrieved.

## 1. Chunking strategy (embedding fixed: bge-small, retrieval: dense)

| config    |   k |   hit@3 |   MRR |   nDCG |   page_hit |   score |   latency_ms |   chunks |   mean_chars |   chunk_s |   index_s |
|:----------|----:|--------:|------:|-------:|-----------:|--------:|-------------:|---------:|-------------:|----------:|----------:|
| fixed     |   3 |   0.944 | 0.944 |  0.944 |      0.722 |   0.87  |         19.2 |     1212 |         1485 |       0.3 |       1.5 |
| recursive |   3 |   1     | 0.935 |  0.955 |      0.778 |   0.889 |          8.3 |     2100 |          860 |       0.1 |       2   |
| semantic  |   3 |   0.944 | 0.889 |  0.894 |      0.611 |   0.798 |          8.7 |     2067 |          790 |      61.5 |       2.7 |

**Selected: `recursive`**

![chunking](figures/chunking_k3.png)

## 2. Embedding model (chunker fixed: recursive, retrieval: dense)

| config    |   k |   hit@3 |   MRR |   nDCG |   page_hit |   score |   latency_ms |   dim | provider    |   index_s |
|:----------|----:|--------:|------:|-------:|-----------:|--------:|-------------:|------:|:------------|----------:|
| minilm    |   3 |       1 | 0.972 |  0.974 |      0.889 |   0.945 |          5.9 |   384 | huggingface |       9   |
| bge-small |   3 |       1 | 0.935 |  0.955 |      0.778 |   0.889 |          8.8 |   384 | huggingface |       2.4 |
| nomic     |   3 |       1 | 0.972 |  0.983 |      0.833 |   0.93  |         11.9 |   768 | lmstudio    |      35.7 |

**Selected: `minilm`**

![embeddings](figures/embeddings_k3.png)

## 3. Retrieval strategy (embedding: minilm, chunker: recursive)

| config     |   k |   hit@3 |   MRR |   nDCG |   page_hit |   score |   latency_ms | description                                              |
|:-----------|----:|--------:|------:|-------:|-----------:|--------:|-------------:|:---------------------------------------------------------|
| dense      |   3 |   1     | 0.972 |  0.974 |      0.889 |   0.945 |          6.1 | Cosine similarity over embeddings (baseline).            |
| mmr        |   3 |   1     | 0.963 |  0.963 |      0.833 |   0.92  |          7   | Maximal Marginal Relevance — relevance with diversity.   |
| bm25       |   3 |   0.944 | 0.917 |  0.923 |      0.833 |   0.891 |          4.7 | Sparse lexical keyword matching.                         |
| hybrid     |   3 |   1     | 0.963 |  0.968 |      0.889 |   0.94  |         12.7 | BM25 + dense fused with weighted Reciprocal Rank Fusion. |
| rerank     |   3 |   1     | 0.972 |  0.979 |      0.889 |   0.947 |        592.5 | Hybrid candidates reranked by a cross-encoder.           |
| multiquery |   3 |   1     | 0.972 |  0.979 |      0.722 |   0.891 |        586.1 | LLM query expansion, results fused with RRF.             |

**Selected: `rerank`**

![retrieval](figures/retrieval_k3.png)

## Final configuration

- chunker: `recursive`
- embedding: `minilm` (dim 384)
- retrieval: `rerank`
- LLM: `liquid/lfm2.5-1.2b` via http://10.42.80.38:1234/v1
