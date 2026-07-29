# Presentation Outline — Research Paper Answer Bot

12 slides, matching section 6.2 of the capstone brief. Numbers marked `⟨…⟩` are filled from
`reports/experiments_k3.md` after the final run — do not hand-type them, copy from the report.

---

## Slide 1 — Title
**Research Paper Answer Bot**
RAG over 15 seminal GenAI papers · grounded answers with paper title + page citations
Name · GenAI Pinnacle Plus Program · 2026

---

## Slide 2 — Problem Statement
- Researchers need answers *from* papers, not a search results page.
- General LLMs hallucinate citations — confidently wrong page numbers are worse than no answer.
- **Goal:** a chatbot that answers only from a curated corpus and shows the top-3 supporting passages
  with paper title and page number.
- **Constraint taken on deliberately:** zero paid APIs. Every model runs locally or on a local
  OpenAI-compatible server.

---

## Slide 3 — Architecture Overview
Render `deck/architecture.mmd` (Mermaid). Four stages:
1. **Ingestion (offline)** — PDFs → per-page Documents → chunks → embeddings → ChromaDB
2. **Retrieval** — condense with history → dense + BM25 → RRF fusion → cross-encoder rerank
3. **Corrective RAG** — grade passages → web fallback when evidence is weak
4. **Generation** — numbered context → grounded prompt → LLM → answer + 3 citations

Say out loud: *metadata (`title`, `page`) is attached at load time and survives every stage* — that
is what makes citation correct by construction.

---

## Slide 4 — Data
- 15 arXiv papers: Attention, BERT, GPT-3, RAG, DPR, InstructGPT, LoRA, Chinchilla, Llama 2, CoT,
  ReAct, Mixtral, FlashAttention, Self-RAG, CRAG
- ⟨514⟩ pages parsed, ⟨2⟩ skipped (cover/figure-only pages, flagged not silently dropped)
- Cleaning: ligature repair, hyphen-across-linebreak rejoin, control-character strip
- Titles from a curated `arxiv_id → title` map, **not** scraped from page 1 (column-shuffled text
  produces wrong titles → wrong citations)

Include the pages-per-paper and characters-per-page charts from the notebook.

---

## Slide 5 — Chunking Strategy
| Strategy | Chunks | Mean chars | Preprocess time |
|---|---|---|---|
| fixed (512 tok / 64) | ⟨1212⟩ | ⟨1485⟩ | ⟨5.7 s⟩ |
| recursive (1000 / 150) | ⟨2100⟩ | ⟨860⟩ | ⟨0.1 s⟩ |
| semantic (percentile 90) | ⟨2067⟩ | ⟨790⟩ | ⟨61.8 s⟩ |

Point: semantic chunking costs ~600× the preprocessing time of recursive. Selection is by measured
retrieval quality, not by which sounds most sophisticated.

---

## Slide 6 — Embeddings
| Model | Dim | Served by | MRR | page_hit | Index time |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | local GPU | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| BAAI/bge-small-en-v1.5 | 384 | local GPU | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| nomic-embed-text-v1.5 | 768 | LM Studio HTTP | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

Two points worth making:
- The brief's "commercial" slot is filled by a **locally served** HTTP embedding model — same
  architectural role, zero cost.
- BGE and Nomic are **asymmetric**: a query-side instruction prefix is part of the model contract.
  Omitting it silently costs recall.

---

## Slide 7 — Vector DB & Retrieval
- **ChromaDB**, persistent, cosine. One collection per (embedding × chunker) — mixing 384-d and
  768-d vectors is invalid, and separate collections make every experiment re-runnable.
- Six strategies compared: dense · MMR · BM25 · hybrid (RRF) · cross-encoder rerank · multi-query
- **RRF, not score blending** — BM25 scores and cosine similarities are on incompatible scales; only
  ranks are comparable.

Include the retrieval-strategy bar chart + the log-scale latency chart.

---

## Slide 8 — RAG Pipeline
LCEL composition:
```
RunnablePassthrough.assign(context=format_context) | ChatPromptTemplate | ChatOpenAI | StrOutputParser
```
Prompt rules: answer only from context · fixed refusal phrase when context is insufficient ·
cite with `[n]` · never invent numbers or pages.
Context block is numbered and carries title + page, so the model has something real to cite.

---

## Slide 9 — Results & Demo
- 5 live queries with top-3 citations (title + page + score)
- One deliberately out-of-corpus question → the bot refuses instead of inventing an answer
- Evaluation: 20 gold questions (18 answerable, 2 unanswerable)
  - retrieval: hit@3 ⟨…⟩, MRR ⟨…⟩, page_hit ⟨…⟩
  - refusal accuracy ⟨…⟩, correct paper cited ⟨…⟩

---

## Slide 10 — Stretch Goals (all three)
1. **Conversational memory** — per-session history + query condensation so "what about *its*
   limitations?" retrieves correctly. Most naive chat-RAG skips the condense step and retrieves on a pronoun.
2. **Streamlit app** — chat UI, source cards, live switches for embedding / strategy / k / CRAG.
3. **Corrective RAG** — grade passages → keep, top up from DuckDuckGo, or fall back entirely.
   Grader is conservative by design: unparseable grade ⇒ keep the passage.

---

## Slide 11 — Challenges & Learnings
- **`hit@k` saturated at 1.00.** 15 distinct papers + paper-level ground truth = a metric that
  cannot discriminate. Switched the decision to MRR and page_hit. *Naming a metric you got wrong is
  stronger than hiding it.*
- **Only a 1.2B chat model would load** on the server (larger models fail: "exited before becoming
  healthy"). Every LLM-dependent stage — grading, rewriting, judging — was made defensive, with the
  model behind one config value.
- **MMR hurt page_hit** — diversification is the wrong objective for pinpoint factual lookup.
- **LLM-as-a-judge shares the generator's weaknesses** — reported as comparative, not absolute.

---

## Slide 12 — Conclusion & Future Work
Final config: ⟨chunker⟩ + ⟨embedding⟩ + ⟨retrieval⟩ + `LLM_MODEL` via LM Studio. Zero paid APIs.
Next: larger chat model (retrieval stack unchanged) · table/figure extraction · parent-document
retrieval (embed small, return large) · passage-level human labels to replace paper-level proxy truth.

---

### Demo script (5 queries, in order)
1. "What is scaled dot-product attention and why is it scaled?" → Attention, p.4
2. "What are the two pre-training objectives used by BERT?" → BERT
3. "In LoRA, what does the rank r control?" → LoRA
4. "What about its limitations?" *(follow-up — shows memory + query condensation)*
5. "Which team won the 2018 FIFA World Cup?" → refusal (then toggle CRAG on to show web fallback)
