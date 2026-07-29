"""The paper corpus: curated arXiv IDs with their exact titles.

Titles are curated rather than scraped from page 1 because PDF text extraction
mangles title lines (ligatures, column order, author blocks) and citations must
show the *correct* paper title.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperSpec:
    arxiv_id: str
    title: str
    year: int
    topic: str

    @property
    def slug(self) -> str:
        return self.arxiv_id.replace(".", "_").replace("/", "_")

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}"

    @property
    def filename(self) -> str:
        return f"{self.slug}.pdf"


PAPERS: tuple[PaperSpec, ...] = (
    PaperSpec("1706.03762", "Attention Is All You Need", 2017, "architecture"),
    PaperSpec(
        "1810.04805",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        2018,
        "pretraining",
    ),
    PaperSpec("2005.14165", "Language Models are Few-Shot Learners", 2020, "scaling"),
    PaperSpec(
        "2005.11401",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        2020,
        "rag",
    ),
    PaperSpec(
        "2004.04906", "Dense Passage Retrieval for Open-Domain Question Answering", 2020, "rag"
    ),
    PaperSpec(
        "2203.02155",
        "Training language models to follow instructions with human feedback",
        2022,
        "alignment",
    ),
    PaperSpec("2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models", 2021, "peft"),
    PaperSpec("2203.15556", "Training Compute-Optimal Large Language Models", 2022, "scaling"),
    PaperSpec("2307.09288", "Llama 2: Open Foundation and Fine-Tuned Chat Models", 2023, "models"),
    PaperSpec(
        "2201.11903",
        "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        2022,
        "prompting",
    ),
    PaperSpec(
        "2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models", 2022, "agents"
    ),
    PaperSpec("2401.04088", "Mixtral of Experts", 2024, "models"),
    PaperSpec(
        "2205.14135",
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        2022,
        "efficiency",
    ),
    PaperSpec(
        "2310.11511",
        "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        2023,
        "rag",
    ),
    PaperSpec(
        "2401.15884", "Corrective Retrieval Augmented Generation", 2024, "rag"
    ),
)

TITLE_BY_ID: dict[str, str] = {p.arxiv_id: p.title for p in PAPERS}
SPEC_BY_FILENAME: dict[str, PaperSpec] = {p.filename: p for p in PAPERS}
