"""Gold evaluation set.

20 questions spanning 10 papers: factual lookups, multi-hop comparisons, and
out-of-corpus questions that the system *must* refuse. Ground truth is the paper the
answer lives in, which makes retrieval measurable without human labelling of passages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    expected_title: str | None  # None => unanswerable from the corpus
    kind: str  # factual | multihop | unanswerable
    expected_pages: tuple[int, ...] = ()
    note: str = ""

    @property
    def answerable(self) -> bool:
        return self.expected_title is not None


GOLD_SET: tuple[GoldQuestion, ...] = (
    GoldQuestion(
        "q01",
        "What is scaled dot-product attention and why is it scaled by the square root of d_k?",
        "Attention Is All You Need",
        "factual",
        (4,),
    ),
    GoldQuestion(
        "q02",
        "How many encoder and decoder layers does the base Transformer use?",
        "Attention Is All You Need",
        "factual",
        (3, 9),
    ),
    GoldQuestion(
        "q03",
        "What are the two pre-training objectives used by BERT?",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "factual",
        (4, 5),
    ),
    GoldQuestion(
        "q04",
        "How many parameters does the largest GPT-3 model have?",
        "Language Models are Few-Shot Learners",
        "factual",
        (8,),
    ),
    GoldQuestion(
        "q05",
        "What is the difference between RAG-Sequence and RAG-Token?",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "factual",
        (3, 4),
    ),
    GoldQuestion(
        "q06",
        "How is the dual-encoder trained in Dense Passage Retrieval and what negatives are used?",
        "Dense Passage Retrieval for Open-Domain Question Answering",
        "factual",
        (3, 4),
    ),
    GoldQuestion(
        "q07",
        "What are the three steps of the InstructGPT training pipeline?",
        "Training language models to follow instructions with human feedback",
        "factual",
        (3,),
    ),
    GoldQuestion(
        "q08",
        "In LoRA, which matrices are decomposed and what does the rank r control?",
        "LoRA: Low-Rank Adaptation of Large Language Models",
        "factual",
        (4, 5),
    ),
    GoldQuestion(
        "q09",
        "What ratio of tokens to parameters does the Chinchilla scaling law recommend?",
        "Training Compute-Optimal Large Language Models",
        "factual",
        (1, 2, 3),
    ),
    GoldQuestion(
        "q10",
        "What is grouped-query attention and why does Llama 2 use it?",
        "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "factual",
        (5, 6),
    ),
    GoldQuestion(
        "q11",
        "How does chain-of-thought prompting differ from standard few-shot prompting?",
        "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "factual",
        (1, 2),
    ),
    GoldQuestion(
        "q12",
        "What does ReAct interleave and what problem does that solve?",
        "ReAct: Synergizing Reasoning and Acting in Language Models",
        "factual",
        (1, 2, 3),
    ),
    GoldQuestion(
        "q13",
        "How many experts does Mixtral use per token and how many are active?",
        "Mixtral of Experts",
        "factual",
        (1, 2),
    ),
    GoldQuestion(
        "q14",
        "Why is FlashAttention faster despite computing exact attention?",
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "factual",
        (1, 4, 5),
    ),
    GoldQuestion(
        "q15",
        "What are reflection tokens in Self-RAG?",
        "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "factual",
        (2, 3, 4),
    ),
    GoldQuestion(
        "q16",
        "What does the retrieval evaluator in Corrective RAG do when retrieval is judged wrong?",
        "Corrective Retrieval Augmented Generation",
        "multihop",
        (3, 4),
    ),
    GoldQuestion(
        "q17",
        "Compare how LoRA and full fine-tuning differ in the number of trainable parameters.",
        "LoRA: Low-Rank Adaptation of Large Language Models",
        "multihop",
        (1, 2, 5),
    ),
    GoldQuestion(
        "q18",
        "How does the sparse mixture-of-experts routing in Mixtral relate to inference cost?",
        "Mixtral of Experts",
        "multihop",
        (1, 2, 3),
    ),
    GoldQuestion(
        "q19",
        "What is the current stock price of NVIDIA?",
        None,
        "unanswerable",
        note="Out of corpus — the system must refuse.",
    ),
    GoldQuestion(
        "q20",
        "Which football team won the 2018 FIFA World Cup?",
        None,
        "unanswerable",
        note="Out of corpus — the system must refuse.",
    ),
)

ANSWERABLE = tuple(q for q in GOLD_SET if q.answerable)
UNANSWERABLE = tuple(q for q in GOLD_SET if not q.answerable)
