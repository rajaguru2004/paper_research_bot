"""Answer-level evaluation with a local LLM-as-a-judge.

Caveat stated up front: the judge is the same small local model that generates the
answers, so scores are indicative, not authoritative. They are used to compare
configurations against each other, never as an absolute quality claim. Refusal
accuracy on out-of-corpus questions is measured separately and *is* objective.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_bot.evaluation.dataset import GOLD_SET, GoldQuestion
from rag_bot.generation.chain import RAGAnswer, RAGPipeline
from rag_bot.generation.prompts import format_context
from rag_bot.logging_utils import get_logger

log = get_logger(__name__)

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict grader. Score the ANSWER on the given criterion from 1 to 5. "
            "Reply with ONLY the digit.",
        ),
        (
            "human",
            "CRITERION: {criterion}\n\nCONTEXT:\n{context}\n\n"
            "QUESTION: {question}\n\nANSWER: {answer}\n\nScore (1-5):",
        ),
    ]
)

CRITERIA = {
    "faithfulness": (
        "Every claim in the answer is supported by the context. Unsupported claims score 1."
    ),
    "relevance": "The answer addresses the question directly and completely.",
    "citation_use": "The answer cites sources with [n] markers where claims are made.",
}

_DIGIT = re.compile(r"[1-5]")


def parse_score(raw: str) -> float:
    match = _DIGIT.search(raw or "")
    return float(match.group()) if match else float("nan")


@dataclass
class AnswerScore:
    question_id: str
    kind: str
    scores: dict[str, float] = field(default_factory=dict)
    refusal_correct: bool | None = None
    cited_expected_paper: bool | None = None
    latency_s: float = 0.0

    def flat(self) -> dict:
        row = {"id": self.question_id, "kind": self.kind, "latency_s": round(self.latency_s, 2)}
        row.update({k: v for k, v in self.scores.items()})
        row["refusal_correct"] = self.refusal_correct
        row["cited_expected_paper"] = self.cited_expected_paper
        return row


class AnswerJudge:
    def __init__(self, llm: BaseChatModel | None = None) -> None:
        if llm is None:
            from rag_bot.generation.llm import get_llm

            llm = get_llm(max_tokens=8)
        self.chain = JUDGE_PROMPT | llm | StrOutputParser()

    def score(self, question: str, answer: str, context: str, criterion_key: str) -> float:
        try:
            raw = self.chain.invoke(
                {
                    "criterion": CRITERIA[criterion_key],
                    "context": context[:4000],
                    "question": question,
                    "answer": answer,
                }
            )
            return parse_score(raw)
        except Exception as exc:
            log.warning("judge failed on %s: %s", criterion_key, exc)
            return float("nan")

    def judge(self, gold: GoldQuestion, result: RAGAnswer) -> AnswerScore:
        score = AnswerScore(question_id=gold.id, kind=gold.kind, latency_s=result.latency_s)

        if not gold.answerable:
            # Objective check: an out-of-corpus question must be refused.
            score.refusal_correct = result.is_refusal
            return score

        context = format_context(result.documents)
        score.cited_expected_paper = any(c.title == gold.expected_title for c in result.citations)
        if result.is_refusal:
            score.scores = dict.fromkeys(CRITERIA, 1.0)
            return score
        score.scores = {
            key: self.score(gold.question, result.answer, context, key) for key in CRITERIA
        }
        return score


def evaluate_answers(
    pipeline: RAGPipeline,
    questions: Sequence[GoldQuestion] = GOLD_SET,
    judge: AnswerJudge | None = None,
) -> tuple[list[AnswerScore], list[RAGAnswer]]:
    """Answer every gold question and grade the results."""
    judge = judge or AnswerJudge()
    scores: list[AnswerScore] = []
    answers: list[RAGAnswer] = []
    for gold in questions:
        result = pipeline.answer(gold.question)
        answers.append(result)
        scores.append(judge.judge(gold, result))
        log.info("judged %s (%s)", gold.id, gold.kind)
    return scores, answers


def summarise(scores: Sequence[AnswerScore]) -> dict:
    """Aggregate judge scores plus objective refusal / citation accuracy."""

    def mean_of(key: str) -> float:
        values = [s.scores[key] for s in scores if key in s.scores and not np.isnan(s.scores[key])]
        return round(float(np.mean(values)), 2) if values else float("nan")

    refusals = [s.refusal_correct for s in scores if s.refusal_correct is not None]
    cited = [s.cited_expected_paper for s in scores if s.cited_expected_paper is not None]
    return {
        **{key: mean_of(key) for key in CRITERIA},
        "refusal_accuracy": round(float(np.mean(refusals)), 2) if refusals else float("nan"),
        "correct_paper_cited": round(float(np.mean(cited)), 2) if cited else float("nan"),
        "mean_latency_s": round(float(np.mean([s.latency_s for s in scores])), 2),
        "n": len(scores),
    }
