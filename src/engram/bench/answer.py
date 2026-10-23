"""Answering and judging prompts, shared by every system so the comparison is fair."""

from __future__ import annotations

import re

from engram.llm import LLM, Completion

ANSWER_SYSTEM = (
    "You answer questions about a long conversation between two people, using only the "
    "memories provided. Each memory starts with the date it was said. Resolve relative "
    'times ("last week", "yesterday") against that date and answer with absolute dates. '
    "Answer in as few words as possible. If the memories do not contain the answer, "
    'say "Not mentioned".'
)

JUDGE_SYSTEM = (
    "You grade answers to questions about a conversation. You get the question, the gold "
    "answer and a generated answer. Be generous: the generated answer is CORRECT if it "
    "refers to the same fact, person, event or date as the gold answer, even if it is "
    "worded differently, longer, or uses a different date format. It is WRONG if it "
    "misses the gold answer, contradicts it, or says the information is not available. "
    "Reply with exactly one word: CORRECT or WRONG."
)


def answer(llm: LLM, question: str, context: str) -> Completion:
    prompt = f"Memories:\n{context}\n\nQuestion: {question}\nAnswer:"
    return llm.complete(ANSWER_SYSTEM, prompt, max_tokens=100)


def judge(llm: LLM, question: str, gold: str, generated: str) -> bool:
    prompt = f"Question: {question}\nGold answer: {gold}\nGenerated answer: {generated}"
    verdict = llm.complete(JUDGE_SYSTEM, prompt, max_tokens=5).text
    return re.search(r"\bCORRECT\b", verdict.upper()) is not None
