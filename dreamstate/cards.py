"""Flashcards: the student writes them (wake), the judge verifies them, the student quizzes itself (sleep)."""

from __future__ import annotations

import asyncio
import logging

from . import prompts
from .llm import ChatModel
from .util import extract_json, parse_verdict

log = logging.getLogger(__name__)


async def make_cards(student: ChatModel, docs: list[dict], n_per_doc: int, temperature: float) -> list[dict]:
    """Student reads each document in context and writes Q/A flashcards."""
    batch = [[{"role": "user", "content": prompts.MAKE_CARDS.format(n=n_per_doc, text=d["text"])}] for d in docs]
    outputs = await student.chat_many(batch, max_tokens=1500, temperature=temperature)
    cards: list[dict] = []
    for doc, raw in zip(docs, outputs):
        try:
            items = extract_json(raw)
        except ValueError:
            log.warning("could not parse cards for %s", doc["doc_id"])
            continue
        if isinstance(items, dict):
            items = [items]
        k = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            q, a = str(it.get("q", "")).strip(), str(it.get("a", "")).strip()
            if q and a:
                cards.append({"card_id": f"{doc['doc_id']}_c{k}", "doc_id": doc["doc_id"], "day": doc["day"],
                              "q": q, "a": a})
                k += 1
    return cards


async def verify_cards(judge: ChatModel, cards: list[dict], docs_by_id: dict[str, dict]) -> list[dict]:
    """Keep only cards the judge confirms against the source. This is the anti-collapse filter."""
    batch = [[{"role": "user", "content": prompts.VERIFY_CARD.format(
        text=docs_by_id[c["doc_id"]]["text"], q=c["q"], a=c["a"])}] for c in cards]
    verdicts = await judge.chat_many(batch, max_tokens=32, temperature=0.0)
    kept = [c for c, v in zip(cards, verdicts) if parse_verdict(v)]
    log.info("verified %d/%d cards", len(kept), len(cards))
    return kept


async def answer_closed_book(student: ChatModel, questions: list[str], max_tokens: int,
                             temperature: float) -> list[str]:
    batch = [[{"role": "system", "content": prompts.QUIZ_SYSTEM}, {"role": "user", "content": q}]
             for q in questions]
    return await student.chat_many(batch, max_tokens=max_tokens, temperature=temperature)


async def grade(judge: ChatModel, questions: list[str], refs: list[str], preds: list[str]) -> list[bool]:
    batch = [[{"role": "user", "content": prompts.GRADE_ANSWER.format(q=q, ref=r, pred=p)}]
             for q, r, p in zip(questions, refs, preds)]
    verdicts = await judge.chat_many(batch, max_tokens=32, temperature=0.0)
    return [parse_verdict(v) for v in verdicts]


async def quiz(student: ChatModel, judge: ChatModel, cards: list[dict], max_tokens: int,
               temperature: float) -> list[bool]:
    """Closed-book self-quiz over cards. Returns correct/incorrect per card."""
    if not cards:
        return []
    qs = [c["q"] for c in cards]
    preds = await answer_closed_book(student, qs, max_tokens, temperature)
    return await grade(judge, qs, [c["a"] for c in cards], preds)
