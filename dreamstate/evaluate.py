"""Evaluation: held-out closed-book recall per day, and a general-ability (forgetting) check."""

from __future__ import annotations

import logging
import random
import re

from .cards import answer_closed_book, grade
from .corpus import load_eval
from .llm import ChatModel

log = logging.getLogger(__name__)


async def recall_by_day(student: ChatModel, judge: ChatModel, corpus_dir: str, days: list[int],
                        max_tokens: int, temperature: float) -> dict[int, float]:
    """Closed-book accuracy on the teacher-written, paraphrased eval questions for each day.

    These questions are never trained on, so a high score means the fact was learned, not the card wording.
    """
    rows = [r for d in days for r in load_eval(corpus_dir, d)]
    if not rows:
        return {}
    qs = [r["question"] for r in rows]
    preds = await answer_closed_book(student, qs, max_tokens, temperature)
    correct = await grade(judge, qs, [r["answer"] for r in rows], preds)
    out: dict[int, list[bool]] = {}
    for r, c in zip(rows, correct):
        out.setdefault(r["day"], []).append(c)
    return {d: sum(v) / len(v) for d, v in sorted(out.items())}


_MMLU_CACHE: list[dict] | None = None


def load_mmlu(n: int, seed: int = 1234) -> list[dict]:
    """A fixed random slice of MMLU test questions (needs the `general-eval` extra)."""
    global _MMLU_CACHE
    if _MMLU_CACHE is None:
        from datasets import load_dataset  # optional dependency

        ds = load_dataset("cais/mmlu", "all", split="test")
        idx = random.Random(seed).sample(range(len(ds)), n)
        _MMLU_CACHE = [ds[i] for i in idx]
    return _MMLU_CACHE[:n]


async def general_ability(student: ChatModel, n: int) -> float | None:
    """Multiple-choice accuracy on an MMLU slice. Graded by letter match, no judge needed."""
    if n <= 0:
        return None
    try:
        items = load_mmlu(n)
    except Exception as e:  # noqa: BLE001 - dataset download can fail many ways; don't kill the run
        log.warning("general eval skipped: %s", e)
        return None
    letters = "ABCD"
    batch = []
    for it in items:
        opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(it["choices"]))
        batch.append([{"role": "user", "content":
                       f"{it['question']}\n\n{opts}\n\nAnswer with the letter only."}])
    outs = await student.chat_many(batch, max_tokens=4, temperature=0.0)
    hits = 0
    for it, o in zip(items, outs):
        m = re.search(r"[ABCD]", o.upper())
        hits += bool(m) and letters.index(m.group(0)) == it["answer"]
    return hits / len(items)
