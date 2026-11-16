"""Generate the frozen fictional corpus and held-out eval set (run once, then commit).

Layout:
    data/corpus/day_01/docs.jsonl   {"doc_id", "day", "title", "genre", "text", "entities"}
    data/corpus/day_01/eval.jsonl   {"qid", "doc_id", "day", "question", "answer"}
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from . import prompts
from .llm import ChatModel
from .util import extract_json, read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def day_dir(corpus_dir: str | Path, day: int) -> Path:
    return Path(corpus_dir) / f"day_{day:02d}"


def load_docs(corpus_dir: str | Path, day: int) -> list[dict]:
    return read_jsonl(day_dir(corpus_dir, day) / "docs.jsonl")


def load_eval(corpus_dir: str | Path, day: int) -> list[dict]:
    return read_jsonl(day_dir(corpus_dir, day) / "eval.jsonl")


async def _generate_day_docs(teacher: ChatModel, day: int, n: int, used: set[str], debug_dir: Path,
                             per_call: int = 2, max_failures: int = 6) -> list[dict]:
    """Ask the teacher for a few documents at a time until the day has n.

    Asking for all n in one reply produced ~2,500 words of JSON, and one truncated or short reply sank the
    whole day. Small batches keep each reply short, keep partial progress, and let each batch see the names
    the previous batches used. Failed replies are saved under debug_dir for inspection.
    """
    docs: list[dict] = []
    failures = 0
    while len(docs) < n:
        k = min(per_call, n - len(docs))
        names = used | {e for d in docs for e in d.get("entities", [])}
        used_str = ", ".join(sorted(names)) or "(none yet)"
        raw = await teacher.chat(
            [{"role": "user", "content": prompts.WORLD_DOCS.format(n=k, day=day, used_names=used_str)}],
            max_tokens=3000, temperature=0.9,
        )
        try:
            got = extract_json(raw)
            if isinstance(got, dict):
                got = [got]
            got = [d for d in got if isinstance(d, dict) and str(d.get("text", "")).strip()]
            if not got:
                raise ValueError("reply parsed but contained no documents")
            docs.extend(got[:k])
            log.info("day %d: %d/%d docs", day, len(docs), n)
        except ValueError as e:
            failures += 1
            debug_dir.mkdir(parents=True, exist_ok=True)
            path = debug_dir / f"day{day:02d}_fail{failures}.txt"
            path.write_text(raw)
            log.warning("day %d: unusable teacher reply (%s); saved to %s", day, e, path)
            if failures >= max_failures:
                raise RuntimeError(f"Teacher failed to produce {n} docs for day {day} after {failures} bad "
                                   f"replies; see {debug_dir}") from e
    return docs


async def _generate_eval(teacher: ChatModel, doc: dict, n: int) -> list[dict]:
    raw = await teacher.chat(
        [{"role": "user", "content": prompts.EVAL_QUESTIONS.format(n=n, text=doc["text"])}],
        max_tokens=1500, temperature=0.3,
    )
    try:
        items = extract_json(raw)
    except ValueError:
        log.warning("eval generation failed for %s", doc["doc_id"])
        return []
    return [i for i in items if isinstance(i, dict) and i.get("question") and i.get("answer")][:n]


async def generate_corpus(teacher: ChatModel, corpus_dir: str, n_days: int, docs_per_day: int,
                          eval_per_doc: int, overwrite: bool = False) -> None:
    used: set[str] = set()
    for day in range(1, n_days + 1):
        d = day_dir(corpus_dir, day)
        if (d / "docs.jsonl").exists() and not overwrite:
            docs = load_docs(corpus_dir, day)
            log.info("day %d exists (%d docs), skipping", day, len(docs))
            for doc in docs:
                used.update(doc.get("entities", []))
            continue

        docs = await _generate_day_docs(teacher, day, docs_per_day, used, Path(corpus_dir) / "_debug")
        for i, doc in enumerate(docs):
            doc.update(doc_id=f"d{day:02d}_{i:02d}", day=day)
            doc.setdefault("entities", [])
            used.update(doc["entities"])

        evals = await asyncio.gather(*(_generate_eval(teacher, doc, eval_per_doc) for doc in docs))
        eval_rows = []
        for doc, items in zip(docs, evals):
            for j, item in enumerate(items):
                eval_rows.append({"qid": f"{doc['doc_id']}_q{j}", "doc_id": doc["doc_id"], "day": day,
                                  "question": item["question"], "answer": str(item["answer"])})

        write_jsonl(d / "docs.jsonl", docs)
        write_jsonl(d / "eval.jsonl", eval_rows)
        log.info("day %d: %d docs, %d eval questions", day, len(docs), len(eval_rows))
