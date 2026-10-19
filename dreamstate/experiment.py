"""One wake/sleep run for a single (condition, seed).

Each cycle ("day"):
  wake   -> the student reads the day's documents in context and writes flashcards
  verify -> the judge drops cards the source doesn't support
  quiz   -> the student answers the cards closed-book; the judge grades
  sleep  -> LoRA training on the condition's pool, compute-matched across conditions
  eval   -> held-out paraphrased questions for every day so far + general-ability check
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

from .cards import make_cards, quiz, verify_cards
from .config import Config
from .corpus import load_docs
from .evaluate import general_ability, recall_by_day
from .llm import ChatModel
from .scheduler import LeitnerScheduler
from .training import card_datum, doc_datum, fill_budget, train_on
from .util import append_jsonl, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

CONDITIONS = {
    "A": "No sleep (baseline)",
    "B": "Fine-tune on raw documents",
    "C": "Fine-tune on all verified cards",
    "D": "Fine-tune on missed cards only",
    "E": "Missed cards + spaced resurfacing of old cards",
}


def run_dir(cfg: Config, condition: str, seed: int) -> Path:
    return Path(cfg.runs_dir) / f"{condition}_seed{seed}"


async def run_condition(cfg: Config, condition: str, seed: int, service_client, judge: ChatModel) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; choose from {list(CONDITIONS)}")
    out = run_dir(cfg, condition, seed)
    metrics_path = out / "metrics.jsonl"
    done_days = {r["day"] for r in read_jsonl(metrics_path)}
    if len(done_days) >= cfg.n_days:
        log.info("%s seed %d already complete, skipping", condition, seed)
        return
    if done_days:
        # Resuming mid-run would need the LoRA state; simplest correct behaviour is to restart the run.
        log.warning("%s seed %d was partial (%d days); restarting from scratch", condition, seed, len(done_days))
        metrics_path.unlink()
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg.to_dict(), "condition": condition, "seed": seed}, indent=2))

    rng = random.Random(seed)
    tc = await service_client.create_lora_training_client_async(
        base_model=cfg.student_model, rank=cfg.lora_rank, seed=seed,
        user_metadata={"project": "dreamstate", "condition": condition, "seed": str(seed)},
    )
    tokenizer = tc.get_tokenizer()
    sampler = await tc.save_weights_and_get_sampling_client_async(name=f"{condition}-s{seed}-day00")
    student = ChatModel(sampler, tokenizer, cfg.concurrency)
    sched = LeitnerScheduler(cfg.leitner_max_box)

    general0 = await general_ability(student, cfg.general_eval_n)
    append_jsonl(out / "general.jsonl", {"day": 0, "general": general0})
    log.info("[%s s%d] general ability before any sleep: %s", condition, seed, general0)

    for day in range(1, cfg.n_days + 1):
        t0 = time.time()
        docs = load_docs(cfg.corpus_dir, day)
        if not docs:
            raise FileNotFoundError(f"No corpus for day {day}. Run `dreamstate corpus` first.")
        docs_by_id = {d["doc_id"]: d for d in docs}
        row: dict = {"condition": condition, "seed": seed, "day": day}

        # ---- wake: make, verify and self-quiz cards (only conditions that use cards) ----
        verified, missed = [], []
        if condition in {"C", "D", "E"}:
            made = await make_cards(student, docs, cfg.cards_per_doc, cfg.card_temperature)
            verified = await verify_cards(judge, made, docs_by_id)
            correct = await quiz(student, judge, verified, cfg.max_answer_tokens, cfg.quiz_temperature)
            missed = [c for c, ok in zip(verified, correct) if not ok]
            write_jsonl(out / f"cards_day{day:02d}.jsonl",
                        [{**c, "correct_before_sleep": ok} for c, ok in zip(verified, correct)])
            row.update(n_cards_made=len(made), n_verified=len(verified), n_missed=len(missed))

        # ---- choose the sleep pool ----
        if condition == "A":
            pool = []
        elif condition == "B":
            pool = [doc_datum(tokenizer, d) for d in docs]
        elif condition == "C":
            pool = [card_datum(tokenizer, c) for c in verified]
        elif condition == "D":
            pool = [card_datum(tokenizer, c) for c in missed]
        else:  # E: today's misses + old cards that are due AND missed on re-quiz
            due = sched.due(cycle=day)
            due_ok = await quiz(student, judge, due, cfg.max_answer_tokens, cfg.quiz_temperature)
            for c, ok in zip(due, due_ok):
                sched.record(c["card_id"], day, ok)
            due_missed = [c for c, ok in zip(due, due_ok) if not ok]
            missed_ids = {c["card_id"] for c in missed}
            for c in verified:
                sched.add(c, day, correct=c["card_id"] not in missed_ids)
            pool = [card_datum(tokenizer, c) for c in missed + due_missed]
            row.update(n_due=len(due), n_due_missed=len(due_missed))

        # ---- sleep ----
        datums = [d for d, _ in pool]
        counts = [n for _, n in pool]
        schedule = fill_budget(list(range(len(datums))), counts, cfg.train_tokens_per_cycle,
                               cfg.match_compute, rng)
        batch = [datums[i] for i in schedule]
        stats = await train_on(tc, batch, cfg.batch_size, cfg.learning_rate)
        row.update(pool_size=len(datums), train_examples=len(batch),
                   train_tokens=sum(counts[i] for i in schedule), **stats)

        if batch:
            sampler = await tc.save_weights_and_get_sampling_client_async(name=f"{condition}-s{seed}-day{day:02d}")
            student = ChatModel(sampler, tokenizer, cfg.concurrency)

        # ---- eval ----
        row["recall"] = await recall_by_day(student, judge, cfg.corpus_dir, list(range(1, day + 1)),
                                            cfg.max_answer_tokens, cfg.quiz_temperature)
        if day == cfg.n_days:
            g = await general_ability(student, cfg.general_eval_n)
            row["general"] = g
            append_jsonl(out / "general.jsonl", {"day": day, "general": g})
        row["seconds"] = round(time.time() - t0, 1)
        append_jsonl(metrics_path, row)
        if condition == "E":
            (out / "scheduler.json").write_text(json.dumps(sched.to_dict()))
        log.info("[%s s%d] day %d: recall=%s train_tokens=%s", condition, seed, day,
                 {k: round(v, 3) for k, v in row["recall"].items()}, row["train_tokens"])
