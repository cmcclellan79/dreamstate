# dreamstate

DNU research lab experiment, Fall 2026.

**Sleep-phase self-quizzing: consolidating new knowledge into LLM weights with targeted flashcards.**

When a language model reads new information in its context window ("wake"), that knowledge is gone once the
context is. This project tests whether an offline "sleep" phase can move it into the model's weights efficiently:
the model writes flashcards from what it read, quizzes itself without the source, and trains only on the cards it
gets wrong, resurfacing old misses on a spaced-repetition schedule.

## Research question and hypotheses

- **H1 (efficiency):** Training only on missed cards gives equal or better closed-book recall than training on all
  cards or on the raw documents, per training token.
- **H2 (retention):** Resurfacing old missed cards across sleep cycles (Leitner spaced repetition) reduces forgetting
  of earlier days' facts.

## One wake/sleep cycle

```
 day N documents
       │
 WAKE  ├─► student reads each doc in context and writes Q/A flashcards
       ├─► judge checks each card against its source, drops unsupported ones   (anti model-collapse)
       ├─► student answers every card closed-book; judge grades
       │
 SLEEP └─► LoRA training on the condition's pool, at a fixed token budget
       │
 EVAL  ──► held-out paraphrased questions for days 1..N  (+ MMLU slice for forgetting)
```

## Conditions

All training conditions spend the same number of training tokens per night (`train_tokens_per_cycle`); a pool is
repeated until the budget is reached. A card condition with nothing missed that night skips sleep, which counts in
its favour on the efficiency metric.

| | Condition | Sleep pool |
|---|---|---|
| A | No sleep | — (baseline) |
| B | Raw documents | the day's documents, plain language modelling |
| C | All cards | every verified card from today |
| D | Missed cards | verified cards answered wrong closed-book |
| E | Missed + spaced | D, plus old cards that are due for review and missed again |

## Metrics

- **Final recall:** closed-book accuracy on all days' held-out questions after the last cycle.
- **Day-1 retention:** recall on day-1 questions after the last cycle.
- **Recall per million training tokens:** efficiency.
- **Δ general:** MMLU-slice accuracy change from day 0 to the last day (forgetting of general ability).

The held-out questions are written by a separate teacher model and paraphrased away from the document wording,
so the student can't score by memorizing its own card phrasing. The corpus is fictional (invented people, places,
numbers), so every fact is guaranteed new to the student.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,analysis,general-eval]"
cp .env.example .env        # add your TINKER_API_KEY
pytest                      # offline tests, no key needed
```

## Running

```bash
dreamstate smoke                      # 1 day, 2 docs, condition D: cheap end-to-end check
dreamstate corpus                     # generate the frozen corpus + eval set, then commit data/corpus/
dreamstate run                        # all conditions x seeds in configs/default.yaml
dreamstate run -c D E -s 0            # a subset
dreamstate analyze                    # results table + runs/retention.png
```

Override any config field: `dreamstate --set n_days=3 --set student_model=Qwen/Qwen3.5-4B run -c D`.

Outputs per run in `runs/<condition>_seed<k>/`: `metrics.jsonl` (one row per day), `cards_dayNN.jsonl`
(every verified card and whether it was answered correctly before sleep), `general.jsonl`, `config.json`,
and for E `scheduler.json`.

## Models and cost

Defaults (Tinker IDs, see `configs/default.yaml`):

- **Student:** `Qwen/Qwen3-8B`, LoRA rank 32. Thinking mode is turned off so answers are short.
- **Teacher and judge:** `Qwen/Qwen3.6-35B-A3B`, a cheap mixture-of-experts model.

Tinker's model list changes often. Several small Llama and Qwen models were retired in mid-2026, so check the
[models page](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/) before a long run.
At current prices, training one run (5 nights × 40k tokens) costs well under a dollar. Judge and student sampling
make up most of the spend. Run `dreamstate smoke` first and check the billing page before launching the full grid.

## Repository layout

```
dreamstate/
  config.py      Config dataclass, YAML + --set overrides
  prompts.py     every prompt template
  llm.py         chat layer over Tinker sampling (templates, stop tokens, concurrency)
  corpus.py      fictional corpus + held-out eval generation (teacher)
  cards.py       make / verify / quiz / grade
  scheduler.py   Leitner spaced repetition
  training.py    Datum construction, compute-matched budgets, LoRA SFT loop
  evaluate.py    held-out recall by day, MMLU forgetting check
  experiment.py  one (condition, seed) wake/sleep run
  analysis.py    results table and retention plot
  cli.py         `dreamstate` command
configs/default.yaml
data/corpus/     frozen corpus (commit it)
tests/           offline tests, including a full pipeline run against fake Tinker clients
docs/related_work.md
```

## Known limitations and next steps

- A partially finished run restarts from scratch. Resuming would need `save_state` checkpoints.
- The judge is a model. Hand-check a sample of verdicts and report agreement.
- Card counts differ between conditions, so report how many unique facts each pool covered as well as tokens.
- Possible extension: add self-synthesized general data to the sleep pool (as in SSR, see related work) to
  reduce general-ability forgetting.
