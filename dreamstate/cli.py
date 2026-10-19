"""Command line entry point.

    dreamstate corpus                 # generate + freeze the fictional corpus and eval set (once)
    dreamstate smoke                  # 1 day, 2 docs, condition D: checks the whole pipeline cheaply
    dreamstate run                    # all conditions x seeds from the config
    dreamstate run -c D E -s 0        # a subset
    dreamstate analyze                # results table (+ retention plot if matplotlib is installed)

Any config field can be overridden: --set n_days=3 --set student_model=Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .config import Config


def _service_client():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY is not set. Copy .env.example to .env and add your key.")
    import tinker

    return tinker.ServiceClient()


async def _corpus(cfg: Config, overwrite: bool) -> None:
    from .corpus import generate_corpus
    from .llm import make_chat_model

    sc = _service_client()
    teacher = await make_chat_model(sc, cfg.teacher_model, cfg.concurrency)
    await generate_corpus(teacher, cfg.corpus_dir, cfg.n_days, cfg.docs_per_day, cfg.eval_questions_per_doc,
                          overwrite=overwrite)


async def _run(cfg: Config, conditions: list[str], seeds: list[int]) -> None:
    from .experiment import run_condition
    from .llm import make_chat_model

    sc = _service_client()
    judge = await make_chat_model(sc, cfg.judge_model, cfg.concurrency)
    for seed in seeds:
        for cond in conditions:
            await run_condition(cfg, cond, seed, sc, judge)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dreamstate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="generate the frozen fictional corpus + eval set")
    c.add_argument("--overwrite", action="store_true")

    r = sub.add_parser("run", help="run wake/sleep experiments")
    r.add_argument("-c", "--conditions", nargs="+")
    r.add_argument("-s", "--seeds", nargs="+", type=int)

    sub.add_parser("smoke", help="tiny end-to-end check (1 day, 2 docs, condition D)")

    a = sub.add_parser("analyze", help="summarize runs/")
    a.add_argument("--plot", default="runs/retention.png")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config.load(args.config, args.set)

    if args.cmd == "corpus":
        asyncio.run(_corpus(cfg, args.overwrite))
    elif args.cmd == "run":
        asyncio.run(_run(cfg, args.conditions or cfg.conditions, args.seeds or cfg.seeds))
    elif args.cmd == "smoke":
        cfg.n_days, cfg.docs_per_day, cfg.eval_questions_per_doc = 1, 2, 3
        cfg.corpus_dir, cfg.runs_dir = "data/smoke_corpus", "runs/smoke"
        cfg.train_tokens_per_cycle, cfg.general_eval_n = 4000, 0
        asyncio.run(_corpus(cfg, overwrite=False))
        asyncio.run(_run(cfg, ["D"], [0]))
        print("Smoke test finished. See runs/smoke/D_seed0/metrics.jsonl")
    elif args.cmd == "analyze":
        from .analysis import plot_retention, summarize

        print(summarize(cfg.runs_dir))
        try:
            plot_retention(cfg.runs_dir, args.plot)
            print(f"Retention plot: {args.plot}")
        except ImportError:
            print("(install the `analysis` extra for the retention plot)")


if __name__ == "__main__":
    main()
