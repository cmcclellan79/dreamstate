"""Summarize runs/ into the numbers the write-up needs.

Headline metrics per condition (mean ± std over seeds):
  final_recall   closed-book accuracy on all days' held-out questions after the last sleep
  day1_retention recall on day-1 questions after the last sleep (forgetting across cycles)
  recall_per_Mtok  final_recall / total training tokens (efficiency; undefined for A)
  general_delta  change in MMLU accuracy from day 0 to the last day
"""

from __future__ import annotations

import statistics
from pathlib import Path

from .experiment import CONDITIONS
from .util import read_jsonl


def _mean_std(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}"


def summarize(runs_dir: str) -> str:
    per_cond: dict[str, dict[str, list]] = {}
    for d in sorted(Path(runs_dir).glob("*_seed*")):
        cond = d.name.split("_seed")[0]
        rows = read_jsonl(d / "metrics.jsonl")
        if not rows:
            continue
        last = rows[-1]
        recall = {int(k): v for k, v in last["recall"].items()}
        final = sum(recall.values()) / len(recall)
        tokens = sum(r.get("train_tokens", 0) for r in rows)
        gen = {r["day"]: r["general"] for r in read_jsonl(d / "general.jsonl")}
        g0, g1 = gen.get(0), gen.get(last["day"])
        m = per_cond.setdefault(cond, {"final": [], "day1": [], "eff": [], "gdelta": [], "days": []})
        m["final"].append(final)
        m["day1"].append(recall.get(1))
        m["eff"].append(final / (tokens / 1e6) if tokens else None)
        m["gdelta"].append(g1 - g0 if g0 is not None and g1 is not None else None)
        m["days"].append(len(rows))

    lines = ["| Cond | Description | Seeds | Final recall | Day-1 retention | Recall / M train tok | Δ general |",
             "|---|---|---|---|---|---|---|"]
    for cond in sorted(per_cond):
        m = per_cond[cond]
        lines.append(f"| {cond} | {CONDITIONS.get(cond, '')} | {len(m['final'])} | {_mean_std(m['final'])} | "
                     f"{_mean_std(m['day1'])} | {_mean_std(m['eff'])} | {_mean_std(m['gdelta'])} |")
    return "\n".join(lines)


def plot_retention(runs_dir: str, out_path: str) -> None:
    """Day-1 recall over cycles, one line per condition (averaged over seeds)."""
    import matplotlib.pyplot as plt  # optional dependency

    curves: dict[str, dict[int, list[float]]] = {}
    for d in sorted(Path(runs_dir).glob("*_seed*")):
        cond = d.name.split("_seed")[0]
        for r in read_jsonl(d / "metrics.jsonl"):
            v = r["recall"].get("1", r["recall"].get(1))
            if v is not None:
                curves.setdefault(cond, {}).setdefault(r["day"], []).append(v)
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, pts in sorted(curves.items()):
        days = sorted(pts)
        ax.plot(days, [statistics.mean(pts[x]) for x in days], marker="o", label=f"{cond}: {CONDITIONS.get(cond, '')}")
    ax.set_xlabel("Cycle (day)")
    ax.set_ylabel("Recall on day-1 facts")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
