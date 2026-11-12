"""Weights & Biases logging (replaces analysis.py from the Claude scaffold).

One wandb run per (condition, seed), grouped by condition, so the wandb UI averages seeds and shows
mean ± std per condition. metrics.jsonl is still written locally as a backup.

Set use_wandb: false in the config, or WANDB_MODE=disabled, to turn it off (tests do this).
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, enabled: bool, project: str, condition: str, seed: int, config: dict,
                 description: str = ""):
        self.run = None
        self.cum_tokens = 0
        self.general0: float | None = None
        if not enabled or os.environ.get("WANDB_MODE") == "disabled":
            return
        try:
            import wandb
        except ImportError:
            log.warning("wandb not installed; logging to metrics.jsonl only")
            return
        self.run = wandb.init(project=project, group=condition, name=f"{condition}_seed{seed}",
                              job_type="wake-sleep", config={**config, "condition": condition, "seed": seed},
                              notes=description, reinit="finish_previous")
        # every per-day metric is plotted against the cycle number
        self.run.define_metric("day")
        self.run.define_metric("*", step_metric="day")

    def log_general_baseline(self, value: float | None) -> None:
        self.general0 = value
        if self.run and value is not None:
            self.run.summary["general/day0"] = value

    def log_day(self, row: dict) -> None:
        """Log one cycle. Derived metrics (mean recall, efficiency) are computed here so wandb can chart them."""
        self.cum_tokens += row.get("train_tokens", 0) or 0
        recall = {int(k): v for k, v in row["recall"].items()}
        mean_recall = sum(recall.values()) / len(recall) if recall else None
        if not self.run:
            return
        data = {"day": row["day"], "recall/mean": mean_recall, "train/cum_tokens": self.cum_tokens}
        data.update({f"recall/day{d}": v for d, v in recall.items()})
        if mean_recall is not None and self.cum_tokens:
            data["recall/per_Mtok"] = mean_recall / (self.cum_tokens / 1e6)
        for k in ("train_tokens", "train_examples", "pool_size", "steps", "loss_per_token"):
            if row.get(k) is not None:
                data[f"train/{k}"] = row[k]
        for k in ("n_cards_made", "n_verified", "n_missed", "n_due", "n_due_missed"):
            if k in row:
                data[f"cards/{k}"] = row[k]
        if row.get("general") is not None:
            data["general/acc"] = row["general"]
        self.run.log(data)

        # headline numbers for the grouped runs table
        s = self.run.summary
        s["final_recall"] = mean_recall
        s["day1_retention"] = recall.get(1)
        s["total_train_tokens"] = self.cum_tokens
        if row.get("general") is not None and self.general0 is not None:
            s["general_delta"] = row["general"] - self.general0

    def finish(self) -> None:
        if self.run:
            self.run.finish()
