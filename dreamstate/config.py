"""Experiment configuration loaded from YAML with CLI overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    student_model: str = "Qwen/Qwen3-8B"
    teacher_model: str = "Qwen/Qwen3.6-35B-A3B"
    judge_model: str = "Qwen/Qwen3.6-35B-A3B"
    lora_rank: int = 32

    corpus_dir: str = "data/corpus"
    n_days: int = 5
    docs_per_day: int = 8
    eval_questions_per_doc: int = 5

    cards_per_doc: int = 8
    card_temperature: float = 0.7

    learning_rate: float = 1e-4
    batch_size: int = 16
    train_tokens_per_cycle: int = 40_000
    match_compute: bool = True

    leitner_max_box: int = 4

    quiz_temperature: float = 0.0
    max_answer_tokens: int = 64
    general_eval_n: int = 200
    concurrency: int = 32

    conditions: list[str] = field(default_factory=lambda: ["A", "B", "C", "D", "E"])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    runs_dir: str = "runs"

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for item in overrides or []:
            key, _, value = item.partition("=")
            raw[key.strip()] = yaml.safe_load(value)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
