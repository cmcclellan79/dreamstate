"""Small helpers: JSON extraction from model output, JSONL IO."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model response.

    Handles ```json fences, leading chatter, and trailing text. Raises ValueError if nothing parses.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for cand in candidates:
        for i, ch in enumerate(cand):
            if ch in "[{":
                try:
                    value, _ = decoder.raw_decode(cand[i:])
                    return value
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"No JSON found in: {text[:200]!r}")


def parse_verdict(text: str) -> bool:
    """Parse a judge verdict. Accepts JSON {"correct": bool} or a bare YES/NO / true/false."""
    try:
        obj = extract_json(text)
        if isinstance(obj, dict):
            for key in ("correct", "valid", "verdict"):
                if key in obj:
                    v = obj[key]
                    return v if isinstance(v, bool) else str(v).strip().lower() in {"yes", "true", "correct"}
    except ValueError:
        pass
    m = re.search(r"\b(yes|no|true|false|correct|incorrect)\b", text, re.IGNORECASE)
    return bool(m) and m.group(1).lower() in {"yes", "true", "correct"}


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
