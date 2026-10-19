"""Leitner-box spaced repetition over flashcards (used by condition E).

A card in box b is due again 2**b cycles after its last review. A miss sends it back to box 0
(due next cycle); a hit promotes it one box, up to max_box.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class CardState:
    card: dict
    box: int = 0
    due: int = 0          # cycle number when the card should next be reviewed
    reviews: int = 0
    misses: int = 0


class LeitnerScheduler:
    def __init__(self, max_box: int = 4):
        self.max_box = max_box
        self.states: dict[str, CardState] = {}

    def add(self, card: dict, cycle: int, correct: bool) -> None:
        """Register a newly made card with the result of its first quiz."""
        st = CardState(card=card)
        self.states[card["card_id"]] = st
        self.record(card["card_id"], cycle, correct)

    def record(self, card_id: str, cycle: int, correct: bool) -> None:
        st = self.states[card_id]
        st.reviews += 1
        if correct:
            st.box = min(st.box + 1, self.max_box)
        else:
            st.misses += 1
            st.box = 0
        st.due = cycle + 2 ** st.box

    def due(self, cycle: int, exclude_day: int | None = None) -> list[dict]:
        """Cards due for review at this cycle (optionally skipping cards made today)."""
        return [s.card for s in self.states.values()
                if s.due <= cycle and (exclude_day is None or s.card["day"] != exclude_day)]

    def to_dict(self) -> dict:
        return {"max_box": self.max_box, "states": {k: asdict(v) for k, v in self.states.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "LeitnerScheduler":
        s = cls(d["max_box"])
        s.states = {k: CardState(**v) for k, v in d["states"].items()}
        return s
