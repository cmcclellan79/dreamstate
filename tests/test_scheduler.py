from dreamstate.scheduler import LeitnerScheduler


def card(i, day=1):
    return {"card_id": f"c{i}", "day": day, "q": "q", "a": "a"}


def test_miss_is_due_next_cycle_and_hit_is_spaced():
    s = LeitnerScheduler(max_box=3)
    s.add(card(0), cycle=1, correct=False)   # box 0 -> due 2
    s.add(card(1), cycle=1, correct=True)    # box 1 -> due 3
    assert [c["card_id"] for c in s.due(2)] == ["c0"]
    assert {c["card_id"] for c in s.due(3)} == {"c0", "c1"}


def test_promotion_caps_and_miss_resets():
    s = LeitnerScheduler(max_box=2)
    s.add(card(0), cycle=1, correct=True)
    s.record("c0", 3, True)
    s.record("c0", 7, True)
    assert s.states["c0"].box == 2
    s.record("c0", 11, False)
    assert s.states["c0"].box == 0 and s.states["c0"].due == 12 and s.states["c0"].misses == 1


def test_roundtrip():
    s = LeitnerScheduler()
    s.add(card(0), 1, False)
    s2 = LeitnerScheduler.from_dict(s.to_dict())
    assert s2.states["c0"].due == s.states["c0"].due
