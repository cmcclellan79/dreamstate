import random

from dreamstate.training import _as_list, card_datum, doc_datum, fill_budget, make_datum


def test_make_datum_masks_prompt():
    d = make_datum([10, 11, 12], [20, 21])
    w = _as_list(d.loss_fn_inputs["weights"])
    t = _as_list(d.loss_fn_inputs["target_tokens"])
    assert t == [11, 12, 20, 21]
    assert w == [0.0, 0.0, 1.0, 1.0]   # loss only where the target is a completion token


def test_card_datum_trains_on_answer_and_end_of_turn(tok):
    d, n = card_datum(tok, {"q": "Who founded Varrowmere?", "a": "Ilsa Brent"})
    w = _as_list(d.loss_fn_inputs["weights"])
    t = [int(x) for x in _as_list(d.loss_fn_inputs["target_tokens"])]
    trained = [tok.decode([i]) for i, wi in zip(t, w) if wi > 0]
    assert trained == ["Ilsa", "Brent", "<|im_end|>"]
    assert n == len(t) + 1


def test_doc_datum_ends_with_eos(tok):
    d, _ = doc_datum(tok, {"title": "T", "text": "a b c"})
    t = [int(x) for x in _as_list(d.loss_fn_inputs["target_tokens"])]
    assert t[-1] == tok.eos_token_id


def test_fill_budget_matches_tokens():
    items, counts = [0, 1, 2], [10, 10, 10]
    out = fill_budget(items, counts, budget=95, match_compute=True, rng=random.Random(0))
    assert len(out) == 10                # cycles the pool until >= 95 tokens
    assert fill_budget([], [], 95, True, random.Random(0)) == []
    assert sorted(fill_budget(items, counts, 95, False, random.Random(0))) == items
