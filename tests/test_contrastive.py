import math

import pytest

from dreamstate.training import _as_list, contrastive_datums


def trained(tok, datum, sign):
    w = _as_list(datum.loss_fn_inputs["weights"])
    t = [int(x) for x in _as_list(datum.loss_fn_inputs["target_tokens"])]
    return [tok.decode([i]) for i, wi in zip(t, w) if (wi > 0 if sign > 0 else wi < 0)]


def test_pair_pushes_up_truth_and_down_only_the_divergent_part(tok):
    card = {"q": "Who founded Varrowmere?", "a": "Ilsa Brent"}
    datums, n = contrastive_datums(tok, card, "Ilsa Brandt")
    assert len(datums) == 2
    pos, neg = datums
    assert trained(tok, pos, +1) == ["Ilsa", "Brent", "<|im_end|>"]
    # the shared first name and end-of-turn are not pushed down; only the confabulated surname is
    assert trained(tok, neg, -1) == ["Brandt"]
    assert trained(tok, neg, +1) == []


def test_first_scope_targets_only_the_actual_mistake(tok):
    card = {"q": "Who founded Varrowmere?", "a": "Ilsa Brent"}
    (_, neg_first), _ = contrastive_datums(tok, card, "Ilsa Brandt Smith", negative_scope="first")
    (_, neg_all), _ = contrastive_datums(tok, card, "Ilsa Brandt Smith", negative_scope="all")
    assert trained(tok, neg_first, -1) == ["Brandt"]          # where the answers split
    assert trained(tok, neg_all, -1) == ["Brandt", "Smith"]   # every divergent token
    with pytest.raises(ValueError):
        contrastive_datums(tok, card, "x", negative_scope="some")


def test_negative_gradient_is_bounded_when_model_is_certain(tok):
    torch = pytest.importorskip("torch")
    from dreamstate.training import make_contrastive_loss

    pos, neg = contrastive_datums(tok, {"q": "Q?", "a": "right"}, "wrong")[0]
    n_pos, n_neg = (len(_as_list(d.loss_fn_inputs["weights"])) for d in (pos, neg))
    lps = [torch.full((n_pos,), -1.0, requires_grad=True),
           torch.full((n_neg,), math.log(1 - 1e-6), requires_grad=True)]   # model ~certain of its wrong token
    alpha, cap = 0.5, 0.95
    loss, _ = make_contrastive_loss(alpha, cap)([pos, neg], lps)
    assert torch.isfinite(loss)
    loss.backward()
    g = lps[1].grad[torch.tensor(_as_list(neg.loss_fn_inputs["weights"])) < 0]
    # still pushed down (positive dL/dlogp), but bounded at alpha * cap / (1 - cap)
    assert (g > 0).all()
    assert g.max() == pytest.approx(alpha * cap / (1 - cap), rel=1e-4)


@pytest.mark.parametrize("wrong", ["", "   ", "Ilsa Brent"])
def test_no_negative_when_nothing_distinctive(tok, wrong):
    datums, _ = contrastive_datums(tok, {"q": "Who founded Varrowmere?", "a": "Ilsa Brent"}, wrong)
    assert len(datums) == 1


def test_loss_matches_formula_and_gradient_signs(tok):
    torch = pytest.importorskip("torch")
    from dreamstate.training import make_contrastive_loss

    pos, neg = contrastive_datums(tok, {"q": "Q?", "a": "right"}, "wrong")[0]
    lps = []
    for d in (pos, neg):
        n = len(_as_list(d.loss_fn_inputs["weights"]))
        lps.append(torch.full((n,), math.log(0.3), requires_grad=True))
    alpha = 0.5
    loss, metrics = make_contrastive_loss(alpha)([pos, neg], lps)
    n_pos = sum(_as_list(pos.loss_fn_inputs["weights"]))
    n_neg = -sum(_as_list(neg.loss_fn_inputs["weights"]))
    expected = n_pos * -math.log(0.3) + alpha * n_neg * -math.log(0.7)
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-5)
    assert metrics["p_wrong_token:mean"] == pytest.approx(0.3, rel=1e-5)
    loss.backward()
    w_pos = torch.tensor(_as_list(pos.loss_fn_inputs["weights"]))
    w_neg = torch.tensor(_as_list(neg.loss_fn_inputs["weights"]))
    # gradient descent raises logprobs of true tokens (dL/dlp < 0) and lowers wrong ones (dL/dlp > 0)
    assert (lps[0].grad[w_pos > 0] < 0).all()
    assert (lps[1].grad[w_neg < 0] > 0).all()
    assert (lps[1].grad[w_neg == 0] == 0).all()
