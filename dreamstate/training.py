"""Sleep-phase training: turn cards/documents into Tinker Datums and run LoRA SFT steps."""

from __future__ import annotations

import logging
import random

from tinker import types

from . import prompts
from .llm import end_of_turn_text, render_prompt

log = logging.getLogger(__name__)


def make_datum(prompt_tokens: list[int], completion_tokens: list[int]) -> types.Datum:
    """Next-token datum with loss only on the completion (per the Tinker SDK cheatsheet)."""
    full = prompt_tokens + completion_tokens
    n_prefix = len(prompt_tokens) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full[:-1]),
        loss_fn_inputs=dict(
            weights=[0.0] * n_prefix + [1.0] * len(completion_tokens),
            target_tokens=full[1:],
        ),
    )


def card_datum(tokenizer, card: dict) -> tuple[types.Datum, int]:
    """Closed-book Q -> A, in the same chat format the quiz uses. Returns (datum, n_tokens)."""
    msgs = [{"role": "system", "content": prompts.QUIZ_SYSTEM}, {"role": "user", "content": card["q"]}]
    prompt = render_prompt(tokenizer, msgs)
    completion = tokenizer.encode(card["a"] + end_of_turn_text(tokenizer), add_special_tokens=False)
    return make_datum(prompt, completion), len(prompt) + len(completion)


def doc_datum(tokenizer, doc: dict) -> tuple[types.Datum, int]:
    """Condition B: plain language modeling on the raw document (loss on the document text)."""
    prompt = tokenizer.encode(f"{doc['title']}\n\n", add_special_tokens=False) or [tokenizer.eos_token_id]
    completion = tokenizer.encode(doc["text"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        completion.append(tokenizer.eos_token_id)
    return make_datum(prompt, completion), len(prompt) + len(completion)


def fill_budget(items: list, token_counts: list[int], budget: int, match_compute: bool,
                rng: random.Random) -> list:
    """Cycle through shuffled epochs of `items` until `budget` tokens are reached.

    With match_compute=False, return one shuffled pass. An empty pool returns [] (nothing to learn).
    """
    if not items:
        return []
    order = list(range(len(items)))
    if not match_compute:
        rng.shuffle(order)
        return [items[i] for i in order]
    out, used = [], 0
    while used < budget:
        rng.shuffle(order)
        for i in order:
            if used >= budget:
                break
            out.append(items[i])
            used += token_counts[i]
    return out


async def train_on(training_client, datums: list[types.Datum], batch_size: int, lr: float) -> dict:
    """One sleep: SGD over the datums in minibatches. Returns simple stats."""
    if not datums:
        return {"steps": 0, "mean_loss": None}
    loss_sum, loss_tokens, steps = 0.0, 0.0, 0
    adam = types.AdamParams(learning_rate=lr)
    for start in range(0, len(datums), batch_size):
        batch = datums[start:start + batch_size]
        fb = await training_client.forward_backward_async(data=batch, loss_fn="cross_entropy")
        opt = await training_client.optim_step_async(adam)
        fb_result = await fb.result_async()
        await opt.result_async()
        batch_loss = _extract_loss_sum(fb_result)
        if batch_loss is not None:
            loss_sum += batch_loss
            loss_tokens += sum(sum(_as_list(d.loss_fn_inputs["weights"])) for d in batch)
        steps += 1
    return {"steps": steps, "loss_per_token": loss_sum / loss_tokens if loss_tokens else None}


def _as_list(x) -> list[float]:
    return x.tolist() if hasattr(x, "tolist") else list(getattr(x, "data", x))


def _extract_loss_sum(fb_result) -> float | None:
    """Cross-entropy sum for the batch, from ForwardBackwardOutput.metrics (e.g. 'loss:sum')."""
    metrics = getattr(fb_result, "metrics", None) or {}
    for k, v in metrics.items():
        if k.startswith("loss") and isinstance(v, (int, float)):
            return float(v)
    return None


# ---------------------------------------------------------------------------
# Condition F: contrastive dreaming (Boltzmann-style positive + negative phase)
# ---------------------------------------------------------------------------
#
# For each missed card we build two datums in the same quiz chat format:
#   positive: question -> correct answer       weights = +1 on answer tokens
#   negative: question -> the model's own wrong answer ("its dream")
#             weights = -1 on tokens from the first one that differs from the correct answer
# The sign of the weight tells the loss which phase a token belongs to. Tokens the wrong answer shares
# with the correct one (e.g. "Ilsa" in "Ilsa Brandt" vs "Ilsa Brent") are not pushed down.
#
# Loss per token:  positive  -log p           (ordinary cross-entropy: raise the true answer)
#                  negative  -alpha*log(1-p)  (unlikelihood: lower the model's confabulation)
#
# Tinker runs this through forward_backward_custom: it returns per-token logprobs, the loss and its
# gradient are computed locally in PyTorch (CPU is fine), and the gradient is sent back for the update.
# Each step therefore costs a forward pass plus a forward-backward pass.


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def contrastive_datums(tokenizer, card: dict, wrong_answer: str) -> tuple[list[types.Datum], int]:
    """Positive + negative datums for one missed card. Returns ([pos, neg?], n_tokens).

    The negative datum is dropped if the wrong answer is empty or adds nothing beyond the correct answer.
    """
    msgs = [{"role": "system", "content": prompts.QUIZ_SYSTEM}, {"role": "user", "content": card["q"]}]
    prompt = render_prompt(tokenizer, msgs)
    eot = end_of_turn_text(tokenizer)
    good = tokenizer.encode(card["a"] + eot, add_special_tokens=False)
    pos = make_datum(prompt, good)
    n_tokens = len(prompt) + len(good)

    bad = tokenizer.encode(wrong_answer.strip() + eot, add_special_tokens=False) if wrong_answer.strip() else []
    shared = _common_prefix_len(good, bad)
    if not bad or shared >= len(bad) - 1:  # nothing distinctive left to push down (only end-of-turn)
        return [pos], n_tokens
    neg = make_datum(prompt, bad)
    n_prefix = len(prompt) - 1
    # -1 on the distinctive wrong tokens; leave the shared prefix and the end-of-turn token alone
    neg_w = [0.0] * len(_as_list(neg.loss_fn_inputs["weights"]))
    for i in range(n_prefix + shared, n_prefix + len(bad) - 1):
        neg_w[i] = -1.0
    neg = types.Datum(model_input=neg.model_input,
                      loss_fn_inputs=dict(weights=neg_w, target_tokens=neg.loss_fn_inputs["target_tokens"]))
    return [pos, neg], n_tokens + len(prompt) + len(bad)


def make_contrastive_loss(alpha: float):
    """Custom loss for TrainingClient.forward_backward_custom (runs locally in PyTorch on returned logprobs)."""
    import torch

    def loss_fn(data, logprobs_list):
        total = torch.zeros(())
        pos_sum = neg_sum = 0.0
        p_neg = []
        for datum, lp in zip(data, logprobs_list):
            w = torch.tensor(_as_list(datum.loss_fn_inputs["weights"]), dtype=lp.dtype).reshape(lp.shape)
            pos_w, neg_w = w.clamp(min=0), (-w).clamp(min=0)
            pos = -(lp * pos_w).sum()
            # log(1 - p) computed stably; the clamp keeps it finite when p -> 1
            neg = -(torch.log1p(-torch.exp(lp.clamp(max=-1e-6))) * neg_w).sum()
            total = total + pos + alpha * neg
            pos_sum += float(pos.detach())
            neg_sum += float(neg.detach())
            if neg_w.sum() > 0:
                p_neg.append(float((torch.exp(lp.detach()) * neg_w).sum() / neg_w.sum()))
        metrics = {"pos_loss:sum": pos_sum, "neg_loss:sum": neg_sum}
        if p_neg:
            metrics["p_wrong_token:mean"] = sum(p_neg) / len(p_neg)
        return total, metrics

    return loss_fn


async def train_contrastive(training_client, datums: list[types.Datum], batch_size: int, lr: float,
                            alpha: float) -> dict:
    """Like train_on, but with the positive/negative-phase loss."""
    if not datums:
        return {"steps": 0, "loss_per_token": None}
    loss_fn = make_contrastive_loss(alpha)
    adam = types.AdamParams(learning_rate=lr)
    pos_sum = neg_sum = pos_tok = neg_tok = 0.0
    p_wrong: list[float] = []
    steps = 0
    for start in range(0, len(datums), batch_size):
        batch = datums[start:start + batch_size]
        fb = await training_client.forward_backward_custom_async(batch, loss_fn)
        fb_result = await fb.result_async()
        opt = await training_client.optim_step_async(adam)
        await opt.result_async()
        m = getattr(fb_result, "metrics", None) or {}
        pos_sum += m.get("pos_loss:sum", 0.0)
        neg_sum += m.get("neg_loss:sum", 0.0)
        if "p_wrong_token:mean" in m:
            p_wrong.append(m["p_wrong_token:mean"])
        for d in batch:
            w = _as_list(d.loss_fn_inputs["weights"])
            pos_tok += sum(x for x in w if x > 0)
            neg_tok += sum(-x for x in w if x < 0)
        steps += 1
    return {"steps": steps,
            "loss_per_token": pos_sum / pos_tok if pos_tok else None,
            "neg_loss_per_token": neg_sum / neg_tok if neg_tok else None,
            "p_wrong_token": sum(p_wrong) / len(p_wrong) if p_wrong else None}
