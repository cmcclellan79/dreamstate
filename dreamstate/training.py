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
