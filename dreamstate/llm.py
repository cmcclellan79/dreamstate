"""Thin chat layer over Tinker sampling clients.

Tinker samples from token IDs, so this module owns chat-template rendering, stop tokens and decoding.
Thinking mode is disabled for hybrid models (Qwen3.x) so answers are short and cheap.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from tinker import types

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


def render_prompt(tokenizer, messages: Sequence[Message]) -> list[int]:
    """Chat-template the messages and return prompt token IDs ending at the assistant turn."""
    kwargs = dict(add_generation_prompt=True, tokenize=True)
    try:
        out = tokenizer.apply_chat_template(list(messages), enable_thinking=False, **kwargs)
    except TypeError:  # template doesn't accept enable_thinking
        out = tokenizer.apply_chat_template(list(messages), **kwargs)
    if hasattr(out, "keys") and "input_ids" in out:  # newer transformers return a BatchEncoding
        out = out["input_ids"]
    return list(out)


def end_of_turn_ids(tokenizer) -> list[int]:
    """Token IDs that end an assistant turn (e.g. <|im_end|> for Qwen), plus EOS."""
    ids: list[int] = []
    for tok in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != getattr(tokenizer, "unk_token_id", None) and tid >= 0:
            ids.append(tid)
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return sorted(set(ids))


def end_of_turn_text(tokenizer) -> str:
    """The string appended after an assistant answer in training targets."""
    for tok in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", None):
            return tok
    return tokenizer.eos_token or ""


class ChatModel:
    """Async chat over a Tinker SamplingClient with bounded concurrency."""

    def __init__(self, sampling_client, tokenizer, concurrency: int = 32):
        self.client = sampling_client
        self.tokenizer = tokenizer
        self.stop = end_of_turn_ids(tokenizer)
        self._sem = asyncio.Semaphore(concurrency)

    async def chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.7,
                   seed: int | None = None) -> str:
        prompt = types.ModelInput.from_ints(render_prompt(self.tokenizer, messages))
        params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=self.stop, seed=seed)
        async with self._sem:
            result = await self.client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        text = self.tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        return text.strip()

    async def chat_many(self, batch: Sequence[Sequence[Message]], **kwargs) -> list[str]:
        return list(await asyncio.gather(*(self.chat(m, **kwargs) for m in batch)))


async def make_chat_model(service_client, base_model: str, concurrency: int = 32) -> ChatModel:
    """A ChatModel over an untrained base model (used for teacher and judge)."""
    sampler = await service_client.create_sampling_client_async(base_model=base_model)
    return ChatModel(sampler, sampler.get_tokenizer(), concurrency)
