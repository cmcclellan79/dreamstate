"""End-to-end run of every condition against fake Tinker clients (no API key, no network)."""

import json
from types import SimpleNamespace

import pytest

from dreamstate.config import Config
from dreamstate.experiment import CONDITIONS, run_condition
from dreamstate.llm import ChatModel
from dreamstate.util import read_jsonl, write_jsonl


class FakeSampler:
    def __init__(self, tok, trained_steps=0):
        self.tok, self.trained = tok, trained_steps

    def get_tokenizer(self):
        return self.tok

    async def sample_async(self, prompt, num_samples, sampling_params):
        text = self.tok.decode(prompt.to_ints())
        if "flashcards" in text:
            reply = json.dumps([{"q": f"Q{i}?", "a": f"A{i}"} for i in range(3)])
        elif "Flashcard question" in text:
            reply = '{"valid": true}'
        elif "Reference answer" in text:
            # untrained model gets everything wrong; trained model gets things right
            reply = '{"correct": %s}' % ("true" if "Student answer: right" in text else "false")
        else:
            reply = "right" if self.trained else "wrong"
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=self.tok.encode(reply))])


class Future:
    def __init__(self, v):
        self.v = v

    async def result_async(self):
        return self.v


class FakeTrainer:
    def __init__(self, tok):
        self.tok, self.steps = tok, 0

    def get_tokenizer(self):
        return self.tok

    async def forward_backward_async(self, data, loss_fn):
        return Future(SimpleNamespace(metrics={"loss:sum": 1.0 * len(data)}))

    async def optim_step_async(self, params):
        self.steps += 1
        return Future(None)

    async def save_weights_and_get_sampling_client_async(self, name):
        return FakeSampler(self.tok, self.steps)


class FakeService:
    def __init__(self, tok):
        self.tok = tok

    async def create_lora_training_client_async(self, **kw):
        return FakeTrainer(self.tok)


@pytest.mark.parametrize("condition", list(CONDITIONS))
async def test_all_conditions_run(tmp_path, tok, condition):
    corpus = tmp_path / "corpus"
    for day in (1, 2, 3):
        docs = [{"doc_id": f"d{day}_{i}", "day": day, "title": "T", "text": "some invented facts"} for i in range(2)]
        write_jsonl(corpus / f"day_{day:02d}" / "docs.jsonl", docs)
        write_jsonl(corpus / f"day_{day:02d}" / "eval.jsonl",
                    [{"qid": f"{d['doc_id']}_q0", "doc_id": d["doc_id"], "day": day, "question": "EQ?", "answer": "right"}
                     for d in docs])
    cfg = Config(corpus_dir=str(corpus), runs_dir=str(tmp_path / "runs"), n_days=3,
                 train_tokens_per_cycle=200, batch_size=4, general_eval_n=0)
    judge = ChatModel(FakeSampler(tok), tok)
    await run_condition(cfg, condition, 0, FakeService(tok), judge)

    rows = read_jsonl(tmp_path / "runs" / f"{condition}_seed0" / "metrics.jsonl")
    assert [r["day"] for r in rows] == [1, 2, 3]
    if condition == "A":
        assert all(r["train_tokens"] == 0 for r in rows)
        assert set(rows[-1]["recall"].values()) == {0.0}
    else:
        # compute-matched: a night with anything to learn spends the full budget. D/E can sleep for free
        # when nothing was missed (here the fake model is perfect after day 1), which is the efficiency claim.
        assert rows[0]["train_tokens"] >= 200
        assert all(r["train_tokens"] == 0 or r["train_tokens"] >= 200 for r in rows)
    if condition == "E":
        assert rows[1]["n_due"] > 0          # day-1 misses resurface on day 2
        assert rows[0]["n_missed"] == rows[0]["n_verified"] == 6
