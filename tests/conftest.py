import pytest


class FakeTokenizer:
    """Whitespace tokenizer with a Qwen-like chat template, enough to test datum construction offline."""

    eos_token = "<eos>"
    unk_token_id = 0

    def __init__(self):
        self.vocab = {"<unk>": 0, "<eos>": 1, "<|im_end|>": 2}

    @property
    def eos_token_id(self):
        return 1

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab))

    def convert_tokens_to_ids(self, tok):
        return self.vocab.get(tok, 0)

    def encode(self, text, add_special_tokens=False):
        text = text.replace("<|im_end|>", " <|im_end|> ")
        return [self._id(w) for w in text.split()]

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kw):
        text = " ".join(f"<{m['role']}> {m['content']} <|im_end|>" for m in messages)
        if add_generation_prompt:
            text += " <assistant>"
        return self.encode(text)


@pytest.fixture
def tok():
    return FakeTokenizer()
