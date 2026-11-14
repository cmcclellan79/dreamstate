import json

from dreamstate.cards import make_cards


class FakeChat:
    def __init__(self, reply):
        self.reply = reply

    async def chat_many(self, batch, **kw):
        return [self.reply for _ in batch]


async def test_long_sentence_answers_are_dropped():
    reply = json.dumps([
        {"q": "Which team inspected Sector 7 of Oria?", "a": "Inspection Team Delta-9"},
        {"q": "Which team inspected Sector 7?", "a": "The team that inspected Sector 7 was Inspection Team Delta-9."},
    ])
    cards = await make_cards(FakeChat(reply), [{"doc_id": "d01_00", "day": 1, "text": "..."}], 2, 0.7,
                             max_answer_words=8)
    assert [c["a"] for c in cards] == ["Inspection Team Delta-9"]
    assert cards[0]["card_id"] == "d01_00_c0"


async def test_no_limit_keeps_everything():
    reply = json.dumps([{"q": "Q?", "a": "a very long answer " * 5}])
    cards = await make_cards(FakeChat(reply), [{"doc_id": "d", "day": 1, "text": "..."}], 1, 0.7)
    assert len(cards) == 1
