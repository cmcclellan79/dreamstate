import json

import pytest

from dreamstate.corpus import _generate_day_docs


class ScriptedTeacher:
    """Returns the scripted replies in order and records each prompt."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    async def chat(self, messages, **kw):
        self.prompts.append(messages[0]["content"])
        return self.replies.pop(0)


def docs(*names):
    return json.dumps([{"title": n, "text": f"{n} did a thing.", "entities": [n]} for n in names])


async def test_batches_survive_a_truncated_reply_and_pass_names_forward(tmp_path):
    teacher = ScriptedTeacher([docs("Avel", "Brisk"), '[{"title": "Cut off", "text": "trunc', docs("Corran", "Dusk")])
    out = await _generate_day_docs(teacher, day=2, n=4, used={"Old"}, debug_dir=tmp_path)
    assert [d["title"] for d in out] == ["Avel", "Brisk", "Corran", "Dusk"]
    assert (tmp_path / "day02_fail1.txt").read_text().startswith('[{"title": "Cut off"')
    # later batches are told not to reuse names from earlier batches the same day
    assert "Avel" in teacher.prompts[-1] and "Old" in teacher.prompts[-1]


async def test_gives_up_after_repeated_failures(tmp_path):
    teacher = ScriptedTeacher(["not json"] * 6)
    with pytest.raises(RuntimeError, match="day 3"):
        await _generate_day_docs(teacher, day=3, n=2, used=set(), debug_dir=tmp_path)
    assert len(list(tmp_path.glob("day03_fail*.txt"))) == 6
