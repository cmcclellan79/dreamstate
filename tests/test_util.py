import pytest

from dreamstate.util import extract_json, parse_verdict


def test_extract_json_fenced():
    assert extract_json('Sure!\n```json\n[{"q": "a", "a": "b"}]\n```') == [{"q": "a", "a": "b"}]


def test_extract_json_with_chatter_and_think():
    assert extract_json('<think>hmm {not json</think> Here: {"valid": true} done') == {"valid": True}


def test_extract_json_fails():
    with pytest.raises(ValueError):
        extract_json("no json here")


@pytest.mark.parametrize("text,expected", [
    ('{"correct": true}', True), ('{"correct": false}', False), ('{"valid": "yes"}', True),
    ("YES", True), ("No.", False), ("incorrect", False), ("", False),
])
def test_parse_verdict(text, expected):
    assert parse_verdict(text) is expected
