"""The Token Factory client against a fake Token Factory that misbehaves on purpose."""

import openai
import pytest
from fake_token_factory import FakeTokenFactory, bad_request, response

from overfit_court.llm import LLM, LLMError, extract_code, parse_json


def no_mock():
    raise AssertionError("demo-mode reply used while a client is attached")


@pytest.fixture
def live(settings):
    """An LLM wired to a scripted fake: live(script) -> (llm, fake)."""

    def make(script):
        fake = FakeTokenFactory(script)
        return LLM(settings, client=fake), fake

    return make


def test_falls_back_when_json_mode_is_rejected_and_remembers(live):
    llm, fake = live([bad_request(), response('{"a": 1}'), response('{"b": 2}')])
    assert llm.chat_json("planner", "ultra", "sys", "u", mock=no_mock) == {"a": 1}
    assert "response_format" in fake.requests[0]
    assert "response_format" not in fake.requests[1]
    assert "JSON object" in fake.requests[1]["messages"][0]["content"]
    assert llm.chat_json("planner", "ultra", "sys", "u", mock=no_mock) == {"b": 2}
    assert "response_format" not in fake.requests[2]  # remembered for this model


def test_retries_with_more_tokens_when_reasoning_eats_the_budget(live):
    llm, fake = live(
        [
            response("<think>long reasoning that never ends", "length"),
            response(
                "<think>done</think>```python\ndef positions(prices):\n    return prices\n```"
            ),
        ]
    )
    reply = llm.chat("coder", "super", "sys", "u", mock=no_mock, max_tokens=1000)
    assert fake.requests[1]["max_tokens"] == 4000
    assert "def positions" in extract_code(reply)
    assert len(llm.ledger) == 2 and "retry" in llm.ledger[1].note
    assert llm.total_cost() > 0


def test_asks_again_once_when_the_reply_is_not_json(live):
    llm, fake = live([response("Sure! Here is the plan: risk is low"), response('{"risk": "low"}')])
    assert llm.chat_json("auditor", "nano", "sys", "code", mock=no_mock) == {"risk": "low"}
    assert "could not be parsed" in fake.requests[1]["messages"][1]["content"]


def test_a_model_that_never_answers_gives_a_clear_error(live):
    llm, _ = live([response("", "length"), response("<think>still thinking", "length")])
    with pytest.raises(LLMError, match="no answer"):
        llm.chat("judge", "ultra", "sys", "u", mock=no_mock)


def test_other_errors_are_not_swallowed(live):
    llm, _ = live([bad_request("model not found")])
    with pytest.raises(openai.BadRequestError):
        llm.chat("judge", "ultra", "sys", "u", mock=no_mock)  # not JSON mode: no fallback


def test_parsing_helpers():
    assert parse_json('<think>x</think>\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json('Here you go: {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("[1, 2]")
    two_blocks = (
        "Helper:\n```python\nx = 1\n```\n"
        "Answer:\n```python\ndef positions(prices):\n    return 0\n```"
    )
    assert extract_code(two_blocks).startswith("def positions")
