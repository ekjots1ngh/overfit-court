"""The live client against a fake Token Factory that misbehaves on purpose."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from overfit_court.config import get_settings
from overfit_court.llm import LLM, LLMError, extract_code, parse_json


def _resp(content, finish="stop", tin=100, tout=50):
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout))


def _bad_request(msg="response_format is not supported for this model"):
    req = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return openai.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


class FakeClient:
    """Replays a script of responses (or exceptions) and records every request."""
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _llm(script):
    s = get_settings()
    s.nebius_api_key = "test"
    client = FakeClient(script)
    return LLM(s, client=client), client


def test_falls_back_when_json_mode_is_rejected_and_remembers():
    llm, c = _llm([_bad_request(), _resp('{"a": 1}'), _resp('{"b": 2}')])
    assert llm.chat_json("planner", "ultra", "sys", "u", mock=lambda: "") == {"a": 1}
    assert "response_format" in c.requests[0] and "response_format" not in c.requests[1]
    assert "JSON object" in c.requests[1]["messages"][0]["content"]
    assert llm.chat_json("planner", "ultra", "sys", "u", mock=lambda: "") == {"b": 2}
    assert "response_format" not in c.requests[2]       # remembered for this model


def test_retries_with_more_tokens_when_reasoning_eats_the_budget():
    llm, c = _llm([_resp("<think>long reasoning that never ends", "length"),
                   _resp("<think>done</think>```python\ndef positions(prices):\n    return prices\n```")])
    out = llm.chat("coder", "super", "sys", "u", mock=lambda: "", max_tokens=1000)
    assert c.requests[1]["max_tokens"] == 4000
    assert "def positions" in extract_code(out)
    assert len(llm.ledger) == 2 and "retry" in llm.ledger[1].note
    assert llm.total_cost() > 0


def test_reasks_once_when_the_reply_is_not_json():
    llm, c = _llm([_resp("Sure! Here is the plan: risk is low"), _resp('{"risk": "low"}')])
    assert llm.chat_json("auditor", "nano", "sys", "code", mock=lambda: "") == {"risk": "low"}
    assert "could not be parsed" in c.requests[1]["messages"][1]["content"]


def test_gives_a_clear_error_when_the_model_never_answers():
    llm, _ = _llm([_resp("", "length"), _resp("<think>still thinking", "length")])
    with pytest.raises(LLMError, match="no answer"):
        llm.chat("judge", "ultra", "sys", "u", mock=lambda: "")


def test_other_errors_are_not_swallowed():
    llm, _ = _llm([_bad_request("model not found")])
    with pytest.raises(openai.BadRequestError):
        llm.chat("judge", "ultra", "sys", "u", mock=lambda: "")      # not JSON mode: no fallback


def test_parsing_helpers():
    assert parse_json('<think>x</think>\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json('Here you go: {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("[1, 2]")
    two = "Helper:\n```python\nx = 1\n```\nAnswer:\n```python\ndef positions(prices):\n    return 0\n```"
    assert extract_code(two).startswith("def positions")
