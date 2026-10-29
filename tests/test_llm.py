"""The Token Factory client against a fake Token Factory that misbehaves on purpose."""

from datetime import date

import openai
import pytest
from fake_token_factory import FakeTokenFactory, bad_request, response

from overfit_court import llm as llm_module
from overfit_court.llm import LLM, LLMError, Spend, extract_code, parse_json


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
    assert len(llm.calls) == 2 and "retry" in llm.calls[1].note
    assert llm.total_cost() > 0


def test_asks_again_once_when_the_reply_is_not_json(live):
    llm, fake = live([response("Sure! Here is the plan: risk is low"), response('{"risk": "low"}')])
    assert llm.chat_json("auditor", "nano", "sys", "code", mock=no_mock) == {"risk": "low"}
    assert "could not be parsed" in fake.requests[1]["messages"][1]["content"]


def test_a_model_that_never_answers_gives_a_clear_error(live):
    llm, _ = live([response("", "length"), response("<think>still thinking", "length")])
    with pytest.raises(LLMError, match="no answer"):
        llm.chat("judge", "ultra", "sys", "u", mock=no_mock)


def test_other_errors_are_not_swallowed_and_failures_are_logged(live):
    llm, _ = live([bad_request("model not found")])
    with pytest.raises(openai.BadRequestError):
        llm.chat("judge", "ultra", "sys", "u", mock=no_mock)  # not JSON mode: no fallback
    assert len(llm.calls) == 1 and llm.calls[0].note.startswith("failed: BadRequestError")


def test_json_mode_is_not_blamed_for_a_request_that_fails_anyway(live):
    """A 400 for another reason (say, a prompt that is too long) fails again without JSON mode."""
    llm, fake = live(
        [bad_request("prompt too long"), bad_request("prompt too long"), response('{"a": 1}')]
    )
    with pytest.raises(openai.BadRequestError, match="prompt too long"):
        llm.chat_json("planner", "ultra", "sys", "u", mock=no_mock)
    assert [call.note.split(":")[0] for call in llm.calls] == [
        "failed",
        "retry without JSON mode; failed",
    ]
    assert llm.chat_json("planner", "ultra", "sys", "u", mock=no_mock) == {"a": 1}
    assert "response_format" in fake.requests[2]  # JSON mode is still used for this model


def test_a_refused_reasoning_switch_is_dropped_alone_and_remembered(live):
    def token_factory(request):
        if "extra_body" in request:
            return bad_request("unexpected keyword chat_template_kwargs")
        return response('{"a": 1}')

    llm, fake = live(token_factory)
    assert llm.chat_json("auditor", "nano", "s", "u", mock=no_mock, reasoning=False) == {"a": 1}
    assert [call.note.split(":")[0] for call in llm.calls] == [
        "failed",
        "retry without the thinking switch",  # JSON mode was kept: it wasn't the problem
    ]
    assert "response_format" in fake.requests[1]
    llm.chat_json("auditor", "nano", "s", "u", mock=no_mock, reasoning=False)
    assert "extra_body" not in fake.requests[2] and "response_format" in fake.requests[2]


def test_thinking_and_quick_roles_ask_differently(live):
    llm, fake = live([response("{}"), response("{}")])
    llm.chat_json("judge", "ultra", "s", "u", mock=no_mock, reasoning=True)
    llm.chat_json("auditor", "nano", "s", "u", mock=no_mock, reasoning=False)
    thinking, quick = fake.requests
    assert thinking["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert (thinking["temperature"], thinking["top_p"]) == (0.6, 0.95)
    assert quick["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert quick["temperature"] == 0.2 and "top_p" not in quick


def test_an_answer_cut_off_mid_way_is_asked_for_again_with_more_room(live):
    llm, fake = live([response("```python\ndef positions(pri", "length"), response("done")])
    assert llm.chat("coder", "super", "s", "u", mock=no_mock, max_tokens=12000) == "done"
    assert fake.requests[1]["max_tokens"] == 32000  # four times, up to the cap


@pytest.mark.parametrize(
    "configured, listed",
    [
        ("nvidia/nemotron-3-ultra-550b-a55b", "nvidia/Nemotron-3-Ultra-550b-a55b"),  # case
        ("nemotron-3-super-120b-a12b", "nvidia/nemotron-3-super-120b-a12b"),  # organisation
        ("nvidia/nemotron-3-nano-30b-a3b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"),  # name
    ],
)
def test_a_model_id_that_differs_only_in_case_or_prefix_is_matched(settings, configured, listed):
    settings.nebius_api_key = "test"
    settings.models["super"] = configured
    others = ["nvidia/nemotron-3-super-120b-a12b-serverless", "nvidia/other-model"]
    fake = FakeTokenFactory([response("ok")], models=[*others, listed])
    LLM(settings, client=fake).chat("coder", "super", "s", "u", mock=no_mock)
    assert fake.requests[0]["model"] == listed


def test_every_call_adds_to_the_days_spend_shared_by_all_clients(settings):
    settings.nebius_api_key = "test"
    for _ in range(2):  # two clients, as two visitors of the app would have
        fake = FakeTokenFactory([response("ok", prompt_tokens=1_000_000, completion_tokens=0)])
        LLM(settings, client=fake).chat("auditor", "nano", "s", "u", mock=no_mock)
    assert llm_module.SPEND.today() == pytest.approx(2 * settings.token_prices["nano"][0])


def test_the_spend_starts_again_each_day():
    clock = {"day": date(2026, 10, 29)}
    spend = Spend(today=lambda: clock["day"])
    spend.add(1.5)
    assert spend.today() == 1.5
    clock["day"] = date(2026, 10, 30)  # midnight UTC has passed
    assert spend.today() == 0.0


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
