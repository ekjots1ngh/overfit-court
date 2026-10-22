"""Model replies are checked before anything uses them, and demo-mode routing matches words."""

import json

from fake_token_factory import FakeTokenFactory, response

from overfit_court import agents, research
from overfit_court.library import route
from overfit_court.llm import LLM

EVIDENCE = {
    "time_travel_status": "passed",
    "time_travel": "passed",
    "trial_number_on_this_data": 1,
    "n_configs": 63,
    "n_distinct_configs_on_this_data": 63,
    "distinct_bets_this_trial": 3.1,
    "distinct_bets_total": 3.1,
    "best_sharpe_annual": 0.81,
    "luck_sharpe_annual": 0.74,
    "p_value": 0.31,
    "years_of_live_trading_to_prove_it": None,
    "prob_oos_loss": 0.41,
    "walk_forward_sharpe": -0.12,
    "pbo": 0.55,
}


def live(settings, *replies):
    settings.nebius_api_key = "test"
    return LLM(settings, client=FakeTokenFactory([response(r) for r in replies]))


def test_an_audit_answer_in_the_wrong_form_is_read_cautiously(settings):
    llm = live(
        settings,
        '{"risk": "High", "issues": [{"line": 3, "problem": "uses prices.mean()"}]}',
        '{"risk": "moderate-ish", "issues": "one long string"}',
    )
    first = agents.audit(llm, "def positions(prices):\n    return prices\n")
    assert first["risk"] == "high"
    assert any("prices.mean()" in issue for issue in first["issues"])
    second = agents.audit(llm, "def positions(prices):\n    return prices\n")
    assert second["risk"] == "medium"  # an answer we can't read is not a clean bill
    assert second["issues"] == ["one long string"]


def test_the_judge_cannot_contradict_the_verdict_or_return_a_string_of_reasons(settings):
    reply = {
        "headline": "ACQUITTED: a genuine edge",
        "reasoning": "A single paragraph instead of a list.",
        "what_would_change_my_mind": [],
    }
    opinion = agents.judge(live(settings, json.dumps(reply)), "CONVICTED", EVIDENCE)
    assert opinion["headline"].startswith("CONVICTED")  # the court's own headline instead
    assert opinion["reasoning"] == ["A single paragraph instead of a list."]
    assert len(opinion["what_would_change_my_mind"]) == 2  # the court's own, as nothing came back


def test_the_clerk_can_only_cite_what_the_search_found(settings):
    hits = [
        {"query": "q", "title": "Momentum (2013)", "url": "https://example.org/a", "content": ""}
    ]
    reply = {
        "already_known": "yes",
        "summary": "Well known.",
        "citations": [
            {"title": "Made-up paper", "url": "https://example.org/invented"},
            {"title": "Renamed by the model", "url": "https://example.org/a"},
            "not even an object",
        ],
    }
    found = agents.triage_prior_art(live(settings, json.dumps(reply)), {}, hits)
    assert found["citations"] == [{"title": "Momentum (2013)", "url": "https://example.org/a"}]
    assert found["already_known"] is None  # "yes" is not a boolean


def test_a_failed_search_is_reported_and_the_trial_goes_on(settings, monkeypatch):
    class BrokenTavily:
        def __init__(self, api_key):
            pass

        def search(self, *args, **kwargs):
            raise ConnectionError("tavily is down")

    monkeypatch.setattr(research, "TavilyClient", BrokenTavily)
    settings.tavily_api_key = "test"
    hits, note = research.prior_art(["momentum anomaly"], settings)
    assert hits == [] and "tavily is down" in note
    found = agents.triage_prior_art(LLM(settings), {}, hits, note)
    assert found["already_known"] is None and "tavily is down" in found["summary"]


def test_demo_routing_matches_whole_words():
    assert route("Buy small caps with strong momentum").name == "momentum"  # not "sma"
    assert route("A systematic momentum strategy").name == "momentum"  # not "ema"
    assert route("Firms that beat expectations keep rising").name == "momentum"  # not "cta"
    assert route("Buy on Mondays, sell on Fridays").name == "weekday"  # plurals still match
    assert route("a golden cross of moving averages").name == "ma_cross"
