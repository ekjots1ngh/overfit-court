"""Model replies are checked before anything uses them, and demo-mode routing matches words."""

import json

import pytest
from fake_token_factory import FakeTokenFactory, response

from overfit_court import agents, research
from overfit_court.fishing import FALLBACK, NemotronFisherman
from overfit_court.library import FISHABLE, route
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
    assert "headline" in opinion["note"]  # the record says whose headline it is


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


def test_no_reply_at_all_never_crashes_a_trial(settings):
    """Unreadable twice over, or empty: the court uses its own answers."""
    audited = agents.audit(live(settings, "risk: low", "risk: low"), "def positions(p): ...")
    assert audited["risk"] == "medium"
    assert any("could not be read" in issue for issue in audited["issues"])
    courts_own = agents.court_opinion("CONVICTED", EVIDENCE)
    for replies in (["Convicted."] * 2, [""] * 4):
        opinion = agents.judge(live(settings, *replies), "CONVICTED", EVIDENCE)
        assert opinion.pop("note").startswith("The court's own opinion")
        assert opinion == courts_own
    assert "note" not in agents.judge(LLM(settings.offline()), "CONVICTED", EVIDENCE)
    hits = [{"query": "q", "title": "t", "url": "https://example.org/a", "content": ""}]
    found = agents.triage_prior_art(live(settings, "no", "no"), {}, hits)
    assert found["already_known"] is None and "triage failed" in found["summary"]


def test_odd_shapes_from_the_clerk_and_the_planner_are_handled(settings):
    hits = [{"query": "q", "title": "t", "url": "https://example.org/a", "content": ""}]
    reply = {"citations": [{"url": ["https://example.org/a"]}]}  # a URL given as a list
    assert agents.triage_prior_art(live(settings, json.dumps(reply)), {}, hits)["citations"] == []
    spec = json.loads(
        '{"param_grid": {"a": [1, 2, NaN, Infinity]}, "search_queries": "momentum anomaly",'
        ' "family": ["momentum"]}'
    )
    clean = agents.validate_spec(spec)
    assert clean["param_grid"] == {"a": [1, 2]}
    assert clean["search_queries"] == ["momentum anomaly"]  # not one search per letter
    assert clean["family"] is None
    assert "def positions" in agents.write_code(LLM(settings.offline()), spec)  # even unvalidated


def test_a_planner_grid_with_absurd_values_or_names_is_rejected_cleanly(settings):
    huge = 10**400  # too big for a float: dropped, never an OverflowError
    assert agents.validate_spec({"param_grid": {"window": [huge, 20, 60]}})["param_grid"] == {
        "window": [20, 60]
    }
    for name in ("prices", "lambda", "class", "2fast"):  # each would break positions()
        with pytest.raises(agents.SpecError):
            agents.validate_spec({"param_grid": {name: [1, 2]}})
    with pytest.raises(agents.SpecError, match="at most 50"):
        agents.validate_spec({"param_grid": {"window": list(range(1, 1000))}})
    reply = '{"param_grid": {"window": [' + "9" * 400 + "]}}"
    with pytest.raises(agents.SpecError, match="planner could not"):
        agents.plan(live(settings, reply, reply), "momentum")


def test_the_audit_flags_code_that_reaches_outside_its_prices(settings):
    llm = LLM(settings.offline())
    honest = "import numpy as np\n\ndef positions(prices):\n    return np.sign(prices.diff())\n"
    assert agents.audit(llm, honest)["escapes"] == []
    sneaky = [
        "import numpy, os",  # a second module hidden after an allowed one
        "import pandas as pd\nx = pd.read_csv('prices.csv')",
        "import numpy as np\nnp.__dict__['stash'] = 1",
        "import pandas as pd\nx = pd.eval('prices', level=3)",
        "x = getattr(object, 'mro')",
    ]
    for code in sneaky:
        audited = agents.audit(llm, code)
        assert audited["risk"] == "high" and audited["escapes"], code


def test_the_nemotron_fisherman_falls_back_rather_than_crashing(settings):
    unusable = [
        json.dumps({"family": ["momentum"], "params": []}),  # a family given as a list
        "no idea",  # not JSON, even when asked again
    ]
    for reply in unusable:
        fisherman = NemotronFisherman(live(settings, reply, reply))
        family, branches, idea = fisherman.propose([], set(), 1.0, 5)
        assert idea.startswith(FALLBACK) and family in FISHABLE and branches


def test_the_nemotron_fisherman_uses_the_catalogues_own_values(settings):
    reply = {
        "family": "momentum",
        "idea": "short lookbacks",
        "params": [
            {"lookback": 20.0, "skip": 0, "top_frac": 0.2, "universe": 1},  # 20.0 means 20
            {"lookback": 20, "skip": 0, "top_frac": 0.2, "universe": True},  # True is no number
        ],
    }
    family, branches, idea = NemotronFisherman(live(settings, json.dumps(reply))).propose(
        [], set(), 1.0, 5
    )
    assert family == "momentum" and idea == "short lookbacks"
    assert branches == [{"lookback": 20, "skip": 0, "top_frac": 0.2, "universe": 1}]
    assert isinstance(branches[0]["lookback"], int)


def test_demo_routing_matches_whole_words():
    assert route("Buy small caps with strong momentum").name == "momentum"  # not "sma"
    assert route("A systematic momentum strategy").name == "momentum"  # not "ema"
    assert route("Firms that beat expectations keep rising").name == "momentum"  # not "cta"
    assert route("Buy on Mondays, sell on Fridays").name == "weekday"  # plurals still match
    assert route("a golden cross of moving averages").name == "ma_cross"
