"""Model replies are checked before anything uses them, and demo-mode routing matches words."""

import json
from unittest.mock import ANY

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
    "luck_equivalent_tries": 12,
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


HITS = [
    {"kind": "papers", "title": "Returns to buying winners (1993)", "url": "https://ex.org/a"},
    {"kind": "practice", "title": "Momentum, after costs", "url": "https://ex.org/b"},
]


def test_the_clerk_can_only_cite_what_the_search_found(settings):
    reply = {
        "already_known": "yes",
        "summary": "Well known.",
        "cite": [1, 7, "0", True, 1, -1],  # out of range, not numbers, a repeat
        "published": {
            "lookback": {"value": 252, "source": 0},
            "skip": {"value": 21, "source": 9},  # a source that was never found
            "made_up": {"value": 1, "source": 0},  # not a parameter of the plan
        },
    }
    spec = {"param_grid": {"lookback": [60, 250], "skip": [0, 21]}}
    found = agents.triage_prior_art(live(settings, json.dumps(reply)), spec, HITS)
    assert found["citations"] == [
        {"title": "Momentum, after costs", "url": "https://ex.org/b", "kind": "practice"}
    ]
    assert found["published"] == {
        "lookback": {
            "value": 252,
            "source": {"title": HITS[0]["title"], "url": "https://ex.org/a", "kind": "papers"},
        }
    }
    assert found["already_known"] is None  # "yes" is not a boolean


def test_a_failed_search_is_reported_and_the_trial_goes_on(settings, monkeypatch):
    class BrokenTavily:
        def __init__(self, api_key):
            pass

        def search(self, *args, **kwargs):
            raise ConnectionError("tavily is down")

    monkeypatch.setattr(research, "TavilyClient", BrokenTavily)
    settings.tavily_api_key = "test"
    search = research.prior_art(["momentum anomaly"], settings)
    assert search.hits == [] and "tavily is down" in search.note
    assert search.usage()["searches"] == search.usage()["failed"] == 2  # papers and practice
    found = agents.triage_prior_art(LLM(settings), {}, search.hits, search.note)
    assert found["already_known"] is None and "tavily is down" in found["summary"]


def test_papers_and_practice_are_searched_merged_and_ranked(settings, monkeypatch):
    class FakeTavily:
        calls = []

        def __init__(self, api_key):
            pass

        def search(self, query, **options):
            FakeTavily.calls.append((query, tuple(options["include_domains"])))
            kind = "papers" if "ssrn.com" in options["include_domains"] else "practice"
            shared = {"url": "https://ex.org/shared", "title": "Both found this", "content": ""}
            results = [
                {**shared, "score": 0.9 if kind == "papers" else 0.4},
                {"url": f"https://ex.org/{kind}/{query}", "title": kind, "score": 0.5},
                {"title": "no url", "score": 1.0},
            ]
            return {"results": results, "usage": {"credits": 2}}

    monkeypatch.setattr(research, "TavilyClient", FakeTavily)
    settings.tavily_api_key = "test"
    search = research.prior_art(["q1", "q2", "q3", "q4"], settings)
    assert len(FakeTavily.calls) == 6  # three queries at most, each in papers and in practice
    urls = [hit["url"] for hit in search.hits]
    assert urls[0] == "https://ex.org/shared" and urls.count("https://ex.org/shared") == 1
    assert search.hits[0]["score"] == 0.9 and search.hits[0]["kind"] == "papers"  # best copy
    assert [hit["score"] for hit in search.hits] == sorted(
        (hit["score"] for hit in search.hits), reverse=True
    )
    assert search.usage() == {"searches": 6, "failed": 0, "credits": 12, "slowest_s": ANY}


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
    found = agents.triage_prior_art(live(settings, "no", "no"), {}, HITS)
    assert found["already_known"] is None and "triage failed" in found["summary"]
    assert [hit["url"] for hit in found["top_results"]] == ["https://ex.org/a", "https://ex.org/b"]


def test_odd_shapes_from_the_clerk_and_the_planner_are_handled(settings):
    reply = {"cite": "0", "published": ["lookback", 252]}  # a string and a list, not the shapes
    found = agents.triage_prior_art(live(settings, json.dumps(reply)), {}, HITS)
    assert found["citations"] == [] and found["published"] == {}
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
