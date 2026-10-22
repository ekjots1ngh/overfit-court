"""A whole trial through a fake Token Factory that answers like a real model:
reasoning tags, prose around code, a sloppy parameter grid, a leaky first draft."""

import json

from fake_token_factory import FakeTokenFactory, response

from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.llm import LLM
from overfit_court.pipeline import run_trial, unseal
from overfit_court.sandbox import LocalExecutor
from overfit_court.stats import CONFIRMED, CONVICTED

PLAN = """<think>The user wants reversion to a long-run anchor. Two parameters.</think>
Here is the spec:
```json
{"name": "Long-run anchor", "hypothesis": "Prices revert to their long-run average.",
 "param_grid": {"band": [0.5, 1.0, "2"], "window": [60.0, 120, 250]},
 "search_queries": ["mean reversion long run average"], "known_risks": ["regime change"]}
```"""
JUDGEMENT = {
    "headline": "The verdict, explained",
    "reasoning": ["a", "b", "c"],
    "what_would_change_my_mind": ["x", "y"],
}


def like_a_real_model(request):
    """Answer each role the way a live model plausibly would."""
    system, user = request["messages"][0]["content"], request["messages"][1]["content"]
    if system.startswith("You are a quantitative research lead"):
        return response(PLAN)
    if system.startswith("You write trading signal code"):
        anchor = FAMILIES["anchor"]
        code = anchor.fixed_code if "Time travel detected" in user else anchor.code
        return response(
            f"<think>ok</think>Sure, here's the strategy:\n```python\n{code}```\n"
            "It normalises each asset."
        )
    if system.startswith("You audit trading code"):
        return response('Looks fine to me. {"risk": "low", "issues": []}')
    if system.startswith("You are the judge"):
        return response(json.dumps(JUDGEMENT))
    raise AssertionError(f"unexpected prompt: {system[:60]}")


def test_a_full_trial_through_a_live_like_model_and_the_real_subprocess_harness(settings, silent):
    settings.nebius_api_key = "test"
    fake = FakeTokenFactory(like_a_real_model)
    trial = run_trial(
        "buy assets below their long-run average",
        synthetic_prices(seed=4),
        settings=settings,
        executor=LocalExecutor(),
        llm=LLM(settings, client=fake),
        log=silent,
    )
    assert trial.spec["param_grid"] == {"band": [0.5, 1], "window": [60, 120, 250]}  # cleaned
    assert trial.returns.shape[1] == 6
    assert trial.caught_and_repaired and "rolling" in trial.code
    models = [request["model"] for request in fake.requests]
    assert models.count(settings.models["super"]) == 2  # the first draft and its repair
    assert settings.models["ultra"] in models and settings.models["nano"] in models
    assert trial.opinion == JUDGEMENT
    assert len(trial.cost_table) == len(fake.requests)
    assert sum(call["cost_usd"] for call in trial.cost_table) > 0
    assert trial.verdict == CONVICTED  # pure noise: the honest version has no edge
    sealed = unseal(trial, executor=LocalExecutor(), log=silent)
    assert sealed.verdict != CONFIRMED and sealed.committed_params == trial.findings.best_params
