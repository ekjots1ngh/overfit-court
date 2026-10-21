"""The whole trial against a fake Token Factory that answers like a real model:
reasoning tags, prose around code, a sloppy grid, a leaky first attempt."""
import json
from types import SimpleNamespace

from overfit_court.config import get_settings
from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.llm import LLM
from overfit_court.pipeline import run_trial, unseal
from overfit_court.sandbox import LocalExecutor

PLAN = """<think>The user wants reversion to a long-run anchor. Two parameters.</think>
Here is the spec:
```json
{"name": "Long-run anchor", "hypothesis": "Prices revert to their long-run average.",
 "param_grid": {"band": [0.5, 1.0, "2"], "window": [60.0, 120, 250]},
 "search_queries": ["mean reversion long run average"], "known_risks": ["regime change"]}
```"""
JUDGE = {"headline": "Verdict explained", "reasoning": ["a", "b", "c"],
         "what_would_change_my_mind": ["x", "y"]}


def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                                    finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=800, completion_tokens=400))


class LiveLikeTokenFactory:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, **kw):
        system, user = messages[0]["content"], messages[1]["content"]
        self.calls.append((model, system[:30]))
        if system.startswith("You are a quantitative research lead"):
            return _resp(PLAN)
        if system.startswith("You write trading signal code"):
            fam = FAMILIES["anchor"]
            code = fam.fixed_code if "Time travel detected" in user else fam.code
            return _resp(f"<think>ok</think>Sure, here's the strategy:\n```python\n{code}```\n"
                         "It normalises each asset.")
        if system.startswith("You audit trading code"):
            return _resp('Looks fine to me. {"risk": "low", "issues": []}')
        if system.startswith("You are the judge"):
            return _resp(json.dumps(JUDGE))
        raise AssertionError(f"unexpected prompt: {system[:60]}")


def test_full_trial_through_a_live_like_model_and_real_subprocess_harness():
    s = get_settings()
    s.nebius_api_key = "test"
    s.tavily_api_key = None
    fake = LiveLikeTokenFactory()
    llm = LLM(s, client=fake)
    t = run_trial("buy assets below their long-run average", synthetic_prices(seed=4), s,
                  executor=LocalExecutor(), llm=llm, log=lambda *a: None)
    assert t.spec["param_grid"] == {"band": [0.5, 1], "window": [60, 120, 250]}  # cleaned
    assert t.returns.shape[1] == 6
    assert t.caught_and_repaired and "rolling" in t.code
    models = [m for m, _ in fake.calls]
    assert models.count(s.models["super"]) == 2          # first draft + repair
    assert s.models["ultra"] in models and s.models["nano"] in models
    assert t.opinion == JUDGE
    cost = sum(c["cost_usd"] for c in t.cost_table)
    assert cost > 0 and len(t.cost_table) == len(fake.calls)
    assert unseal(t, LocalExecutor(), settings=s, log=lambda *a: None).verdict
