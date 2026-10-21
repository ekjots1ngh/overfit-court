import numpy as np
import pytest

from overfit_court import _harness
from overfit_court.config import get_settings
from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.pipeline import (TimeTravelError, expand_grid, run_trial, time_travel_probes,
                                    unseal)
from overfit_court.sandbox import InProcessExecutor, LocalExecutor, NebiusSandboxExecutor

from fake_contree import FakeContree

quiet = lambda *a: None
LEAKS = {
    "full-sample mean": "z = (prices - prices.mean()) / prices.std()\n    return -np.sign(z)",
    "negative shift": "return np.sign(prices.shift(-1) - prices).fillna(0)",
    "back-fill": "nxt = prices.where(prices.pct_change() > 0).bfill()\n    return np.sign(nxt - prices).fillna(0)",
    "centred window": "return np.sign(prices - prices.rolling(21, center=True).mean()).fillna(0)",
    "whole-series quantile": "return (prices > prices.quantile(0.5)).astype(float)",
}


def _fn(body):
    ns = {}
    exec(f"import numpy as np\nimport pandas as pd\ndef positions(prices):\n    {body}\n", ns)
    return ns["positions"]


def _settings():
    s = get_settings()
    s.nebius_api_key = None
    s.tavily_api_key = None
    return s


@pytest.mark.parametrize("name", LEAKS)
def test_catches_every_kind_of_leak(name):
    leak = _harness.time_travel(_fn(LEAKS[name]), synthetic_prices(n_days=600, seed=1), [{}])
    assert leak["leaked"], name
    assert "Time travel detected" in _harness.describe_leak(leak)


def test_never_flags_an_honest_strategy():
    px = synthetic_prices(n_days=800, seed=2)
    for f in FAMILIES.values():
        code = f.fixed_code or f.code
        ns = {}
        exec(code, ns)
        g = expand_grid(f.param_grid, 500)
        assert not _harness.time_travel(ns["positions"], px, time_travel_probes(g))["leaked"], f.name


def test_probes_span_the_grid():
    g = expand_grid({"a": list(range(10))}, 100)
    p = time_travel_probes(g)
    assert len(p) == 3 and p[0] == {"a": 0} and p[-1] == {"a": 9}


def test_time_travel_runs_as_its_own_sandbox_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(NebiusSandboxExecutor, "CACHE", tmp_path / "img")
    monkeypatch.delenv("OC_SANDBOX_IMAGE", raising=False)
    anchor = FAMILIES["anchor"]
    px = synthetic_prices(n_days=500, seed=3)
    grid = expand_grid(anchor.param_grid, 8)
    fake = FakeContree()
    res = NebiusSandboxExecutor(n_workers=2, client=fake).run(anchor.code, px, grid, 5.0,
                                                             leak_check=time_travel_probes(grid))
    assert res.leak["leaked"]
    assert sum(m["time_travel"] for m in res.metrics) == 1
    assert len(fake.created) == 1 + 1 + 1 + 2 + 1      # base, deps, trial, 2 grid branches, time travel
    local = LocalExecutor(n_workers=2).run(anchor.code, px, grid, 5.0,
                                           leak_check=time_travel_probes(grid))
    assert np.allclose(res.returns.values, local.returns.values)
    assert local.leak == res.leak


def test_leak_is_repaired_then_judged_honestly():
    t = run_trial("buy assets trading below their long-run average price", synthetic_prices(seed=11),
                  _settings(), executor=InProcessExecutor(), log=quiet)
    assert [bool(h["problem"]) for h in t.code_history] == [True, False]
    assert "Time travel detected" in t.code_history[0]["problem"]
    assert t.caught_and_repaired and not t.time_traveller
    assert t.evidence()["time_travel"] == "passed after repair"
    assert "prices.mean()" not in t.code
    assert t.verdict != "ACQUITTED"                     # the honest version has no edge on noise
    unseal(t, InProcessExecutor(), settings=_settings(), log=quiet)


def test_unrepaired_time_traveller_is_convicted_and_the_room_stays_shut():
    t = run_trial("buy assets trading below their long-run average price", synthetic_prices(seed=11),
                  _settings(), executor=InProcessExecutor(), max_repairs=0, log=quiet)
    e = t.evidence()
    assert t.time_traveller and t.verdict == "CONVICTED"
    assert e["p_real"] > 0.95                            # the statistics alone were fooled
    assert e["time_travel"].startswith("FAILED")
    assert "reads the future" in t.opinion["headline"]
    with pytest.raises(TimeTravelError):
        unseal(t, InProcessExecutor(), settings=_settings(), log=quiet)
