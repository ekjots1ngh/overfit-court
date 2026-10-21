"""The time-travel test: catching strategies that read the future."""

import numpy as np
import pytest
from fake_contree import FakeContree

from overfit_court import _harness
from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.pipeline import (
    TimeTravelError,
    expand_grid,
    run_trial,
    time_travel_probes,
    unseal,
)
from overfit_court.sandbox import InProcessExecutor, LocalExecutor, NebiusSandboxExecutor
from overfit_court.stats import ACQUITTED, CONVICTED

LEAKS = {
    "full-sample mean": "z = (prices - prices.mean()) / prices.std()\n    return -np.sign(z)",
    "negative shift": "return np.sign(prices.shift(-1) - prices).fillna(0)",
    "back-fill": (
        "nxt = prices.where(prices.pct_change() > 0).bfill()\n"
        "    return np.sign(nxt - prices).fillna(0)"
    ),
    "centred window": "return np.sign(prices - prices.rolling(21, center=True).mean()).fillna(0)",
    "whole-series quantile": "return (prices > prices.quantile(0.5)).astype(float)",
}
LEAKY_IDEA = "buy assets trading below their long-run average price"


def strategy(body: str):
    namespace = {}
    exec(
        f"import numpy as np\nimport pandas as pd\ndef positions(prices):\n    {body}\n", namespace
    )
    return namespace["positions"]


@pytest.mark.parametrize("kind", LEAKS)
def test_catches_every_kind_of_leak(kind):
    report = _harness.time_travel(strategy(LEAKS[kind]), synthetic_prices(n_days=600, seed=1), [{}])
    assert report["leaked"], kind
    assert "Time travel detected" in _harness.describe_leak(report)


def test_never_flags_an_honest_strategy():
    prices = synthetic_prices(n_days=800, seed=2)
    for family in FAMILIES.values():
        namespace = {}
        exec(family.fixed_code or family.code, namespace)
        probes = time_travel_probes(expand_grid(family.param_grid, 500))
        assert not _harness.time_travel(namespace["positions"], prices, probes)["leaked"]


def test_probes_are_the_first_middle_and_last_configurations():
    probes = time_travel_probes(expand_grid({"a": list(range(10))}, 100))
    assert probes == [{"a": 0}, {"a": 4}, {"a": 9}]


def test_time_travel_runs_as_its_own_sandbox_branch():
    anchor = FAMILIES["anchor"]
    prices = synthetic_prices(n_days=500, seed=3)
    grid = expand_grid(anchor.param_grid, 8)
    fake = FakeContree()
    remote = NebiusSandboxExecutor(n_chunks=2, client=fake).run(
        anchor.code, prices, grid, 5.0, leak_check=time_travel_probes(grid)
    )
    assert remote.leak["leaked"]
    assert sum(metric["time_travel"] for metric in remote.metrics) == 1
    # base, dependencies, trial checkpoint, two grid branches, one time-travel branch
    assert len(fake.created) == 1 + 1 + 1 + 2 + 1
    local = LocalExecutor(n_chunks=2).run(
        anchor.code, prices, grid, 5.0, leak_check=time_travel_probes(grid)
    )
    assert np.allclose(remote.returns.to_numpy(), local.returns.to_numpy())
    assert local.leak == remote.leak


def test_a_leak_is_repaired_then_judged_honestly(settings, silent):
    trial = run_trial(
        LEAKY_IDEA, synthetic_prices(seed=11), settings, executor=InProcessExecutor(), log=silent
    )
    assert [bool(attempt["problem"]) for attempt in trial.code_history] == [True, False]
    assert "Time travel detected" in trial.code_history[0]["problem"]
    assert trial.caught_and_repaired and not trial.time_traveller
    assert trial.evidence()["time_travel"] == "passed after repair"
    assert "prices.mean()" not in trial.code
    assert trial.verdict != ACQUITTED  # the honest version has no edge on pure noise
    unseal(trial, InProcessExecutor(), settings=settings, log=silent)


def test_an_unrepaired_time_traveller_is_convicted_and_the_room_stays_shut(settings, silent):
    trial = run_trial(
        LEAKY_IDEA,
        synthetic_prices(seed=11),
        settings,
        executor=InProcessExecutor(),
        max_repairs=0,
        log=silent,
    )
    evidence = trial.evidence()
    assert trial.time_traveller and trial.verdict == CONVICTED
    assert trial.judgement.verdict == ACQUITTED  # the statistics alone were fooled
    assert evidence["p_real"] > 0.95
    assert evidence["time_travel"].startswith("FAILED")
    assert "reads the future" in trial.opinion["headline"]
    with pytest.raises(TimeTravelError):
        unseal(trial, InProcessExecutor(), settings=settings, log=silent)
