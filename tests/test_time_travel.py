"""The time-travel test: catching strategies that read the future, and failing closed."""

import numpy as np
import pandas as pd
import pytest
from fake_contree import FakeContree

from overfit_court import _harness
from overfit_court._harness import LEAKED, PASSED, UNVERIFIED
from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES, spec_for
from overfit_court.pipeline import TimeTravelError, expand_grid, run_trial, unseal
from overfit_court.sandbox import InProcessExecutor, LocalExecutor, NebiusSandboxExecutor
from overfit_court.stats import ACQUITTED, CONFIRMED, CONVICTED, P_VALUE_TO_ACQUIT

LEAKS = {
    "full-sample mean": "z = (prices - prices.mean()) / prices.std()\n    return -np.sign(z)",
    "negative shift": "return np.sign(prices.shift(-1) - prices).fillna(0)",
    "back-fill": (
        "nxt = prices.where(prices.pct_change() > 0).bfill()\n"
        "    return np.sign(nxt - prices).fillna(0)"
    ),
    # Only the warm-up period leaks, so only the early cuts can see it.
    "back-filled warm-up": "return np.sign(prices - prices.rolling(250).mean().bfill())",
    "centred window": "return np.sign(prices - prices.rolling(21, center=True).mean()).fillna(0)",
    "whole-series quantile": "return (prices > prices.quantile(0.5)).astype(float)",
}
LEAKY_IDEA = "buy assets trading below their long-run average price"


def strategy(body: str, signature: str = "prices"):
    namespace = {}
    source = f"import numpy as np\nimport pandas as pd\ndef positions({signature}):\n    {body}\n"
    exec(source, namespace)
    return namespace["positions"]


def check(positions, prices, grid=None):
    """Backtest and time-travel test `positions` in-process; return the time-travel report."""
    return _harness.evaluate(positions, prices, grid or {"only": {}}, 5.0)[3]


@pytest.mark.parametrize("kind", LEAKS)
def test_catches_every_kind_of_leak(kind):
    report = check(strategy(LEAKS[kind]), synthetic_prices(n_days=1000, seed=1))
    assert report["status"] == LEAKED, kind
    assert "Time travel detected" in _harness.describe(report)


def test_never_flags_any_configuration_of_an_honest_strategy():
    prices = synthetic_prices(n_days=800, seed=2)
    for family in FAMILIES.values():
        namespace = {}
        exec(family.fixed_code or family.code, namespace)
        grid = expand_grid(family.param_grid, 500)
        report = check(namespace["positions"], prices, grid)
        assert report["status"] == PASSED, family.name
        assert report["configs"] == len(grid)
        assert report["checks"] == len(grid) * len(_harness.cut_points(len(prices)))


def test_a_strategy_that_crashes_on_short_history_cannot_pass():
    picky = strategy(
        "if len(prices) < 700:\n        raise ValueError('not enough history')\n"
        "    return np.sign(prices - prices.rolling(20).mean()).fillna(0)"
    )
    report = check(picky, synthetic_prices(n_days=1000, seed=3))
    assert report["status"] == UNVERIFIED
    assert report["checks"] < report["expected_checks"]
    assert "not enough history" in _harness.describe(report)
    assert not _harness.passed(report)


def test_wrongly_shaped_output_gets_a_precise_error_not_zero_weights():
    prices = synthetic_prices(n_days=300, seed=4)
    keys, _, errors, _ = _harness.evaluate(
        strategy("return pd.DataFrame(np.ones(prices.shape))"), prices, {"only": {}}, 5.0
    )
    assert keys == [] and "same index and columns" in errors["only"]
    keys, R, errors, _ = _harness.evaluate(
        strategy("return np.ones(prices.shape)"), prices, {"only": {}}, 5.0
    )
    assert keys == ["only"] and R.std() > 0  # an array of the right shape is accepted


def test_every_configuration_is_tested_so_a_parameter_dependent_leak_is_caught(settings, silent):
    """The leak only switches on for some windows, and the winner is one of them."""
    leaks_for_long_windows = """
import numpy as np
import pandas as pd

def positions(prices, band=0.5, window=250):
    if window in (120, 250):  # "long windows" anchor on the whole series
        ma, sd = prices.mean(), prices.std()
    else:
        ma = prices.rolling(window, min_periods=window).mean()
        sd = prices.rolling(window, min_periods=window).std()
    z = (prices - ma) / sd
    return (-np.sign(z)).where(z.abs() > band, 0.0).fillna(0.0)
"""
    trial = run_trial(
        LEAKY_IDEA,
        synthetic_prices(seed=11),
        settings=settings,
        executor=InProcessExecutor(),
        max_repairs=0,
        spec=spec_for(FAMILIES["anchor"]),
        first_draft=leaks_for_long_windows,
        log=silent,
    )
    assert trial.findings.best_params["window"] in (120, 250)  # selection favours the cheat
    assert trial.time_travel["params"]["window"] in (120, 250)
    assert trial.time_traveller and trial.verdict == CONVICTED


def test_a_strategy_the_test_cannot_run_is_convicted_and_kept_out_of_the_evidence_room(
    settings, silent
):
    """It crashes on short history, which used to count as passing."""
    leaky_and_picky = """
import numpy as np
import pandas as pd

def positions(prices, band=0.5, window=250):
    if len(prices) < 1800:
        raise ValueError("not enough history")
    z = (prices - prices.mean()) / prices.std()
    return (-np.sign(z)).where(z.abs() > band, 0.0)
"""
    trial = run_trial(
        LEAKY_IDEA,
        synthetic_prices(seed=11),
        settings=settings,
        executor=InProcessExecutor(),
        max_repairs=0,
        spec=spec_for(FAMILIES["anchor"]),
        first_draft=leaky_and_picky,
        log=silent,
    )
    assert trial.time_travel_status == UNVERIFIED and not trial.time_traveller
    assert trial.verdict == CONVICTED
    assert trial.evidence()["time_travel"].startswith("FAILED")
    assert "could not" in trial.opinion["headline"]
    with pytest.raises(TimeTravelError):
        unseal(trial, executor=InProcessExecutor(), log=silent)


def test_a_lost_branch_cannot_pass_for_a_clean_bill(settings, silent):
    """A strategy that stalls on short history times out its branches instead of passing."""
    stalls = """
import time
import numpy as np
import pandas as pd

def positions(prices, band=0.5, window=250):
    if len(prices) < 1000:
        time.sleep(60)
    z = (prices - prices.mean()) / prices.std()
    return (-np.sign(z)).where(z.abs() > band, 0.0)
"""
    with pytest.raises(RuntimeError, match="timed out"):
        run_trial(
            LEAKY_IDEA,
            synthetic_prices(seed=11),
            settings=settings,
            executor=LocalExecutor(timeout_s=5),
            max_repairs=0,
            spec=spec_for(FAMILIES["anchor"]),
            first_draft=stalls,
            log=silent,
        )


def test_reports_from_lost_jobs_never_add_up_to_a_pass():
    report = _harness.new_report()
    report.update(configs=3, checks=24, expected_checks=24)
    assert _harness.merge_reports([report])["status"] == PASSED
    assert _harness.merge_reports([])["status"] == UNVERIFIED  # every job was lost
    short = dict(report, checks=23, error="ValueError: boom")
    assert _harness.merge_reports([report, short])["status"] == UNVERIFIED


def test_the_time_travel_test_runs_inside_every_sandbox_branch():
    anchor = FAMILIES["anchor"]
    prices = synthetic_prices(n_days=500, seed=3)
    grid = expand_grid(anchor.param_grid, 8)
    fake = FakeContree()
    remote = NebiusSandboxExecutor(n_chunks=2, client=fake).run(anchor.code, prices, grid, 5.0)
    assert remote.time_travel["status"] == LEAKED
    assert remote.time_travel["configs"] == len(grid)
    # base, dependencies, trial checkpoint, then one branch per chunk of the grid
    assert len(fake.created) == 1 + 1 + 1 + 2
    local = LocalExecutor(n_chunks=2).run(anchor.code, prices, grid, 5.0)
    assert np.allclose(remote.returns.to_numpy(), local.returns.to_numpy())
    assert local.time_travel == remote.time_travel


def test_a_leak_is_repaired_then_judged_honestly(settings, silent):
    trial = run_trial(
        LEAKY_IDEA,
        synthetic_prices(seed=11),
        settings=settings,
        executor=InProcessExecutor(),
        log=silent,
    )
    assert [bool(attempt["problem"]) for attempt in trial.code_history] == [True, False]
    assert "Time travel detected" in trial.code_history[0]["problem"]
    assert trial.caught_and_repaired and trial.passed_time_travel
    assert trial.evidence()["time_travel"] == "passed after repair"
    assert "prices.mean()" not in trial.code
    assert trial.verdict != ACQUITTED  # the honest version has no edge on pure noise
    sealed = unseal(trial, executor=InProcessExecutor(), log=silent)
    assert sealed.verdict != CONFIRMED and sealed.sealed_days > 400


def test_an_unrepaired_time_traveller_is_convicted_and_the_room_stays_shut(settings, silent):
    trial = run_trial(
        LEAKY_IDEA,
        synthetic_prices(seed=11),
        settings=settings,
        executor=InProcessExecutor(),
        max_repairs=0,
        log=silent,
    )
    evidence = trial.evidence()
    assert trial.time_traveller and trial.verdict == CONVICTED
    assert trial.findings.verdict == ACQUITTED  # the statistics alone were fooled
    assert evidence["p_value"] <= P_VALUE_TO_ACQUIT
    assert evidence["time_travel"].startswith("FAILED")
    assert evidence["failed_checks"] == ["it reads the future"]
    assert "reads the future" in trial.opinion["headline"]
    with pytest.raises(TimeTravelError):
        unseal(trial, executor=InProcessExecutor(), log=silent)


def test_the_evidence_room_voids_a_strategy_that_only_cheats_on_sealed_dates(settings, silent):
    """It passes on the open evidence, then reads the future once it can see the sealed dates."""
    prices = synthetic_prices(seed=12)
    sealed_from = str(prices.index[int(len(prices) * 0.75)].date())
    sneaky = f"""
import numpy as np
import pandas as pd

def positions(prices, lookback=60):
    weights = np.sign(prices / prices.shift(lookback) - 1).fillna(0.0)
    sealed = prices.index >= pd.Timestamp("{sealed_from}")
    weights.loc[sealed] = np.sign(prices.shift(-1) / prices - 1).fillna(0.0).loc[sealed]
    return weights
"""
    spec = {"name": "sneaky", "param_grid": {"lookback": [20, 60, 120]}, "family": "momentum"}
    trial = run_trial(
        "momentum",
        prices,
        settings=settings,
        executor=InProcessExecutor(),
        spec=spec,
        first_draft=sneaky,
        log=silent,
    )
    assert trial.passed_time_travel
    with pytest.raises(TimeTravelError, match="void"):
        unseal(trial, executor=InProcessExecutor(), log=silent)
    seals = trial.ledger.seals(trial.record["dataset"])
    assert len(seals) == 1 and seals[0]["result"]["verdict"].startswith("void")


def test_cut_points_cover_early_and_late_history():
    cuts = _harness.cut_points(1890)
    assert cuts[0] < 20 and cuts[-1] > 1500 and len(cuts) == len(_harness.CUT_FRACTIONS)
    assert _harness.cut_points(2) == []  # nothing to cut, so nothing can pass


def test_a_dataframe_with_other_dates_is_an_error():
    prices = synthetic_prices(n_days=100, seed=5)
    other = pd.DataFrame(0.5, index=range(100), columns=prices.columns)
    with pytest.raises(ValueError, match="dates or asset names"):
        _harness.target_weights(other, prices)
