"""The time-travel test: catching strategies that read the future, and failing closed."""

import numpy as np
import pandas as pd
import pytest
from fake_contree import FakeContree
from fake_token_factory import FakeTokenFactory, response

from overfit_court import _harness
from overfit_court._harness import LEAKED, PASSED, UNVERIFIED
from overfit_court.data import synthetic_prices
from overfit_court.ledger import Ledger, record_key
from overfit_court.library import FAMILIES, spec_for
from overfit_court.llm import LLM
from overfit_court.pipeline import (
    Mistrial,
    TimeTravelError,
    expand_grid,
    run_trial,
    sealed_cuts,
    split_sealed,
    unseal,
)
from overfit_court.sandbox import InProcessExecutor, LocalExecutor, NebiusSandboxExecutor
from overfit_court.stats import ACQUITTED, CONFIRMED, CONVICTED, MISTRIAL, P_VALUE_TO_ACQUIT

LEAKS = {
    "full-sample mean": "z = (prices - prices.mean()) / prices.std()\n    return -np.sign(z)",
    "negative shift": "return np.sign(prices.shift(-1) - prices).fillna(0)",
    "back-fill": (
        "nxt = prices.where(prices.pct_change() > 0).bfill()\n"
        "    return np.sign(nxt - prices).fillna(0)"
    ),
    # Only the warm-up period leaks, so only the early cuts can see these.
    "back-filled warm-up": "return np.sign(prices - prices.rolling(250).mean().bfill())",
    "short back-filled warm-up": "return np.sign(prices - prices.rolling(5).mean().bfill())",
    "centred window": "return np.sign(prices - prices.rolling(21, center=True).mean()).fillna(0)",
    "whole-series quantile": "return (prices > prices.quantile(0.5)).astype(float)",
    # Computes the full-sample anchor on its first call and reuses it on every later
    # one, which would hide the leak if runs shared the module.
    "anchor cached between calls": (
        "if not CACHE:\n        CACHE['mean'], CACHE['sd'] = prices.mean(), prices.std()\n"
        "    return -np.sign((prices - CACHE['mean']) / CACHE['sd'])"
    ),
}
LEAKY_IDEA = "buy assets trading below their long-run average price"


def strategy(body: str):
    """A loader for a strategy whose positions(prices) function has this body."""
    source = (
        "import numpy as np\nimport pandas as pd\n\nCACHE = {}\n\n"
        f"def positions(prices):\n    {body}\n"
    )
    return _harness.strategy_loader(source)


def check(load_strategy, prices, grid=None, extra_cuts=()):
    """Backtest and time-travel test a strategy in-process; return the time-travel report."""
    grid = grid or {"only": {}}
    cuts = _harness.cut_points(len(prices), extra_cuts)
    return _harness.evaluate(load_strategy, prices, grid, 5.0, True, cuts)[3]


@pytest.mark.parametrize("kind", LEAKS)
def test_catches_every_kind_of_leak(kind):
    report = check(strategy(LEAKS[kind]), synthetic_prices(n_days=1000, seed=1))
    assert report["status"] == LEAKED, kind
    assert "Time travel detected" in _harness.describe(report)


def test_never_flags_any_configuration_of_an_honest_strategy():
    prices = synthetic_prices(n_days=800, seed=2)
    for family in FAMILIES.values():
        load = _harness.strategy_loader(family.fixed_code or family.code)
        grid = expand_grid(family.param_grid, 500)
        report = check(load, prices, grid)
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


def test_a_strategy_that_is_not_deterministic_cannot_pass():
    """Two runs on the same data disagree, so a change on shorter history would prove nothing."""
    coin = strategy(
        "return pd.DataFrame(np.random.choice([-1.0, 1.0], prices.shape), prices.index, "
        "prices.columns)"
    )
    report = check(coin, synthetic_prices(n_days=300, seed=3))
    assert report["status"] == UNVERIFIED
    assert "deterministic" in _harness.describe(report)


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


def test_a_strategy_that_stalls_on_short_history_cannot_pass(settings, silent):
    """It times out its jobs instead of passing. Nothing ran, so nothing joins the record."""
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
    ledger, prices = Ledger(), synthetic_prices(seed=11)
    with pytest.raises(Mistrial, match="timed out"):
        run_trial(
            LEAKY_IDEA,
            prices,
            settings=settings,
            ledger=ledger,
            executor=LocalExecutor(timeout_s=5),
            max_repairs=0,
            spec=spec_for(FAMILIES["anchor"]),
            first_draft=stalls,
            log=silent,
        )
    assert ledger.history(record_key(prices, 0.25)) == []


def test_reports_from_failed_jobs_never_add_up_to_a_pass():
    report = _harness.new_report()
    report.update(configs=3, checks=24, expected_checks=24)
    assert _harness.merge_reports([report])["status"] == PASSED
    assert _harness.merge_reports([])["status"] == UNVERIFIED  # every job was lost
    short = dict(report, checks=23, error="ValueError: boom")
    assert _harness.merge_reports([report, short])["status"] == UNVERIFIED
    lost = _harness.unverified_report("ConnectionError", lost_configs=4)
    merged = _harness.merge_reports([report, lost])
    assert merged["status"] == UNVERIFIED and merged["lost_configs"] == 4
    # Every check counted, but something went wrong on the way: still not a pass.
    assert _harness.finish(dict(report, error="the comparison failed"))["status"] == UNVERIFIED


def test_one_lost_branch_among_several_fails_the_test():
    momentum = FAMILIES["momentum"]
    prices = synthetic_prices(n_days=400, seed=1)
    grid = expand_grid(momentum.param_grid, 16)
    fake = FakeContree(dead_config="c000")  # every try of the branch holding c000 is lost
    result = NebiusSandboxExecutor(n_chunks=4, client=fake).run(momentum.code, prices, grid, 5.0)
    assert fake.lost == 2  # the first try and its retry
    assert result.returns.shape[1] == 12 and len(result.errors) == 4
    assert result.time_travel["status"] == UNVERIFIED
    assert result.time_travel["lost_configs"] == 4
    assert "were lost" in next(iter(result.errors.values()))


def test_a_trial_whose_results_are_lost_is_a_mistrial_not_the_codes_fault(settings, silent):
    """No repair is asked for: the code did nothing wrong. The trial is still on the record."""
    settings.nebius_api_key = "test"
    requests = []

    def model(request):
        requests.append(request["messages"][0]["content"][:30])
        return response("{}")

    ledger, prices = Ledger(), synthetic_prices(n_days=600, seed=3)
    momentum = FAMILIES["momentum"]
    small = {**spec_for(momentum), "param_grid": {"lookback": [20, 60], "skip": [0, 5]}}
    executor = NebiusSandboxExecutor(n_chunks=4, client=FakeContree(dead_config="c000"))
    with pytest.raises(Mistrial, match="lost"):
        run_trial(
            "momentum",
            prices,
            settings=settings,
            ledger=ledger,
            executor=executor,
            llm=LLM(settings, client=FakeTokenFactory(model)),
            spec=small,
            first_draft=momentum.code,
            log=silent,
        )
    assert not any(r.startswith("You write trading signal code") for r in requests)
    [entry] = ledger.history(record_key(prices, 0.25))
    assert entry["verdict"] == MISTRIAL and entry["n"] > 0  # the branches that came back


def test_a_branch_that_runs_too_long_is_not_retried_and_is_the_codes_fault():
    momentum = FAMILIES["momentum"]
    prices = synthetic_prices(n_days=400, seed=1)
    grid = expand_grid(momentum.param_grid, 16)
    fake = FakeContree(slow_config="c000")  # the sandbox reports a timeout for that branch
    result = NebiusSandboxExecutor(n_chunks=4, client=fake).run(momentum.code, prices, grid, 5.0)
    assert sorted(metric["tries"] for metric in result.metrics) == [1, 1, 1, 1]
    assert result.time_travel["status"] == UNVERIFIED
    assert result.time_travel["lost_configs"] == 0
    assert "timed out" in result.time_travel["error"]


def test_a_branch_lost_once_is_retried():
    momentum = FAMILIES["momentum"]
    prices = synthetic_prices(n_days=400, seed=1)
    grid = expand_grid(momentum.param_grid, 16)
    fake = FakeContree(flaky=1)
    result = NebiusSandboxExecutor(n_chunks=4, client=fake).run(momentum.code, prices, grid, 5.0)
    assert fake.lost == 1 and not result.errors
    assert result.time_travel["status"] == PASSED
    assert sorted(metric["tries"] for metric in result.metrics) == [1, 1, 1, 2]
    [retried] = [metric for metric in result.metrics if metric["tries"] == 2]
    assert "ConnectionError" in retried["lost_tries"][0]


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


@pytest.mark.parametrize("starts", [0.75, 0.81, 0.97])
def test_the_evidence_room_voids_a_strategy_that_only_cheats_on_sealed_dates(
    starts, settings, silent
):
    """It passes on the open evidence, then reads the future once it can see sealed dates."""
    prices = synthetic_prices(seed=12)
    cheat_from = str(prices.index[int(len(prices) * starts)].date())
    sneaky = f"""
import numpy as np
import pandas as pd

def positions(prices, lookback=60):
    weights = np.sign(prices / prices.shift(lookback) - 1).fillna(0.0)
    late = prices.index >= pd.Timestamp("{cheat_from}")
    weights.loc[late] = np.sign(prices.shift(-1) / prices - 1).fillna(0.0).loc[late]
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


def test_a_cheat_confined_to_a_few_sealed_weeks_is_caught():
    """It reads tomorrow's price only on days 1970 to 2010, between the research cuts."""
    prices = synthetic_prices(seed=12)
    start, end = (str(prices.index[day].date()) for day in (1970, 2010))
    windowed = strategy(
        "weights = np.sign(prices / prices.shift(60) - 1).fillna(0.0)\n"
        f"    window = (prices.index >= '{start}') & (prices.index <= '{end}')\n"
        "    weights.loc[window] = np.sign(prices.shift(-1) / prices - 1).fillna(0.0).loc[window]\n"
        "    return weights"
    )
    assert check(windowed, prices)["status"] == PASSED  # the research cuts miss it
    _, cut = split_sealed(prices)
    report = check(windowed, prices, extra_cuts=sealed_cuts(prices, cut))
    assert report["status"] == LEAKED and start <= report["first_changed_date"] <= end


def test_a_future_stashed_in_another_module_cannot_reach_a_shorter_history():
    """A fresh module per run can't stop this; running shorter histories first can."""
    stasher = strategy(
        "seen = np.__dict__.setdefault('_overfit_court_test_stash', {})\n"
        "    if len(prices) >= len(seen.get('prices', ())):\n"
        "        seen['prices'] = prices  # the longest history seen so far\n"
        "    anchor = seen['prices']\n"
        "    return -np.sign((prices - anchor.mean()) / anchor.std())"
    )
    try:
        report = check(stasher, synthetic_prices(n_days=600, seed=1))
    finally:
        np.__dict__.pop("_overfit_court_test_stash", None)
    assert report["status"] == LEAKED


def test_a_strategy_cannot_read_the_data_file():
    """The runner deletes its inputs before any strategy code runs."""
    reads_the_file = (
        "import pandas as pd\n\n"
        "def positions(prices, lookback=5):\n"
        "    full = pd.read_csv('prices.csv', index_col=0, parse_dates=True)\n"
        "    return (full.mean() > prices).astype(float).reindex(prices.index)\n"
    )
    prices = synthetic_prices(n_days=300, seed=2)
    result = LocalExecutor(n_chunks=1).run(reads_the_file, prices, {"a": {}, "b": {}}, 5.0)
    assert result.returns.shape[1] == 0
    assert all("FileNotFoundError" in error for error in result.errors.values())


def test_cut_points_cover_early_and_late_history():
    cuts = _harness.cut_points(1890)
    assert cuts[:4] == [2, 3, 5, 10] and cuts[-1] > 0.9 * 1890 and len(cuts) == 14
    assert _harness.cut_points(2) == []  # nothing to cut, so nothing can pass
    assert _harness.cut_points(100, extra=[40, 41, 99, 100]) == [
        2,
        3,
        5,
        10,
        20,
        40,
        41,
        60,
        80,
        90,
        95,
        99,
    ]


def test_a_dataframe_with_other_dates_is_an_error():
    prices = synthetic_prices(n_days=100, seed=5)
    other = pd.DataFrame(0.5, index=range(100), columns=prices.columns)
    with pytest.raises(ValueError, match="dates or asset names"):
        _harness.target_weights(other, prices)
