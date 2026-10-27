"""Trials end to end, and the rules that turn statistics into a verdict."""

import itertools

import numpy as np
import pandas as pd
import pytest
from fake_token_factory import FakeTokenFactory, response

from overfit_court import pipeline, stats
from overfit_court._harness import LEAKED, PASSED, UNVERIFIED
from overfit_court.agents import SpecError, audit, validate_spec
from overfit_court.data import PLANTED_SIGNAL, synthetic_prices
from overfit_court.ledger import Ledger, record_key
from overfit_court.library import FAMILIES, spec_for
from overfit_court.llm import LLM
from overfit_court.pipeline import (
    IDENTICAL,
    Mistrial,
    TimeTravelError,
    expand_grid,
    final_verdict,
    judge_statistics,
    run_trial,
    split_sealed,
    unseal,
)
from overfit_court.report import court_record_markdown
from overfit_court.sandbox import InProcessExecutor, LocalExecutor
from overfit_court.stats import ACQUITTED, CONVICTED, HUNG_JURY, MISTRIAL


def test_large_grids_are_sampled_down_without_listing_every_combination():
    small = {"a": [1, 2, 3], "b": [10, 20]}
    assert list(expand_grid(small, 50).values()) == [
        dict(zip(small, values, strict=True)) for values in itertools.product(*small.values())
    ]
    big = {name: list(range(50)) for name in "abcd"}  # 6.25 million combinations
    sample = expand_grid(big, 50)
    assert len(sample) == 50 and sample == expand_grid(big, 50)  # reproducible
    assert all(set(params) == set("abcd") for params in sample.values())


def test_a_grid_of_one_configuration_is_rejected_up_front():
    with pytest.raises(SpecError, match="only one configuration"):
        validate_spec({"param_grid": {"lookback": [60], "skip": [5.0]}})
    assert validate_spec({"param_grid": {"lookback": [60, 120.0]}})["param_grid"] == {
        "lookback": [60, 120]
    }


def test_parameters_that_change_nothing_go_back_for_repair(settings, silent):
    ignores_its_parameters = (
        "import numpy as np\n\n"
        "def positions(prices, lookback=60, skip=5, top_frac=0.3):\n"
        "    return np.sign(prices / prices.shift(20) - 1).fillna(0.0)\n"
    )
    trial = run_trial(
        "momentum",
        synthetic_prices(seed=3),
        settings=settings,
        executor=InProcessExecutor(),
        first_draft=ignores_its_parameters,
        log=silent,
    )
    assert trial.code_history[0]["problem"] == IDENTICAL
    assert trial.code != ignores_its_parameters and trial.passed_time_travel


def test_a_repair_that_breaks_the_strategy_falls_back_to_the_last_version_that_ran(
    settings, silent
):
    """The leaky draft ran; its 'repair' crashes. The draft is judged (convicted), not lost."""
    settings.nebius_api_key = "test"

    def model(request):
        if request["messages"][0]["content"].startswith("You write trading signal code"):
            return response(
                "```python\ndef positions(prices, **params):\n    raise KeyError(1)\n```"
            )
        return response("{}")  # the auditor and the judge say nothing useful

    anchor = FAMILIES["anchor"]
    ledger = Ledger()
    trial = run_trial(
        "buy below the long-run average",
        synthetic_prices(seed=11),
        settings=settings,
        ledger=ledger,
        executor=InProcessExecutor(),
        llm=LLM(settings, client=FakeTokenFactory(model)),
        spec=spec_for(anchor),
        first_draft=anchor.code,
        max_repairs=1,
        log=silent,
    )
    assert [bool(a["problem"]) for a in trial.code_history] == [True, True]
    assert trial.judged_attempt == 1 and trial.code_history[0].get("judged")
    assert trial.code == anchor.code and trial.time_traveller
    assert trial.verdict == CONVICTED
    assert ledger.history(trial.record["dataset"])[0]["verdict"] == CONVICTED


def test_a_trial_that_fails_after_its_backtests_is_still_recorded(settings, silent, monkeypatch):
    def broken(*args, **kwargs):
        raise ZeroDivisionError("the statistics broke")

    monkeypatch.setattr(pipeline, "judge_statistics", broken)
    ledger, prices = Ledger(), synthetic_prices(seed=3)
    with pytest.raises(ZeroDivisionError):
        run_trial(
            "momentum",
            prices,
            settings=settings,
            ledger=ledger,
            executor=InProcessExecutor(),
            log=silent,
        )
    [entry] = ledger.history(record_key(prices, 0.25))
    assert entry["verdict"] == MISTRIAL and entry["n"] > 0


def test_a_trial_that_fails_while_being_built_is_still_recorded(settings, silent):
    """The leaky draft's configurations ran, then the run of its repair failed."""

    class FailsOnTheSecondRun(InProcessExecutor):
        def __init__(self):
            self.runs = 0

        def run(self, *args, **kwargs):
            self.runs += 1
            if self.runs == 2:
                raise ConnectionError("the sandbox service went away")
            return super().run(*args, **kwargs)

    ledger, prices = Ledger(), synthetic_prices(seed=11)
    with pytest.raises(ConnectionError):
        run_trial(
            "buy below the long-run average",
            prices,
            settings=settings,
            ledger=ledger,
            executor=FailsOnTheSecondRun(),
            log=silent,
        )
    [entry] = ledger.history(record_key(prices, 0.25))
    assert entry["verdict"] == MISTRIAL and entry["n"] > 0


def test_a_strategy_that_never_works_is_a_mistrial_with_nothing_to_record(settings, silent):
    """No configuration ever produced returns, so nothing was tried on this data."""
    ledger, prices = Ledger(), synthetic_prices(seed=3)
    broken = "def positions(prices, **params):\n    raise KeyError('nothing works')\n"
    with pytest.raises(Mistrial, match="No version"):
        run_trial(
            "momentum",
            prices,
            settings=settings,
            ledger=ledger,
            executor=InProcessExecutor(),
            first_draft=broken,
            max_repairs=0,
            log=silent,
        )
    assert ledger.history(record_key(prices, 0.25)) == []


def test_code_that_reaches_outside_its_prices_goes_back_for_repair(settings, silent):
    momentum = FAMILIES["momentum"]
    trial = run_trial(
        "momentum",
        synthetic_prices(seed=7, signal=PLANTED_SIGNAL),
        settings=settings,
        executor=InProcessExecutor(),
        first_draft="import os\n" + momentum.code,
        log=silent,
    )
    assert "reaches outside" in trial.code_history[0]["problem"]
    assert "import os" not in trial.code and trial.verdict == ACQUITTED


def test_code_still_reaching_outside_its_prices_is_kept_out_of_the_evidence_room(settings, silent):
    """Here a harmless import, but the time-travel test can't vouch for code that does this."""
    momentum = FAMILIES["momentum"]
    trial = run_trial(
        "momentum",
        synthetic_prices(seed=7, signal=PLANTED_SIGNAL),
        settings=settings,
        executor=InProcessExecutor(),
        first_draft="import os\n" + momentum.code,
        max_repairs=0,
        log=silent,
    )
    assert trial.passed_time_travel and trial.audit["risk"] == "high"
    assert trial.findings.verdict == ACQUITTED and trial.verdict == HUNG_JURY
    with pytest.raises(TimeTravelError, match="reaches outside"):
        unseal(trial, executor=InProcessExecutor(), log=silent)


def test_a_trial_that_never_ran_anything_leaves_no_record(settings, silent):
    settings.nebius_api_key = "test"
    no_code = LLM(settings, client=FakeTokenFactory(lambda request: response("No code today.")))
    ledger, prices = Ledger(), synthetic_prices(seed=3)
    with pytest.raises(SyntaxError):  # in-process, the reply can't even be compiled
        run_trial(
            "momentum",
            prices,
            settings=settings,
            ledger=ledger,
            executor=InProcessExecutor(),
            llm=no_code,
            spec=spec_for(FAMILIES["momentum"]),
            log=silent,
        )
    assert ledger.history(record_key(prices, 0.25)) == []


def test_out_of_sample_checks_judge_this_trials_own_configurations():
    """An earlier trial that read the future must never stand in for this one."""
    open_prices, _ = split_sealed(synthetic_prices(seed=3))
    momentum, anchor = FAMILIES["momentum"], FAMILIES["anchor"]

    def returns_of(family):
        grid = expand_grid(family.param_grid, 120)
        return InProcessExecutor().run(family.code, open_prices, grid, 5.0, False).returns

    honest, leaky = returns_of(momentum), returns_of(anchor)
    grid = dict.fromkeys(honest.columns, {})
    alone = judge_statistics(honest, grid, seed=1, n_boot=200)
    after_a_cheat = judge_statistics(honest, grid, prior=[leaky.to_numpy()], seed=1, n_boot=200)
    assert after_a_cheat.walk_forward.oos_sharpe_annual == alone.walk_forward.oos_sharpe_annual
    assert after_a_cheat.cscv.prob_oos_loss == alone.cscv.prob_oos_loss
    assert after_a_cheat.reality_check.n_configs > alone.reality_check.n_configs  # still counted


def test_luck_equivalent_tries_grow_with_the_search():
    rng = np.random.default_rng(0)
    counts = []
    for n in (10, 100, 1000):
        noise = rng.normal(0, 0.01, (1500, n))
        luck = stats.reality_check(noise, 0.0, n_boot=200, seed=1).null_median
        counts.append(stats.luck_equivalent_tries(luck, 1500))
    assert counts == sorted(counts) and 5 < counts[0] < 15 and 500 < counts[-1] < 1500


def test_every_library_family_runs_and_passes_the_static_audit(settings):
    prices = synthetic_prices(n_days=400, seed=1)
    llm = LLM(settings)
    for family in FAMILIES.values():
        grid = expand_grid(family.param_grid, 200)
        result = InProcessExecutor().run(family.code, prices, grid, 5.0, time_travel=False)
        assert not result.errors, family.name
        assert audit(llm, family.code)["risk"] == "low", family.name


def test_noise_is_not_acquitted(settings, silent):
    trial = run_trial(
        "momentum",
        synthetic_prices(seed=3),
        settings=settings,
        executor=InProcessExecutor(),
        log=silent,
    )
    assert trial.verdict != ACQUITTED
    assert trial.data_source.startswith("12 assets, 2520 days")
    assert "Distinct bets" in court_record_markdown(trial)


def test_a_strong_edge_is_acquitted_through_the_subprocess_harness(settings, silent):
    prices = synthetic_prices(seed=7, signal=PLANTED_SIGNAL)
    trial = run_trial("momentum", prices, settings=settings, executor=LocalExecutor(), log=silent)
    assert trial.verdict == ACQUITTED
    assert trial.passed_time_travel and trial.time_travel["configs"] == trial.returns.shape[1]
    assert trial.evidence()["years_of_live_trading_to_prove_it"] < 5


def test_overrides_only_ever_make_the_verdict_stricter():
    assert final_verdict(ACQUITTED, LEAKED, audit_risk="low") == CONVICTED
    assert final_verdict(ACQUITTED, UNVERIFIED, audit_risk="low") == CONVICTED
    assert final_verdict(ACQUITTED, None, audit_risk="low") == CONVICTED  # the test never ran
    assert final_verdict(ACQUITTED, PASSED, audit_risk="high") == HUNG_JURY
    assert final_verdict(CONVICTED, PASSED, audit_risk="low") == CONVICTED
    assert final_verdict(HUNG_JURY, PASSED, audit_risk="high") == HUNG_JURY
    assert final_verdict(ACQUITTED, PASSED, audit_risk="low") == ACQUITTED


def test_the_literatures_own_configuration_is_tested_on_its_own():
    combos = itertools.product([60, 120, 250], [0, 21])
    grid = {f"k{i}": {"lookback": lb, "skip": sk} for i, (lb, sk) in enumerate(combos)}
    returns = pd.DataFrame(
        np.random.default_rng(0).normal(0.0005, 0.01, (1000, 6)), columns=list(grid)
    )
    paper = {"title": "Returns to buying winners", "url": "https://ex.org/a", "kind": "papers"}
    published = {
        "lookback": {"value": 252, "source": paper},  # a year of trading days
        "skip": {"value": 21, "source": paper},
    }
    test = pipeline.literature_test(published, grid, returns)
    assert test["params"] == {"lookback": 250, "skip": 21} and not test["exact"]
    assert test["sources"] == [paper]
    assert test["p_value"] == pytest.approx(stats.single_test_p_value(returns["k5"]), abs=1e-6)
    # Every parameter needs a published value, or there is nothing pre-registered.
    assert pipeline.literature_test({"lookback": published["lookback"]}, grid, returns) is None
    # If that configuration never ran, the nearest one that did is used.
    nearest = pipeline.literature_test(published, grid, returns.drop(columns="k5"))
    assert nearest["params"] in ({"lookback": 120, "skip": 21}, {"lookback": 250, "skip": 0})
