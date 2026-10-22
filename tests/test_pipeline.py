"""Trials end to end, and the rules that turn statistics into a verdict."""

import pytest

from overfit_court._harness import LEAKED, PASSED, UNVERIFIED
from overfit_court.agents import SpecError, audit, validate_spec
from overfit_court.data import PLANTED_SIGNAL, synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.llm import LLM
from overfit_court.pipeline import IDENTICAL, expand_grid, final_verdict, run_trial
from overfit_court.report import court_record_markdown
from overfit_court.sandbox import InProcessExecutor, LocalExecutor
from overfit_court.stats import ACQUITTED, CONVICTED, HUNG_JURY


def test_large_grids_are_sampled_down():
    assert len(expand_grid({"a": list(range(20)), "b": list(range(20))}, 50)) == 50


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
