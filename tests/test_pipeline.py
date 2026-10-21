from overfit_court.agents import audit
from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.llm import LLM
from overfit_court.pipeline import expand_grid, final_verdict, run_trial
from overfit_court.report import court_record_markdown
from overfit_court.sandbox import InProcessExecutor, LocalExecutor
from overfit_court.stats import ACQUITTED, CONVICTED, HUNG_JURY


def test_large_grids_are_sampled_down():
    assert len(expand_grid({"a": list(range(20)), "b": list(range(20))}, 50)) == 50


def test_every_library_family_runs_and_passes_the_static_audit(settings):
    prices = synthetic_prices(n_days=400, seed=1)
    llm = LLM(settings)
    for family in FAMILIES.values():
        result = InProcessExecutor().run(
            family.code, prices, expand_grid(family.param_grid, 200), 5.0
        )
        assert not result.errors, family.name
        assert audit(llm, family.code)["risk"] == "low", family.name


def test_noise_is_not_acquitted(settings, silent):
    trial = run_trial("momentum", synthetic_prices(seed=3), settings, log=silent)
    assert trial.verdict != ACQUITTED
    assert "Distinct bets" in court_record_markdown(trial)


def test_a_strong_edge_is_acquitted_through_the_subprocess_harness(settings, silent):
    prices = synthetic_prices(seed=7, signal=0.0008)
    trial = run_trial("momentum", prices, settings, executor=LocalExecutor(), log=silent)
    assert trial.verdict == ACQUITTED


def test_overrides_only_ever_make_the_verdict_stricter():
    assert final_verdict(ACQUITTED, reads_the_future=True, audit_risk="low") == CONVICTED
    assert final_verdict(ACQUITTED, reads_the_future=False, audit_risk="high") == HUNG_JURY
    assert final_verdict(CONVICTED, reads_the_future=False, audit_risk="low") == CONVICTED
    assert final_verdict(HUNG_JURY, reads_the_future=False, audit_risk="high") == HUNG_JURY
