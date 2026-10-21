"""The court remembers."""

from overfit_court.data import synthetic_prices
from overfit_court.ledger import Ledger, fingerprint, record_key
from overfit_court.pipeline import run_trial
from overfit_court.sandbox import InProcessExecutor


def test_fingerprints_identify_the_exact_data():
    assert fingerprint(synthetic_prices(seed=1)) != fingerprint(synthetic_prices(seed=2))
    assert fingerprint(synthetic_prices(seed=1)) == fingerprint(synthetic_prices(seed=1))


def test_rerunning_the_same_grid_is_not_penalised(tmp_path, settings, silent):
    """Pressing the button twice is not more research."""
    prices = synthetic_prices(seed=7, signal=0.0008)
    ledger = Ledger(tmp_path)
    first = run_trial("momentum", prices, settings, ledger, InProcessExecutor(), log=silent)
    again = run_trial("momentum", prices, settings, ledger, InProcessExecutor(), log=silent)
    a, b = first.evidence(), again.evidence()
    assert b["trial_number_on_this_data"] == 2
    assert b["n_configs_total_on_this_data"] == 2 * a["n_configs"]
    assert abs(b["distinct_bets_total"] - a["distinct_bets_total"]) < 0.5
    assert b["p_real"] == a["p_real"]


def test_new_ideas_raise_the_bar_and_the_record_survives_a_restart(tmp_path, settings, silent):
    prices = synthetic_prices(seed=7)
    ledger = Ledger(tmp_path)
    first = run_trial("momentum", prices, settings, ledger, InProcessExecutor(), log=silent)
    for idea in ["buy on mondays", "golden cross", "calm stocks", "new highs"]:
        last = run_trial(idea, prices, settings, ledger, InProcessExecutor(), log=silent)
    e = last.evidence()
    # Several of these families are largely the same market bet: about 3 new bets, not 4+.
    assert e["distinct_bets_total"] > first.evidence()["distinct_bets_total"] + 2
    assert e["luck_sharpe_annual"] > e["luck_sharpe_annual_this_trial_only"]

    reopened = Ledger(tmp_path)
    key = record_key(prices, 0.25)
    assert len(reopened.history(key)) == 5
    tried = sum(array.shape[1] for array in reopened.prior_returns(key))
    assert tried == e["n_configs_total_on_this_data"]


def test_clearing_forgets_the_data_and_its_files(tmp_path, settings, silent):
    prices = synthetic_prices(seed=7)
    ledger = Ledger(tmp_path)
    run_trial("momentum", prices, settings, ledger, InProcessExecutor(), log=silent)
    key = record_key(prices, 0.25)
    ledger.clear(key)
    assert ledger.history(key) == []
    assert list(tmp_path.glob("*.npy")) == []
