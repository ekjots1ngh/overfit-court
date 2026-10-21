from overfit_court.config import get_settings
from overfit_court.data import synthetic_prices
from overfit_court.ledger import Ledger, fingerprint
from overfit_court.pipeline import record_key, run_trial
from overfit_court.sandbox import InProcessExecutor

quiet = lambda *a: None


def _settings():
    s = get_settings()
    s.nebius_api_key = None
    s.tavily_api_key = None
    return s


def test_fingerprint_changes_with_data():
    assert fingerprint(synthetic_prices(seed=1)) != fingerprint(synthetic_prices(seed=2))
    assert fingerprint(synthetic_prices(seed=1)) == fingerprint(synthetic_prices(seed=1))


def test_rerunning_the_same_grid_is_not_penalised(tmp_path):
    """Pressing the button twice is not more research."""
    px = synthetic_prices(seed=7, signal=0.0008)
    ledger = Ledger(tmp_path)
    a = run_trial("momentum", px, _settings(), ledger, InProcessExecutor(), log=quiet).evidence()
    b = run_trial("momentum", px, _settings(), ledger, InProcessExecutor(), log=quiet).evidence()
    assert b["trial_number_on_this_data"] == 2
    assert b["n_configs_total_on_this_data"] == 2 * a["n_configs"]
    assert abs(b["distinct_bets_total"] - a["distinct_bets_total"]) < 0.5
    assert b["p_real"] == a["p_real"]


def test_new_ideas_raise_the_bar(tmp_path):
    px = synthetic_prices(seed=7)
    ledger = Ledger(tmp_path)
    first = run_trial("momentum", px, _settings(), ledger, InProcessExecutor(), log=quiet).evidence()
    for idea in ["buy on mondays", "golden cross", "calm stocks", "new highs"]:
        last = run_trial(idea, px, _settings(), ledger, InProcessExecutor(), log=quiet).evidence()
    # several of these families are largely the same market bet, so ~3 new bets, not 4+
    assert last["distinct_bets_total"] > first["distinct_bets_total"] + 2
    assert last["luck_sharpe_annual"] > last["luck_sharpe_annual_this_trial_only"]
    # the record survives a restart
    again = Ledger(tmp_path)
    key = record_key(px, 0.25)
    assert len(again.history(key)) == 5
    assert sum(m.shape[1] for m in again.prior_returns(key)) == last["n_configs_total_on_this_data"]


def test_min_track_record():
    from overfit_court import stats
    assert stats.min_track_record_years(0.02, 0.05) is None
    assert 0 < stats.min_track_record_years(0.2, 0.05) < stats.min_track_record_years(0.07, 0.05)
