from overfit_court.config import get_settings
from overfit_court.data import synthetic_prices
from overfit_court.fishing import run_expedition
from overfit_court.ledger import Ledger
from overfit_court.library import FAMILIES
from overfit_court.pipeline import commitment_hash, run_trial, split_sealed, unseal
from overfit_court.sandbox import InProcessExecutor

quiet = lambda *a: None


class SpyExecutor(InProcessExecutor):
    """Records the last date of every dataset an executor is given."""
    def __init__(self):
        self.seen = []

    def run(self, code, prices, grid, cost_bps, leak_check=None):
        self.seen.append((prices.index.max(), len(grid)))
        return super().run(code, prices, grid, cost_bps, leak_check)


def _settings():
    s = get_settings()
    s.nebius_api_key = None
    s.tavily_api_key = None
    return s


def test_research_never_sees_the_sealed_evidence():
    px = synthetic_prices(seed=7, signal=0.0008)
    _, cut = split_sealed(px)
    spy = SpyExecutor()
    ledger = Ledger()
    t = run_trial("momentum", px, _settings(), ledger, spy, log=quiet)
    assert all(last < cut for last, _ in spy.seen)            # research phase
    sealed = unseal(t, spy, ledger, _settings(), log=quiet)
    assert spy.seen[-1] == (px.index.max(), 1)                # one strategy, full data
    assert sealed.sealed_days == (px.index >= cut).sum()
    assert sealed.commitment == commitment_hash(t.code, t.j.best_params, t.record["dataset"], cut)
    assert sealed.verdict == "CONFIRMED"
    again = unseal(t, spy, ledger, _settings(), log=quiet)    # the court remembers openings
    assert again.times_opened_before == 1
    assert again.p_real <= sealed.p_real


def test_fishing_expedition_on_noise():
    ledger = Ledger()
    e = run_expedition(synthetic_prices(seed=201), target=1.0, max_attempts=8,
                       settings=_settings(), ledger=ledger, executor=InProcessExecutor(),
                       seed=1, log=quiet)
    assert len(e.attempts) == 8
    assert e.n_configs == sum(a.n for a in e.attempts)
    assert e.distinct_bets < e.n_configs
    assert e.court.verdict != "ACQUITTED"
    assert e.sealed is not None and e.sealed.verdict != "CONFIRMED"
    assert e.claimed_family in FAMILIES


def test_fishing_expedition_finds_a_real_edge():
    e = run_expedition(synthetic_prices(seed=305, signal=0.0008), target=1.0, max_attempts=40,
                       settings=_settings(), executor=InProcessExecutor(), seed=5, log=quiet)
    assert e.hit_target
    assert e.court.verdict == "ACQUITTED"
