"""The evidence room, and the fishing expedition."""

from overfit_court.data import synthetic_prices
from overfit_court.fishing import run_expedition
from overfit_court.ledger import Ledger
from overfit_court.library import FISHABLE
from overfit_court.pipeline import commitment_hash, run_trial, split_sealed, unseal
from overfit_court.sandbox import InProcessExecutor
from overfit_court.stats import ACQUITTED, CONFIRMED


class SpyExecutor(InProcessExecutor):
    """Records the last date of every dataset handed to it."""

    def __init__(self):
        self.seen = []

    def run(self, code, prices, grid, cost_bps, leak_check=None):
        self.seen.append((prices.index.max(), len(grid)))
        return super().run(code, prices, grid, cost_bps, leak_check)


def test_research_never_sees_the_sealed_evidence(settings, silent):
    prices = synthetic_prices(seed=7, signal=0.0008)
    _, cut = split_sealed(prices)
    spy, ledger = SpyExecutor(), Ledger()
    trial = run_trial("momentum", prices, settings, ledger, spy, log=silent)
    assert all(last_date < cut for last_date, _ in spy.seen)  # the research phase

    sealed = unseal(trial, spy, ledger, settings, log=silent)
    assert spy.seen[-1] == (prices.index.max(), 1)  # one strategy, the full data
    assert sealed.sealed_days == (prices.index >= cut).sum()
    assert sealed.commitment == commitment_hash(
        trial.code, trial.judgement.best_params, trial.record["dataset"], cut
    )
    assert sealed.verdict == CONFIRMED

    again = unseal(trial, spy, ledger, settings, log=silent)  # the court remembers openings
    assert again.times_opened_before == 1
    assert again.p_real <= sealed.p_real


def test_a_fishing_expedition_on_pure_noise(settings, silent):
    expedition = run_expedition(
        synthetic_prices(seed=201),
        target=1.0,
        max_attempts=8,
        settings=settings,
        executor=InProcessExecutor(),
        seed=1,
        log=silent,
    )
    assert len(expedition.attempts) == 8
    assert expedition.n_configs == sum(a.n for a in expedition.attempts)
    assert expedition.distinct_bets < expedition.n_configs
    assert expedition.court.verdict != ACQUITTED
    assert expedition.sealed is not None and expedition.sealed.verdict != CONFIRMED
    assert expedition.claimed_family in FISHABLE


def test_a_fishing_expedition_finds_a_real_edge(settings, silent):
    expedition = run_expedition(
        synthetic_prices(seed=305, signal=0.0008),
        target=1.0,
        max_attempts=40,
        settings=settings,
        executor=InProcessExecutor(),
        seed=5,
        log=silent,
    )
    assert expedition.hit_target
    assert expedition.court.verdict == ACQUITTED
