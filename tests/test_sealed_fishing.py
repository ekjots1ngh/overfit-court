"""The evidence room, and the fishing expedition."""

import json

import numpy as np
import pytest

from overfit_court.data import PLANTED_SIGNAL, synthetic_prices
from overfit_court.fishing import run_expedition
from overfit_court.ledger import Ledger, record_key, record_seed
from overfit_court.library import FISHABLE
from overfit_court.pipeline import (
    commitment_hash,
    judge_statistics,
    run_trial,
    split_sealed,
    unseal,
)
from overfit_court.report import court_record_markdown
from overfit_court.sandbox import InProcessExecutor
from overfit_court.stats import ACQUITTED, CONFIRMED


class SpyExecutor(InProcessExecutor):
    """Records every dataset's last date and cost, and how many seal openings were on record."""

    def __init__(self, ledger=None):
        self.seen, self.ledger = [], ledger

    def run(self, code, prices, grid, cost_bps, time_travel=True, extra_cuts=()):
        seals = None
        if self.ledger is not None:
            seals = sum(len(entry["seals"]) for entry in self.ledger.index.values())
        self.seen.append((prices.index.max(), len(grid), cost_bps, seals))
        return super().run(code, prices, grid, cost_bps, time_travel, extra_cuts)


def test_research_never_sees_the_sealed_evidence(settings, silent):
    prices = synthetic_prices(seed=7, signal=PLANTED_SIGNAL)
    _, cut = split_sealed(prices)
    ledger = Ledger()
    spy = SpyExecutor(ledger)
    trial = run_trial(
        "momentum", prices, settings=settings, ledger=ledger, executor=spy, log=silent
    )
    assert all(last_date < cut for last_date, *_ in spy.seen)  # the research phase

    sealed = unseal(trial, executor=spy, log=silent)
    last_date, n_configs, _, seals_when_run = spy.seen[-1]
    assert (last_date, n_configs) == (prices.index.max(), 1)  # one strategy, the full data
    assert seals_when_run == 1  # the opening was on record before the test ran
    assert sealed.sealed_days == (prices.index >= cut).sum()
    assert sealed.commitment == commitment_hash(
        trial.code,
        trial.findings.best_params,
        key=trial.record["dataset"],
        cut=cut,
        cost_bps=trial.cost_bps,
    )
    assert len(sealed.commitment) == 64  # the full SHA-256
    assert sealed.verdict == CONFIRMED

    again = unseal(trial, executor=spy, log=silent)  # the court remembers openings
    assert again.times_opened_before == 1
    assert again.p_value >= sealed.p_value


def test_the_evidence_room_uses_the_trials_own_costs(settings, silent):
    """Lowering the cost slider after the trial must not flatter the sealed test."""
    trial = run_trial(
        "momentum",
        synthetic_prices(seed=7, signal=PLANTED_SIGNAL),
        settings=settings,
        executor=InProcessExecutor(),
        log=silent,
    )
    settings.cost_bps = 0.0
    spy = SpyExecutor()
    sealed = unseal(trial, executor=spy, log=silent)
    assert spy.seen[-1][2] == trial.cost_bps == sealed.cost_bps == 5.0
    other_cost = commitment_hash(
        trial.code,
        sealed.committed_params,
        key=trial.record["dataset"],
        cut=trial.cut,
        cost_bps=0.0,
    )
    assert other_cost != sealed.commitment


def test_a_failed_opening_still_counts(settings, silent):
    trial = run_trial(
        "momentum",
        synthetic_prices(seed=7, signal=PLANTED_SIGNAL),
        settings=settings,
        executor=InProcessExecutor(),
        log=silent,
    )
    trial.code = "def positions(prices, **params):\n    raise KeyError('gone')\n"
    with pytest.raises(RuntimeError, match="committed strategy failed"):
        unseal(trial, executor=InProcessExecutor(), log=silent)
    seals = trial.ledger.seals(trial.record["dataset"])
    assert len(seals) == 1 and seals[0]["result"] == {"verdict": "failed to run"}
    assert "KeyError" in trial.failed_opening  # and the trial's own record says so
    assert "Opened, with no valid result" in court_record_markdown(trial)
    assert json.loads(trial.to_json())["sealed_evidence"]["valid_result"] is False


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


def test_an_expedition_counts_each_branch_once(settings, silent):
    prices = synthetic_prices(seed=201)
    expedition = run_expedition(
        prices,
        target=1.0,
        max_attempts=8,
        settings=settings,
        ledger=Ledger(),
        executor=InProcessExecutor(),
        seed=1,
        open_seal=False,
        track_luck=True,
        log=silent,
    )
    assert expedition.court.n_tried == expedition.n_configs
    grid = dict.fromkeys(expedition.returns.columns, {})
    by_hand = judge_statistics(
        expedition.returns, grid, seed=record_seed(record_key(prices, 0.25)), n_boot=500
    )
    assert expedition.court.reality_check == by_hand.reality_check
    luck = [attempt.luck_annual for attempt in expedition.attempts]
    assert luck == sorted(luck)  # the bar for luck only rises as branches accumulate


def test_a_fishing_expedition_finds_a_real_edge(settings, silent):
    expedition = run_expedition(
        synthetic_prices(seed=305, signal=PLANTED_SIGNAL),
        target=1.0,
        max_attempts=40,
        settings=settings,
        executor=InProcessExecutor(),
        seed=5,
        log=silent,
    )
    assert expedition.hit_target
    assert expedition.court.verdict == ACQUITTED
    assert np.isfinite(expedition.naive_p_value) and expedition.naive_p_value < 0.05
