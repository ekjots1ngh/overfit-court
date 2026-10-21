"""The three demo acts behind the app's "Start here" tab and the demo video.

Each act uses its own fixed synthetic market, so the story is the same for
every viewer: an AI fooling itself on noise, a time traveller caught in the
act, and a real edge surviving the sealed evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import synthetic_prices
from .fishing import ScriptedFisherman, run_expedition
from .library import FAMILIES, spec_for
from .pipeline import (drop_flat, expand_grid, judge_statistics, run_trial, split_sealed,
                       time_travel_probes, unseal)
from .sandbox import InProcessExecutor, LocalExecutor

NOISE_SEED, EDGE_SEED = 11, 7
FISH_MARKET = 244     # a noise market where the fisherman "hits" Sharpe >= 1 after 14 attempts
ANCHOR_IDEA = "Buy assets trading below their long-run average price and short those above it."
EDGE_IDEA = "Buy the assets that went up most over the last few months and short the losers."


def act_fishing(settings, ledger, executor=None, fisherman=None, log=print):
    px = synthetic_prices(seed=FISH_MARKET)
    return run_expedition(px, 1.0, 40, fisherman or ScriptedFisherman(FISH_MARKET), settings,
                          ledger, executor or InProcessExecutor(), track_luck=True,
                          seed=FISH_MARKET, log=log)


@dataclass
class TimeTravelAct:
    statistics_alone: object      # Judgement of the leaky first draft, as if no time-travel test
    leaky_sharpe_annual: float
    trial: object                 # the real trial: caught, repaired, judged honestly


def act_time_travel(settings, ledger, executor=None, log=print, llm=None) -> TimeTravelAct:
    px = synthetic_prices(seed=NOISE_SEED)
    open_px, _ = split_sealed(px)
    anchor = FAMILIES["anchor"]
    grid = expand_grid(anchor.param_grid, settings.max_configs)
    leaky = InProcessExecutor().run(anchor.code, open_px, grid, settings.cost_bps,
                                    leak_check=time_travel_probes(grid))
    M = drop_flat(leaky.returns)
    alone = judge_statistics(M, leaky.grid, [], "0000beef", "low", 300)
    # the court is handed a plausible draft that reads the future; Nemotron Super does the repair
    trial = run_trial(ANCHOR_IDEA, px, settings, ledger, executor or LocalExecutor(), log=log,
                      llm=llm, spec=spec_for(anchor), first_draft=anchor.code)
    from . import stats
    return TimeTravelAct(alone, stats.annualise(alone.best_sharpe), trial)


def act_real_edge(settings, ledger, executor=None, log=print, llm=None):
    px = synthetic_prices(seed=EDGE_SEED, signal=0.0008)
    trial = run_trial(EDGE_IDEA, px, settings, ledger, executor or LocalExecutor(), log=log,
                      llm=llm)
    if not trial.time_traveller:
        unseal(trial, executor or LocalExecutor(), ledger, settings, log=log)
    return trial
