"""The three demo acts behind the app's "Start here" tab (and the demo video).

Each act uses its own fixed synthetic market, so every viewer sees the same story:

1. an AI told to find alpha in pure noise, and four judges of what it brings back;
2. a strategy that reads the future: what the statistics alone say, and what
   the time-travel test catches;
3. a real edge surviving the sealed evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import stats
from .data import synthetic_prices
from .fishing import Expedition, ScriptedFisherman, run_expedition
from .library import FAMILIES, spec_for
from .pipeline import (
    Judgement,
    Log,
    Trial,
    drop_flat,
    expand_grid,
    judge_statistics,
    run_trial,
    split_sealed,
    unseal,
)
from .sandbox import InProcessExecutor, LocalExecutor

FISHING_MARKET = 244  # pure noise; the fisherman "hits" Sharpe 1 after 14 attempts
LEAKY_MARKET = 11  # pure noise
EDGE_MARKET = 7  # a planted, persistent edge
STATISTICS_ALONE_SEED = 0xBEEF  # bootstrap seed for act 2's "statistics alone" panel

ANCHOR_IDEA = "Buy assets trading below their long-run average price and short those above it."
EDGE_IDEA = "Buy the assets that went up most over the last few months and short the losers."


def act_fishing(settings, ledger, executor=None, fisherman=None, log: Log = print) -> Expedition:
    """Act 1. The scripted fisherman by default, so the story is reproducible."""
    return run_expedition(
        synthetic_prices(seed=FISHING_MARKET),
        target=1.0,
        max_attempts=40,
        fisherman=fisherman or ScriptedFisherman(FISHING_MARKET),
        settings=settings,
        ledger=ledger,
        executor=executor or InProcessExecutor(),  # library code only, so in-process is safe
        track_luck=True,
        seed=FISHING_MARKET,
        log=log,
    )


@dataclass
class TimeTravelAct:
    statistics_alone: Judgement  # the leaky draft judged with no time-travel test
    leaky_sharpe_annual: float
    trial: Trial  # the real trial: caught, repaired by Nemotron Super, judged honestly


def act_time_travel(settings, ledger, executor=None, log: Log = print, llm=None) -> TimeTravelAct:
    """Act 2. The court is handed a plausible draft that reads the future."""
    prices = synthetic_prices(seed=LEAKY_MARKET)
    anchor = FAMILIES["anchor"]

    # What the statistics would say if nobody checked for time travel.
    open_prices, _ = split_sealed(prices)
    grid = expand_grid(anchor.param_grid, settings.max_configs)
    leaky = InProcessExecutor().run(anchor.code, open_prices, grid, settings.cost_bps)
    alone = judge_statistics(drop_flat(leaky.returns), leaky.grid, [], STATISTICS_ALONE_SEED, 300)

    # What the court does: catch it, send it back for repair, judge the honest version.
    trial = run_trial(
        ANCHOR_IDEA,
        prices,
        settings,
        ledger,
        executor or LocalExecutor(),
        log=log,
        llm=llm,
        spec=spec_for(anchor),
        first_draft=anchor.code,
    )
    return TimeTravelAct(alone, stats.annualise(alone.best_sharpe), trial)


def act_real_edge(settings, ledger, executor=None, log: Log = print, llm=None) -> Trial:
    """Act 3. Nemotron plans and writes the strategy; the winner then faces the sealed evidence."""
    executor = executor or LocalExecutor()
    trial = run_trial(
        EDGE_IDEA,
        synthetic_prices(seed=EDGE_MARKET, signal=0.0008),
        settings,
        ledger,
        executor,
        log=log,
        llm=llm,
    )
    if not trial.time_traveller:
        unseal(trial, executor, ledger, settings, log=log)
    return trial
