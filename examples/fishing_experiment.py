"""The fishing expedition, many times over: Result 1 in the README.

A fisherman is told to find a strategy with annualised Sharpe >= 1 and given
40 attempts. Each attempt is a new branch: a strategy family, parameter values
and an asset universe. It stops as soon as it hits the target. Its claim is
then judged by a standard significance test, by a check of only its final
grid, by Overfit Court (counting every branch), and by the sealed evidence.

    python -m examples.fishing_experiment [n_markets] [--nemotron]

The default fisherman is scripted (exploration plus hill-climbing). With
--nemotron and NEBIUS_API_KEY set, Nemotron 3 Super chooses where to fish.
Either way only built-in library code runs, so the in-process executor is safe.
"""

import sys
import time

import pandas as pd

from overfit_court.config import get_settings
from overfit_court.data import synthetic_prices
from overfit_court.fishing import NemotronFisherman, ScriptedFisherman, run_expedition
from overfit_court.llm import LLM
from overfit_court.sandbox import InProcessExecutor

MARKETS = [("pure noise", 0.0, 200), ("planted edge", 0.0008, 300)]  # (label, signal, first seed)


def silent(*_args):
    pass


def percent(flags) -> str:
    return f"{100 * flags.mean():.0f}%"


def main(n_markets: int, use_nemotron: bool) -> None:
    settings = get_settings()
    executor = InProcessExecutor()
    rows = []
    start = time.perf_counter()
    for label, signal, first_seed in MARKETS:
        for i in range(n_markets):
            if use_nemotron:
                fisherman = NemotronFisherman(LLM(settings), fallback_seed=i)
            else:
                fisherman = ScriptedFisherman(i)
            expedition = run_expedition(
                synthetic_prices(seed=first_seed + i, signal=signal),
                target=1.0,
                max_attempts=40,
                fisherman=fisherman,
                settings=settings,
                executor=executor,
                seed=i,
                log=silent,
            )
            rows.append({"market": label, **expedition.summary()})

    results = pd.DataFrame(rows)
    results.to_csv("fishing_results.csv", index=False)
    print(f"Fisherman: {rows[0]['fisherman']}\n")
    print(
        "| Market | Expeditions | Median branches | Median distinct bets | Hit Sharpe >= 1 "
        "| Standard test says real | Final-grid check says real | Court acquits "
        "| Sealed evidence confirms |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for label, group in results.groupby("market", sort=False):
        print(
            f"| {label} | {len(group)} | {group.configs.median():.0f} | "
            f"{group.distinct_bets.median():.0f} | {percent(group.hit_target)} | "
            f"{percent(group.naive_p_real >= 0.95)} | {percent(group.final_grid_p_real >= 0.95)} | "
            f"{percent(group.court_verdict == 'ACQUITTED')} | "
            f"{percent(group.sealed == 'CONFIRMED')} |"
        )
    print(
        f"\n{len(results)} expeditions in {time.perf_counter() - start:.0f}s; "
        "one row per expedition in fishing_results.csv"
    )


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(int(arguments[0]) if arguments else 10, "--nemotron" in sys.argv)
