"""Can statistics catch a strategy that reads the future? No. Time travel can.

The 'anchor' idea (buy assets below their long-run average) is plausible, and
its obvious implementation normalises by the mean and standard deviation of the
WHOLE series, so every position quietly uses prices that haven't happened yet.

On pure-noise markets this experiment asks:
  - how often would the court's statistics alone acquit the leaky version?
  - how often does the time-travel test catch it?
  - how often does the time-travel test wrongly flag an honest strategy?
    (every library family, every market)

Usage: python examples/time_travel_experiment.py [n_markets]
"""
import sys
import time
from collections import Counter

import numpy as np

from overfit_court import _harness, stats
from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.pipeline import (drop_flat, expand_grid, judge_statistics, split_sealed,
                                    time_travel_probes)
from overfit_court.sandbox import InProcessExecutor

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ex = InProcessExecutor()
anchor = FAMILIES["anchor"]
grid = expand_grid(anchor.param_grid, 200)
probes = time_travel_probes(grid)
verdicts, caught, sharpes, luck = Counter(), 0, [], []
false_flags, honest_checks = 0, 0
t0 = time.perf_counter()
for i in range(n):
    px = synthetic_prices(seed=500 + i)
    open_px, _ = split_sealed(px)
    res = ex.run(anchor.code, open_px, grid, 5.0, leak_check=probes)
    M = drop_flat(res.returns)
    j = judge_statistics(M, res.grid, [], f"{i:08x}", "low", n_boot=300)
    verdicts[j.verdict] += 1
    caught += bool(res.leak["leaked"])
    sharpes.append(stats.annualise(j.best_sharpe))
    luck.append(j.rc.null_median_annual)
    for f in FAMILIES.values():
        if f.name == "anchor":
            continue
        ns = {}
        exec(f.code, ns)
        g = expand_grid(f.param_grid, 500)
        leak = _harness.time_travel(ns["positions"], open_px, time_travel_probes(g))
        honest_checks += 1
        false_flags += bool(leak["leaked"])
    ns = {}
    exec(anchor.fixed_code, ns)
    leak = _harness.time_travel(ns["positions"], open_px, probes)
    honest_checks += 1
    false_flags += bool(leak["leaked"])

print(f"Leaky 'long-run average' strategy on {n} pure-noise markets ({len(grid)} configs each):\n")
print("| | Result |")
print("|---|---|")
print(f"| Median best backtest Sharpe | {np.median(sharpes):.2f} |")
print(f"| Median Sharpe luck alone would give | {np.median(luck):.2f} |")
print(f"| Statistics alone: acquitted | {verdicts['ACQUITTED']} of {n} |")
print(f"| Statistics alone: hung jury | {verdicts['HUNG JURY']} of {n} |")
print(f"| Statistics alone: convicted | {verdicts['CONVICTED']} of {n} |")
print(f"| Time-travel test caught it | {caught} of {n} |")
print(f"| Honest strategies wrongly flagged | {false_flags} of {honest_checks} |")
print(f"\n{time.perf_counter() - t0:.0f}s")
