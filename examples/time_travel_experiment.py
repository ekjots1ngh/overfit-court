"""Can statistics catch a strategy that reads the future? Result 2 in the README.

The "anchor" idea (buy assets below their long-run average) is plausible, and
its natural first draft normalises by the mean and standard deviation of the
WHOLE series, so every position quietly uses prices that haven't happened yet.
On pure-noise markets this asks:

- how often would the court's statistics alone acquit the leaky draft?
- how often does the time-travel test catch it?
- how often does the time-travel test wrongly flag an honest strategy (every
  configuration of every library family, and of the repaired draft, on every
  market)?

    python -m examples.time_travel_experiment [n_markets]
"""

import sys
import time
from collections import Counter

import numpy as np

from overfit_court import _harness, stats
from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.pipeline import drop_flat, expand_grid, judge_statistics, split_sealed
from overfit_court.sandbox import InProcessExecutor


def main(n_markets: int) -> None:
    anchor = FAMILIES["anchor"]
    grid = expand_grid(anchor.param_grid, 200)
    # Every honest family, plus the repaired draft: none of them should ever be flagged.
    honest = [(f, f.code) for f in FAMILIES.values() if f.name != "anchor"]
    honest.append((anchor, anchor.fixed_code))
    verdicts, caught, best_sharpes, luck = Counter(), 0, [], []
    false_alarms, honest_configs, honest_checks = 0, 0, 0
    start = time.perf_counter()

    for i in range(n_markets):
        open_prices, _ = split_sealed(synthetic_prices(seed=500 + i))
        result = InProcessExecutor().run(anchor.code, open_prices, grid, 5.0)
        findings = judge_statistics(drop_flat(result.returns), result.grid, seed=i, n_boot=300)
        verdicts[findings.verdict] += 1
        caught += result.time_travel["status"] == _harness.LEAKED
        best_sharpes.append(stats.annualise(findings.best_sharpe))
        luck.append(findings.reality_check.null_median_annual)

        for family, code in honest:
            family_grid = expand_grid(family.param_grid, 500)
            checked = InProcessExecutor().run(code, open_prices, family_grid, 5.0).time_travel
            false_alarms += checked["status"] != _harness.PASSED
            honest_configs += checked["configs"]
            honest_checks += checked["checks"]

    print(
        f"Leaky 'long-run average' draft on {n_markets} pure-noise markets "
        f"({len(grid)} configs each):\n"
    )
    print("| | Result |")
    print("|---|---|")
    print(f"| Median best backtest Sharpe | {np.median(best_sharpes):.2f} |")
    print(f"| Median Sharpe luck alone would give | {np.median(luck):.2f} |")
    print(f"| Statistics alone: acquitted | {verdicts['ACQUITTED']} of {n_markets} |")
    print(f"| Statistics alone: hung jury | {verdicts['HUNG JURY']} of {n_markets} |")
    print(f"| Statistics alone: convicted | {verdicts['CONVICTED']} of {n_markets} |")
    print(f"| Time-travel test caught it | {caught} of {n_markets} |")
    print(
        f"| Honest strategies wrongly flagged | {false_alarms} of {len(honest) * n_markets} "
        f"({honest_configs} configurations, {honest_checks} checks) |"
    )
    print(f"\n{time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
