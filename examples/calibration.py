"""Is the court calibrated? Run it on markets where we KNOW the truth.

signal = 0   pure noise: every 'edge' is overfitting (should never be acquitted)
signal > 0   planted persistent drifts that momentum can genuinely detect

Each trial is judged on the open 75% of the data, then the committed
strategy is tested once on the sealed 25%.

Usage: python examples/calibration.py [n_seeds]
"""
import sys
import time
from collections import Counter

from overfit_court.data import synthetic_prices
from overfit_court.pipeline import run_trial, unseal
from overfit_court.sandbox import InProcessExecutor

levels = {"none (noise)": 0.0, "weak": 0.0004, "strong": 0.0008}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ex = InProcessExecutor()          # library code is trusted; model-written code never runs here
quiet = lambda *a: None
print("| Planted edge | Runs | Convicted | Hung jury | Acquitted | Sealed: confirmed | Sealed: refuted "
      "| Median best Sharpe (open) | Median Sharpe (sealed) |")
print("|---|---|---|---|---|---|---|---|---|")
t0 = time.perf_counter()
for label, sig in levels.items():
    v, sealed_v, naive, sealed_sr = Counter(), Counter(), [], []
    for seed in range(n):
        px = synthetic_prices(signal=sig, seed=100 + seed)
        t = run_trial("buy recent winners, short losers", px, executor=ex, log=quiet)
        s = unseal(t, executor=ex, log=quiet)
        v[t.verdict] += 1
        sealed_v[s.verdict] += 1
        naive.append(t.evidence()["best_sharpe_annual"])
        sealed_sr.append(s.sealed_sharpe_annual)
    med = lambda xs: sorted(xs)[len(xs) // 2]
    print(f"| {label} | {n} | {v['CONVICTED']} | {v['HUNG JURY']} | {v['ACQUITTED']} | "
          f"{sealed_v['CONFIRMED']} | {sealed_v['REFUTED']} | {med(naive):.2f} | {med(sealed_sr):.2f} |")
print(f"\n{3 * n} trials in {time.perf_counter() - t0:.0f}s")
