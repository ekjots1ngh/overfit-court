"""The Fishing Expedition experiment.

A fisherman agent is told to find a strategy with annualised Sharpe >= 1 and
given 40 attempts. Each attempt is a new branch: a strategy family, parameter
values and a choice of asset universe. It stops as soon as it hits the target.
Its best strategy is then judged by:

  - a standard backtest significance test (the claimed strategy alone)
  - an honest-looking check of only the final grid it reports
  - Overfit Court, counting every branch it walked
  - the sealed evidence, which no branch ever saw

Usage: python examples/fishing_experiment.py [n_markets] [--nemotron]
The default fisherman is scripted (random exploration + hill-climbing).
With --nemotron and NEBIUS_API_KEY set, Nemotron 3 Super does the fishing.
"""
import sys
import time

import pandas as pd

from overfit_court.config import get_settings
from overfit_court.data import synthetic_prices
from overfit_court.fishing import NemotronFisherman, ScriptedFisherman, run_expedition
from overfit_court.llm import LLM
from overfit_court.sandbox import InProcessExecutor

args = [a for a in sys.argv[1:] if not a.startswith("--")]
n = int(args[0]) if args else 10
use_llm = "--nemotron" in sys.argv
s = get_settings()
ex = InProcessExecutor()          # library code only; never used for model-written code
rows = []
t0 = time.perf_counter()
for market, sig, base in [("pure noise", 0.0, 200), ("planted edge", 0.0008, 300)]:
    for i in range(n):
        fisher = NemotronFisherman(LLM(s), fallback_seed=i) if use_llm else ScriptedFisherman(i)
        e = run_expedition(synthetic_prices(seed=base + i, signal=sig), target=1.0,
                           max_attempts=40, fisherman=fisher, settings=s, executor=ex,
                           seed=i, log=lambda *a: None)
        rows.append({"market": market, **e.summary()})
df = pd.DataFrame(rows)
df.to_csv("fishing_results.csv", index=False)

pct = lambda x: f"{100 * x.mean():.0f}%"
print(f"Fisherman: {rows[0]['fisherman']}\n")
print("| Market | Expeditions | Median branches | Median distinct bets | Hit Sharpe >= 1 "
      "| Standard test says real | Final-grid check says real | Court acquits | Sealed evidence confirms |")
print("|---|---|---|---|---|---|---|---|---|")
for market, g in df.groupby("market", sort=False):
    print(f"| {market} | {len(g)} | {g.configs.median():.0f} | {g.distinct_bets.median():.0f} | "
          f"{pct(g.hit_target)} | {pct(g.naive_p_real >= 0.95)} | {pct(g.final_grid_p_real >= 0.95)} | "
          f"{pct(g.court_verdict == 'ACQUITTED')} | {pct(g.sealed == 'CONFIRMED')} |")
print(f"\n{len(df)} expeditions in {time.perf_counter() - t0:.0f}s; per-expedition rows in fishing_results.csv")
