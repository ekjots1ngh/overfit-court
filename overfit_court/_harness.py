"""The backtest harness: the single source of truth for how a strategy is scored.

This file depends only on numpy and pandas, because its source is uploaded
verbatim into Token Factory Sandboxes as runner.py. Local subprocesses run
the same file, and in-process experiments import it, so every executor
applies exactly the same lag, costs and time-travel test.

Two jobs:

backtest     Score every parameter configuration. The harness, not the
             strategy, lags positions by one day and charges turnover costs,
             so model-written code cannot trade on the close it just saw.

time_travel  Prove the strategy is causal. Run it on history truncated at
             several cut dates and check that every position before the cut
             is identical to the full-history run. If a position on a past
             date changes when later data is added, the code read the future.
             This catches leaks a code review misses: full-sample means and
             standard deviations, negative shifts, back-fills, centred windows,
             quantiles over the whole series.
"""
import json

import numpy as np
import pandas as pd

TOLERANCE = 1e-9


def _clean(pos, prices):
    pos = pd.DataFrame(pos).reindex(index=prices.index, columns=prices.columns)
    return pos.astype(float).fillna(0.0).clip(-1.0, 1.0)


def backtest(fn, prices, grid, cost_bps):
    """Return (keys, T x N array of daily returns, errors) for every config in grid."""
    rets = prices.pct_change().fillna(0.0)
    cost = cost_bps / 1e4
    keys, cols, errors = [], [], {}
    for key, params in grid.items():
        try:
            pos = _clean(fn(prices.copy(), **params), prices)
            gross = pos.abs().sum(axis=1).clip(lower=1.0)
            pos = pos.div(gross, axis=0)                 # gross exposure <= 1
            held = pos.shift(1).fillna(0.0)              # harness-enforced lag
            turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
            r = ((held * rets).sum(axis=1) - turnover * cost).to_numpy(dtype=float)
            if not np.all(np.isfinite(r)):
                raise ValueError("non-finite returns")
            keys.append(key)
            cols.append(r)
        except Exception as e:
            errors[key] = f"{type(e).__name__}: {e}"[:500]
    R = np.column_stack(cols) if cols else np.zeros((len(prices), 0))
    return keys, R, errors


def time_travel(fn, prices, params_list, cut_fracs=(0.4, 0.6, 0.8)):
    """Check that positions up to each cut do not depend on data after it."""
    worst = {"leaked": False, "max_diff": 0.0, "params": None, "cut_date": None,
             "first_changed_date": None, "n_changed_days": 0, "checks": 0, "error": None}
    for params in params_list:
        try:
            full = _clean(fn(prices.copy(), **params), prices)
        except Exception as e:
            worst["error"] = f"{type(e).__name__}: {e}"[:300]
            continue
        for frac in cut_fracs:
            k = int(len(prices) * frac)
            if k < 2:
                continue
            part_px = prices.iloc[:k]
            try:
                part = _clean(fn(part_px.copy(), **params), part_px)
            except Exception as e:
                worst["error"] = f"{type(e).__name__}: {e}"[:300]
                continue
            worst["checks"] += 1
            diff = (part - full.iloc[:k]).abs().max(axis=1)
            bad = diff[diff > TOLERANCE]
            if len(bad) and float(diff.max()) > worst["max_diff"]:
                worst.update(leaked=True, max_diff=float(diff.max()), params=params,
                             cut_date=str(part_px.index[-1].date()),
                             first_changed_date=str(bad.index[0].date()),
                             n_changed_days=int(len(bad)))
    return worst


def describe_leak(leak):
    """Plain-English description, also used as the repair message for the engineer."""
    if not leak or not leak.get("leaked"):
        return "No time travel detected."
    return (f"Time travel detected: with parameters {leak['params']}, the position on "
            f"{leak['first_changed_date']} changed (by up to {leak['max_diff']:.3g}) when data "
            f"after {leak['cut_date']} was added, on {leak['n_changed_days']} past days. "
            "The code uses information from the future, for example a mean, standard "
            "deviation or quantile over the whole series, a negative shift, a back-fill or "
            "a centred window. Use only rolling or expanding statistics of past data.")


def _main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("strategy", "strategy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    with open("job.json") as fh:
        job = json.load(fh)
    keys, R, errors = backtest(mod.positions, prices, job.get("grid", {}), job.get("cost_bps", 0))
    meta = {"keys": keys, "errors": errors}
    if job.get("leak_check"):
        meta["leak"] = time_travel(mod.positions, prices, job["leak_check"])
    np.savez_compressed("result.npz", R=R, meta=np.array(json.dumps(meta)))


if __name__ == "__main__":
    _main()
