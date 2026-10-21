"""The backtest harness: the single source of truth for how a strategy is scored.

This file depends only on numpy and pandas because its source is uploaded
verbatim into Token Factory Sandboxes as `runner.py`. Local subprocesses run
the same file, and in-process experiments import it, so every executor applies
exactly the same lag, costs and time-travel test.

A strategy is any module defining

    positions(prices: DataFrame, **params) -> DataFrame of target weights

where row t may only use prices up to and including row t.

Two jobs:

backtest     Score every parameter configuration. The harness, not the
             strategy, lags positions by one day and charges turnover costs,
             so model-written code cannot trade on the close it just saw.

time_travel  Prove the strategy is causal. Run it on history truncated at
             several cut dates and check that every position before the cut is
             identical to the full-history run. If a position on a past date
             changes when later data is added, the code read the future. This
             catches leaks that code review misses: full-sample means, standard
             deviations or quantiles, negative shifts, back-fills, centred
             windows.

Run as a script (inside a sandbox) it reads strategy.py, prices.csv and
job.json from the working directory and writes result.npz.
"""

import importlib.util
import json

import numpy as np
import pandas as pd

TOLERANCE = 1e-9  # weights are in [-1, 1]; anything bigger than this is a real change
CUT_FRACTIONS = (0.4, 0.6, 0.8)  # where history is truncated for the time-travel test


def _clean(weights, prices):
    """Align a strategy's output to the price grid: same dates and assets, finite, in [-1, 1]."""
    weights = pd.DataFrame(weights).reindex(index=prices.index, columns=prices.columns)
    return weights.astype(float).fillna(0.0).clip(-1.0, 1.0)


def backtest(positions_fn, prices, grid, cost_bps):
    """Daily returns for every configuration in `grid` ({key: params}).

    Returns (keys, R, errors): R is a T x len(keys) array of daily returns for
    the configurations that ran; errors maps the others to their exception.
    """
    asset_returns = prices.pct_change().fillna(0.0)
    cost_per_unit_turnover = cost_bps / 1e4
    keys, columns, errors = [], [], {}
    for key, params in grid.items():
        try:
            target = _clean(positions_fn(prices.copy(), **params), prices)
            # Scale down so total gross exposure never exceeds 1 (no hidden leverage).
            gross = target.abs().sum(axis=1).clip(lower=1.0)
            target = target.div(gross, axis=0)
            # Decided at today's close, held from tomorrow: the lag the strategy can't skip.
            held = target.shift(1).fillna(0.0)
            # Day 0 holds nothing (positions are lagged), so its turnover is 0.
            turnover = held.diff().abs().sum(axis=1)
            daily = (held * asset_returns).sum(axis=1) - turnover * cost_per_unit_turnover
            daily = daily.to_numpy(dtype=float)
            if not np.all(np.isfinite(daily)):
                raise ValueError("non-finite returns")
            keys.append(key)
            columns.append(daily)
        except Exception as e:  # one bad configuration must not sink the rest
            errors[key] = f"{type(e).__name__}: {e}"[:500]
    R = np.column_stack(columns) if columns else np.zeros((len(prices), 0))
    return keys, R, errors


def time_travel(positions_fn, prices, params_list, cut_fractions=CUT_FRACTIONS):
    """Check that positions up to each cut date do not depend on data after it.

    Returns a report dict; `leaked` is True if any past position changed. The
    worst case found (largest change) is the one reported.
    """
    report = {
        "leaked": False,
        "max_diff": 0.0,
        "params": None,
        "cut_date": None,
        "first_changed_date": None,
        "n_changed_days": 0,
        "checks": 0,
        "error": None,
    }
    for params in params_list:
        try:
            full = _clean(positions_fn(prices.copy(), **params), prices)
        except Exception as e:
            report["error"] = f"{type(e).__name__}: {e}"[:300]
            continue
        for fraction in cut_fractions:
            k = int(len(prices) * fraction)
            if k < 2:
                continue
            past = prices.iloc[:k]
            try:
                truncated = _clean(positions_fn(past.copy(), **params), past)
            except Exception as e:
                report["error"] = f"{type(e).__name__}: {e}"[:300]
                continue
            report["checks"] += 1
            change = (truncated - full.iloc[:k]).abs().max(axis=1)
            changed = change[change > TOLERANCE]
            if len(changed) and float(change.max()) > report["max_diff"]:
                report.update(
                    leaked=True,
                    max_diff=float(change.max()),
                    params=params,
                    cut_date=str(past.index[-1].date()),
                    first_changed_date=str(changed.index[0].date()),
                    n_changed_days=int(len(changed)),
                )
    return report


def describe_leak(report):
    """Plain-English description of a time-travel report.

    Also used verbatim as the repair request sent to the engineer model, so it
    names the parameters and dates and says what kind of fix is needed.
    """
    if not report or not report.get("leaked"):
        return "No time travel detected."
    return (
        f"Time travel detected: with parameters {report['params']}, the position on "
        f"{report['first_changed_date']} changed (by up to {report['max_diff']:.3g}) when data "
        f"after {report['cut_date']} was added, on {report['n_changed_days']} past days. "
        "The code uses information from the future, for example a mean, standard deviation "
        "or quantile over the whole series, a negative shift, a back-fill or a centred "
        "window. Use only rolling or expanding statistics of past data."
    )


def _main():
    """Sandbox entry point: strategy.py + prices.csv + job.json -> result.npz."""
    spec = importlib.util.spec_from_file_location("strategy", "strategy.py")
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    with open("job.json") as fh:
        job = json.load(fh)

    keys, R, errors = backtest(
        strategy.positions, prices, job.get("grid", {}), job.get("cost_bps", 0)
    )
    meta = {"keys": keys, "errors": errors}
    if job.get("leak_check"):
        meta["leak"] = time_travel(strategy.positions, prices, job["leak_check"])
    np.savez_compressed("result.npz", R=R, meta=np.array(json.dumps(meta)))


if __name__ == "__main__":
    _main()
