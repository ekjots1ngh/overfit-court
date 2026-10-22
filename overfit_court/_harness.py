"""The backtest harness: the single source of truth for how a strategy is scored.

This file depends only on numpy and pandas because its source is uploaded
verbatim into Token Factory Sandboxes as `runner.py`. Local subprocesses run
the same file, and in-process experiments import it, so every executor applies
exactly the same lag, costs and time-travel test.

A strategy is any module defining

    positions(prices: DataFrame, **params) -> DataFrame of target weights

where row t may only use prices up to and including row t.

`evaluate` does two things for every configuration (one set of params):

backtest     Score it. The harness, not the strategy, lags positions by one
             day, caps gross exposure at 1 and charges turnover costs, so
             model-written code cannot trade on the close it just saw.

time travel  Check that it is causal. Re-run it on history cut short at eight
             points, from the first 1% of the days to the first 80%, and
             require every position before each cut to match the full run. If
             a position on a past date changes when later data is added, the
             code read the future. This catches leaks that code review misses:
             full-sample means, standard deviations or quantiles, negative
             shifts, back-fills (including a back-filled warm-up period, which
             only the early cuts can see) and centred windows.

The time-travel test fails closed. Its status is "passed" only when every
check of every configuration ran and found nothing. A strategy that crashes on
shorter history is "unverified", and the court treats that like a leak.

Run as a script (inside a sandbox) it reads strategy.py, prices.csv and
job.json from the working directory and writes result.npz.
"""

import importlib.util
import json

import numpy as np
import pandas as pd

TOLERANCE = 1e-9  # weights are in [-1, 1]; a change bigger than this is a real change
CUT_FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)  # where history is cut short
PASSED, LEAKED, UNVERIFIED = "passed", "leaked", "unverified"  # time-travel statuses


# --- scoring ---
def target_weights(raw, prices):
    """The weights a strategy actually trades.

    Aligned to the price grid, finite, each in [-1, 1], and scaled down so that
    gross exposure never exceeds 1 (no hidden leverage). Output of the wrong
    shape raises an error precise enough for the engineer model to fix, rather
    than silently turning into zero weights.
    """
    if isinstance(raw, np.ndarray) and raw.shape == prices.shape:
        raw = pd.DataFrame(raw, index=prices.index, columns=prices.columns)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(
            f"positions() returned a {type(raw).__name__}; it must return a DataFrame with "
            "the same index and columns as prices"
        )
    if raw.index.intersection(prices.index).empty or raw.columns.intersection(prices.columns).empty:
        raise ValueError(
            "positions() returned a DataFrame whose dates or asset names don't match prices; "
            "return one with the same index and columns"
        )
    weights = raw.reindex(index=prices.index, columns=prices.columns).astype(float)
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
    gross = weights.abs().sum(axis=1).clip(lower=1.0)
    return weights.div(gross, axis=0)


def daily_returns(weights, asset_returns, cost_per_unit_turnover):
    """Net daily returns of `weights`, each decided at a close and held from the next day."""
    held = weights.shift(1).fillna(0.0)  # the one-day lag the strategy can't skip
    turnover = held.diff().abs().sum(axis=1)  # day 0 holds nothing, so its turnover is 0
    daily = (held * asset_returns).sum(axis=1) - turnover * cost_per_unit_turnover
    daily = daily.to_numpy(dtype=float)
    if not np.all(np.isfinite(daily)):
        raise ValueError("non-finite returns")
    return daily


# --- the time-travel test ---
def cut_points(n_days, fractions=CUT_FRACTIONS):
    """The history lengths (in days) that the time-travel test re-runs a strategy on."""
    return sorted({int(n_days * f) for f in fractions if 2 <= int(n_days * f) < n_days})


def new_report():
    """An empty time-travel report. It stays "unverified" until checks have run."""
    return {
        "status": UNVERIFIED,
        "configs": 0,  # configurations tested
        "checks": 0,  # (configuration, cut) comparisons completed
        "expected_checks": 0,
        "leaky_configs": 0,
        # The worst leak found: how far a past weight moved, with which parameters,
        # when history was cut at which date, first on which date, on how many days.
        "max_diff": 0.0,
        "params": None,
        "cut_date": None,
        "first_changed_date": None,
        "n_changed_days": 0,
        "error": None,  # why a check could not run, if one couldn't
    }


def time_travel(positions_fn, prices, params, weights, report, cuts):
    """Time-travel test one configuration whose full-history weights are `weights`.

    Updates `report` in place: counts the checks that ran and keeps the worst leak.
    """
    report["configs"] += 1
    report["expected_checks"] += len(cuts)
    leaked = False
    for k in cuts:
        past = prices.iloc[:k]
        try:
            truncated = target_weights(positions_fn(past.copy(), **params), past)
        except Exception as e:  # recorded; a check that didn't run can never count as passed
            if report["error"] is None:
                report["error"] = f"with parameters {params}, on the first {k} days: {_error(e)}"
            continue
        report["checks"] += 1
        change = (truncated - weights.iloc[:k]).abs().max(axis=1)
        changed = change[change > TOLERANCE]
        if changed.empty:
            continue
        leaked = True
        if float(changed.max()) > report["max_diff"]:
            report.update(
                max_diff=float(changed.max()),
                params=params,
                cut_date=_day(past.index[-1]),
                first_changed_date=_day(changed.index[0]),
                n_changed_days=int(len(changed)),
            )
    report["leaky_configs"] += int(leaked)


def finish(report):
    """Set the report's status from its counts. Anything short of a full pass is not a pass."""
    if report["leaky_configs"]:
        report["status"] = LEAKED
    elif report["checks"] > 0 and report["checks"] == report["expected_checks"]:
        report["status"] = PASSED
    else:
        report["status"] = UNVERIFIED
    return report


def merge_reports(reports):
    """Combine the reports of several jobs into one, keeping the worst leak."""
    merged = new_report()
    for report in reports:
        for count in ("configs", "checks", "expected_checks", "leaky_configs"):
            merged[count] += report[count]
        if report["max_diff"] > merged["max_diff"]:
            for detail in (
                "max_diff",
                "params",
                "cut_date",
                "first_changed_date",
                "n_changed_days",
            ):
                merged[detail] = report[detail]
        merged["error"] = merged["error"] or report["error"]
    return finish(merged)


def passed(report):
    """Did the time-travel test run in full and find nothing?"""
    return bool(report) and report.get("status") == PASSED


def describe(report):
    """A time-travel report in plain English.

    Also sent verbatim to the engineer model as the repair request, so it names
    the parameters and dates involved and says what kind of fix is needed.
    """
    if not report:
        return "The time-travel test did not run."
    if report["status"] == PASSED:
        return (
            f"No time travel: all {report['checks']} checks of {report['configs']} "
            "configurations passed."
        )
    if report["status"] == LEAKED:
        others = report["leaky_configs"] - 1
        also = f" ({others} other configurations leak too)" if others > 0 else ""
        return (
            f"Time travel detected: with parameters {report['params']}, the position on "
            f"{report['first_changed_date']} changed (by up to {report['max_diff']:.3g}) when data "
            f"after {report['cut_date']} was added, on {report['n_changed_days']} past days{also}. "
            "The code uses information from the future, for example a mean, standard deviation "
            "or quantile over the whole series, a negative shift, a back-fill or a centred "
            "window. Use only rolling or expanding statistics of past data."
        )
    reason = report["error"] or "no configuration could be tested"
    return (
        f"The time-travel test could not verify this strategy: {reason}. The test re-runs "
        "positions() on shorter histories, down to the first few days of data, so it must "
        "work on any length of history: return zero weights where there is not enough data yet."
    )


# --- one job ---
def evaluate(positions_fn, prices, grid, cost_bps, check_time_travel=True):
    """Backtest every configuration in `grid` ({key: params}) and time-travel test each one.

    Returns (keys, R, errors, report). R is a T x len(keys) array of daily
    returns for the configurations that ran; errors maps the others to their
    exception; report is the time-travel report (None if not requested).
    """
    asset_returns = prices.pct_change().fillna(0.0)
    cost_per_unit_turnover = cost_bps / 1e4
    cuts = cut_points(len(prices))
    report = new_report() if check_time_travel else None
    keys, columns, errors = [], [], {}
    for key, params in grid.items():
        try:
            weights = target_weights(positions_fn(prices.copy(), **params), prices)
            columns.append(daily_returns(weights, asset_returns, cost_per_unit_turnover))
        except Exception as e:  # one bad configuration must not sink the rest
            errors[key] = _error(e)
            continue
        keys.append(key)
        if report is not None:
            time_travel(positions_fn, prices, params, weights, report, cuts)
    R = np.column_stack(columns) if columns else np.zeros((len(prices), 0))
    return keys, R, errors, None if report is None else finish(report)


def _error(e):
    return f"{type(e).__name__}: {e}"[:500]


def _day(timestamp):
    return str(timestamp.date()) if hasattr(timestamp, "date") else str(timestamp)


def _main():
    """Sandbox entry point: strategy.py + prices.csv + job.json -> result.npz."""
    spec = importlib.util.spec_from_file_location("strategy", "strategy.py")
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    with open("job.json") as fh:
        job = json.load(fh)

    keys, R, errors, report = evaluate(
        strategy.positions, prices, job["grid"], job["cost_bps"], job["time_travel"]
    )
    meta = {"keys": keys, "errors": errors, "time_travel": report}
    np.savez_compressed("result.npz", R=R, meta=np.array(json.dumps(meta)))


if __name__ == "__main__":
    _main()
