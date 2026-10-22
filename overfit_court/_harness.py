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

time travel  Check that it is causal. Re-run it on history cut short at a
             series of points (fourteen on the default data, from the first
             two days to the first 95% of them) and require every position to
             stay the same as data is added. If a position on a past date
             changes when later data arrives, the code read the future. This
             catches leaks that code review misses: full-sample means,
             standard deviations or quantiles, negative shifts, back-fills
             (including a back-filled warm-up period, which only the early
             cuts can see) and centred windows.

For each configuration, the shorter histories run first, shortest first, each
with a freshly loaded strategy, and the full history runs last. So nothing one
of its runs leaves behind, in its own module or anywhere else in the process,
can carry information from a longer history into a shorter one. (A later
configuration in the same job could pick up what an earlier one left. Only code
written to cheat would look, and it can't reach the evidence room's result: the
room tests a single configuration, in fresh sandboxes.) Each history is compared with
the next longer one, which catches the same leaks as comparing every history
with the full run, while keeping only one earlier result in memory. The full
history also runs twice: a strategy that answers differently on the same data
can't be checked this way.

The time-travel test fails closed. Its status is "passed" only when every
check of every configuration ran and found nothing. A strategy that crashes on
shorter history, isn't deterministic, or whose results were lost is
"unverified", and the court treats that like a leak.

Run as a script (inside a sandbox) it reads strategy.py, prices.csv and
job.json from the working directory, deletes them before any strategy code
runs, and writes result.npz. A job can be given its own list of cuts, so the
evidence room can share the checks of its one configuration across sandboxes.
"""

import json
import os

import numpy as np
import pandas as pd

TOLERANCE = 1e-9  # weights are in [-1, 1]; a change bigger than this is a real change
EARLY_CUTS = (2, 3, 5, 10)  # in days: short warm-up periods that a back-fill could leak into
CUT_FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)  # of the history
PASSED, LEAKED, UNVERIFIED = "passed", "leaked", "unverified"  # time-travel statuses


# --- loading a strategy ---
def strategy_loader(source, filename="strategy.py"):
    """A function that returns a freshly loaded `positions` every time it is called.

    The source is compiled once and executed into a new namespace per call, so
    each run starts from a clean module.
    """
    code = compile(source, filename, "exec")

    def load():
        namespace = {"__name__": "strategy"}
        exec(code, namespace)
        return namespace["positions"]

    return load


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
def cut_points(n_days, extra=()):
    """The history lengths (in days) that the time-travel test re-runs a strategy on.

    The early cuts and a fraction of the history for each of CUT_FRACTIONS
    (fourteen lengths on the default data; fewer on a short history, where
    some coincide), plus any `extra` lengths: the evidence room asks for one
    ending on every sealed day.
    """
    lengths = set(EARLY_CUTS) | {int(n_days * f) for f in CUT_FRACTIONS} | {int(k) for k in extra}
    return sorted(k for k in lengths if 2 <= k < n_days)


def new_report():
    """An empty time-travel report. It stays "unverified" until checks have run."""
    return {
        "status": UNVERIFIED,
        "configs": 0,  # configurations tested (set by finish, from "tested")
        "checks": 0,  # comparisons completed, one per cut per configuration
        "expected_checks": 0,
        "leaky_configs": 0,  # configurations that leaked (set by finish, from "leaky")
        "lost_configs": 0,  # configurations whose results the sandbox service lost
        # Keys rather than counts, so a configuration whose cuts were shared between
        # several jobs (the evidence room's) still counts once.
        "tested": [],
        "leaky": [],
        # The worst leak found: how far a past weight moved, with which parameters,
        # when history was extended past which date, first on which date, on how many days.
        "max_diff": 0.0,
        "params": None,
        "cut_date": None,
        "first_changed_date": None,
        "n_changed_days": 0,
        "error": None,  # why a check could not run, if one couldn't
    }


def unverified_report(why, lost_configs=0):
    """The report for a job that produced no results, so verified nothing.

    `lost_configs` counts configurations the sandbox service lost (a network or
    API failure, which is no fault of the code). Otherwise the strategy itself
    failed: it didn't import, crashed the runner, or ran out of time.
    """
    report = new_report()
    report["lost_configs"] = lost_configs
    report["error"] = why
    return report


def _new_tally(n_cuts):
    """One configuration's time-travel results, before they join the report."""
    return {"checks": 0, "expected": n_cuts, "leak": None, "error": None}


def _compare(shorter, longer, tally):
    """Every position in `shorter` must reappear unchanged in the longer history's run."""
    change = (longer.iloc[: len(shorter)] - shorter).abs().max(axis=1)
    changed = change[change > TOLERANCE]
    if not changed.empty and float(changed.max()) > (tally["leak"] or {}).get("max_diff", 0.0):
        tally["leak"] = {
            "max_diff": float(changed.max()),
            "cut_date": _day(shorter.index[-1]),
            "first_changed_date": _day(changed.index[0]),
            "n_changed_days": int(len(changed)),
        }
    tally["checks"] += 1  # counted only once the comparison has run


def run_shorter_histories(positions_on, prices, cuts, tally):
    """Run the strategy on every cut of history, shortest first, each compared with the last.

    `positions_on(history)` returns the weights of a freshly loaded strategy.
    A cut that crashes is recorded, and its check never counts as passed.
    Returns the weights from the longest cut that ran (None if none did), for
    the final comparison with the full history.
    """
    previous = None
    for k in cuts:
        try:
            weights = positions_on(prices.iloc[:k])
            if previous is not None:
                _compare(previous, weights, tally)
        except Exception as e:  # recorded; a check that didn't run can never count as passed
            tally["error"] = tally["error"] or f"on the first {k} days: {_error(e)}"
            continue
        previous = weights
    return previous


def finish_time_travel(positions_on, prices, weights, longest_cut, tally):
    """Run the full history again, and compare the longest cut with the full run.

    Two runs on the same data must agree. If they don't, differences between
    histories prove nothing, so the configuration is unverified, not leaky.
    """
    again = positions_on(prices)
    if float((again - weights).abs().to_numpy().max(initial=0.0)) > TOLERANCE:
        tally["leak"] = None
        tally["error"] = (
            "positions() gave different weights on the same data twice, so it isn't "
            "deterministic (no randomness, and no state kept between calls)"
        )
        return
    if longest_cut is not None:
        _compare(longest_cut, weights, tally)


def add_tally(report, key, params, tally):
    """Add one configuration's time-travel results to the report, keeping the worst leak."""
    report["tested"].append(key)
    report["checks"] += tally["checks"]
    report["expected_checks"] += tally["expected"]
    if tally["error"] and report["error"] is None:
        report["error"] = f"with parameters {params}, {tally['error']}"
    leak = tally["leak"]
    if leak is not None:
        report["leaky"].append(key)
        if leak["max_diff"] > report["max_diff"]:
            report.update(leak, params=params)


def finish(report):
    """Set the report's status from its counts. Anything short of a full pass is not a pass."""
    report["tested"], report["leaky"] = sorted(set(report["tested"])), sorted(set(report["leaky"]))
    report["configs"], report["leaky_configs"] = len(report["tested"]), len(report["leaky"])
    complete = report["checks"] > 0 and report["checks"] == report["expected_checks"]
    if report["leaky_configs"]:
        report["status"] = LEAKED
    elif complete and report["lost_configs"] == 0 and report["error"] is None:
        report["status"] = PASSED
    else:
        report["status"] = UNVERIFIED
    return report


def merge_reports(reports):
    """Combine the reports of several jobs into one, keeping the worst leak."""
    merged = new_report()
    for report in reports:
        for count in ("checks", "expected_checks", "lost_configs"):
            merged[count] += report[count]
        merged["tested"] += report["tested"]
        merged["leaky"] += report["leaky"]
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
        f"The time-travel test could not verify this strategy: {reason}. Every configuration "
        "is re-run on shorter histories, down to the first two days of data, so positions() "
        "must be deterministic, run quickly and work on any length of history: return zero "
        "weights where there is not enough data yet."
    )


# --- one job ---
def evaluate(load_strategy, prices, grid, cost_bps, check_time_travel=True, cuts=None):
    """Backtest every configuration in `grid` ({key: params}) and time-travel test each one.

    `load_strategy()` must return a freshly loaded `positions` (see
    strategy_loader). `cuts` are the history lengths the time-travel test
    re-runs each configuration on (by default, cut_points). Returns (keys, R,
    errors, report): R is a T x len(keys) array of daily returns for the
    configurations that ran; errors maps the others to their exception; report
    is the time-travel report (None if not requested).
    """
    asset_returns = prices.pct_change().fillna(0.0)
    cost_per_unit_turnover = cost_bps / 1e4
    cuts = cut_points(len(prices)) if cuts is None else sorted(int(k) for k in cuts)
    report = new_report() if check_time_travel else None
    keys, columns, errors = [], [], {}
    for key, params in grid.items():

        def positions_on(history, params=params):
            return target_weights(load_strategy()(history.copy(), **params), history)

        tally = _new_tally(len(cuts))
        longest_cut = None
        try:
            if report is not None:  # shorter histories first: see the module docstring
                longest_cut = run_shorter_histories(positions_on, prices, cuts, tally)
            weights = positions_on(prices)
            columns.append(daily_returns(weights, asset_returns, cost_per_unit_turnover))
        except Exception as e:  # one bad configuration must not sink the rest
            errors[key] = _error(e)
            continue
        keys.append(key)
        if report is not None:
            try:
                finish_time_travel(positions_on, prices, weights, longest_cut, tally)
            except Exception as e:  # the repeat run or the last comparison failed: unverifiable
                tally["error"] = tally["error"] or f"on the full history, run again: {_error(e)}"
            add_tally(report, key, params, tally)
    R = np.column_stack(columns) if columns else np.zeros((len(prices), 0))
    return keys, R, errors, None if report is None else finish(report)


def _error(e):
    return f"{type(e).__name__}: {e}"[:500]


def _day(timestamp):
    return str(timestamp.date()) if hasattr(timestamp, "date") else str(timestamp)


def _main():
    """Sandbox entry point: strategy.py + prices.csv + job.json -> result.npz.

    The inputs are deleted once read, before any strategy code runs, so the
    strategy sees only the prices it is handed, never the file they came from.
    """
    with open("strategy.py") as fh:
        source = fh.read()
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    with open("job.json") as fh:
        job = json.load(fh)
    for name in ("strategy.py", "prices.csv", "job.json"):
        os.remove(name)

    keys, R, errors, report = evaluate(
        strategy_loader(source),
        prices,
        job["grid"],
        job["cost_bps"],
        job["time_travel"],
        job["cuts"],
    )
    meta = {"keys": keys, "errors": errors, "time_travel": report}
    np.savez_compressed("result.npz", R=R, meta=np.array(json.dumps(meta)))


if __name__ == "__main__":
    _main()
