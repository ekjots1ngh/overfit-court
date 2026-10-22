"""A trial, from idea to verdict, and the evidence room.

run_trial(idea, prices)
    1. plan      Nemotron Ultra turns the idea into a hypothesis and a parameter grid.
    2. research  Tavily searches for prior art; Nemotron Nano triages it.
    3. build     Nemotron Super writes the strategy. The executor backtests every
                 configuration on the OPEN evidence and time-travel tests each
                 one. A crash, a leak, or code the test can't verify goes back to
                 Super for repair.
    4. audit     Static patterns plus a Nemotron Nano review of the final code.
    5. judge     Statistics over the whole court record decide the verdict;
                 Nemotron Ultra explains it.
    6. record    The trial's configurations join the court record.

unseal(trial)
    Commit to the winning configuration and record the opening first, then test
    it once on the SEALED evidence in a fresh executor run. Research never
    receives the sealed dates, so neither the model nor its code can have seen them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from . import _harness, agents, rmt, stats
from ._harness import LEAKED, PASSED
from .config import Settings, get_settings
from .data import synthetic_prices
from .ledger import Ledger, record_key, record_seed
from .llm import LLM
from .research import prior_art
from .sandbox import ExecutionResult, get_executor
from .stats import ACQUITTED, CONFIRMED, CONVICTED, HUNG_JURY, INCONCLUSIVE, REFUTED

Log = Callable[[str], None]

SEALED_FRACTION = 0.25  # the most recent quarter of the data is the sealed evidence
MAX_REPAIRS = 2  # how many times the engineer may repair a crash or a time-travel failure
IDENTICAL = (
    "Every configuration produced exactly the same returns, so the parameters have no "
    "effect on the strategy. Make each parameter change the positions."
)

# CSCV cost grows as C(splits, splits / 2) x configurations: C(16, 8) = 12,870
# combinations is fine for one trial; past 150 distinct configurations, C(12, 6) = 924.
_CSCV_SPLITS, _CSCV_SPLITS_MANY, _MANY_CONFIGS = 16, 12, 150

# The harness defines how a strategy is scored, so the sealed-evidence commitment covers it.
HARNESS_SHA256 = hashlib.sha256(Path(_harness.__file__).read_bytes()).hexdigest()


# --- small helpers ---
def expand_grid(param_grid: dict, max_configs: int, seed: int = 0) -> dict:
    """Every combination of parameter values, as {"c000": params, ...}.

    Grids larger than `max_configs` are sampled down reproducibly.
    """
    names = list(param_grid)
    combos = [
        dict(zip(names, values, strict=True)) for values in itertools.product(*param_grid.values())
    ]
    if len(combos) > max_configs:
        combos = random.Random(seed).sample(combos, max_configs)
    return {f"c{i:03d}": params for i, params in enumerate(combos)}


def split_sealed(prices: pd.DataFrame, sealed_fraction: float = SEALED_FRACTION):
    """(open evidence, first sealed date). With no seal, the cut is None."""
    if sealed_fraction <= 0:
        return prices, None
    cut = int(len(prices) * (1 - sealed_fraction))
    return prices.iloc[:cut], prices.index[cut]


def drop_flat(returns: pd.DataFrame) -> pd.DataFrame:
    """Configurations that never trade (zero variance) are not strategies."""
    return returns.loc[:, returns.std(ddof=1) > stats.FLAT]


def describe_data(prices: pd.DataFrame) -> str:
    """What the evidence is, in a few words."""
    first, last = prices.index[0].date(), prices.index[-1].date()
    return f"{prices.shape[1]} assets, {prices.shape[0]} days from {first} to {last}"


@contextmanager
def _timed(timings: dict, name: str):
    start = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - start, 3)


# --- the statistics ---
@dataclass
class Findings:
    """What the statistics found about the best configuration among everything tried."""

    best_key: str
    best_params: dict
    best_sharpe: float  # per day, on the open evidence
    n_tried: int  # configurations run on this data so far, repeats included
    reality_check: stats.RealityCheckResult  # over every distinct configuration in the record
    reality_check_this_trial: stats.RealityCheckResult  # over this trial's configurations alone
    spectrum: rmt.Spectrum  # distinct bets in the whole court record
    spectrum_this_trial: rmt.Spectrum
    deflated_sharpe: float  # analytic DSR with distinct bets as trials (for reference)
    cscv: stats.CSCVResult
    walk_forward: stats.WalkForwardResult
    years_to_prove_live: float | None  # track record needed at the walk-forward Sharpe
    verdict: str  # the statistics' ruling

    @property
    def reasons(self) -> list[str]:
        """The checks that clearly failed."""
        return stats.verdict_reasons(
            self.reality_check.p_value,
            self.walk_forward.oos_sharpe_annual,
            self.cscv.prob_oos_loss,
        )


def judge_statistics(
    returns: pd.DataFrame,
    grid: dict,
    *,
    prior: Sequence[np.ndarray] = (),
    seed: int,
    n_boot: int = 500,
) -> Findings:
    """Judge the best configuration in `returns`, remembering everything in `prior`.

    The Reality Check asks whether the pick beats the best that luck would
    produce across EVERY configuration ever tried on this data. CSCV and
    walk-forward also run over the whole record: they judge the research
    process ("pick the best of everything tried"), not just the last grid.
    Exact repeats of a configuration are counted once: running the same grid
    twice is not more research.
    """
    sharpes = returns.apply(stats.sharpe)
    best = str(sharpes.idxmax())
    pick = float(sharpes.max())
    this_trial = stats.distinct_columns(returns.to_numpy())
    everything = np.column_stack([*prior, returns.to_numpy()]) if prior else returns.to_numpy()
    n_tried = everything.shape[1]
    everything = stats.distinct_columns(everything)

    reality = stats.reality_check(everything, pick, n_boot=n_boot, seed=seed)
    spectrum = rmt.spectrum(everything)
    if prior:
        reality_this = stats.reality_check(this_trial, pick, n_boot=n_boot, seed=seed)
        spectrum_this = rmt.spectrum(this_trial)
    else:
        reality_this, spectrum_this = reality, spectrum

    every_sharpe = [stats.sharpe(everything[:, j]) for j in range(everything.shape[1])]
    deflated = stats.deflated_sharpe_ratio(
        returns[best], every_sharpe, n_trials=max(2, round(spectrum.distinct_bets))
    )
    splits = _CSCV_SPLITS if everything.shape[1] <= _MANY_CONFIGS else _CSCV_SPLITS_MANY
    cscv = stats.pbo_cscv(everything, n_splits=splits)
    walk = stats.walk_forward(everything)
    # A committed strategy trading live faces no selection. So: at the Sharpe that
    # walk-forward selection actually earned, how long a live record would it take
    # to show the true Sharpe is above zero?
    earned = walk.returns
    years = stats.min_track_record_years(
        stats.sharpe(earned), 0.0, float(st.skew(earned)), float(st.kurtosis(earned, fisher=False))
    )
    verdict = stats.verdict_from_stats(reality.p_value, walk.oos_sharpe_annual, cscv.prob_oos_loss)
    return Findings(
        best_key=best,
        best_params=grid[best],
        best_sharpe=pick,
        n_tried=n_tried,
        reality_check=reality,
        reality_check_this_trial=reality_this,
        spectrum=spectrum,
        spectrum_this_trial=spectrum_this,
        deflated_sharpe=deflated,
        cscv=cscv,
        walk_forward=walk,
        years_to_prove_live=years,
        verdict=verdict,
    )


def final_verdict(statistical: str, time_travel_status: str | None, audit_risk: str) -> str:
    """The statistics' ruling, overridden only ever to be stricter.

    A strategy that reads the future, or that the time-travel test could not
    verify, is convicted whatever its statistics: the burden of proof is on it.
    """
    if time_travel_status != PASSED:
        return CONVICTED
    if audit_risk == "high" and statistical == ACQUITTED:
        return HUNG_JURY
    return statistical


# --- results ---
@dataclass
class SealedResult:
    commitment: str  # SHA-256 of everything committed to, recorded before the test
    committed_params: dict
    cost_bps: float
    sealed_from: str
    sealed_days: int
    open_sharpe_annual: float
    sealed_sharpe_annual: float
    p_value: float  # one pre-registered test of "Sharpe above zero", corrected for earlier openings
    times_opened_before: int
    verdict: str  # CONFIRMED, INCONCLUSIVE or REFUTED
    sealed_returns: list = field(repr=False, default_factory=list)
    execution: dict = field(default_factory=dict)

    def to_dict(self, include_returns: bool = False) -> dict:
        d = asdict(self)
        if not include_returns:
            d.pop("sealed_returns")
        return d


@dataclass
class Trial:
    idea: str
    spec: dict
    code: str  # the final version, after any repairs
    verdict: str  # CONVICTED, HUNG JURY or ACQUITTED
    findings: Findings
    returns: pd.DataFrame  # daily returns of every configuration, open evidence only
    grid: dict
    errors: dict  # configurations that failed or never traded
    audit: dict
    prior_art: dict
    record: dict  # this dataset's court record at the time of the trial
    cut: pd.Timestamp | None  # first sealed date
    sealed_fraction: float
    cost_bps: float  # trading cost of the trial; the evidence room uses the same
    time_travel: dict | None  # the time-travel report for the final code
    code_history: list  # every attempt: code, problem found, time-travel report
    ledger: Ledger = field(repr=False)  # the court record, where seal openings are noted too
    prices_full: pd.DataFrame = field(repr=False)  # open and sealed evidence
    opinion: dict = field(default_factory=dict)  # the judge's explanation
    cost_table: list = field(default_factory=list)  # every model call, with tokens and cost
    execution: dict = field(default_factory=dict)  # sandbox branches, time and cost
    timings: dict = field(default_factory=dict)
    data_source: str = ""
    executor: str = "local"
    sealed: SealedResult | None = None

    @property
    def time_travel_status(self) -> str:
        """passed, leaked or unverified."""
        return (self.time_travel or {}).get("status", _harness.UNVERIFIED)

    @property
    def passed_time_travel(self) -> bool:
        """Every configuration of the final code was checked, and none reads the future."""
        return self.time_travel_status == PASSED

    @property
    def time_traveller(self) -> bool:
        """The final code still reads the future."""
        return self.time_travel_status == LEAKED

    @property
    def caught_and_repaired(self) -> bool:
        """An earlier draft read the future, and the repaired code passes the test."""
        earlier = self.code_history[:-1]
        leaked_before = any((a["time_travel"] or {}).get("status") == LEAKED for a in earlier)
        return leaked_before and self.time_travel_status == PASSED

    def evidence(self) -> dict:
        """The numbers behind the verdict: what the judge model sees and the app shows."""
        f = self.findings
        status = self.time_travel_status
        if status == PASSED:
            time_travel = "passed after repair" if self.caught_and_repaired else "passed"
            failed_checks = f.reasons
        elif status == LEAKED:
            time_travel = "FAILED: the strategy reads the future"
            failed_checks = ["it reads the future"]
        else:
            time_travel = "FAILED: could not be verified"
            failed_checks = ["the time-travel test could not verify it"]
        years = f.years_to_prove_live
        return {
            "trial_number_on_this_data": self.record["trial_number"],
            "n_configs": int(self.returns.shape[1]),
            "n_configs_tried_on_this_data": f.n_tried,
            "n_distinct_configs_on_this_data": f.reality_check.n_configs,
            "distinct_bets_this_trial": round(f.spectrum_this_trial.distinct_bets, 1),
            "distinct_bets_total": round(f.spectrum.distinct_bets, 1),
            "n_days_open": int(self.returns.shape[0]),
            "sealed_from": None if self.cut is None else str(self.cut.date()),
            "best_params": f.best_params,
            "best_sharpe_annual": round(stats.annualise(f.best_sharpe), 3),
            "median_sharpe_annual": round(
                stats.annualise(float(self.returns.apply(stats.sharpe).median())), 3
            ),
            "luck_sharpe_annual": round(f.reality_check.null_median_annual, 3),
            "luck_sharpe_q95_annual": round(f.reality_check.null_q95_annual, 3),
            "luck_sharpe_annual_this_trial_only": round(
                f.reality_check_this_trial.null_median_annual, 3
            ),
            "p_value": round(f.reality_check.p_value, 4),
            "p_value_this_trial_only": round(f.reality_check_this_trial.p_value, 4),
            "deflated_sharpe": round(f.deflated_sharpe, 3),
            "pbo": round(f.cscv.pbo, 3),
            "degradation_slope": round(f.cscv.degradation_slope, 3),
            "prob_oos_loss": round(f.cscv.prob_oos_loss, 3),
            "walk_forward_sharpe": round(f.walk_forward.oos_sharpe_annual, 3),
            "years_of_live_trading_to_prove_it": None if years is None else round(years, 1),
            "failed_checks": failed_checks,
            "lookahead_risk": self.audit["risk"],
            "time_travel_status": status,
            "time_travel": time_travel,
            "time_travel_detail": None if status == PASSED else _harness.describe(self.time_travel),
            "repairs": max(0, len(self.code_history) - 1),
            "prior_art_known": self.prior_art.get("already_known"),
        }

    def to_json(self) -> str:
        """The full report: everything needed to reproduce or audit the verdict."""
        return json.dumps(
            {
                "idea": self.idea,
                "spec": self.spec,
                "verdict": self.verdict,
                "opinion": self.opinion,
                "evidence": self.evidence(),
                "audit": self.audit,
                "prior_art": self.prior_art,
                "errors": self.errors,
                "code": self.code,
                "cost_bps": self.cost_bps,
                "distinct_bets": self.findings.spectrum.to_dict(),
                "time_travel": self.time_travel,
                "code_history": self.code_history,
                "sealed_evidence": self.sealed.to_dict() if self.sealed else "still sealed",
                "cost_table": self.cost_table,
                "execution": self.execution,
                "timings": self.timings,
                "data_source": self.data_source,
                "executor": self.executor,
                "court_record": self.record,
            },
            indent=2,
            default=str,
        )


# --- the trial ---
def run_trial(
    idea: str,
    prices: pd.DataFrame | None = None,
    *,
    settings: Settings | None = None,
    ledger: Ledger | None = None,
    executor=None,
    sealed_fraction: float = SEALED_FRACTION,
    max_repairs: int = MAX_REPAIRS,
    n_boot: int = 500,
    log: Log = print,
    llm: LLM | None = None,
    spec: dict | None = None,
    first_draft: str | None = None,
) -> Trial:
    """Put a trading idea on trial.

    `prices` defaults to a pure-noise synthetic market. `spec` and
    `first_draft` let a caller supply the plan or the first version of the
    code (a researcher's own draft, or the demo's leaky one); the court still
    time-travel tests, repairs and judges it in exactly the same way.
    """
    settings = settings or get_settings()
    llm = llm or LLM(settings)
    ledger = ledger if ledger is not None else Ledger()
    executor = executor or get_executor(settings.executor)
    prices_full = prices if prices is not None else synthetic_prices()
    key = record_key(prices_full, sealed_fraction)
    open_prices, cut = split_sealed(prices_full, sealed_fraction)
    timings: dict = {}

    # 1. plan
    with _timed(timings, "plan_s"):
        if spec is None:
            log("Planning the trial (Nemotron Ultra)...")
            spec = agents.plan(llm, idea)
        else:
            spec = agents.validate_spec(spec)
        grid = expand_grid(spec["param_grid"], settings.max_configs)

    # 2. research
    log("Searching for prior art (Tavily) and triaging it (Nemotron Nano)...")
    with _timed(timings, "prior_art_s"):
        hits, search_note = prior_art(spec["search_queries"], settings)
        found = agents.triage_prior_art(llm, spec, hits, search_note)

    # 3. build: write, run, time-travel test, repair
    with _timed(timings, "build_s"):
        code, result, attempts = _build(
            llm=llm,
            spec=spec,
            grid=grid,
            first_draft=first_draft,
            executor=executor,
            open_prices=open_prices,
            cut=cut,
            cost_bps=settings.cost_bps,
            max_repairs=max_repairs,
            log=log,
        )
    time_travel_status = (result.time_travel or {}).get("status")

    # 4. audit
    log("Auditing the code for look-ahead bias (Nemotron Nano)...")
    audit = agents.audit(llm, code)
    if time_travel_status != PASSED:
        audit["risk"] = "high"
        audit["issues"] = [_harness.describe(result.time_travel), *audit["issues"]]

    # 5. judge
    log("Reality Check over the whole court record, distinct bets, CSCV, walk-forward...")
    with _timed(timings, "stats_s"):
        findings = judge_statistics(
            result.returns,
            result.grid,
            prior=ledger.prior_returns(key),
            seed=record_seed(key),
            n_boot=n_boot,
        )
    history = ledger.history(key)
    data_source = describe_data(prices_full)
    if prices is None:
        data_source = f"synthetic pure noise, {data_source}"
    trial = Trial(
        idea=idea,
        spec=spec,
        code=code,
        verdict=final_verdict(findings.verdict, time_travel_status, audit["risk"]),
        findings=findings,
        returns=result.returns,
        grid=result.grid,
        errors=result.errors,
        audit=audit,
        prior_art=found,
        record={
            "dataset": key,
            "trial_number": len(history) + 1,
            "previous": [
                {"idea": h["idea"], "configs": h["n"], "verdict": h["verdict"]} for h in history
            ],
            "sealed_openings": len(ledger.seals(key)),
        },
        cut=cut,
        sealed_fraction=sealed_fraction,
        cost_bps=settings.cost_bps,
        time_travel=result.time_travel,
        code_history=attempts,
        ledger=ledger,
        prices_full=prices_full,
        execution={**result.summary(), "setup": result.setup},
        timings=timings,
        data_source=data_source,
        executor=executor.name,
    )
    log("Writing the judgment (Nemotron Ultra)...")
    trial.opinion = agents.judge(llm, trial.verdict, trial.evidence())
    trial.cost_table = llm.cost_table()

    # 6. record
    ledger.record(key, idea, result.returns.to_numpy(), trial.verdict, spec.get("family"))
    return trial


def _build(
    *, llm, spec, grid, first_draft, executor, open_prices, cut, cost_bps, max_repairs, log
) -> tuple[str, ExecutionResult, list[dict]]:
    """Write the strategy, backtest and time-travel test every configuration, repair on failure.

    A crash, parameters that change nothing, or a time-travel test that finds a
    leak or cannot run, goes back to Nemotron Super with the exact problem (for
    a leak: the parameters and dates). After `max_repairs` repairs the last
    attempt stands. Returns (final code, its results, every attempt).
    """
    if first_draft is None:
        log("Writing the strategy (Nemotron Super)...")
        code = agents.write_code(llm, spec)
    else:
        code = first_draft
    withheld = "" if cut is None else f", sealed evidence from {cut.date()} withheld"

    attempts = []
    for attempt in range(1, max_repairs + 2):
        log(
            f"Running {len(grid)} configurations, each with the time-travel test, in parallel "
            f"({executor.name}{withheld})..."
        )
        result = executor.run(code, open_prices, grid, cost_bps)
        flat = result.returns.columns.difference(drop_flat(result.returns).columns)
        result.errors.update(dict.fromkeys(flat, "never trades (zero variance)"))
        result.returns = drop_flat(result.returns)

        problem = None
        if result.returns.shape[1] < max(2, len(grid) // 2):
            problem = next(iter(result.errors.values()), "no configurations produced returns")
        elif stats.distinct_columns(result.returns.to_numpy()).shape[1] < 2:
            problem = IDENTICAL
        elif not _harness.passed(result.time_travel):
            problem = _harness.describe(result.time_travel)
        attempts.append(
            {
                "attempt": attempt,
                "code": code,
                "problem": problem,
                "time_travel": result.time_travel,
            }
        )
        if problem is None or attempt > max_repairs:
            break
        log(f"{problem[:200]} Sending it back to Nemotron Super for repair...")
        code = agents.write_code(llm, spec, error=problem, previous=code)

    if result.returns.shape[1] < 2:
        raise RuntimeError(
            f"Fewer than two configurations ever produced returns. Errors: {result.errors}"
        )
    if stats.distinct_columns(result.returns.to_numpy()).shape[1] < 2:
        raise RuntimeError(IDENTICAL)
    return code, result, attempts


# --- the evidence room ---
class TimeTravelError(ValueError):
    """The evidence room refuses, or voids, a strategy that reads the future."""


def commitment_hash(code: str, params: dict, *, key: str, cut, cost_bps: float) -> str:
    """SHA-256 of everything the sealed test depends on, fixed before the seal is opened.

    That is the strategy's code and parameters, the data (its fingerprint and
    the cut date), the trading cost, and the harness that scores it.
    """
    blob = json.dumps(
        {
            "code": code,
            "params": params,
            "data": key,
            "cut": str(cut),
            "cost_bps": cost_bps,
            "harness": HARNESS_SHA256,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def unseal_strategy(
    code: str,
    params: dict,
    *,
    open_sharpe: float,
    key: str,
    cut,
    cost_bps: float,
    prices: pd.DataFrame,
    executor,
    ledger: Ledger,
    log: Log = print,
) -> SealedResult:
    """One pre-registered test of one committed strategy on the sealed evidence.

    The opening is recorded with its commitment before anything runs, so it
    counts whatever happens next. The strategy then runs on the full history,
    time-travel test included, and only the days from `cut` onwards are scored.
    Each earlier opening was another chance to get lucky, so the p-value is
    corrected for them (Sidak).
    """
    commitment = commitment_hash(code, params, key=key, cut=cut, cost_bps=cost_bps)
    earlier_openings = len(ledger.seals(key))
    ledger.record_seal_opening(key, commitment, params)
    log(
        f"Committed to {params} (commitment {commitment[:16]}). Opening the evidence room: a "
        "fresh run with the sealed data and only this one strategy..."
    )

    result = executor.run(code, prices, {"committed": params}, cost_bps)
    if "committed" not in result.returns:
        ledger.record_seal_result(key, commitment, {"verdict": "failed to run"})
        raise RuntimeError(f"The committed strategy failed on the full data: {result.errors}")
    if not _harness.passed(result.time_travel):
        ledger.record_seal_result(key, commitment, {"verdict": "void: failed the time-travel test"})
        raise TimeTravelError(
            "The committed strategy failed the time-travel test on the full history, so its "
            "sealed result is void (the opening still counts). "
            + _harness.describe(result.time_travel)
        )
    daily = result.returns["committed"]
    sealed = daily[daily.index >= cut].to_numpy()

    sr = stats.sharpe(sealed)
    confidence = stats.probabilistic_sharpe_ratio(
        sr, 0.0, len(sealed), float(st.skew(sealed)), float(st.kurtosis(sealed, fisher=False))
    )
    # Sidak over every opening so far: with p = 1 - confidence for this one test,
    # the chance that at least one of k openings looks this good is 1 - (1 - p)^k.
    p_value = 1.0 - confidence ** (earlier_openings + 1)
    if p_value <= stats.P_VALUE_TO_CONFIRM:
        verdict = CONFIRMED
    elif sr <= 0:
        verdict = REFUTED
    else:
        verdict = INCONCLUSIVE

    outcome = SealedResult(
        commitment=commitment,
        committed_params=params,
        cost_bps=cost_bps,
        sealed_from=str(pd.Timestamp(cut).date()),
        sealed_days=len(sealed),
        open_sharpe_annual=round(stats.annualise(open_sharpe), 3),
        sealed_sharpe_annual=round(stats.annualise(sr), 3),
        p_value=round(p_value, 4),
        times_opened_before=earlier_openings,
        verdict=verdict,
        sealed_returns=sealed.tolist(),
        execution=result.summary(),
    )
    ledger.record_seal_result(
        key, commitment, {"verdict": verdict, "sealed_sharpe_annual": outcome.sealed_sharpe_annual}
    )
    return outcome


def unseal(trial: Trial, *, executor=None, log: Log = print) -> SealedResult:
    """Open the evidence room for the trial's winning configuration.

    Refused unless the strategy passed the time-travel test: one that reads
    the future would read the sealed evidence too, so the test would prove
    nothing. The opening is recorded in the trial's own court record.
    """
    if trial.cut is None:
        raise ValueError("This trial was run without sealed evidence.")
    if not trial.passed_time_travel:
        if trial.time_traveller:
            why = "this strategy reads the future"
        else:
            why = "the time-travel test could not verify this strategy"
        raise TimeTravelError(
            f"The evidence room stays shut: {why}, so it could read the sealed evidence too."
        )
    trial.sealed = unseal_strategy(
        trial.code,
        trial.findings.best_params,
        open_sharpe=trial.findings.best_sharpe,
        key=trial.record["dataset"],
        cut=trial.cut,
        cost_bps=trial.cost_bps,
        prices=trial.prices_full,
        executor=executor or get_executor(get_settings().executor),
        ledger=trial.ledger,
        log=log,
    )
    return trial.sealed
