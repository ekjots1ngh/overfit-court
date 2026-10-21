"""A trial, from idea to verdict, and the evidence room.

run_trial(idea, prices)
    1. plan      Nemotron Ultra turns the idea into a hypothesis and a parameter grid.
    2. research  Tavily searches for prior art; Nemotron Nano triages it.
    3. build     Nemotron Super writes the strategy. The executor runs every
                 configuration on the OPEN evidence, plus the time-travel test.
                 A crash or a leak goes back to Super for repair.
    4. audit     Static patterns plus a Nemotron Nano review of the final code.
    5. judge     Statistics over the whole court record decide the verdict;
                 Nemotron Ultra explains it.
    6. record    The trial's configurations join the court record.

unseal(trial)
    Commit to the winning configuration (its hash is recorded first), then test
    it once on the SEALED evidence in a fresh sandbox. Research never receives
    the sealed dates, so neither the model nor its code can have seen them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as st

from . import _harness, agents, rmt, stats
from .config import Settings, get_settings
from .data import synthetic_prices
from .ledger import Ledger, record_key, record_seed
from .llm import LLM
from .research import prior_art
from .sandbox import ExecutionResult, get_executor
from .stats import ACQUITTED, CONFIRMED, CONVICTED, HUNG_JURY, INCONCLUSIVE, REFUTED

Log = Callable[[str], None]

SEALED_FRACTION = 0.25  # the most recent quarter of the data is the sealed evidence
MAX_REPAIRS = 2  # how many times the engineer may repair a crash or a leak
_FLAT = 1e-12  # a configuration whose returns vary less than this never trades

# CSCV cost grows as C(splits, splits / 2) x configurations: C(16, 8) = 12,870
# splits is fine for one trial; past a few hundred configurations C(12, 6) = 924.
_CSCV_SPLITS, _CSCV_SPLITS_MANY, _MANY_CONFIGS = 16, 12, 150


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


def time_travel_probes(grid: dict, n: int = 3) -> list[dict]:
    """The configurations the time-travel test runs: first, middle and last of the grid.

    Leaks are usually structural (they appear for every parameter value), but a
    parameter can switch a leaky code path on, so the probes span the grid.
    """
    params = list(grid.values())
    if len(params) <= n:
        return params
    picks = sorted({round(i * (len(params) - 1) / (n - 1)) for i in range(n)})
    return [params[i] for i in picks]


def split_sealed(prices: pd.DataFrame, sealed_fraction: float = SEALED_FRACTION):
    """(open evidence, first sealed date). With no seal, the cut is None."""
    if sealed_fraction <= 0:
        return prices, None
    cut = int(len(prices) * (1 - sealed_fraction))
    return prices.iloc[:cut], prices.index[cut]


def drop_flat(returns: pd.DataFrame) -> pd.DataFrame:
    """Configurations that never trade (zero variance) are not strategies."""
    return returns.loc[:, returns.std(ddof=1) > _FLAT]


@contextmanager
def _timed(timings: dict, name: str):
    start = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - start, 3)


# --- the statistics ---
@dataclass
class Judgement:
    """Everything the statistics say about one pick from a body of tried configurations."""

    best_key: str
    best_params: dict
    best_sharpe: float  # per day, on the open evidence
    reality_check: stats.RealityCheckResult  # against the whole court record
    reality_check_this_trial: stats.RealityCheckResult  # against this trial alone
    spectrum: rmt.Spectrum  # distinct bets in the whole court record
    spectrum_this_trial: rmt.Spectrum
    deflated_sharpe: float  # analytic DSR with distinct bets as trials (for reference)
    cscv: stats.CSCVResult
    walk_forward: stats.WalkForwardResult
    min_track_record_years: float | None
    verdict: str  # the statistics' ruling

    @property
    def reasons(self) -> list[str]:
        """The checks that clearly failed."""
        return stats.verdict_reasons(
            self.reality_check.p_real,
            self.walk_forward.oos_sharpe_annual,
            self.cscv.prob_oos_loss,
        )


def judge_statistics(
    returns: pd.DataFrame, grid: dict, prior: list[np.ndarray], seed: int, n_boot: int = 500
) -> Judgement:
    """Judge the best configuration in `returns`, remembering everything in `prior`.

    The Reality Check asks whether the pick beats the best that luck would
    produce across EVERY configuration ever tried on this data. CSCV and
    walk-forward also run over the whole record: they judge the research
    process ("pick the best of everything tried"), not just the last grid.
    """
    sharpes = returns.apply(stats.sharpe).to_numpy()
    best = str(returns.columns[int(np.argmax(sharpes))])
    pick = float(sharpes.max())
    everything = np.column_stack([*prior, returns.to_numpy()]) if prior else returns.to_numpy()

    reality = stats.reality_check(everything, pick, n_boot=n_boot, seed=seed)
    spectrum = rmt.spectrum(everything)
    if prior:
        reality_this = stats.reality_check(returns.to_numpy(), pick, n_boot=n_boot, seed=seed)
        spectrum_this = rmt.spectrum(returns.to_numpy())
    else:
        reality_this, spectrum_this = reality, spectrum

    every_sharpe = [stats.sharpe(everything[:, j]) for j in range(everything.shape[1])]
    deflated = stats.deflated_sharpe_ratio(
        returns[best], every_sharpe, n_trials=max(2, round(spectrum.distinct_bets))
    )
    splits = _CSCV_SPLITS if everything.shape[1] <= _MANY_CONFIGS else _CSCV_SPLITS_MANY
    cscv = stats.pbo_cscv(everything, n_splits=splits)
    walk = stats.walk_forward(everything)
    r = returns[best].to_numpy()
    years = stats.min_track_record_years(
        pick, reality.null_median, float(st.skew(r)), float(st.kurtosis(r, fisher=False))
    )
    verdict = stats.verdict_from_stats(reality.p_real, walk.oos_sharpe_annual, cscv.prob_oos_loss)
    return Judgement(
        best_key=best,
        best_params=grid[best],
        best_sharpe=pick,
        reality_check=reality,
        reality_check_this_trial=reality_this,
        spectrum=spectrum,
        spectrum_this_trial=spectrum_this,
        deflated_sharpe=deflated,
        cscv=cscv,
        walk_forward=walk,
        min_track_record_years=years,
        verdict=verdict,
    )


def final_verdict(statistical: str, reads_the_future: bool, audit_risk: str) -> str:
    """The statistics' ruling, overridden only ever to be stricter."""
    if reads_the_future:
        return CONVICTED  # a strategy that reads the future has no edge to judge
    if audit_risk == "high" and statistical == ACQUITTED:
        return HUNG_JURY
    return statistical


# --- results ---
@dataclass
class SealedResult:
    commitment: str  # hash of code + parameters + data, recorded before the test
    committed_params: dict
    sealed_from: str
    sealed_days: int
    open_sharpe_annual: float
    sealed_sharpe_annual: float
    p_real: float  # one pre-registered test, corrected for earlier openings
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
    judgement: Judgement
    returns: pd.DataFrame  # daily returns of every configuration, open evidence only
    grid: dict
    errors: dict  # configurations that failed or never traded
    audit: dict
    prior_art: dict
    record: dict  # the court record for this data at the time of the trial
    cut: pd.Timestamp | None  # first sealed date
    sealed_fraction: float
    time_travel: dict | None  # the time-travel report for the final code
    code_history: list  # every attempt: code, problem found, time-travel report
    opinion: dict = field(default_factory=dict)  # the judge's explanation
    cost_table: list = field(default_factory=list)  # every model call, with tokens and cost
    execution: dict = field(default_factory=dict)  # sandbox branches, time and cost
    timings: dict = field(default_factory=dict)
    data_source: str = "user data"
    executor: str = "local"
    prices_full: pd.DataFrame | None = field(default=None, repr=False)
    sealed: SealedResult | None = None

    @property
    def time_traveller(self) -> bool:
        """The final code still reads the future."""
        return bool(self.time_travel and self.time_travel.get("leaked"))

    @property
    def caught_and_repaired(self) -> bool:
        """An earlier draft read the future, and the repair fixed it."""
        earlier_leak = any((a.get("time_travel") or {}).get("leaked") for a in self.code_history)
        return earlier_leak and not self.time_traveller

    def evidence(self) -> dict:
        """The numbers behind the verdict: what the judge model sees and the app shows."""
        j = self.judgement
        if self.time_traveller:
            time_travel = "FAILED: the strategy reads the future"
        elif self.caught_and_repaired:
            time_travel = "passed after repair"
        else:
            time_travel = "passed"
        years = j.min_track_record_years
        return {
            "trial_number_on_this_data": self.record["trial_number"],
            "n_configs": int(self.returns.shape[1]),
            "n_configs_total_on_this_data": j.reality_check.n_configs,
            "distinct_bets_this_trial": round(j.spectrum_this_trial.distinct_bets, 1),
            "distinct_bets_total": round(j.spectrum.distinct_bets, 1),
            "n_days_open": int(self.returns.shape[0]),
            "sealed_from": None if self.cut is None else str(self.cut.date()),
            "best_params": j.best_params,
            "best_sharpe_annual": round(stats.annualise(j.best_sharpe), 3),
            "median_sharpe_annual": round(
                stats.annualise(float(self.returns.apply(stats.sharpe).median())), 3
            ),
            "luck_sharpe_annual": round(j.reality_check.null_median_annual, 3),
            "luck_sharpe_q95_annual": round(j.reality_check.null_q95_annual, 3),
            "luck_sharpe_annual_this_trial_only": round(
                j.reality_check_this_trial.null_median_annual, 3
            ),
            "p_real": round(j.reality_check.p_real, 3),
            "p_real_this_trial_only": round(j.reality_check_this_trial.p_real, 3),
            "deflated_sharpe": round(j.deflated_sharpe, 3),
            "pbo": round(j.cscv.pbo, 3),
            "degradation_slope": round(j.cscv.degradation_slope, 3),
            "prob_oos_loss": round(j.cscv.prob_oos_loss, 3),
            "walk_forward_sharpe": round(j.walk_forward.oos_sharpe_annual, 3),
            "failed_checks": j.reasons,
            "years_of_live_trading_to_prove_it": None if years is None else round(years, 1),
            "lookahead_risk": self.audit["risk"],
            "time_travel": time_travel,
            "time_travel_detail": (
                _harness.describe_leak(self.time_travel) if self.time_traveller else None
            ),
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
                "distinct_bets": self.judgement.spectrum.to_dict(),
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
    settings: Settings | None = None,
    ledger: Ledger | None = None,
    executor=None,
    *,
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
        found = agents.triage_prior_art(llm, spec, prior_art(spec["search_queries"], settings))

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
    reads_the_future = bool(result.leak and result.leak.get("leaked"))

    # 4. audit
    log("Auditing the code for look-ahead bias (Nemotron Nano)...")
    audit = agents.audit(llm, code)
    if reads_the_future:
        audit["risk"] = "high"
        audit["issues"] = [_harness.describe_leak(result.leak), *audit["issues"]]

    # 5. judge
    log("Reality Check over the whole court record, distinct bets, CSCV, walk-forward...")
    with _timed(timings, "stats_s"):
        judgement = judge_statistics(
            result.returns, result.grid, ledger.prior_returns(key), record_seed(key), n_boot
        )
    history = ledger.history(key)
    trial = Trial(
        idea=idea,
        spec=spec,
        code=code,
        verdict=final_verdict(judgement.verdict, reads_the_future, audit["risk"]),
        judgement=judgement,
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
        time_travel=result.leak,
        code_history=attempts,
        execution={**result.summary(), "setup": result.setup},
        timings=timings,
        data_source="synthetic (pure noise)" if prices is None else "user data",
        executor=executor.name,
        prices_full=prices_full,
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
    """Write the strategy, run every configuration plus the time-travel test, repair on failure.

    A crash, or a leak found by the time-travel test, goes back to Nemotron
    Super with the exact error (for a leak: the parameters and dates). After
    `max_repairs` repairs the last attempt stands. Returns (final code, its
    results, every attempt).
    """
    if first_draft is None:
        log("Writing the strategy (Nemotron Super)...")
        code = agents.write_code(llm, spec)
    else:
        code = first_draft
    probes = time_travel_probes(grid)
    withheld = "" if cut is None else f", sealed evidence from {cut.date()} withheld"

    attempts = []
    for attempt in range(1, max_repairs + 2):
        log(
            f"Running {len(grid)} configurations plus the time-travel test in parallel "
            f"({executor.name}{withheld})..."
        )
        result = executor.run(code, open_prices, grid, cost_bps, leak_check=probes)
        flat = result.returns.columns.difference(drop_flat(result.returns).columns)
        result.errors.update(dict.fromkeys(flat, "never trades (zero variance)"))
        result.returns = drop_flat(result.returns)

        problem = None
        if result.returns.shape[1] < max(2, len(grid) // 2):
            problem = next(iter(result.errors.values()), "no configurations produced returns")
        elif result.leak and result.leak.get("leaked"):
            problem = _harness.describe_leak(result.leak)
        attempts.append(
            {"attempt": attempt, "code": code, "problem": problem, "time_travel": result.leak}
        )
        if problem is None or attempt > max_repairs:
            break
        log(f"{problem[:200]} Sending it back to Nemotron Super for repair...")
        code = agents.write_code(llm, spec, error=problem, previous=code)

    if result.returns.shape[1] < 2:
        raise RuntimeError(f"Strategy never ran successfully: {result.errors}")
    return code, result, attempts


# --- the evidence room ---
class TimeTravelError(ValueError):
    """The evidence room refuses a strategy that reads the future."""


def commitment_hash(code: str, params: dict, key: str, cut) -> str:
    """What is committed to before the seal is opened: code, parameters and data."""
    blob = json.dumps(
        {"code": code, "params": params, "data": key, "cut": str(cut)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def unseal_strategy(
    code: str,
    params: dict,
    *,
    open_sharpe: float,
    key: str,
    cut,
    prices: pd.DataFrame,
    executor=None,
    ledger: Ledger | None = None,
    settings: Settings | None = None,
    log: Log = print,
) -> SealedResult:
    """One pre-registered test of one committed strategy on the sealed evidence.

    The strategy runs on the full history in a fresh executor, and only the
    days from `cut` onwards are scored. Every opening of the seal is recorded;
    each earlier opening was another chance to get lucky, so the probability
    is corrected for them (Sidak).
    """
    if cut is None:
        raise ValueError("This run had no sealed evidence.")
    settings = settings or get_settings()
    ledger = ledger if ledger is not None else Ledger()
    executor = executor or get_executor(settings.executor)
    commitment = commitment_hash(code, params, key, cut)
    earlier_openings = len(ledger.seals(key))
    log(
        f"Committed to {params} (commitment {commitment}). Opening the evidence room: a fresh "
        "sandbox with the sealed data and only this one strategy..."
    )

    result = executor.run(code, prices, {"committed": params}, settings.cost_bps)
    if "committed" not in result.returns:
        raise RuntimeError(f"Committed strategy failed on sealed data: {result.errors}")
    daily = result.returns["committed"]
    sealed = daily[daily.index >= cut].to_numpy()

    sr = stats.sharpe(sealed)
    p_one_test = stats.probabilistic_sharpe_ratio(
        sr, 0.0, len(sealed), float(st.skew(sealed)), float(st.kurtosis(sealed, fisher=False))
    )
    p_real = p_one_test ** (earlier_openings + 1)
    if p_real >= stats.P_REAL_TO_CONFIRM:
        verdict = CONFIRMED
    elif sr <= 0:
        verdict = REFUTED
    else:
        verdict = INCONCLUSIVE

    outcome = SealedResult(
        commitment=commitment,
        committed_params=params,
        sealed_from=str(pd.Timestamp(cut).date()),
        sealed_days=len(sealed),
        open_sharpe_annual=round(stats.annualise(open_sharpe), 3),
        sealed_sharpe_annual=round(stats.annualise(sr), 3),
        p_real=round(p_real, 3),
        times_opened_before=earlier_openings,
        verdict=verdict,
        sealed_returns=sealed.tolist(),
        execution=result.summary(),
    )
    ledger.record_seal(
        key,
        commitment,
        {
            "params": params,
            "verdict": verdict,
            "sealed_sharpe_annual": outcome.sealed_sharpe_annual,
        },
    )
    return outcome


def unseal(
    trial: Trial,
    executor=None,
    ledger: Ledger | None = None,
    settings: Settings | None = None,
    log: Log = print,
    *,
    allow_time_travel: bool = False,
) -> SealedResult:
    """Open the evidence room for the trial's winning configuration.

    Refused for a strategy that failed the time-travel test: it would read the
    sealed future inside the evidence room too, so the test would prove nothing.
    (`allow_time_travel` exists only so an experiment can show exactly that.)
    """
    if trial.cut is None:
        raise ValueError("This trial was run without sealed evidence.")
    if trial.time_traveller and not allow_time_travel:
        raise TimeTravelError(
            "The evidence room stays shut: this strategy reads the future, "
            "so it would read the sealed evidence too."
        )
    trial.sealed = unseal_strategy(
        trial.code,
        trial.judgement.best_params,
        open_sharpe=trial.judgement.best_sharpe,
        key=trial.record["dataset"],
        cut=trial.cut,
        prices=trial.prices_full,
        executor=executor,
        ledger=ledger,
        settings=settings,
        log=log,
    )
    return trial.sealed
