"""End-to-end trial, and the evidence room.

run_trial: idea -> spec -> code -> audit -> sandboxed grid on OPEN evidence
           -> Reality Check + distinct bets over the whole court record
           -> CSCV, walk-forward -> verdict -> judgment
unseal:    commit to one configuration (hash), then test it once on the SEALED
           evidence in a fresh sandbox. Research sandboxes never receive the
           sealed dates, so neither the model nor its code can have seen them.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as st

from . import agents, rmt, stats
from .config import Settings, get_settings
from .data import synthetic_prices
from .ledger import Ledger, fingerprint
from .llm import LLM
from .research import prior_art
from .sandbox import get_executor

SEALED_FRAC = 0.25


def expand_grid(param_grid: dict, max_configs: int, seed: int = 0) -> dict:
    names = list(param_grid)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*param_grid.values())]
    if len(combos) > max_configs:
        combos = random.Random(seed).sample(combos, max_configs)
    return {f"c{i:03d}": c for i, c in enumerate(combos)}


def split_sealed(prices: pd.DataFrame, frac: float = SEALED_FRAC):
    """Open evidence (the past) and the cut date where sealed evidence starts."""
    if frac <= 0:
        return prices, None
    cut_i = int(len(prices) * (1 - frac))
    return prices.iloc[:cut_i], prices.index[cut_i]


def drop_flat(M: pd.DataFrame) -> pd.DataFrame:
    """Configurations that never trade (zero variance) are not strategies."""
    return M.loc[:, M.std(ddof=1) > 1e-12]


def record_key(prices: pd.DataFrame, frac: float) -> str:
    return f"{fingerprint(prices)}-s{int(round(frac * 100))}"


def _seed(key: str) -> int:
    return int(key[:8], 16)


@dataclass
class Judgement:
    """Statistics for one pick against a body of tried configurations."""
    best_key: str
    best_params: dict
    best_sharpe: float
    rc: stats.RealityCheckResult
    rc_this: stats.RealityCheckResult
    spec_all: rmt.Spectrum
    spec_this: rmt.Spectrum
    dsr_analytic: stats.DSRResult
    pbo: stats.PBOResult
    wf: stats.WalkForwardResult
    min_trl_years: float | None
    verdict: str


def judge_statistics(M: pd.DataFrame, grid: dict, prior: list[np.ndarray], key: str,
                     audit_risk: str = "low", n_boot: int = 500) -> Judgement:
    srs = M.apply(stats.sharpe).values
    best = str(M.columns[int(np.argmax(srs))])
    pick = float(srs.max())
    M_all = np.column_stack([*prior, M.values]) if prior else M.values
    rc = stats.reality_check(M_all, pick, n_boot=n_boot, seed=_seed(key))
    rc_this = rc if not prior else stats.reality_check(M.values, pick, n_boot=n_boot,
                                                       seed=_seed(key))
    spec_all = rmt.spectrum(M_all)
    spec_this = spec_all if not prior else rmt.spectrum(M.values)
    all_srs = np.array([stats.sharpe(M_all[:, j]) for j in range(M_all.shape[1])])
    dsr = stats.deflated_sharpe_ratio(M[best].values, all_srs)
    n_eff = max(2, int(round(spec_all.distinct_bets)))
    sr0 = stats.expected_max_sharpe(float(np.var(all_srs, ddof=1)), n_eff)
    dsr.benchmark_sharpe, dsr.benchmark_sharpe_annual = sr0, stats.annualise(sr0)
    dsr.n_trials = n_eff
    dsr.dsr = stats.probabilistic_sharpe_ratio(dsr.sharpe, sr0, dsr.n_obs, dsr.skew, dsr.kurtosis)
    pbo = stats.pbo_cscv(M_all, n_splits=16 if M_all.shape[1] <= 150 else 12)
    wf = stats.walk_forward(M_all)
    r = M[best].values
    min_trl = stats.min_track_record_years(pick, rc.null_median, float(st.skew(r)),
                                           float(st.kurtosis(r, fisher=False)))
    verdict = stats.verdict_from_stats(rc.p_real, wf.oos_sharpe_annual, pbo.prob_oos_loss)
    if audit_risk == "high" and verdict == "ACQUITTED":
        verdict = "HUNG JURY"
    return Judgement(best, grid[best], pick, rc, rc_this, spec_all, spec_this, dsr, pbo, wf,
                     min_trl, verdict)


@dataclass
class SealedResult:
    commitment: str
    committed_params: dict
    sealed_from: str
    sealed_days: int
    open_sharpe_annual: float
    sealed_sharpe_annual: float
    p_real: float                  # one pre-registered test, corrected for repeat openings
    times_opened_before: int
    verdict: str                   # CONFIRMED / INCONCLUSIVE / REFUTED
    sealed_returns: list = field(repr=False, default_factory=list)
    execution: dict = field(default_factory=dict)

    def to_dict(self, include_series: bool = False) -> dict:
        d = asdict(self)
        if not include_series:
            d.pop("sealed_returns")
        return d


@dataclass
class Trial:
    idea: str
    spec: dict
    code: str
    audit: dict
    prior_art: dict
    grid: dict
    errors: dict
    returns: pd.DataFrame          # open evidence only
    j: Judgement
    record: dict
    cut: pd.Timestamp | None
    sealed_frac: float
    opinion: dict = field(default_factory=dict)
    cost_table: list = field(default_factory=list)
    exec_summary: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    data_source: str = "synthetic"
    executor: str = "local"
    sealed: SealedResult | None = None
    prices_full: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def verdict(self) -> str:
        return self.j.verdict

    @property
    def best_key(self) -> str:
        return self.j.best_key

    def evidence(self) -> dict:
        j = self.j
        srs = self.returns.apply(stats.sharpe)
        return {
            "trial_number_on_this_data": self.record["trial_number"],
            "n_configs": int(self.returns.shape[1]),
            "n_configs_total_on_this_data": j.rc.n_configs,
            "distinct_bets_this_trial": round(j.spec_this.distinct_bets, 1),
            "distinct_bets_total": round(j.spec_all.distinct_bets, 1),
            "n_days_open": int(self.returns.shape[0]),
            "sealed_from": None if self.cut is None else str(self.cut.date()),
            "best_params": j.best_params,
            "best_sharpe_annual": round(stats.annualise(j.best_sharpe), 3),
            "median_sharpe_annual": round(stats.annualise(float(srs.median())), 3),
            "luck_sharpe_annual": round(j.rc.null_median_annual, 3),
            "luck_sharpe_q95_annual": round(j.rc.null_q95_annual, 3),
            "luck_sharpe_annual_this_trial_only": round(j.rc_this.null_median_annual, 3),
            "p_real": round(j.rc.p_real, 3),
            "p_real_this_trial_only": round(j.rc_this.p_real, 3),
            "dsr_analytic": round(j.dsr_analytic.dsr, 3),
            "pbo": round(j.pbo.pbo, 3),
            "degradation_slope": round(j.pbo.degradation_slope, 3),
            "prob_oos_loss": round(j.pbo.prob_oos_loss, 3),
            "walk_forward_sharpe": round(j.wf.oos_sharpe_annual, 3),
            "years_of_live_trading_to_prove_it": (None if j.min_trl_years is None
                                                  else round(j.min_trl_years, 1)),
            "lookahead_risk": self.audit["risk"],
            "prior_art_known": self.prior_art.get("already_known"),
        }

    def to_json(self) -> str:
        return json.dumps({
            "idea": self.idea, "spec": self.spec, "verdict": self.verdict,
            "opinion": self.opinion, "evidence": self.evidence(), "audit": self.audit,
            "prior_art": self.prior_art, "errors": self.errors, "code": self.code,
            "distinct_bets": self.j.spec_all.to_dict(),
            "sealed_evidence": self.sealed.to_dict() if self.sealed else "still sealed",
            "cost_table": self.cost_table, "execution": self.exec_summary,
            "timings": self.timings, "data_source": self.data_source,
            "executor": self.executor, "court_record": self.record,
        }, indent=2, default=str)

    def to_markdown(self) -> str:
        e = self.evidence()
        total = sum(c["cost_usd"] for c in self.cost_table)
        yrs = e["years_of_live_trading_to_prove_it"]
        lines = [
            f"# {self.verdict}: {self.spec.get('name', 'strategy')}",
            f"_{self.opinion.get('headline', '')}_", "",
            "| Metric | Value |", "|---|---|",
            f"| Trial number on this data | {e['trial_number_on_this_data']} |",
            f"| Configurations (this trial / whole record) | "
            f"{e['n_configs']} / {e['n_configs_total_on_this_data']} |",
            f"| Distinct bets, random matrix theory (this trial / whole record) | "
            f"{e['distinct_bets_this_trial']} / {e['distinct_bets_total']} |",
            f"| Best backtest Sharpe (ann., open evidence) | {e['best_sharpe_annual']:.2f} |",
            f"| Best Sharpe luck alone would give | {e['luck_sharpe_annual']:.2f} |",
            f"| Probability the edge is real (Reality Check) | {e['p_real']:.0%} |",
            f"| Probability of backtest overfitting | {e['pbo']:.0%} |",
            f"| Walk-forward Sharpe (ann.) | {e['walk_forward_sharpe']:.2f} |",
            f"| Years of live trading to prove it | {'never at this Sharpe' if yrs is None else yrs} |",
            f"| Look-ahead risk | {e['lookahead_risk']} |",
            f"| Model spend | ${total:.4f} |",
            f"| Execution ({self.executor}) | {self.exec_summary} |",
        ]
        if self.sealed:
            s = self.sealed
            lines += [f"| Sealed evidence ({s.sealed_days} days from {s.sealed_from}) | "
                      f"{s.verdict}: Sharpe {s.sealed_sharpe_annual:.2f}, "
                      f"P(real) {s.p_real:.0%}, commitment {s.commitment} |"]
        lines += ["", "**Reasoning**", *[f"- {r}" for r in self.opinion.get("reasoning", [])], "",
                  "**What would change the verdict**",
                  *[f"- {r}" for r in self.opinion.get("what_would_change_my_mind", [])]]
        if self.prior_art.get("summary"):
            lines += ["", f"**Prior art:** {self.prior_art['summary']}"]
        return "\n".join(lines)


def run_trial(idea: str, prices: pd.DataFrame | None = None, settings: Settings | None = None,
              ledger: Ledger | None = None, executor=None, sealed_frac: float = SEALED_FRAC,
              max_repairs: int = 2, n_boot: int = 500, log=print) -> Trial:
    s = settings or get_settings()
    llm = LLM(s)
    ledger = ledger if ledger is not None else Ledger()
    t = {}
    data_source = "synthetic (pure noise control)" if prices is None else "user data"
    full = synthetic_prices() if prices is None else prices
    key = record_key(full, sealed_frac)
    open_px, cut = split_sealed(full, sealed_frac)

    t0 = time.perf_counter()
    log("Planning the trial (Nemotron Ultra)...")
    spec = agents.plan(llm, idea)
    grid = expand_grid(spec["param_grid"], s.max_configs)
    t["plan_s"] = time.perf_counter() - t0

    log("Searching for prior art (Tavily) and triaging it (Nemotron Nano)...")
    t0 = time.perf_counter()
    pa = agents.triage_prior_art(llm, spec, prior_art(spec.get("search_queries", []), s))
    t["prior_art_s"] = time.perf_counter() - t0

    log("Writing the strategy (Nemotron Super)...")
    t0 = time.perf_counter()
    code = agents.write_code(llm, spec)
    executor = executor or get_executor(s.executor)
    sealed_note = "" if cut is None else f", sealed evidence from {cut.date()} withheld"
    log(f"Running {len(grid)} configurations in parallel ({executor.name}{sealed_note})...")
    res = None
    for attempt in range(max_repairs + 1):
        res = executor.run(code, open_px, grid, s.cost_bps)
        flat = [c for c in res.returns.columns if c not in drop_flat(res.returns).columns]
        res.errors.update({c: "never trades (zero variance)" for c in flat})
        res.returns = drop_flat(res.returns)
        if res.returns.shape[1] >= max(2, len(grid) // 2):
            break
        err = next(iter(res.errors.values()), "no configurations produced returns")
        log(f"Run failed ({err[:160]}). Repairing (Nemotron Super), attempt {attempt + 1}...")
        code = agents.write_code(llm, spec, error=err, previous=code)
    t["code_and_backtest_s"] = time.perf_counter() - t0
    if res is None or res.returns.shape[1] < 2:
        raise RuntimeError(f"Strategy never ran successfully: {res.errors if res else ''}")

    log("Auditing the code for look-ahead bias (Nemotron Nano)...")
    audit = agents.audit(llm, code)

    log("Reality Check over the whole court record, distinct bets, CSCV, walk-forward...")
    t0 = time.perf_counter()
    prior = ledger.prior_returns(key)
    j = judge_statistics(res.returns, res.grid, prior, key, audit["risk"], n_boot)
    t["stats_s"] = time.perf_counter() - t0

    history = ledger.history(key)
    record = {"dataset": key, "trial_number": len(history) + 1,
              "previous": [{"idea": h["idea"], "configs": h["n"], "verdict": h["verdict"]}
                           for h in history],
              "sealed_openings": len(ledger.seals(key))}
    trial = Trial(idea, spec, code, audit, pa, res.grid, res.errors, res.returns, j, record,
                  cut, sealed_frac, exec_summary=res.summary(), timings=t,
                  data_source=data_source, executor=executor.name, prices_full=full)
    log("Writing the judgment (Nemotron Ultra)...")
    trial.opinion = agents.judge(llm, j.verdict, trial.evidence())
    trial.cost_table = llm.cost_table()
    ledger.record(key, idea, res.returns.values, res.returns.apply(stats.sharpe).values,
                  j.verdict, spec.get("family"))
    return trial


def commitment_hash(code: str, params: dict, key: str, cut) -> str:
    blob = json.dumps({"code": code, "params": params, "data": key, "cut": str(cut)},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def unseal_strategy(code: str, params: dict, open_sharpe: float, key: str, cut, prices_full,
                    executor=None, ledger: Ledger | None = None, settings: Settings | None = None,
                    log=print) -> SealedResult:
    """One pre-registered test of one committed strategy on the sealed evidence."""
    if cut is None:
        raise ValueError("This run had no sealed evidence.")
    s = settings or get_settings()
    ledger = ledger if ledger is not None else Ledger()
    commitment = commitment_hash(code, params, key, cut)
    before = len(ledger.seals(key))
    log(f"Committed to {params} (commitment {commitment}). Opening the evidence room: a fresh "
        f"sandbox with the sealed data and only this one strategy...")
    executor = executor or get_executor(s.executor)
    res = executor.run(code, prices_full, {"committed": params}, s.cost_bps)
    if "committed" not in res.returns:
        raise RuntimeError(f"Committed strategy failed on sealed data: {res.errors}")
    r = res.returns["committed"]
    sealed = r[r.index >= cut].values
    sr = stats.sharpe(sealed)
    psr = stats.probabilistic_sharpe_ratio(sr, 0.0, len(sealed), float(st.skew(sealed)),
                                           float(st.kurtosis(sealed, fisher=False)))
    p_real = psr ** (before + 1)            # Sidak: every earlier opening was another test
    verdict = "CONFIRMED" if p_real >= 0.95 else ("REFUTED" if sr <= 0 else "INCONCLUSIVE")
    out = SealedResult(commitment, params, str(pd.Timestamp(cut).date()), len(sealed),
                       round(stats.annualise(open_sharpe), 3), round(stats.annualise(sr), 3),
                       round(p_real, 3), before, verdict, sealed.tolist(), res.summary())
    ledger.record_seal(key, commitment, {"params": params, "verdict": verdict,
                                         "sealed_sharpe_annual": out.sealed_sharpe_annual})
    return out


def unseal(trial: Trial, executor=None, ledger: Ledger | None = None,
           settings: Settings | None = None, log=print) -> SealedResult:
    if trial.cut is None:
        raise ValueError("This trial was run without sealed evidence.")
    out = unseal_strategy(trial.code, trial.j.best_params, trial.j.best_sharpe,
                          trial.record["dataset"], trial.cut, trial.prices_full,
                          executor, ledger, settings, log)
    trial.sealed = out
    return out
