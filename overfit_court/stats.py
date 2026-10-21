"""Statistics that decide whether a backtest is real or overfit.

The maths decides the verdict; the language models only explain it.

References
- Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR, min TRL)
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
- Bailey, Borwein, Lopez de Prado & Zhu (2016), "The Probability of Backtest
  Overfitting" (CSCV / PBO)
- White (2000), "A Reality Check for Data Snooping"
- Politis & Romano (1994), "The Stationary Bootstrap"
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329
PERIODS_PER_YEAR = 252


def sharpe(returns, ddof: int = 1) -> float:
    """Per-period (non-annualised) Sharpe ratio."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=ddof)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(r.mean() / sd)


def annualise(sr: float, periods: int = PERIODS_PER_YEAR) -> float:
    return float(sr * math.sqrt(periods))


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int,
                               skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(true SR > benchmark) given sample SR, adjusting for non-normal returns.

    kurt is raw (non-excess) kurtosis, so 3.0 for a normal distribution.
    """
    if n_obs < 2:
        return 0.5
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    denom = max(denom, 1e-12)
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(st.norm.cdf(z))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """Expected maximum Sharpe among n_trials skill-less strategies."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    a = st.norm.ppf(1.0 - 1.0 / n_trials)
    b = st.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


@dataclass
class DSRResult:
    sharpe: float                 # per period
    sharpe_annual: float
    benchmark_sharpe: float       # expected max SR under the null, per period
    benchmark_sharpe_annual: float
    n_trials: int
    n_obs: int
    skew: float
    kurtosis: float
    dsr: float                    # probability the edge survives multiple testing

    def to_dict(self):
        return asdict(self)


def deflated_sharpe_ratio(returns, trial_sharpes) -> DSRResult:
    """Analytic DSR of the selected strategy given the Sharpes of every trial tried."""
    r = np.asarray(returns, dtype=float)
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    sr = sharpe(r)
    var_sr = float(np.var(trials, ddof=1)) if trials.size > 1 else 0.0
    sr0 = expected_max_sharpe(var_sr, int(trials.size))
    skew = float(st.skew(r)) if r.size > 2 else 0.0
    kurt = float(st.kurtosis(r, fisher=False)) if r.size > 3 else 3.0
    dsr = probabilistic_sharpe_ratio(sr, sr0, int(r.size), skew, kurt)
    return DSRResult(sr, annualise(sr), sr0, annualise(sr0), int(trials.size),
                     int(r.size), skew, kurt, dsr)


@dataclass
class PBOResult:
    pbo: float                    # probability of backtest overfitting
    n_splits: int
    n_combinations: int
    logits: list                  # lambda for every IS/OOS split
    is_sharpe_of_selected: list
    oos_sharpe_of_selected: list
    degradation_slope: float      # OOS SR regressed on IS SR of the pick
    prob_oos_loss: float          # P(selected strategy has OOS SR < 0)

    def to_dict(self, include_series: bool = False):
        d = asdict(self)
        if not include_series:
            for k in ("logits", "is_sharpe_of_selected", "oos_sharpe_of_selected"):
                d.pop(k)
        return d


def pbo_cscv(returns_matrix, n_splits: int = 16, min_obs_per_split: int = 20) -> PBOResult:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    returns_matrix: T x N array, one column per configuration tried.
    """
    M = np.asarray(returns_matrix, dtype=float)
    T, N = M.shape
    if N < 2:
        raise ValueError("PBO needs at least two configurations")
    S = min(n_splits, T // min_obs_per_split)
    S -= S % 2
    if S < 4:
        raise ValueError(f"Not enough observations ({T}) for CSCV")

    blocks = np.array_split(np.arange(T), S)
    cnt = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.stack([M[b].sum(axis=0) for b in blocks])        # S x N
    s2 = np.stack([(M[b] ** 2).sum(axis=0) for b in blocks])  # S x N

    def sr_of(idx):
        n = cnt[idx].sum()
        a = s1[idx].sum(axis=0)
        b = s2[idx].sum(axis=0)
        mean = a / n
        var = np.clip((b - n * mean ** 2) / (n - 1), 0.0, None)
        sd = np.sqrt(var)
        return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)

    logits, is_sel, oos_sel = [], [], []
    everything = set(range(S))
    for combo in combinations(range(S), S // 2):
        is_idx = list(combo)
        oos_idx = sorted(everything.difference(combo))
        sr_is, sr_oos = sr_of(is_idx), sr_of(oos_idx)
        pick = int(np.argmax(sr_is))
        rank = st.rankdata(sr_oos)[pick]            # 1 = worst, N = best
        w = rank / (N + 1.0)
        logits.append(float(math.log(w / (1.0 - w))))
        is_sel.append(float(sr_is[pick]))
        oos_sel.append(float(sr_oos[pick]))

    logits_a = np.array(logits)
    is_a, oos_a = np.array(is_sel), np.array(oos_sel)
    slope = float(np.polyfit(is_a, oos_a, 1)[0]) if np.ptp(is_a) > 0 else 0.0
    return PBOResult(
        pbo=float(np.mean(logits_a <= 0)),
        n_splits=S,
        n_combinations=len(logits),
        logits=logits,
        is_sharpe_of_selected=is_sel,
        oos_sharpe_of_selected=oos_sel,
        degradation_slope=slope,
        prob_oos_loss=float(np.mean(oos_a < 0)),
    )


@dataclass
class WalkForwardResult:
    oos_returns: list
    selected_columns: list
    oos_sharpe_annual: float
    n_folds: int

    def to_dict(self, include_series: bool = False):
        d = asdict(self)
        if not include_series:
            d.pop("oos_returns")
        return d


def walk_forward(returns_matrix, n_folds: int = 6) -> WalkForwardResult:
    """Expanding-window selection: pick the best config on the past, trade it next."""
    M = np.asarray(returns_matrix, dtype=float)
    T = M.shape[0]
    bounds = np.linspace(0, T, n_folds + 1).astype(int)
    oos, picks = [], []
    for k in range(1, n_folds):
        train = M[: bounds[k]]
        test = M[bounds[k]: bounds[k + 1]]
        srs = np.array([sharpe(train[:, j]) for j in range(M.shape[1])])
        j = int(np.argmax(srs))
        picks.append(j)
        oos.extend(test[:, j].tolist())
    return WalkForwardResult(oos, picks, annualise(sharpe(oos)), n_folds)


def verdict_from_stats(p_real: float, wf_sharpe_annual: float, prob_oos_loss: float) -> str:
    """Deterministic ruling. The LLM judge explains it but cannot overturn it.

    p_real: probability the edge is real after every configuration tried
            (Reality Check), so fishing lowers it.
    wf_sharpe_annual: Sharpe when the choice is made walk-forward.
    prob_oos_loss: how often the in-sample winner loses money out of sample
            (CSCV). PBO is reported but not used: when every configuration is
            genuinely profitable, which one wins in-sample is random, so PBO
            sits near 50% even though nothing is wrong.
    """
    if p_real < 0.5 or wf_sharpe_annual <= 0 or prob_oos_loss >= 0.5:
        return "CONVICTED"
    if p_real >= 0.95 and wf_sharpe_annual > 0 and prob_oos_loss < 0.2:
        return "ACQUITTED"
    return "HUNG JURY"


def fmt_prob(p: float) -> str:
    """99.7% is not 100%: never round a probability to certainty."""
    if p >= 0.995:
        return "over 99%"
    if p <= 0.005:
        return "under 1%"
    return f"{p:.0%}"


def verdict_reasons(p_real: float, wf_sharpe_annual: float, prob_oos_loss: float) -> list[str]:
    """Which checks failed, in plain words (empty when nothing failed outright)."""
    out = []
    if p_real < 0.5:
        out.append(f"luck explains it (P(real) {p_real:.0%})")
    if wf_sharpe_annual <= 0:
        out.append(f"chosen walk-forward it loses money (Sharpe {wf_sharpe_annual:.2f})")
    if prob_oos_loss >= 0.5:
        out.append(f"the in-sample winner loses out of sample in {prob_oos_loss:.0%} of splits")
    return out


def min_track_record_years(sr: float, sr_benchmark: float, skew: float = 0.0,
                           kurt: float = 3.0, alpha: float = 0.05,
                           periods: int = PERIODS_PER_YEAR) -> float | None:
    """Years of live data needed to be (1 - alpha) confident true SR > benchmark.

    Bailey & Lopez de Prado (2012) minimum track record length. Returns None
    when the observed Sharpe does not beat the benchmark at all (no amount of
    data would confirm it).
    """
    if sr <= sr_benchmark:
        return None
    z = st.norm.ppf(1.0 - alpha)
    var_term = max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12)
    n_obs = 1.0 + var_term * (z / (sr - sr_benchmark)) ** 2
    return float(n_obs / periods)


# ---------------------------------------------------------------- reality check
def stationary_bootstrap_indices(T: int, B: int, mean_block: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: random blocks with geometric lengths."""
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    jump = rng.random((B, T)) < 1.0 / mean_block
    starts = rng.integers(0, T, (B, T))
    for t in range(1, T):
        idx[:, t] = np.where(jump[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


@dataclass
class RealityCheckResult:
    p_value: float               # P(best of all tried looks this good with zero edge)
    p_real: float                # 1 - p_value
    pick_sharpe_annual: float
    null_median_annual: float    # what luck alone typically produces as the best
    null_q95_annual: float
    null_median: float           # per period, for min track record
    n_configs: int
    n_boot: int
    mean_block: float

    def to_dict(self):
        return asdict(self)


def reality_check(returns_matrix, pick_sharpe: float, n_boot: int = 500,
                  mean_block: float = 10.0, seed: int = 0) -> RealityCheckResult:
    """White's (2000) Reality Check with studentised (Sharpe) statistics.

    Every configuration is demeaned so none has any edge, then time blocks
    are resampled to preserve volatility clustering, fat tails and the
    correlation between configurations. The p-value is how often the best
    Sharpe among ALL configurations, in those edgeless worlds, beats the one
    you picked. Duplicated or near-identical configurations barely move the
    null, so re-running the same grid is not penalised; genuinely new bets are.
    """
    M = np.asarray(returns_matrix, dtype=float)
    sd = M.std(axis=0, ddof=1)
    M = M[:, sd > 1e-12]
    T, N = M.shape
    Mc = M - M.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(T, n_boot, mean_block, rng)
    maxes = np.empty(n_boot)
    for b in range(n_boot):
        R = Mc[idx[b]]
        mu = R.mean(axis=0)
        s = R.std(axis=0, ddof=1)
        maxes[b] = np.max(np.divide(mu, s, out=np.zeros_like(mu), where=s > 0))
    p = float((1 + np.sum(maxes >= pick_sharpe)) / (n_boot + 1))
    med = float(np.median(maxes))
    return RealityCheckResult(p, 1.0 - p, annualise(pick_sharpe), annualise(med),
                              annualise(float(np.quantile(maxes, 0.95))), med, N, n_boot,
                              mean_block)
