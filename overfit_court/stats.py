"""The statistics that decide whether a backtest is real or overfit.

The maths decides the verdict; the language models only explain it. Every
function here is pure: no I/O, and no randomness without an explicit seed.
Inputs are daily returns. A "returns matrix" is T x N: one row per day and one
column per strategy configuration that was tried.

Contents, in the order a trial uses them:

1. Sharpe basics             sharpe, annualise
2. Sharpe inference          probabilistic_sharpe_ratio, expected_max_sharpe,
                             deflated_sharpe_ratio, min_track_record_years
3. The Reality Check         reality_check (the court's main test of luck)
4. Overfitting diagnostics   pbo_cscv, walk_forward
5. The ruling                verdict_from_stats, verdict_reasons, fmt_prob

References
- Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier"
  (probabilistic Sharpe ratio, minimum track record length)
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
- Bailey, Borwein, Lopez de Prado & Zhu (2016), "The Probability of Backtest
  Overfitting" (combinatorially symmetric cross-validation)
- White (2000), "A Reality Check for Data Snooping"
- Politis & Romano (1994), "The Stationary Bootstrap"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import stats as st

PERIODS_PER_YEAR = 252
EULER_GAMMA = 0.5772156649015329
_FLAT = 1e-12  # a standard deviation below this means "never trades"

# Verdicts on the open evidence...
CONVICTED, HUNG_JURY, ACQUITTED = "CONVICTED", "HUNG JURY", "ACQUITTED"
# ...and on the sealed evidence.
CONFIRMED, INCONCLUSIVE, REFUTED = "CONFIRMED", "INCONCLUSIVE", "REFUTED"

# The thresholds behind the ruling (see verdict_from_stats).
P_REAL_TO_ACQUIT = 0.95  # Reality Check: at least 95% sure the edge is not luck
P_REAL_TO_CONVICT = 0.50  # below this, luck is the better explanation
OOS_LOSS_TO_CONVICT = 0.50  # the in-sample winner loses out of sample in most splits
OOS_LOSS_TO_ACQUIT = 0.20
P_REAL_TO_CONFIRM = 0.95  # sealed evidence: one pre-registered test at the 5% level


# --- 1. Sharpe basics ---
def sharpe(returns) -> float:
    """Per-period (not annualised) Sharpe ratio; 0 for a series that never moves."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(r.mean() / sd)


def annualise(sr: float, periods: int = PERIODS_PER_YEAR) -> float:
    """Daily Sharpe to annual Sharpe."""
    return float(sr * math.sqrt(periods))


# --- 2. Sharpe inference ---
def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """Probability that the true Sharpe beats `sr_benchmark`, given an observed `sr`.

    Adjusts the standard error of the Sharpe ratio for skewed, fat-tailed
    returns. `kurt` is raw kurtosis (3.0 for a normal distribution).
    """
    if n_obs < 2:
        return 0.5
    variance = max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12)
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(variance)
    return float(st.norm.cdf(z))


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """Expected best Sharpe among `n_trials` independent strategies with no skill.

    Uses the extreme-value approximation from Bailey & Lopez de Prado (2014).
    """
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    a = st.norm.ppf(1.0 - 1.0 / n_trials)
    b = st.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe_ratio(returns, trial_sharpes, n_trials: int | None = None) -> float:
    """Probability the chosen strategy's Sharpe survives the number of trials run.

    The analytic version from Bailey & Lopez de Prado (2014). It assumes the
    trials are independent, so pass `n_trials` (for example the number of
    distinct bets) when they are not. Overfit Court reports it for comparison
    with the literature; the verdict uses the bootstrap Reality Check, which
    handles correlated trials directly.
    """
    r = np.asarray(returns, dtype=float)
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    var_sr = float(np.var(trials, ddof=1)) if trials.size > 1 else 0.0
    benchmark = expected_max_sharpe(var_sr, n_trials if n_trials is not None else trials.size)
    skew = float(st.skew(r)) if r.size > 2 else 0.0
    kurt = float(st.kurtosis(r, fisher=False)) if r.size > 3 else 3.0
    return probabilistic_sharpe_ratio(sharpe(r), benchmark, r.size, skew, kurt)


def min_track_record_years(
    sr: float,
    sr_benchmark: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    confidence: float = 0.95,
    periods: int = PERIODS_PER_YEAR,
) -> float | None:
    """Years of live trading needed to be `confidence` sure the true Sharpe beats the benchmark.

    The minimum track record length of Bailey & Lopez de Prado (2012). Returns
    None when the observed Sharpe does not beat the benchmark at all: then no
    amount of live data would confirm it.
    """
    if sr <= sr_benchmark:
        return None
    z = st.norm.ppf(confidence)
    variance = max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12)
    n_obs = 1.0 + variance * (z / (sr - sr_benchmark)) ** 2
    return float(n_obs / periods)


# --- 3. The Reality Check ---
@dataclass
class RealityCheckResult:
    p_value: float  # P(the best of everything tried looks this good with zero edge)
    p_real: float  # 1 - p_value: the probability the edge is real
    null_median: float  # the best Sharpe luck alone typically produces (per day)
    null_median_annual: float
    null_q95_annual: float  # luck gets this far one time in twenty
    n_configs: int
    n_boot: int


def stationary_bootstrap_indices(T: int, n_boot: int, mean_block: float, rng) -> np.ndarray:
    """Resampled row indices for the stationary bootstrap (Politis & Romano, 1994).

    Each resample stitches together blocks of consecutive days whose lengths are
    geometric with mean `mean_block`, so volatility clustering and short-range
    dependence survive the resampling.
    """
    idx = np.empty((n_boot, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, n_boot)
    start_new_block = rng.random((n_boot, T)) < 1.0 / mean_block
    new_starts = rng.integers(0, T, (n_boot, T))
    for t in range(1, T):
        idx[:, t] = np.where(start_new_block[:, t], new_starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def reality_check(
    returns_matrix, pick_sharpe: float, n_boot: int = 500, mean_block: float = 10.0, seed: int = 0
) -> RealityCheckResult:
    """White's (2000) Reality Check, using Sharpe ratios as the test statistic.

    Question: if none of the strategies tried had any edge at all, how often
    would the best of them look as good as the one picked?

    Every configuration is demeaned (so none has an edge), then whole blocks of
    days are resampled, which keeps fat tails, volatility clustering and the
    correlation between configurations. Because the null uses the maximum over
    all columns, near-identical configurations barely move it: re-running the
    same grid is not penalised, while genuinely new bets raise the bar.

    `pick_sharpe` is the per-day Sharpe of the chosen strategy on the original data.
    """
    M = np.asarray(returns_matrix, dtype=float)
    M = M[:, M.std(axis=0, ddof=1) > _FLAT]
    T, N = M.shape
    edgeless = M - M.mean(axis=0)
    rng = np.random.default_rng(seed)
    resamples = stationary_bootstrap_indices(T, n_boot, mean_block, rng)

    best_by_luck = np.empty(n_boot)
    for b in range(n_boot):
        R = edgeless[resamples[b]]
        mu, sd = R.mean(axis=0), R.std(axis=0, ddof=1)
        best_by_luck[b] = np.max(np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0))

    p_value = float((1 + np.sum(best_by_luck >= pick_sharpe)) / (n_boot + 1))
    median = float(np.median(best_by_luck))
    return RealityCheckResult(
        p_value=p_value,
        p_real=1.0 - p_value,
        null_median=median,
        null_median_annual=annualise(median),
        null_q95_annual=annualise(float(np.quantile(best_by_luck, 0.95))),
        n_configs=N,
        n_boot=n_boot,
    )


# --- 4. Overfitting diagnostics ---
@dataclass
class CSCVResult:
    pbo: float  # probability of backtest overfitting: the in-sample winner ranks below median OOS
    prob_oos_loss: float  # how often the in-sample winner loses money out of sample
    degradation_slope: float  # OOS Sharpe regressed on IS Sharpe of the winner
    n_splits: int
    n_combinations: int


def pbo_cscv(returns_matrix, n_splits: int = 16, min_days_per_block: int = 20) -> CSCVResult:
    """Combinatorially symmetric cross-validation (Bailey et al., 2016).

    Cut the history into `n_splits` blocks. For every way of choosing half the
    blocks as "in sample", pick the configuration with the best in-sample
    Sharpe and see how it does on the other half. That gives two numbers:

    - pbo: how often the winner ranks in the bottom half out of sample, i.e.
      whether the exact parameters chosen were luck;
    - prob_oos_loss: how often the winner actually loses money out of sample.

    The cost grows as C(n_splits, n_splits / 2) x N, so callers use fewer
    splits when they have many configurations.
    """
    M = np.asarray(returns_matrix, dtype=float)
    T, N = M.shape
    if N < 2:
        raise ValueError("CSCV needs at least two configurations")
    S = min(n_splits, T // min_days_per_block)
    S -= S % 2
    if S < 4:
        raise ValueError(f"Not enough days ({T}) for CSCV")

    # Per-block sums let every split's Sharpe be computed without re-scanning the data.
    blocks = np.array_split(np.arange(T), S)
    days = np.array([len(b) for b in blocks], dtype=float)
    sums = np.stack([M[b].sum(axis=0) for b in blocks])  # S x N
    sums_sq = np.stack([(M[b] ** 2).sum(axis=0) for b in blocks])  # S x N

    def sharpe_over(block_ids: list[int]) -> np.ndarray:
        n = days[block_ids].sum()
        mean = sums[block_ids].sum(axis=0) / n
        var = np.clip((sums_sq[block_ids].sum(axis=0) - n * mean**2) / (n - 1), 0.0, None)
        sd = np.sqrt(var)
        return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)

    logits, winner_is, winner_oos = [], [], []
    all_blocks = set(range(S))
    for in_sample in combinations(range(S), S // 2):
        out_of_sample = sorted(all_blocks.difference(in_sample))
        sr_is, sr_oos = sharpe_over(list(in_sample)), sharpe_over(out_of_sample)
        winner = int(np.argmax(sr_is))
        relative_rank = st.rankdata(sr_oos)[winner] / (N + 1.0)  # in (0, 1); 1 = best
        logits.append(math.log(relative_rank / (1.0 - relative_rank)))
        winner_is.append(sr_is[winner])
        winner_oos.append(sr_oos[winner])

    winner_is, winner_oos = np.array(winner_is), np.array(winner_oos)
    slope = float(np.polyfit(winner_is, winner_oos, 1)[0]) if np.ptp(winner_is) > 0 else 0.0
    return CSCVResult(
        pbo=float(np.mean(np.array(logits) <= 0)),
        prob_oos_loss=float(np.mean(winner_oos < 0)),
        degradation_slope=slope,
        n_splits=S,
        n_combinations=len(logits),
    )


@dataclass
class WalkForwardResult:
    oos_sharpe_annual: float  # the Sharpe you would actually have earned
    selected_columns: list  # which configuration was picked for each fold
    n_folds: int


def walk_forward(returns_matrix, n_folds: int = 6) -> WalkForwardResult:
    """Honest selection: pick the best configuration on the past, then trade it next.

    The history is cut into `n_folds` equal pieces. For each piece after the
    first, choose the configuration with the best Sharpe on everything before
    it (an expanding window) and record what it earned on that piece.
    """
    M = np.asarray(returns_matrix, dtype=float)
    bounds = np.linspace(0, M.shape[0], n_folds + 1).astype(int)
    earned, picks = [], []
    for k in range(1, n_folds):
        history = M[: bounds[k]]
        choice = int(np.argmax([sharpe(history[:, j]) for j in range(M.shape[1])]))
        picks.append(choice)
        earned.extend(M[bounds[k] : bounds[k + 1], choice])
    return WalkForwardResult(annualise(sharpe(earned)), picks, n_folds)


# --- 5. The ruling ---
def verdict_from_stats(p_real: float, wf_sharpe_annual: float, prob_oos_loss: float) -> str:
    """The court's ruling. Deterministic: the judge model explains it but cannot change it.

    Convicted if any check clearly fails, acquitted only if all clearly pass:

    - p_real: Reality Check over every configuration tried. Fishing lowers it.
    - wf_sharpe_annual: the Sharpe earned when the choice is made walk-forward.
    - prob_oos_loss: how often the in-sample winner loses money out of sample.

    PBO is deliberately not used. When every configuration is genuinely
    profitable, which one wins in sample is random, so PBO sits near 50% even
    though nothing is wrong (we found this on known-truth markets).
    """
    if p_real < P_REAL_TO_CONVICT or wf_sharpe_annual <= 0 or prob_oos_loss >= OOS_LOSS_TO_CONVICT:
        return CONVICTED
    if p_real >= P_REAL_TO_ACQUIT and prob_oos_loss < OOS_LOSS_TO_ACQUIT:
        return ACQUITTED
    return HUNG_JURY


def verdict_reasons(p_real: float, wf_sharpe_annual: float, prob_oos_loss: float) -> list[str]:
    """The checks that clearly failed, in plain words (empty if none did)."""
    reasons = []
    if p_real < P_REAL_TO_CONVICT:
        reasons.append(f"luck explains it (P(real) {fmt_prob(p_real)})")
    if wf_sharpe_annual <= 0:
        reasons.append(f"chosen walk-forward it loses money (Sharpe {wf_sharpe_annual:.2f})")
    if prob_oos_loss >= OOS_LOSS_TO_CONVICT:
        reasons.append(f"the in-sample winner loses out of sample in {prob_oos_loss:.0%} of splits")
    return reasons


def fmt_prob(p: float) -> str:
    """Format a probability without ever rounding it to certainty (99.7% is not 100%)."""
    if p >= 0.995:
        return "over 99%"
    if p <= 0.005:
        return "under 1%"
    return f"{p:.0%}"
