"""How many genuinely different bets did you make? Random matrix theory answers.

A grid of 63 momentum variants is not 63 independent tries: lookback 60 and
lookback 90 are nearly the same bet. The eigenvalues of the correlation matrix
of everything tried say how many there really are. Their participation ratio,

    distinct bets = (sum of eigenvalues)^2 / (sum of squared eigenvalues)
                  = N / (1 + (N - 1) x mean squared correlation),

counts N identical strategies as 1 bet and N unrelated ones as N.

One correction matters. In a finite sample even unrelated strategies look a
little correlated, which inflates the squared eigenvalues. For independent
returns with constant volatility the inflation is the second moment of the
Marchenko-Pastur law: a mean squared correlation of about 1/T. Strategies on
the same market are not like that. They share its volatility regimes, and
shared volatility makes unrelated strategies look more correlated than 1/T
(about twice as much in this project's synthetic markets). So the noise level
is measured from the data itself: it is the squared correlation the same
strategies would show if every daily return kept its size but got a random
sign, that is, if they shared their volatility and nothing else. Only
correlation beyond that counts as a shared bet.

The chart's noise edge uses the same idea: the largest eigenvalue among 19
sign-randomised copies of the strategies, which luck alone exceeds one time in
twenty. The textbook Marchenko-Pastur band, which assumes constant volatility
and is narrower, is kept for reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .stats import FLAT

NOISE_DRAWS = 19  # sign-randomised copies behind the noise edge (exceeded by luck 1 time in 20)
_DENSE_UP_TO = 1500  # up to this many columns a dense eigensolver is faster than Lanczos


@dataclass
class Spectrum:
    n_configs: int
    n_days: int
    distinct_bets: float  # participation ratio of the spectrum, net of sampling noise
    eigenvalues: list  # of the correlation matrix, largest first
    noise_edge: float  # the largest eigenvalue luck produces, allowing for shared volatility
    mp_lower: float  # textbook Marchenko-Pastur band for this N / T (constant volatility)
    mp_upper: float
    n_above_noise: int  # eigenvalues above the noise edge: real common structure
    top_eigen_share: float  # share of all variance in the single biggest factor

    def to_dict(self, include_eigenvalues: bool = False) -> dict:
        d = asdict(self)
        if not include_eigenvalues:
            d.pop("eigenvalues")
        return d


def _standardise(X: np.ndarray) -> np.ndarray:
    """Columns with mean 0 and sample standard deviation 1."""
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)


def _random_signs(rng, shape) -> np.ndarray:
    """A matrix of independent random +1s and -1s."""
    return rng.integers(0, 2, size=shape, dtype=np.int8) * 2 - 1


def _top_eigenvalue(Z: np.ndarray, rng) -> float:
    """The largest eigenvalue of the correlation matrix of standardised columns Z.

    For many columns, Lanczos iteration on x -> Z'Zx finds it without ever
    forming the N x N matrix.
    """
    T, N = Z.shape
    if N <= _DENSE_UP_TO:
        return float(np.linalg.eigvalsh(Z.T @ Z / (T - 1))[-1])
    operator = LinearOperator((N, N), matvec=lambda x: Z.T @ (Z @ x) / (T - 1), dtype=float)
    start = rng.normal(size=N)
    return float(eigsh(operator, k=1, which="LA", v0=start, return_eigenvectors=False)[0])


def spectrum(returns_matrix, seed: int = 0) -> Spectrum:
    """Correlation spectrum, noise edge and distinct-bet count of a T x N returns matrix."""
    M = np.asarray(returns_matrix, dtype=float)
    M = M[:, M.std(axis=0, ddof=1) > FLAT]
    T, N = M.shape
    if N == 0:
        return Spectrum(0, T, 0.0, [], 0.0, 0.0, 0.0, 0, 0.0)
    if N == 1:
        return Spectrum(1, T, 1.0, [1.0], 0.0, 0.0, 0.0, 0, 1.0)

    Z = _standardise(M)
    corr = Z.T @ Z / (T - 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(corr)[::-1], 0.0, None)
    pairs = ~np.eye(N, dtype=bool)

    # The squared correlation two columns would have by chance if each daily
    # return kept its size but got a random sign: the sum over days of the
    # product of the two columns' shares of their own variance on that day.
    share = Z**2 / (Z**2).sum(axis=0)
    by_chance = (share.T @ share)[pairs].mean()
    shared = max(0.0, (corr[pairs] ** 2).mean() - by_chance)
    distinct_bets = N / (1.0 + (N - 1) * shared)

    rng = np.random.default_rng(seed)
    noise_edge = max(
        _top_eigenvalue(_standardise(Z * _random_signs(rng, Z.shape)), rng)
        for _ in range(NOISE_DRAWS)
    )
    q = N / T
    return Spectrum(
        n_configs=N,
        n_days=T,
        distinct_bets=float(distinct_bets),
        eigenvalues=eigenvalues.tolist(),
        noise_edge=float(noise_edge),
        mp_lower=float((1 - np.sqrt(q)) ** 2) if q < 1 else 0.0,
        mp_upper=float((1 + np.sqrt(q)) ** 2),
        n_above_noise=int((eigenvalues > noise_edge).sum()),
        top_eigen_share=float(eigenvalues[0] / eigenvalues.sum()),
    )
