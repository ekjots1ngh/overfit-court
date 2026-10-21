"""How many genuinely different bets did you make? Random matrix theory answers.

A grid of 63 momentum variants is not 63 independent tries: lookback 60 and
lookback 90 are nearly the same bet. To count the real number of bets we look
at the correlation matrix of everything tried.

Even unrelated strategies show some correlation in a finite sample. The
Marchenko-Pastur law says how much: for T days and N series of pure noise, the
eigenvalues of the sample correlation matrix fall inside a known band. Anything
above the band is real common structure.

`distinct_bets` removes the correlation that sampling noise alone would create
(an average squared correlation of about 1/(T-1)) and then applies the usual
effective-number formula N / (1 + (N - 1) * mean squared correlation). N
identical strategies count as about 1 bet; N independent ones count as about N.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

_FLAT = 1e-12  # columns with a standard deviation below this never trade


@dataclass
class Spectrum:
    n_configs: int
    n_days: int
    distinct_bets: float  # noise-corrected effective number of independent bets
    eigenvalues: list  # of the correlation matrix, largest first
    mp_lower: float  # Marchenko-Pastur noise band for this N / T
    mp_upper: float
    n_above_noise: int  # eigenvalues above the band: real common structure
    top_eigen_share: float  # share of all variance in the single biggest factor

    def to_dict(self, include_eigenvalues: bool = False) -> dict:
        d = asdict(self)
        if not include_eigenvalues:
            d.pop("eigenvalues")
        return d


def spectrum(returns_matrix) -> Spectrum:
    """Correlation spectrum, Marchenko-Pastur band and distinct-bet count of a T x N matrix."""
    M = np.asarray(returns_matrix, dtype=float)
    M = M[:, M.std(axis=0, ddof=1) > _FLAT]
    T, N = M.shape
    if N == 0:
        return Spectrum(0, T, 0.0, [], 0.0, 0.0, 0, 0.0)
    if N == 1:
        return Spectrum(1, T, 1.0, [1.0], 0.0, 0.0, 0, 1.0)

    X = (M - M.mean(axis=0)) / M.std(axis=0, ddof=1)
    corr = (X.T @ X) / (T - 1)
    eigenvalues = np.clip(np.sort(np.linalg.eigvalsh(corr))[::-1], 0.0, None)

    q = N / T
    mp_upper = (1 + np.sqrt(q)) ** 2
    mp_lower = (1 - np.sqrt(q)) ** 2 if q < 1 else 0.0

    off_diagonal = corr[~np.eye(N, dtype=bool)]
    true_rho_sq = np.clip(off_diagonal**2 - 1.0 / (T - 1), 0.0, None)  # remove sampling noise
    distinct_bets = N / (1.0 + (N - 1) * true_rho_sq.mean())

    return Spectrum(
        n_configs=N,
        n_days=T,
        distinct_bets=float(distinct_bets),
        eigenvalues=eigenvalues.tolist(),
        mp_lower=float(mp_lower),
        mp_upper=float(mp_upper),
        n_above_noise=int((eigenvalues > mp_upper).sum()),
        top_eigen_share=float(eigenvalues[0] / eigenvalues.sum()),
    )
