"""How many genuinely different bets did you make? Random matrix theory answers.

A grid of 63 momentum variants is not 63 independent tries: lookback 60 and
lookback 90 are nearly the same bet. We compare the correlation spectrum of
everything tried against the Marchenko-Pastur law (what pure sampling noise
produces for T observations of N series), remove the correlation noise alone
would create, and count what is left as effective independent bets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Spectrum:
    n_configs: int
    n_obs: int
    q: float                   # N / T
    distinct_bets: float       # noise-corrected effective number of independent trials
    eigenvalues: list
    mp_lower: float
    mp_upper: float
    n_above_mp: int            # eigenvalues beyond the noise edge (common structure)
    top_eigen_share: float     # variance share of the largest eigenvalue

    def to_dict(self, include_eigs: bool = False) -> dict:
        d = asdict(self)
        if not include_eigs:
            d.pop("eigenvalues")
        return d


def _standardise(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    sd = M.std(axis=0, ddof=1)
    keep = sd > 1e-12
    X = M[:, keep]
    return (X - X.mean(axis=0)) / sd[keep]


def spectrum(M) -> Spectrum:
    X = _standardise(M)
    T, N = X.shape
    if N == 0:
        return Spectrum(0, T, 0.0, 0.0, [], 0.0, 0.0, 0, 0.0)
    if N == 1:
        return Spectrum(1, T, 1 / T, 1.0, [1.0], 0.0, 0.0, 0, 1.0)
    C = (X.T @ X) / (T - 1)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    lam = np.clip(lam, 0.0, None)
    q = N / T
    upper = (1 + np.sqrt(q)) ** 2
    lower = (1 - np.sqrt(q)) ** 2 if q < 1 else 0.0
    off = C[~np.eye(N, dtype=bool)]
    rho2 = np.clip(off ** 2 - 1.0 / (T - 1), 0.0, None)   # remove sampling-noise correlation
    n_eff = N / (1.0 + (N - 1) * rho2.mean())
    return Spectrum(N, T, float(q), float(n_eff), lam.tolist(), float(lower), float(upper),
                    int((lam > upper).sum()), float(lam[0] / lam.sum()))


def mp_density(q: float, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Marchenko-Pastur density for unit-variance noise, ratio q = N/T < 1."""
    q = min(q, 0.999)
    lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    x = np.linspace(lo, hi, n_points)
    y = np.sqrt(np.clip((hi - x) * (x - lo), 0, None)) / (2 * np.pi * q * x)
    return x, y
