"""Price data. Synthetic regime-switching markets by default, CSV for real data.

Pure-noise synthetic data is the court's control group: any "edge" found on
it is overfitting by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(n_assets: int = 12, n_days: int = 2520, seed: int = 7,
                     signal: float = 0.0, persistence: float = 0.997) -> pd.DataFrame:
    """Two-state Markov volatility regimes plus an optional planted edge.

    signal > 0 gives each asset a slowly drifting expected return (AR(1) with
    the given persistence, stationary daily sd = signal). Momentum can
    genuinely detect that, which makes it the court's positive control.
    """
    rng = np.random.default_rng(seed)
    vols = np.array([0.008, 0.022])
    stay = np.array([0.99, 0.97])
    state = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        state[t] = state[t - 1] if rng.random() < stay[state[t - 1]] else 1 - state[t - 1]
    common = rng.normal(0, 1, n_days)
    eps = 0.5 * common[:, None] + np.sqrt(0.75) * rng.normal(0, 1, (n_days, n_assets))
    eps *= vols[state][:, None]
    mu = np.zeros_like(eps)
    if signal > 0:
        shock_sd = signal * np.sqrt(1 - persistence ** 2)
        mu[0] = rng.normal(0, signal, n_assets)
        for t in range(1, n_days):
            mu[t] = persistence * mu[t - 1] + rng.normal(0, shock_sd, n_assets)
    r = mu + eps
    idx = pd.bdate_range("2014-01-01", periods=n_days)
    cols = [f"ASSET_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)


def load_csv(path) -> pd.DataFrame:
    """Wide CSV: first column dates, one column of close prices per asset."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df.ffill().dropna(how="all", axis=1).dropna()
