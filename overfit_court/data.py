"""Price data: synthetic markets where the truth is known, or a CSV of real prices.

Synthetic markets are what make calibration possible. With signal=0 they are
pure noise, so any "edge" found in them is overfitting by construction. With
signal>0 a real, persistent edge is planted that momentum-style strategies can
genuinely detect. Either way the noise is realistic: volatility switches
between calm and turbulent regimes, and assets move together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CALM_VOL, TURBULENT_VOL = 0.008, 0.022  # daily volatility in each regime
STAY_CALM, STAY_TURBULENT = 0.99, 0.97  # probability of staying in the regime each day
MARKET_LOADING = 0.5  # every asset's exposure to a common market shock (correlation 0.25)


def synthetic_prices(
    n_assets: int = 12,
    n_days: int = 2520,
    seed: int = 7,
    signal: float = 0.0,
    persistence: float = 0.997,
) -> pd.DataFrame:
    """Daily closes for `n_assets` over `n_days` business days (2520 is ten years).

    signal > 0 gives each asset a slowly drifting expected return: an AR(1)
    process with the given persistence whose long-run daily standard deviation
    is `signal`. The same seed always gives the same market.
    """
    rng = np.random.default_rng(seed)

    # Volatility regimes: a two-state Markov chain, calm (0) or turbulent (1).
    vols = np.array([CALM_VOL, TURBULENT_VOL])
    stay = np.array([STAY_CALM, STAY_TURBULENT])
    state = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        state[t] = state[t - 1] if rng.random() < stay[state[t - 1]] else 1 - state[t - 1]

    # Shocks: a common market factor plus each asset's own noise, scaled by the regime.
    market = rng.normal(0, 1, n_days)
    own = rng.normal(0, 1, (n_days, n_assets))
    shocks = MARKET_LOADING * market[:, None] + np.sqrt(1 - MARKET_LOADING**2) * own
    shocks *= vols[state][:, None]

    # The planted edge (if any): each asset's expected return drifts slowly over time.
    drift = np.zeros_like(shocks)
    if signal > 0:
        innovation_sd = signal * np.sqrt(1 - persistence**2)
        drift[0] = rng.normal(0, signal, n_assets)
        for t in range(1, n_days):
            drift[t] = persistence * drift[t - 1] + rng.normal(0, innovation_sd, n_assets)

    log_returns = drift + shocks
    dates = pd.bdate_range("2014-01-01", periods=n_days)
    names = [f"ASSET_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(100 * np.exp(np.cumsum(log_returns, axis=0)), index=dates, columns=names)


def load_csv(path) -> pd.DataFrame:
    """A wide CSV of daily closes: first column dates, then one column per asset.

    Gaps are forward-filled (a missing close means the price didn't change),
    then any asset with no data at all, and any leading rows with gaps, are dropped.
    """
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return prices.ffill().dropna(how="all", axis=1).dropna()
