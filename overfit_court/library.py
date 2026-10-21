"""A garden of forking paths: strategy families an agent can fish in.

Used three ways: demo mode (no API key) routes ideas to a family by keyword,
the scripted fisherman explores them, and the Nemotron fisherman chooses
among them. Every function follows the harness contract: weights at row t
use data up to row t only; the harness lags them and charges costs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    name: str
    title: str
    hypothesis: str
    keywords: tuple
    param_grid: dict
    code: str
    search_queries: tuple = ()


def _f(name, title, hypothesis, keywords, grid, code, queries=()):
    return Family(name, title, hypothesis, tuple(keywords), grid, code.strip() + "\n", tuple(queries))


FAMILIES = {f.name: f for f in [
    _f("weekday", "Day-of-week effect",
       "Returns on one weekday are systematically different from the others.",
       ["monday", "tuesday", "wednesday", "thursday", "friday", "weekday", "day of week",
        "day-of-week", "weekend"],
       {"day": [0, 1, 2, 3, 4], "side": [1, -1]},
       '''
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, day: int = 0, side: int = 1) -> pd.DataFrame:
    # decide at today's close whether to hold tomorrow; the next business
    # day's weekday is known from the calendar, no future prices needed
    tomorrow = (prices.index.dayofweek + 1) % 5
    hold = (tomorrow == day).astype(float)
    w = np.repeat(hold[:, None] * side, prices.shape[1], axis=1)
    return pd.DataFrame(w, index=prices.index, columns=prices.columns)
''', ["day of week effect stock returns", "weekend effect anomaly"]),

    _f("ma_cross", "Moving-average crossover",
       "Assets trading above a slower moving average keep rising.",
       ["moving average", "crossover", "golden cross", "death cross", "sma", "ema"],
       {"fast": [5, 10, 20, 50], "slow": [50, 100, 150, 200]},
       '''
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.DataFrame:
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    return np.sign(f - s).where(s.notna(), 0.0)
''', ["moving average crossover trading rule profitability", "technical trading rules data snooping"]),

    _f("breakout", "Channel breakout",
       "New highs lead to further gains; new lows to further losses.",
       ["breakout", "donchian", "new high", "channel", "turtle"],
       {"window": [20, 40, 55, 100, 150, 200]},
       '''
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, window: int = 55) -> pd.DataFrame:
    hi = prices.rolling(window).max()
    lo = prices.rolling(window).min()
    sig = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    sig[prices >= hi] = 1.0
    sig[prices <= lo] = -1.0
    return sig.ffill().fillna(0.0)
''', ["donchian channel breakout performance", "turtle trading rules evidence"]),

    _f("zscore", "Mean reversion from stretched prices",
       "Prices far from their recent average snap back.",
       ["mean reversion", "mean-reversion", "z-score", "zscore", "stretched", "snap back",
        "oversold", "overbought"],
       {"window": [10, 20, 40, 60], "entry": [0.5, 1.0, 1.5, 2.0]},
       '''
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, window: int = 20, entry: float = 1.0) -> pd.DataFrame:
    ma = prices.rolling(window).mean()
    sd = prices.rolling(window).std()
    z = (prices - ma) / sd
    return (-np.sign(z)).where(z.abs() > entry, 0.0).fillna(0.0)
''', ["short-term mean reversion equities", "z-score mean reversion strategy"]),

    _f("reversal", "Short-term reversal",
       "Last week's losers beat last week's winners.",
       ["reversal", "buy losers", "buy the losers", "last week's losers", "buy last week",
        "overreaction", "contrarian"],
       {"lookback": [1, 2, 3, 5, 10], "top_frac": [0.2, 0.3, 0.4]},
       '''
import pandas as pd


def positions(prices: pd.DataFrame, lookback: int = 5, top_frac: float = 0.3) -> pd.DataFrame:
    past = prices / prices.shift(lookback) - 1.0
    ranks = past.rank(axis=1, pct=True)
    longs = (ranks <= top_frac).astype(float)
    shorts = (ranks >= 1.0 - top_frac).astype(float)
    return (longs - shorts).where(past.notna(), 0.0)
''', ["short-term reversal anomaly Jegadeesh 1990", "weekly reversal effect"]),

    _f("low_vol", "Low-volatility anomaly",
       "Calm assets beat volatile ones on a risk-adjusted basis.",
       ["volatility", "low vol", "low-vol", "calm", "defensive", "betting against beta"],
       {"vol_window": [20, 60, 120, 250], "top_frac": [0.2, 0.3, 0.4]},
       '''
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, vol_window: int = 60, top_frac: float = 0.3) -> pd.DataFrame:
    vol = np.log(prices).diff().rolling(vol_window).std()
    ranks = vol.rank(axis=1, pct=True)
    longs = (ranks <= top_frac).astype(float)
    shorts = (ranks >= 1.0 - top_frac).astype(float)
    return (longs - shorts).where(vol.notna(), 0.0)
''', ["low volatility anomaly", "betting against beta Frazzini Pedersen"]),

    _f("trend", "Time-series momentum",
       "An asset that has risen over the past months keeps rising, regardless of the others.",
       ["time-series", "time series", "trend following", "trend-following", "trend", "cta"],
       {"lookback": [20, 60, 120, 250], "vol_window": [20, 60, 120]},
       '''
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, lookback: int = 120, vol_window: int = 60) -> pd.DataFrame:
    past = prices / prices.shift(lookback) - 1.0
    vol = np.log(prices).diff().rolling(vol_window).std()
    raw = np.sign(past) / vol
    raw = raw.replace([np.inf, -np.inf], np.nan)
    return raw.div(raw.abs().sum(axis=1), axis=0).fillna(0.0)
''', ["time series momentum Moskowitz Ooi Pedersen", "trend following century of evidence"]),

    _f("momentum", "Cross-sectional momentum",
       "Assets that outperformed over the past months keep outperforming.",
       ["momentum", "winners", "went up", "outperform", "relative strength"],
       {"lookback": [20, 40, 60, 90, 120, 180, 250], "skip": [0, 5, 21],
        "top_frac": [0.2, 0.3, 0.4]},
       '''
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, lookback: int = 60, skip: int = 5,
              top_frac: float = 0.3) -> pd.DataFrame:
    past = prices.shift(skip) / prices.shift(skip + lookback) - 1.0
    ranks = past.rank(axis=1, pct=True)
    longs = (ranks >= 1.0 - top_frac).astype(float)
    shorts = (ranks <= top_frac).astype(float)
    return (longs - shorts).where(past.notna(), 0.0)
''', ["cross-sectional momentum anomaly Jegadeesh Titman", "momentum crash risk"]),
]}

DEFAULT = "momentum"


def route(idea: str) -> Family:
    """Pick the family whose keywords best match the idea (demo mode only)."""
    text = idea.lower()
    for fam in FAMILIES.values():          # dict order puts specific families first
        if any(k in text for k in fam.keywords):
            return fam
    return FAMILIES[DEFAULT]


def spec_for(fam: Family) -> dict:
    return {"name": fam.title, "hypothesis": fam.hypothesis, "family": fam.name,
            "signal_description": fam.hypothesis, "param_grid": dict(fam.param_grid),
            "search_queries": list(fam.search_queries), "known_risks": []}


# ------------------------------------------------------------------ universes
UNIVERSES = list(range(0, 25))     # 0 = every asset; 1..24 = a fixed random half

UNIVERSE_WRAPPER = '''

_base_positions = positions


def positions(prices, universe=0, **params):
    """Universe selection: 'the effect only works on these assets'."""
    import numpy as _np
    w = _base_positions(prices, **params)
    if universe:
        n = prices.shape[1]
        keep = _np.random.default_rng(int(universe)).choice(n, size=max(2, n // 2), replace=False)
        mask = _np.zeros(n, dtype=bool)
        mask[keep] = True
        w = w.copy()
        w.loc[:, ~mask] = 0.0
    return w
'''


def with_universe(code: str) -> str:
    return code + UNIVERSE_WRAPPER
