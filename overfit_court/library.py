"""The strategy library: a small garden of forking paths.

Eight honest strategy families, plus one deliberately leaky draft, used in
three places:

- demo mode (no API key): an idea is routed to a family by keyword, and the
  family's code stands in for what Nemotron would write;
- the scripted fisherman, which explores the families;
- the Nemotron fisherman, which chooses among them.

Every strategy follows the harness contract: the weights for day t may use
prices up to day t only. The harness lags them by a day and charges costs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    name: str  # short identifier, e.g. "momentum"
    title: str  # human-readable name
    hypothesis: str  # the effect the strategy bets on
    keywords: tuple  # demo mode routes an idea here if it contains one of these words
    param_grid: dict  # parameter name -> values to try
    code: str  # source defining positions(prices, **params)
    search_queries: tuple = ()  # prior-art searches for this effect
    fixed_code: str = ""  # demo mode only: what a repair of `code` produces
    fishable: bool = True  # offered to the fishing expedition


def _source(text: str) -> str:
    return text.strip() + "\n"


_ALL = [
    # A plausible idea whose natural first draft reads the future: the mean and
    # standard deviation are taken over the whole series, including dates that
    # haven't happened yet (and the `window` parameter it was meant to use is
    # ignored). Code review usually misses it; the time-travel test doesn't. In
    # demo mode the repair swaps in `fixed_code`, which uses rolling windows.
    Family(
        name="anchor",
        title="Reversion to the long-run average",
        hypothesis="Assets trading below their long-run average price drift back up to it.",
        keywords=(
            "long-run average",
            "long run average",
            "average price",
            "below their average",
            "below average",
            "fair value",
            "anchor",
        ),
        param_grid={"band": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5], "window": [60, 120, 250, 500]},
        code=_source("""
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, band: float = 0.5, window: int = 250) -> pd.DataFrame:
    # how far is each asset from its long-run average price?
    z = (prices - prices.mean()) / prices.std()
    return (-np.sign(z)).where(z.abs() > band, 0.0)
"""),
        fixed_code=_source("""
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, band: float = 0.5, window: int = 250) -> pd.DataFrame:
    # long-run average and spread from PAST prices only (rolling window)
    ma = prices.rolling(window, min_periods=window).mean()
    sd = prices.rolling(window, min_periods=window).std()
    z = (prices - ma) / sd
    return (-np.sign(z)).where(z.abs() > band, 0.0).fillna(0.0)
"""),
        search_queries=(
            "mean reversion to long run average price stocks",
            "anchoring fair value reversion",
        ),
        fishable=False,
    ),
    Family(
        name="weekday",
        title="Day-of-week effect",
        hypothesis="Returns on one weekday are systematically different from the others.",
        keywords=(
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "weekday",
            "day of week",
            "day-of-week",
            "weekend",
        ),
        param_grid={"day": [0, 1, 2, 3, 4], "side": [1, -1]},
        code=_source("""
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, day: int = 0, side: int = 1) -> pd.DataFrame:
    # decide at today's close whether to hold tomorrow; the next business
    # day's weekday is known from the calendar, no future prices needed
    tomorrow = (prices.index.day_of_week + 1) % 5
    hold = (tomorrow == day).astype(float)
    w = np.repeat(hold[:, None] * side, prices.shape[1], axis=1)
    return pd.DataFrame(w, index=prices.index, columns=prices.columns)
"""),
        search_queries=("day of week effect stock returns", "weekend effect anomaly"),
    ),
    Family(
        name="ma_cross",
        title="Moving-average crossover",
        hypothesis="Assets trading above a slower moving average keep rising.",
        keywords=("moving average", "crossover", "golden cross", "death cross", "sma", "ema"),
        param_grid={"fast": [5, 10, 20, 50], "slow": [50, 100, 150, 200]},
        code=_source("""
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.DataFrame:
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    return np.sign(f - s).where(s.notna(), 0.0)
"""),
        search_queries=(
            "moving average crossover trading rule profitability",
            "technical trading rules data snooping",
        ),
    ),
    Family(
        name="breakout",
        title="Channel breakout",
        hypothesis="New highs lead to further gains; new lows to further losses.",
        keywords=("breakout", "donchian", "new high", "channel", "turtle"),
        param_grid={"window": [20, 40, 55, 100, 150, 200]},
        code=_source("""
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, window: int = 55) -> pd.DataFrame:
    hi = prices.rolling(window).max()
    lo = prices.rolling(window).min()
    sig = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    sig[prices >= hi] = 1.0
    sig[prices <= lo] = -1.0
    return sig.ffill().fillna(0.0)
"""),
        search_queries=("donchian channel breakout performance", "turtle trading rules evidence"),
    ),
    Family(
        name="zscore",
        title="Mean reversion from stretched prices",
        hypothesis="Prices far from their recent average snap back.",
        keywords=(
            "mean reversion",
            "mean-reversion",
            "z-score",
            "zscore",
            "stretched",
            "snap back",
            "oversold",
            "overbought",
        ),
        param_grid={"window": [10, 20, 40, 60], "entry": [0.5, 1.0, 1.5, 2.0]},
        code=_source("""
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, window: int = 20, entry: float = 1.0) -> pd.DataFrame:
    ma = prices.rolling(window).mean()
    sd = prices.rolling(window).std()
    z = (prices - ma) / sd
    return (-np.sign(z)).where(z.abs() > entry, 0.0).fillna(0.0)
"""),
        search_queries=("short-term mean reversion equities", "z-score mean reversion strategy"),
    ),
    Family(
        name="reversal",
        title="Short-term reversal",
        hypothesis="Last week's losers beat last week's winners.",
        keywords=(
            "reversal",
            "buy losers",
            "buy the losers",
            "last week's losers",
            "buy last week",
            "overreaction",
            "contrarian",
        ),
        param_grid={"lookback": [1, 2, 3, 5, 10], "top_frac": [0.2, 0.3, 0.4]},
        code=_source("""
import pandas as pd


def positions(prices: pd.DataFrame, lookback: int = 5, top_frac: float = 0.3) -> pd.DataFrame:
    past = prices / prices.shift(lookback) - 1.0
    ranks = past.rank(axis=1, pct=True)
    longs = (ranks <= top_frac).astype(float)
    shorts = (ranks >= 1.0 - top_frac).astype(float)
    return (longs - shorts).where(past.notna(), 0.0)
"""),
        search_queries=("short-term reversal anomaly Jegadeesh 1990", "weekly reversal effect"),
    ),
    Family(
        name="low_vol",
        title="Low-volatility anomaly",
        hypothesis="Calm assets beat volatile ones on a risk-adjusted basis.",
        keywords=("volatility", "low vol", "low-vol", "calm", "defensive", "betting against beta"),
        param_grid={"vol_window": [20, 60, 120, 250], "top_frac": [0.2, 0.3, 0.4]},
        code=_source("""
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, vol_window: int = 60, top_frac: float = 0.3) -> pd.DataFrame:
    vol = np.log(prices).diff().rolling(vol_window).std()
    ranks = vol.rank(axis=1, pct=True)
    longs = (ranks <= top_frac).astype(float)
    shorts = (ranks >= 1.0 - top_frac).astype(float)
    return (longs - shorts).where(vol.notna(), 0.0)
"""),
        search_queries=("low volatility anomaly", "betting against beta Frazzini Pedersen"),
    ),
    Family(
        name="trend",
        title="Time-series momentum",
        hypothesis="An asset that has risen over the past months keeps rising, "
        "regardless of the others.",
        keywords=(
            "time-series",
            "time series",
            "trend following",
            "trend-following",
            "trend",
            "cta",
        ),
        param_grid={"lookback": [20, 60, 120, 250], "vol_window": [20, 60, 120]},
        code=_source("""
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, lookback: int = 120, vol_window: int = 60) -> pd.DataFrame:
    past = prices / prices.shift(lookback) - 1.0
    vol = np.log(prices).diff().rolling(vol_window).std()
    raw = np.sign(past) / vol
    raw = raw.replace([np.inf, -np.inf], np.nan)
    return raw.div(raw.abs().sum(axis=1), axis=0).fillna(0.0)
"""),
        search_queries=(
            "time series momentum Moskowitz Ooi Pedersen",
            "trend following century of evidence",
        ),
    ),
    Family(
        name="momentum",
        title="Cross-sectional momentum",
        hypothesis="Assets that outperformed over the past months keep outperforming.",
        keywords=("momentum", "winners", "went up", "outperform", "relative strength"),
        param_grid={
            "lookback": [20, 40, 60, 90, 120, 180, 250],
            "skip": [0, 5, 21],
            "top_frac": [0.2, 0.3, 0.4],
        },
        code=_source("""
import numpy as np
import pandas as pd


def positions(prices: pd.DataFrame, lookback: int = 60, skip: int = 5,
              top_frac: float = 0.3) -> pd.DataFrame:
    past = prices.shift(skip) / prices.shift(skip + lookback) - 1.0
    ranks = past.rank(axis=1, pct=True)
    longs = (ranks >= 1.0 - top_frac).astype(float)
    shorts = (ranks <= top_frac).astype(float)
    return (longs - shorts).where(past.notna(), 0.0)
"""),
        search_queries=(
            "cross-sectional momentum anomaly Jegadeesh Titman",
            "momentum crash risk",
        ),
    ),
]

FAMILIES: dict[str, Family] = {f.name: f for f in _ALL}
FISHABLE: dict[str, Family] = {name: f for name, f in FAMILIES.items() if f.fishable}
DEFAULT = "momentum"


def route(idea: str) -> Family:
    """Demo mode: the first family with a keyword in the idea (most specific families first).

    Keywords match whole words, optionally plural: "mondays" finds "monday",
    but "small" does not find "sma" and "systematic" does not find "ema".
    """
    text = idea.lower()
    for family in _ALL:
        if any(re.search(rf"\b{re.escape(k)}s?\b", text) for k in family.keywords):
            return family
    return FAMILIES[DEFAULT]


def family_of(spec: dict) -> Family:
    """The family a spec names, or the default one: demo mode needs code for every spec."""
    name = spec.get("family")
    if isinstance(name, str) and name in FAMILIES:
        return FAMILIES[name]
    return FAMILIES[DEFAULT]


def spec_for(family: Family) -> dict:
    """The planner's spec for a family, in the same shape Nemotron Ultra returns."""
    return {
        "name": family.title,
        "hypothesis": family.hypothesis,
        "family": family.name,
        "signal_description": family.hypothesis,
        "param_grid": dict(family.param_grid),
        "search_queries": list(family.search_queries),
        "known_risks": [],
    }


# --- universe selection ---
# One more fork in the garden: "the effect only works on these assets". Universe
# 0 trades every asset; universes 1 to 24 each trade a fixed random half. The
# wrapper is appended to a strategy's source, so it runs inside the sandbox too.
UNIVERSES = list(range(25))

_UNIVERSE_WRAPPER = '''

import numpy as _np

_unrestricted_positions = positions


def positions(prices, universe=0, **params):
    """Trade only a fixed random half of the assets, chosen by `universe` (0 = all)."""
    weights = _unrestricted_positions(prices, **params)
    if universe:
        n = prices.shape[1]
        chosen = _np.random.default_rng(int(universe)).choice(n, size=max(2, n // 2), replace=False)
        keep = _np.zeros(n, dtype=bool)
        keep[chosen] = True
        weights = weights.copy()
        weights.loc[:, ~keep] = 0.0
    return weights
'''


def with_universe(code: str) -> str:
    """A strategy's source with the universe-selection parameter added."""
    return code + _UNIVERSE_WRAPPER
