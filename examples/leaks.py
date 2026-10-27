"""Eight ways a strategy can read the future, shared by the tests and the tier experiment.

Each is the body of a `positions(prices)` function; `source()` wraps it into a
module like the ones the engineer writes.
"""

LEAKS = {
    "full-sample mean": "z = (prices - prices.mean()) / prices.std()\n    return -np.sign(z)",
    "negative shift": "return np.sign(prices.shift(-1) - prices).fillna(0)",
    "back-fill": (
        "nxt = prices.where(prices.pct_change() > 0).bfill()\n"
        "    return np.sign(nxt - prices).fillna(0)"
    ),
    # Only the warm-up period leaks, so only the early cuts can see these.
    "back-filled warm-up": "return np.sign(prices - prices.rolling(250).mean().bfill())",
    "short back-filled warm-up": "return np.sign(prices - prices.rolling(5).mean().bfill())",
    "centred window": "return np.sign(prices - prices.rolling(21, center=True).mean()).fillna(0)",
    "whole-series quantile": "return (prices > prices.quantile(0.5)).astype(float)",
    # Computes the full-sample anchor on its first call and reuses it on every later
    # one, which would hide the leak if runs shared the module.
    "anchor cached between calls": (
        "if not CACHE:\n        CACHE['mean'], CACHE['sd'] = prices.mean(), prices.std()\n"
        "    return -np.sign((prices - CACHE['mean']) / CACHE['sd'])"
    ),
}


def source(body: str) -> str:
    """A strategy module whose positions(prices) function has this body."""
    return (
        "import numpy as np\nimport pandas as pd\n\nCACHE = {}\n\n"
        f"def positions(prices):\n    {body}\n"
    )
