"""Download real daily closes into the wide CSV format the court expects.

    pip install yfinance
    python examples/fetch_prices.py sectors      # 11 US sector ETFs
    python examples/fetch_prices.py XLK XLF XLE  # any tickers

Uses adjusted closes so dividends and splits do not create fake returns.
"""
import sys

import yfinance as yf

PRESETS = {
    "sectors": ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"],
    "countries": ["EWU", "EWG", "EWQ", "EWJ", "EWC", "EWA", "EWH", "EWS", "EWL", "EWN", "EWD"],
}

args = sys.argv[1:] or ["sectors"]
tickers = PRESETS.get(args[0], args)
name = args[0] if args[0] in PRESETS else "prices"
df = yf.download(tickers, start="2012-01-01", auto_adjust=True, progress=False)["Close"]
df = df.dropna(axis=1, thresh=int(0.9 * len(df))).dropna()
df.to_csv(f"{name}.csv")
print(f"Saved {name}.csv: {df.shape[0]} days x {df.shape[1]} assets, "
      f"{df.index[0].date()} to {df.index[-1].date()}")
