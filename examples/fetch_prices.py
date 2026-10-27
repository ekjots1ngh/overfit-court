"""Download real daily closes into the wide CSV format the court expects.

    pip install yfinance
    python -m examples.fetch_prices sectors      # 11 US sector ETFs
    python -m examples.fetch_prices XLK XLF XLE  # any tickers

Adjusted closes are used, so dividends and splits don't show up as fake returns.
"""

import sys

import yfinance as yf

PRESETS = {
    "sectors": ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"],
    "countries": ["EWU", "EWG", "EWQ", "EWJ", "EWC", "EWA", "EWH", "EWS", "EWL", "EWN", "EWD"],
}


def main(arguments: list[str]) -> None:
    arguments = arguments or ["sectors"]
    tickers = PRESETS.get(arguments[0], arguments)
    name = arguments[0] if arguments[0] in PRESETS else "prices"
    download = yf.download(tickers, start="2012-01-01", auto_adjust=True, progress=False)
    if download is None or download.empty:
        sys.exit(f"Nothing came back for {', '.join(tickers)}: check the tickers and your network.")
    closes = download["Close"]
    # Keep assets with at least 90% of days, then only days where every asset traded.
    closes = closes.dropna(axis=1, thresh=int(0.9 * len(closes))).dropna()
    if closes.shape[1] < 2 or closes.empty:
        sys.exit("Fewer than two assets have enough history in common; try other tickers.")
    closes.to_csv(f"{name}.csv")
    print(
        f"Saved {name}.csv: {closes.shape[0]} days x {closes.shape[1]} assets, "
        f"{closes.index[0].date()} to {closes.index[-1].date()}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
