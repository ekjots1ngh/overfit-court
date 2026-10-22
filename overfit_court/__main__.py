"""Command line.

    python -m overfit_court "your idea" [--csv prices.csv] [--unseal] [--out report]
    python -m overfit_court fish [--csv prices.csv] [--target 1.0] [--attempts 40]

Without --csv the data is a synthetic market (pure noise, or --planted for a
real edge). The court record persists in ~/.overfit_court/ (see --ledger), so
every run counts against the next one on the same data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .data import PLANTED_SIGNAL, load_csv, synthetic_prices
from .fishing import run_expedition
from .ledger import Ledger, record_key
from .pipeline import SEALED_FRACTION, TimeTravelError, run_trial, unseal
from .report import court_record_markdown

DEFAULT_LEDGER = os.getenv("OC_LEDGER_DIR", "~/.overfit_court")


def _add_data_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", help="wide CSV of daily closes (default: a synthetic market)")
    parser.add_argument("--planted", action="store_true", help="synthetic market with a real edge")
    parser.add_argument("--seed", type=int, default=7, help="synthetic market seed")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER, help="directory for the court record")
    parser.add_argument("--forget", action="store_true", help="wipe the record for this data first")


def _prices_and_ledger(args):
    if args.csv:
        prices = load_csv(args.csv)
    else:
        prices = synthetic_prices(seed=args.seed, signal=PLANTED_SIGNAL if args.planted else 0.0)
    ledger = Ledger(args.ledger)
    if args.forget:
        ledger.clear(record_key(prices, SEALED_FRACTION))
    return prices, ledger


def trial_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m overfit_court")
    parser.add_argument("idea", help="the trading idea, in plain English")
    parser.add_argument("--unseal", action="store_true", help="then test the winner on sealed data")
    parser.add_argument("--out", default="report", help="write <out>.md and <out>.json")
    _add_data_options(parser)
    args = parser.parse_args(argv)

    prices, ledger = _prices_and_ledger(args)
    trial = run_trial(args.idea, prices, ledger=ledger)
    problem = None
    if args.unseal:
        try:
            unseal(trial)
        except TimeTravelError as e:
            problem = f"The evidence room was not opened, or its result is void: {e}"
        except Exception as e:  # a sandbox or network failure: the trial's report is still written
            problem = f"The evidence room could not be opened ({type(e).__name__}: {e})."
    markdown = court_record_markdown(trial)
    Path(f"{args.out}.md").write_text(markdown)
    Path(f"{args.out}.json").write_text(trial.to_json())
    print("\n" + markdown)
    if problem:
        print(f"\n{problem}")


def fish_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m overfit_court fish")
    parser.add_argument("--target", type=float, default=1.0, help="Sharpe the fisherman must hit")
    parser.add_argument("--attempts", type=int, default=40, help="attempts it is allowed")
    _add_data_options(parser)
    args = parser.parse_args(argv)

    prices, ledger = _prices_and_ledger(args)
    expedition = run_expedition(
        prices, target=args.target, max_attempts=args.attempts, ledger=ledger, seed=args.seed
    )
    print(json.dumps(expedition.summary(), indent=2, default=str))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "fish":
        fish_command(sys.argv[2:])
    else:
        trial_command(sys.argv[1:])


if __name__ == "__main__":
    main()
