"""CLI.

  python -m overfit_court "your idea" [--csv prices.csv] [--unseal] [--out report]
  python -m overfit_court fish [--csv prices.csv] [--target 1.0] [--attempts 40]

The court record persists in ~/.overfit_court/ (override with --ledger).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from .data import load_csv, synthetic_prices
from .ledger import Ledger
from .pipeline import SEALED_FRAC, record_key, run_trial, unseal

DEFAULT_LEDGER = os.getenv("OC_LEDGER_DIR", "~/.overfit_court")


def _common(ap):
    ap.add_argument("--csv", help="wide CSV of daily closes (default: synthetic noise)")
    ap.add_argument("--planted", action="store_true", help="synthetic data with a real edge")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ledger", default=DEFAULT_LEDGER, help="directory for the court record")
    ap.add_argument("--forget", action="store_true", help="wipe the court record for this data")


def _prices(a):
    if a.csv:
        return load_csv(a.csv)
    return synthetic_prices(seed=a.seed, signal=0.0008 if a.planted else 0.0)


def trial_main(argv):
    ap = argparse.ArgumentParser(prog="overfit_court")
    ap.add_argument("idea", help="trading idea in plain English")
    ap.add_argument("--unseal", action="store_true", help="then test the winner on sealed evidence")
    ap.add_argument("--out", default="report", help="output path prefix")
    _common(ap)
    a = ap.parse_args(argv)
    px, ledger = _prices(a), Ledger(a.ledger)
    if a.forget:
        ledger.clear(record_key(px, SEALED_FRAC))
    trial = run_trial(a.idea, px, ledger=ledger)
    if a.unseal:
        unseal(trial, ledger=ledger)
    md = trial.to_markdown()
    Path(a.out + ".md").write_text(md)
    Path(a.out + ".json").write_text(trial.to_json())
    print("\n" + md)


def fish_main(argv):
    from .fishing import run_expedition
    ap = argparse.ArgumentParser(prog="overfit_court fish")
    ap.add_argument("--target", type=float, default=1.0)
    ap.add_argument("--attempts", type=int, default=40)
    _common(ap)
    a = ap.parse_args(argv)
    px, ledger = _prices(a), Ledger(a.ledger)
    if a.forget:
        ledger.clear(record_key(px, SEALED_FRAC))
    e = run_expedition(px, a.target, a.attempts, ledger=ledger, seed=a.seed)
    print(json.dumps(e.summary(), indent=2, default=str))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fish":
        fish_main(sys.argv[2:])
    else:
        trial_main(sys.argv[1:])


if __name__ == "__main__":
    main()
