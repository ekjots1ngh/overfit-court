"""Calibration: is the court right on markets where we know the truth?

Three kinds of synthetic market, many random seeds each:
    none (noise)   every "edge" is overfitting; the court should never acquit
    weak, strong   a real persistent edge is planted; momentum can find it

Each trial is judged on the open 75% of the data, then its winner is tested
once on the sealed 25%. Runs in demo mode with the built-in strategy library,
so it is reproducible and no model-written code ever runs in-process.

    python -m examples.calibration [n_seeds]
"""

import sys
import time
from collections import Counter

from overfit_court.config import get_settings
from overfit_court.data import synthetic_prices
from overfit_court.pipeline import run_trial, unseal
from overfit_court.sandbox import InProcessExecutor

PLANTED_EDGE = {"none (noise)": 0.0, "weak": 0.0004, "strong": 0.0008}
IDEA = "buy recent winners, short losers"


def silent(*_args):
    pass


def median(values):
    return sorted(values)[len(values) // 2]


def main(n_seeds: int) -> None:
    settings = get_settings().offline()
    executor = InProcessExecutor()
    print(
        "| Planted edge | Runs | Convicted | Hung jury | Acquitted | Sealed: confirmed "
        "| Sealed: refuted | Median best Sharpe (open) | Median Sharpe (sealed) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    start = time.perf_counter()
    for label, signal in PLANTED_EDGE.items():
        verdicts, sealed_verdicts, open_sharpes, sealed_sharpes = Counter(), Counter(), [], []
        for seed in range(n_seeds):
            prices = synthetic_prices(signal=signal, seed=100 + seed)
            trial = run_trial(IDEA, prices, settings, executor=executor, log=silent)
            sealed = unseal(trial, executor, settings=settings, log=silent)
            verdicts[trial.verdict] += 1
            sealed_verdicts[sealed.verdict] += 1
            open_sharpes.append(trial.evidence()["best_sharpe_annual"])
            sealed_sharpes.append(sealed.sealed_sharpe_annual)
        print(
            f"| {label} | {n_seeds} | {verdicts['CONVICTED']} | {verdicts['HUNG JURY']} | "
            f"{verdicts['ACQUITTED']} | {sealed_verdicts['CONFIRMED']} | "
            f"{sealed_verdicts['REFUTED']} | {median(open_sharpes):.2f} | "
            f"{median(sealed_sharpes):.2f} |"
        )
    print(f"\n{3 * n_seeds} trials in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
