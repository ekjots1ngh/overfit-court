"""Which Nemotron tier should audit code, and which should repair it? Needs a live key.

For each of the eight leaks in examples/leaks.py, each tier (Nano, Super, Ultra):

- reviews the leaky code, asked exactly as the court's auditor is asked. Only
  the model's own review counts here, not the court's pattern checks. A leak is
  spotted when the review says medium or high risk;
- reviews every honest strategy in the library too, so a tier that calls
  everything risky gets no credit: those are its false alarms;
- repairs the leaky code from the time-travel test's own description of the
  leak, asked exactly as the court's engineer is. A repair counts when the new
  code passes the time-travel test, still trades, and reaches for nothing
  beyond its prices.

    python -m examples.tier_experiment

It prints a summary per tier (with cost and time) and the result per leak.
The court's choice is in agents.py: Nano audits, Super writes and repairs, and
a leak that survives a Super repair goes to Ultra.
"""

import statistics
import sys

from examples.leaks import LEAKS, source
from overfit_court import _harness, agents
from overfit_court.config import get_settings
from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.llm import LLM

TIERS = ("nano", "super", "ultra")
ONE_CONFIG = {"only": {}}


def time_travel(code: str, prices) -> tuple[dict, bool]:
    """(the time-travel report, whether the strategy trades at all) for one configuration."""
    cuts = _harness.cut_points(len(prices))
    _, returns, errors, report = _harness.evaluate(
        _harness.strategy_loader(code), prices, ONE_CONFIG, 5.0, True, cuts
    )
    trades = not errors and returns.shape[1] == 1 and float(returns[:, 0].std()) > 0
    return report, trades


def spots_a_problem(llm: LLM, tier: str, code: str) -> bool:
    """The model's own review of the code, asked as the court's auditor asks it.

    A review that can't be had or read counts as not spotting anything.
    """
    try:
        review = llm.chat_json(
            "auditor",
            tier,
            agents.AUDITOR_SYS,
            code,
            mock=lambda: "{}",
            reasoning=False,
            max_tokens=agents.QUICK_BUDGET,
        )
    except Exception:
        return False
    return str(review.get("risk", "")).strip().lower() in ("medium", "high")


def repairs(llm: LLM, tier: str, kind: str, code: str, problem: str, prices) -> bool:
    """Whether the tier's repair passes the time-travel test, trades, and stays in bounds."""
    spec = {"name": kind, "hypothesis": "", "param_grid": {}}
    try:
        fixed = agents.write_code(llm, spec, error=problem, previous=code, tier=tier)
        report, trades = time_travel(fixed, prices)
    except Exception:  # no usable code came back
        return False
    return _harness.passed(report) and trades and not agents.escapes(fixed)


def main() -> None:
    settings = get_settings()
    if settings.mock_llm:
        sys.exit("This experiment asks the live models: set NEBIUS_API_KEY first.")
    prices = synthetic_prices(n_days=1000, seed=1)
    honest = [family.fixed_code or family.code for family in FAMILIES.values()]
    leaky = {}
    for kind, body in LEAKS.items():
        code = source(body)
        report, _ = time_travel(code, prices)
        leaky[kind] = (code, _harness.describe(report))

    rows, per_leak = [], {kind: {} for kind in LEAKS}
    for tier in TIERS:
        llm = LLM(settings)
        spotted = false_alarms = repaired = 0
        for kind, (code, problem) in leaky.items():
            seen = spots_a_problem(llm, tier, code)
            fixed = repairs(llm, tier, kind, code, problem, prices)
            spotted, repaired = spotted + seen, repaired + fixed
            per_leak[kind][tier] = ("spotted" if seen else "missed") + (
                ", repaired" if fixed else ", not repaired"
            )
        for code in honest:
            false_alarms += spots_a_problem(llm, tier, code)
        latency = statistics.median(call.latency_s for call in llm.calls)
        rows.append(
            f"| {llm.model_for(tier)} | {spotted} of {len(LEAKS)} | {false_alarms} of "
            f"{len(honest)} | {repaired} of {len(LEAKS)} | ${llm.total_cost():.4f} | "
            f"{latency:.1f}s |"
        )
        print(f"{tier} done", file=sys.stderr)

    print(
        "| Model | Leaks spotted in review | Honest code called risky | Leaks repaired "
        "| Cost | Median call |"
    )
    print("|---|---|---|---|---|---|")
    print("\n".join(rows))
    print("\n| Leak | " + " | ".join(tier.title() for tier in TIERS) + " |")
    print("|---|" + "---|" * len(TIERS))
    for kind, results in per_leak.items():
        print(f"| {kind} | " + " | ".join(results[tier] for tier in TIERS) + " |")


if __name__ == "__main__":
    main()
