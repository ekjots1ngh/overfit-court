"""Pre-flight check for live keys: python -m overfit_court.preflight

Checks, in order:

1. Token Factory: the key works, and the three Nemotron model ids exist
   (if not, lists the Nemotron ids your account can see).
2. One real call per tier, asked the way the court's roles ask it: JSON mode,
   and thinking on for Super and Ultra, off for Nano. The reply must parse.
   Reports latency, tokens and cost, and any request part the model refused
   (the client then works without it, as it would in a trial).
3. Token Factory Sandboxes: a real small job through the real code path
   (dependencies image, trial checkpoint, parallel branches, each with the
   time-travel test, results read back), compared with the same job run locally.
4. Tavily: one search.

It ends with what to do next.
"""

from __future__ import annotations

import os
import time

import numpy as np

from . import _harness, research
from .config import Settings, get_settings
from .data import synthetic_prices
from .library import FAMILIES
from .llm import LLM
from .pipeline import expand_grid
from .sandbox import LocalExecutor, NebiusSandboxExecutor

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
TIERS = ("nano", "super", "ultra")
THINKS = {"nano": False, "super": True, "ultra": True}  # as the court's roles on each tier ask
failures: list[str] = []  # every FAIL line, for the closing advice


def report(status: str, what: str, detail: str = "") -> None:
    if status == FAIL:
        failures.append(what)
    print(f"[{status}] {what}" + (f"  ({detail})" if detail else ""))


def check_models(settings: Settings, llm: LLM) -> bool:
    try:
        available = {m.id for m in llm.client.models.list().data}
    except Exception as e:
        report(
            FAIL,
            f"list models at {settings.base_url}",
            f"{type(e).__name__}: {e}. Check NEBIUS_BASE_URL against the Token Factory quickstart.",
        )
        return False
    report(PASS, f"Token Factory key works, {len(available)} models visible")
    all_found = True
    for tier, configured in settings.models.items():
        model = llm.model_for(tier)  # the listed id, if it differs only in case or prefix
        if model in available:
            report(PASS, f"{tier}: {model}", "" if model == configured else f"set as {configured}")
        else:
            nemotron = sorted(m for m in available if "nemotron" in m.lower())
            report(
                FAIL, f"{tier}: {model} not found", f"set MODEL_{tier.upper()} to one of {nemotron}"
            )
            all_found = False
    return all_found


def check_calls(llm: LLM) -> None:
    for tier in TIERS:
        thinks = THINKS[tier]
        first = len(llm.calls)
        try:
            reply = llm.chat_json(
                "preflight",
                tier,
                'Reply with this JSON object: {"ready": true}',
                "Status?",
                mock=lambda: '{"ready": true}',
                reasoning=thinks,
                max_tokens=4000 if thinks else 400,
            )
            ok = reply.get("ready") is True
        except Exception as e:
            ok, reply = False, f"{type(e).__name__}: {e}"
        calls = llm.calls[first:]
        report(
            PASS if ok else FAIL,
            f"{tier} call, thinking {'on' if thinks else 'off'}, JSON reply",
            f"{len(calls)} request(s), {sum(c.latency_s for c in calls):.1f}s, "
            f"{sum(c.input_tokens for c in calls)}+{sum(c.output_tokens for c in calls)} tokens, "
            f"${sum(c.cost_usd for c in calls):.6f}, replied {str(reply)[:60]!r}",
        )
        for part in sorted(llm.refused.get(llm.model_for(tier), ())):
            report(
                SKIP,
                f"{tier} refused {part}",
                "the client now works without it, as it will in trials",
            )


def check_sandboxes() -> None:
    if not (os.getenv("NEBIUS_PROJECT_ID") or os.getenv("CONTREE_PROJECT")):
        report(
            FAIL, "Token Factory Sandboxes", "set NEBIUS_PROJECT_ID (a project with sandbox access)"
        )
        return
    family = FAMILIES["momentum"]
    prices = synthetic_prices(n_days=300, seed=1)
    grid = expand_grid(family.param_grid, 6)
    try:
        t0 = time.perf_counter()
        remote = NebiusSandboxExecutor(n_chunks=2).run(family.code, prices, grid, 5.0)
        wall = time.perf_counter() - t0
    except Exception as e:
        report(FAIL, "Token Factory Sandboxes", f"{type(e).__name__}: {e}")
        return
    for step in remote.setup:
        report(PASS, f"sandbox {step['step']}", f"{step['wall_s']:.1f}s")

    local = LocalExecutor(n_chunks=2).run(family.code, prices, grid, 5.0)
    same = remote.returns.shape == local.returns.shape and np.allclose(
        remote.returns.to_numpy(), local.returns.to_numpy()
    )
    summary = remote.summary()
    report(
        PASS if same and not remote.errors else FAIL,
        f"{summary['branches']} branches in parallel, results identical to a local run",
        f"{wall:.1f}s end to end, {summary['compute_s_total']:.1f}s compute, "
        f"${summary['sandbox_cost_usd']:.6f}, errors: {remote.errors or 'none'}",
    )
    checks = remote.time_travel or {}
    report(
        PASS if _harness.passed(checks) and checks == local.time_travel else FAIL,
        "time-travel test ran in every branch, same result as locally",
        f"{checks.get('checks', 0)} checks of {checks.get('configs', 0)} configurations: "
        f"{checks.get('status', 'did not run')}",
    )


def check_tavily(settings: Settings) -> None:
    """The court's own search (advanced depth, both domain lists, usage), one kind per line."""
    search = research.prior_art(["time series momentum"], settings)
    for record in search.searches:
        ok = record["results"] > 0 and not record["error"]
        detail = f"{record['results']} results, {record['credits']} credits, {record['latency_s']}s"
        if record["error"]:
            detail += f", {record['error']}"
        elif not ok:
            detail += "; check the domain list in research.SOURCES against a search without it"
        report(PASS if ok else FAIL, f"Tavily {record['kind']} search", detail)
    if not search.searches:
        report(FAIL, "Tavily search", search.note or "no search ran")


def main() -> None:
    failures.clear()
    settings = get_settings()
    print(f"Token Factory endpoint: {settings.base_url}\n")
    if settings.mock_llm:
        report(SKIP, "Token Factory and Sandboxes", "NEBIUS_API_KEY not set")
    else:
        llm = LLM(settings)
        if check_models(settings, llm):
            check_calls(llm)
        check_sandboxes()
    if settings.mock_search:
        report(SKIP, "Tavily", "TAVILY_API_KEY not set")
    else:
        check_tavily(settings)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed. Fix the FAIL lines above, then run this again.")
    else:
        print(
            "Ready. Next: streamlit run app.py, or\n"
            '  python -m overfit_court "buy assets that rose most over the past year"'
        )


if __name__ == "__main__":
    main()
