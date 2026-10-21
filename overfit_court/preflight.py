"""Pre-flight check for live keys: python -m overfit_court.preflight

Checks, in order:

1. Token Factory: the key works, and the three Nemotron model ids exist
   (if not, lists the Nemotron ids your account can see).
2. One real call per tier, with latency, tokens and cost; and whether each tier
   accepts JSON mode (if not, the client falls back to asking in the prompt).
3. Token Factory Sandboxes: a real small job through the real code path
   (dependencies image, trial checkpoint, parallel branches, time-travel
   branch, results read back), compared against the same job run locally.
4. Tavily: one search.
"""

from __future__ import annotations

import os
import time

import numpy as np
from tavily import TavilyClient

from .config import Settings, get_settings
from .data import synthetic_prices
from .library import FAMILIES
from .llm import LLM
from .pipeline import expand_grid, time_travel_probes
from .sandbox import LocalExecutor, NebiusSandboxExecutor

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
TIERS = ("nano", "super", "ultra")


def report(status: str, what: str, detail: str = "") -> None:
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
    for tier, model in settings.models.items():
        if model in available:
            report(PASS, f"{tier}: {model}")
        else:
            nemotron = sorted(m for m in available if "nemotron" in m.lower())
            report(
                FAIL, f"{tier}: {model} not found", f"set MODEL_{tier.upper()} to one of {nemotron}"
            )
            all_found = False
    return all_found


def check_calls(settings: Settings, llm: LLM) -> None:
    for tier in TIERS:
        try:
            reply = llm.chat(
                "preflight",
                tier,
                "Reply with the single word: ready.",
                "Status?",
                mock=lambda: "ready",
                max_tokens=400,
            )
            call = llm.ledger[-1]
            report(
                PASS,
                f"{tier} call",
                f"{call.latency_s:.1f}s, {call.input_tokens}+{call.output_tokens} tokens, "
                f"${call.cost_usd:.6f}, replied {reply[:30]!r}",
            )
        except Exception as e:
            report(FAIL, f"{tier} call", f"{type(e).__name__}: {e}")

    for tier in TIERS:
        try:
            llm.client.chat.completions.create(
                model=settings.models[tier],
                max_tokens=200,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": 'Reply with {"ok": true}'}],
            )
            report(PASS, f"{tier} accepts JSON mode")
        except Exception as e:
            report(
                SKIP,
                f"{tier} rejects JSON mode; the client will ask for JSON in the prompt instead",
                f"{type(e).__name__}: {str(e)[:120]}",
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
    probes = time_travel_probes(grid)
    try:
        t0 = time.perf_counter()
        remote = NebiusSandboxExecutor(n_chunks=2).run(family.code, prices, grid, 5.0, probes)
        wall = time.perf_counter() - t0
    except Exception as e:
        report(FAIL, "Token Factory Sandboxes", f"{type(e).__name__}: {e}")
        return
    for step in remote.setup:
        report(PASS, f"sandbox {step['step']}", f"{step['wall_s']:.1f}s")

    local = LocalExecutor(n_chunks=2).run(family.code, prices, grid, 5.0, probes)
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
    leak = remote.leak or {}
    report(
        PASS if leak and leak == local.leak else FAIL,
        "time-travel test ran as its own branch",
        f"{leak.get('checks', 0)} checks, leaked: {leak.get('leaked')}",
    )


def check_tavily(settings: Settings) -> None:
    try:
        t0 = time.perf_counter()
        response = TavilyClient(api_key=settings.tavily_api_key).search(
            "time series momentum anomaly", max_results=2
        )
        report(
            PASS,
            "Tavily search",
            f"{len(response.get('results', []))} results, {time.perf_counter() - t0:.1f}s",
        )
    except Exception as e:
        report(FAIL, "Tavily search", f"{type(e).__name__}: {e}")


def main() -> None:
    settings = get_settings()
    print(f"Token Factory endpoint: {settings.base_url}\n")
    if settings.mock_llm:
        report(SKIP, "Token Factory and Sandboxes", "NEBIUS_API_KEY not set")
    else:
        llm = LLM(settings)
        if check_models(settings, llm):
            check_calls(settings, llm)
        check_sandboxes()
    if settings.mock_search:
        report(SKIP, "Tavily", "TAVILY_API_KEY not set")
    else:
        check_tavily(settings)


if __name__ == "__main__":
    main()
