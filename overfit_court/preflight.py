"""Pre-flight check for live keys: python -m overfit_court.preflight

Verifies, in order: Token Factory auth and model IDs, one real call per
Nemotron tier (with measured latency and cost), Token Factory Sandboxes
(import image, run python, branch, read a file back), and Tavily search.
"""
from __future__ import annotations

import asyncio
import os
import time

from .config import get_settings
from .llm import LLM

OK, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def line(status: str, what: str, detail: str = "") -> None:
    print(f"[{status}] {what}" + (f"  ({detail})" if detail else ""))


def check_models(s) -> bool:
    if not s.nebius_api_key:
        line(SKIP, "Token Factory", "NEBIUS_API_KEY not set")
        return False
    from openai import OpenAI
    client = OpenAI(base_url=s.base_url, api_key=s.nebius_api_key)
    try:
        ids = {m.id for m in client.models.list().data}
    except Exception as e:
        line(FAIL, f"list models at {s.base_url}", f"{type(e).__name__}: {e}. "
             "Check NEBIUS_BASE_URL against the Token Factory quickstart.")
        return False
    line(OK, f"Token Factory auth, {len(ids)} models visible")
    good = True
    for tier, model in s.models.items():
        if model in ids:
            line(OK, f"{tier}: {model}")
        else:
            nemo = sorted(i for i in ids if "nemotron" in i.lower())
            line(FAIL, f"{tier}: {model} not found", f"set MODEL_{tier.upper()} to one of {nemo}")
            good = False
    return good


def check_calls(s) -> None:
    llm = LLM(s)
    for tier in ("nano", "super", "ultra"):
        try:
            out = llm.chat("preflight", tier, "Reply with the single word: ready.", "Status?",
                           mock=lambda: "ready", max_tokens=400)
            c = llm.ledger[-1]
            line(OK, f"{tier} call", f"{c.latency_s:.1f}s, {c.input_tokens}+{c.output_tokens} tokens, "
                 f"${c.cost_usd:.6f}, said {out[:30]!r}")
        except Exception as e:
            line(FAIL, f"{tier} call", f"{type(e).__name__}: {e}")
    # JSON mode: if a tier rejects response_format the client falls back to asking in the
    # prompt; worth knowing, and worth reporting in the hackathon feedback
    for tier in ("nano", "super", "ultra"):
        try:
            llm.client.chat.completions.create(
                model=s.models[tier], max_tokens=200, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": 'Reply with {"ok": true}'}])
            line(OK, f"{tier} JSON mode accepted")
        except Exception as e:
            line(SKIP, f"{tier} JSON mode not accepted; the client will fall back",
                 f"{type(e).__name__}: {str(e)[:120]}")


async def _sandbox():
    from .sandbox import NebiusSandboxExecutor
    ex = NebiusSandboxExecutor()
    client = ex._make_client()
    t0 = time.perf_counter()
    deps = await ex._deps_image(client)
    line(OK, "sandbox image with numpy + pandas", f"{deps.uuid}, {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    a = await deps.run(shell="mkdir -p /workspace && echo branch-a > /workspace/x.txt",
                       disposable=False, timeout=120)
    b = await a.run(shell="cat /workspace/x.txt && python -c 'import pandas; print(pandas.__version__)'",
                    disposable=False, timeout=120)
    blob = await b.read("/workspace/x.txt")
    line(OK if blob.strip() == b"branch-a" else FAIL, "checkpoint, branch and read back",
         f"{time.perf_counter() - t0:.1f}s, stdout {str(b.stdout).strip()!r}, "
         f"cost ${float(b.result.cost or 0):.6f}")


def check_sandbox(s) -> None:
    if not s.nebius_api_key:
        line(SKIP, "Token Factory Sandboxes", "NEBIUS_API_KEY not set")
        return
    if not (os.getenv("NEBIUS_PROJECT_ID") or os.getenv("CONTREE_PROJECT")):
        line(FAIL, "Token Factory Sandboxes", "set NEBIUS_PROJECT_ID (project with sandbox access)")
        return
    try:
        asyncio.run(_sandbox())
    except Exception as e:
        line(FAIL, "Token Factory Sandboxes", f"{type(e).__name__}: {e}")


def check_tavily(s) -> None:
    if s.mock_search:
        line(SKIP, "Tavily", "TAVILY_API_KEY not set")
        return
    from tavily import TavilyClient
    try:
        t0 = time.perf_counter()
        res = TavilyClient(api_key=s.tavily_api_key).search("time series momentum anomaly",
                                                              max_results=2)
        line(OK, "Tavily search", f"{len(res.get('results', []))} results, "
             f"{time.perf_counter() - t0:.1f}s")
    except Exception as e:
        line(FAIL, "Tavily search", f"{type(e).__name__}: {e}")


def main():
    s = get_settings()
    print(f"Base URL: {s.base_url}\n")
    if check_models(s):
        check_calls(s)
    check_sandbox(s)
    check_tavily(s)


if __name__ == "__main__":
    main()
