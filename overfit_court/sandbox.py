"""Run model-written strategy code over a parameter grid, in isolation.

The harness, not the model, applies a one-bar lag to positions and charges
transaction costs, so generated code cannot trade on today's close.

Executors
- LocalExecutor: subprocesses on your machine. Dev only; it runs untrusted
  LLM code, which is exactly why real runs belong in a sandbox.
- NebiusSandboxExecutor: Token Factory Sandboxes (ConTree microVMs, via
  contree-sdk). Three levels of checkpoint:
      python:3.12-slim -> +numpy/pandas (cached) -> +data+code (per trial)
  and every parameter chunk branches from the per-trial checkpoint in
  parallel. Setup happens once; each branch pays only for its own backtests.
- InProcessExecutor: trusted library code only, for experiments.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RUNNER = r'''
import json, importlib.util
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("strategy", "strategy.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
job = json.load(open("job.json"))
rets = prices.pct_change().fillna(0.0)
cost = job["cost_bps"] / 1e4
keys, cols, errors = [], [], {}
for key, params in job["grid"].items():
    try:
        pos = mod.positions(prices.copy(), **params)
        pos = pd.DataFrame(pos).reindex(index=prices.index, columns=prices.columns)
        pos = pos.astype(float).fillna(0.0).clip(-1.0, 1.0)
        gross = pos.abs().sum(axis=1).clip(lower=1.0)
        pos = pos.div(gross, axis=0)                 # gross exposure <= 1
        held = pos.shift(1).fillna(0.0)              # harness-enforced lag
        turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
        r = ((held * rets).sum(axis=1) - turnover * cost).to_numpy(dtype=float)
        if not np.all(np.isfinite(r)):
            raise ValueError("non-finite returns")
        keys.append(key)
        cols.append(r)
    except Exception as e:
        errors[key] = f"{type(e).__name__}: {e}"[:500]
R = np.column_stack(cols) if cols else np.zeros((len(prices), 0))
np.savez_compressed("result.npz", R=R, meta=np.array(json.dumps({"keys": keys, "errors": errors})))
'''

WORKDIR = "/workspace"


@dataclass
class ExecutionResult:
    returns: pd.DataFrame           # T x N, columns are config keys
    errors: dict                    # key -> error string
    grid: dict                      # key -> params
    metrics: list = field(default_factory=list)   # one dict per chunk

    def summary(self) -> dict:
        m = self.metrics
        return {
            "chunks": len(m),
            "wall_s": round(max((x.get("wall_s", 0) for x in m), default=0.0), 2),
            "compute_s_total": round(sum(x.get("elapsed_s", 0) for x in m), 2),
            "sandbox_cost_usd": round(sum(x.get("cost_usd", 0) for x in m), 6),
        }


def decode_result(blob: bytes) -> tuple[list, np.ndarray, dict]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return meta["keys"], z["R"], meta["errors"]


def merge(parts: list[tuple[list, np.ndarray, dict]], grid: dict, index) -> ExecutionResult:
    cols, errors = {}, {}
    for keys, R, errs in parts:
        errors.update(errs)
        for j, k in enumerate(keys):
            cols[k] = R[:, j]
    ordered = [k for k in grid if k in cols]
    mat = pd.DataFrame({k: cols[k] for k in ordered}, index=index)
    return ExecutionResult(mat, errors, grid)


def split(grid: dict, n: int) -> list[dict]:
    keys = list(grid)
    n = max(1, min(n, len(keys)))
    return [{k: grid[k] for k in keys[i::n]} for i in range(n)]


def _tail(text, n: int = 400) -> str:
    text = text.decode(errors="replace") if isinstance(text, bytes) else str(text or "")
    return text.strip()[-n:] or "no output"


# ------------------------------------------------------------------ local
class LocalExecutor:
    name = "local-subprocess"

    def __init__(self, n_workers: int = 4, timeout_s: int = 300):
        self.n_workers = n_workers
        self.timeout_s = timeout_s

    def _run_chunk(self, code: str, prices_csv: str, chunk: dict, cost_bps: float):
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "strategy.py").write_text(code)
            (p / "runner.py").write_text(RUNNER)
            (p / "prices.csv").write_text(prices_csv)
            (p / "job.json").write_text(json.dumps({"grid": chunk, "cost_bps": cost_bps}))
            try:
                proc = subprocess.run([sys.executable, "runner.py"], cwd=d, capture_output=True,
                                      text=True, timeout=self.timeout_s)
                ok = proc.returncode == 0 and (p / "result.npz").exists()
                err = None if ok else _tail(proc.stderr)
            except subprocess.TimeoutExpired:
                ok, err = False, f"timed out after {self.timeout_s}s"
            wall = time.perf_counter() - t0
            metric = {"configs": len(chunk), "wall_s": round(wall, 3), "elapsed_s": round(wall, 3)}
            if not ok:
                return ([], np.zeros((0, 0)), {k: err for k in chunk}), metric
            return decode_result((p / "result.npz").read_bytes()), metric

    def run(self, code: str, prices: pd.DataFrame, grid: dict, cost_bps: float) -> ExecutionResult:
        csv = prices.to_csv()
        chunks = split(grid, self.n_workers)
        with ThreadPoolExecutor(len(chunks)) as pool:
            out = list(pool.map(lambda c: self._run_chunk(code, csv, c, cost_bps), chunks))
        res = merge([o[0] for o in out], grid, prices.index)
        res.metrics = [o[1] for o in out]
        return res


# ------------------------------------------------------------------ nebius
class NebiusSandboxExecutor:
    """Token Factory Sandboxes via contree-sdk.

    Needs NEBIUS_API_KEY and a project with sandbox access enabled
    (NEBIUS_PROJECT_ID or CONTREE_PROJECT), or a profile written by
    `contree auth`. Set OC_SANDBOX_IMAGE to an image UUID that already has
    numpy and pandas to skip the one-off setup (useful if your sandboxes
    have no outbound network).
    """
    name = "nebius-sandboxes"
    BASE_OCI = "docker.io/library/python:3.12-slim"
    CACHE = Path(os.getenv("OC_SANDBOX_CACHE", ".overfit_court_sandbox_image"))

    def __init__(self, n_workers: int = 8, timeout_s: int = 900, client=None):
        self.n_workers = n_workers
        self.timeout_s = timeout_s
        self._client = client
        self.setup_log: list[dict] = []

    # client ---------------------------------------------------------
    def _make_client(self):
        from contree_sdk import Contree
        from contree_sdk.auth import IAMAuth
        from contree_sdk.config import ContreeConfig
        project = os.getenv("NEBIUS_PROJECT_ID") or os.getenv("CONTREE_PROJECT")
        auth = IAMAuth(project_id=project) if project else IAMAuth()
        return Contree(ContreeConfig(auth=auth, operation_timeout=float(self.timeout_s)))

    # checkpoints ----------------------------------------------------
    async def _deps_image(self, client):
        preset = os.getenv("OC_SANDBOX_IMAGE") or (
            self.CACHE.read_text().strip() if self.CACHE.exists() else None)
        if preset:
            return await client.images.use(preset)
        t0 = time.perf_counter()
        py = await client.images.oci(self.BASE_OCI)
        deps = await py.run(shell="mkdir -p /workspace && pip install --no-cache-dir --quiet numpy pandas",
                            disposable=False, timeout=self.timeout_s)
        if deps.exit_code != 0:
            raise RuntimeError(
                "pip install failed inside the sandbox (outbound network may be off). "
                "Build an image with numpy and pandas, then set OC_SANDBOX_IMAGE. "
                f"stderr: {_tail(deps.stderr)}")
        self.CACHE.write_text(str(deps.uuid))
        self.setup_log.append({"step": "deps image", "wall_s": round(time.perf_counter() - t0, 2),
                               "image": str(deps.uuid)})
        return deps

    async def _trial_image(self, client, deps, code: str, prices_csv: str):
        t0 = time.perf_counter()
        img = await deps.run(
            shell=f"cd {WORKDIR} && python -c 'import numpy, pandas, strategy'",
            files={f"{WORKDIR}/runner.py": RUNNER.encode(),
                   f"{WORKDIR}/strategy.py": code.encode(),
                   f"{WORKDIR}/prices.csv": prices_csv.encode()},
            disposable=False, timeout=300)
        self.setup_log.append({"step": "trial checkpoint", "wall_s": round(time.perf_counter() - t0, 2),
                               "exit_code": img.exit_code})
        return img

    async def _run_chunk(self, trial_img, chunk: dict, cost_bps: float, sem):
        async with sem:
            t0 = time.perf_counter()
            job = json.dumps({"grid": chunk, "cost_bps": cost_bps}).encode()
            try:
                out = await trial_img.run(shell=f"cd {WORKDIR} && python runner.py",
                                          files={f"{WORKDIR}/job.json": job},
                                          disposable=False, timeout=self.timeout_s)
                metric = {"configs": len(chunk), "wall_s": round(time.perf_counter() - t0, 3),
                          "elapsed_s": out.result.elapsed_time.total_seconds(),
                          "cost_usd": float(out.result.cost or 0.0),
                          "image": str(out.uuid)}
                if out.exit_code != 0:
                    return ([], np.zeros((0, 0)), {k: _tail(out.stderr) for k in chunk}), metric
                blob = await out.read(f"{WORKDIR}/result.npz")
                return decode_result(blob), metric
            except Exception as e:  # network or API failure for this branch only
                metric = {"configs": len(chunk), "wall_s": round(time.perf_counter() - t0, 3),
                          "error": f"{type(e).__name__}: {e}"}
                return ([], np.zeros((0, 0)), {k: metric["error"] for k in chunk}), metric

    async def _run_async(self, code, prices, grid, cost_bps) -> ExecutionResult:
        client = self._client or self._make_client()
        deps = await self._deps_image(client)
        trial_img = await self._trial_image(client, deps, code, prices.to_csv())
        if trial_img.exit_code != 0:
            err = f"strategy failed to import: {_tail(trial_img.stderr)}"
            return ExecutionResult(pd.DataFrame(index=prices.index), {k: err for k in grid}, grid,
                                   [{"step": "trial checkpoint", "error": err}])
        sem = asyncio.Semaphore(self.n_workers)
        chunks = split(grid, self.n_workers)
        out = await asyncio.gather(*(self._run_chunk(trial_img, c, cost_bps, sem) for c in chunks))
        res = merge([o[0] for o in out], grid, prices.index)
        res.metrics = [o[1] for o in out]
        return res

    def run(self, code, prices, grid, cost_bps) -> ExecutionResult:
        return asyncio.run(self._run_async(code, prices, grid, cost_bps))


# ------------------------------------------------------------------ in-process
class InProcessExecutor:
    """Runs strategy code inside this Python process. Trusted code only.

    For experiments over the built-in strategy library, where spawning a
    subprocess or sandbox per run would dominate the runtime. Never use it
    for model-written code.
    """
    name = "in-process (trusted code only)"

    def run(self, code: str, prices: pd.DataFrame, grid: dict, cost_bps: float) -> ExecutionResult:
        t0 = time.perf_counter()
        ns: dict = {}
        exec(compile(code, "strategy.py", "exec"), ns)       # noqa: S102 (trusted library code)
        fn = ns["positions"]
        rets = prices.pct_change().fillna(0.0)
        cost = cost_bps / 1e4
        cols, errors = {}, {}
        for key, params in grid.items():
            try:
                pos = pd.DataFrame(fn(prices.copy(), **params)).reindex(
                    index=prices.index, columns=prices.columns).astype(float).fillna(0.0).clip(-1, 1)
                pos = pos.div(pos.abs().sum(axis=1).clip(lower=1.0), axis=0)
                held = pos.shift(1).fillna(0.0)
                turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
                cols[key] = ((held * rets).sum(axis=1) - turnover * cost).to_numpy(dtype=float)
            except Exception as e:
                errors[key] = f"{type(e).__name__}: {e}"[:500]
        mat = pd.DataFrame({k: cols[k] for k in grid if k in cols}, index=prices.index)
        wall = time.perf_counter() - t0
        return ExecutionResult(mat, errors, grid, [{"configs": len(grid), "wall_s": round(wall, 3),
                                                   "elapsed_s": round(wall, 3)}])


def get_executor(name: str):
    return NebiusSandboxExecutor() if name == "nebius" else LocalExecutor()
