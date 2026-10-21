"""Run model-written strategy code over a parameter grid, in isolation.

Every executor runs the same harness (overfit_court/_harness.py): the harness,
not the model, applies the one-day lag and transaction costs, and it runs the
time-travel test that proves the strategy never reads the future.

Executors
- NebiusSandboxExecutor: Token Factory Sandboxes (ConTree microVMs, via
  contree-sdk). Checkpoints:
      python:3.12-slim -> +numpy/pandas (cached) -> +data+code (per trial)
  Every chunk of the grid, and the time-travel test, is its own branch of the
  per-trial checkpoint, run in parallel. Setup happens once.
- LocalExecutor: subprocesses on your machine. Dev only; it runs untrusted
  model code, which is exactly why real runs belong in a sandbox.
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

from . import _harness

RUNNER = Path(_harness.__file__).read_text()     # uploaded verbatim as runner.py
WORKDIR = "/workspace"


@dataclass
class ExecutionResult:
    returns: pd.DataFrame           # T x N, columns are config keys
    errors: dict                    # key -> error string
    grid: dict                      # key -> params
    metrics: list = field(default_factory=list)   # one dict per branch
    leak: dict | None = None        # time-travel report, if requested

    def summary(self) -> dict:
        m = self.metrics
        return {
            "branches": len(m),
            "wall_s": round(max((x.get("wall_s", 0) for x in m), default=0.0), 2),
            "compute_s_total": round(sum(x.get("elapsed_s", 0) for x in m), 2),
            "sandbox_cost_usd": round(sum(x.get("cost_usd", 0) for x in m), 6),
        }


Part = tuple  # (keys, R, errors, leak)


def decode_result(blob: bytes) -> Part:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return meta["keys"], z["R"], meta["errors"], meta.get("leak")


def _failed(keys, err) -> Part:
    return [], np.zeros((0, 0)), {k: err for k in keys}, None


def merge(parts: list[Part], grid: dict, index) -> ExecutionResult:
    cols, errors, leak = {}, {}, None
    for keys, R, errs, lk in parts:
        errors.update(errs)
        leak = lk or leak
        for j, k in enumerate(keys):
            cols[k] = R[:, j]
    ordered = [k for k in grid if k in cols]
    mat = pd.DataFrame({k: cols[k] for k in ordered}, index=index)
    return ExecutionResult(mat, errors, grid, leak=leak)


def jobs_for(grid: dict, n: int, cost_bps: float, leak_check: list | None) -> list[dict]:
    keys = list(grid)
    n = max(1, min(n, len(keys))) if keys else 0
    jobs = [{"grid": {k: grid[k] for k in keys[i::n]}, "cost_bps": cost_bps} for i in range(n)]
    if leak_check:
        jobs.append({"grid": {}, "cost_bps": cost_bps, "leak_check": leak_check})
    return jobs


def _tail(text, n: int = 400) -> str:
    text = text.decode(errors="replace") if isinstance(text, bytes) else str(text or "")
    return text.strip()[-n:] or "no output"


# ------------------------------------------------------------------ local
class LocalExecutor:
    name = "local-subprocess"

    def __init__(self, n_workers: int = 4, timeout_s: int = 300):
        self.n_workers = n_workers
        self.timeout_s = timeout_s

    def _run_job(self, code: str, prices_csv: str, job: dict):
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "strategy.py").write_text(code)
            (p / "runner.py").write_text(RUNNER)
            (p / "prices.csv").write_text(prices_csv)
            (p / "job.json").write_text(json.dumps(job))
            try:
                proc = subprocess.run([sys.executable, "runner.py"], cwd=d, capture_output=True,
                                      text=True, timeout=self.timeout_s)
                ok = proc.returncode == 0 and (p / "result.npz").exists()
                err = None if ok else _tail(proc.stderr)
            except subprocess.TimeoutExpired:
                ok, err = False, f"timed out after {self.timeout_s}s"
            wall = time.perf_counter() - t0
            metric = {"configs": len(job["grid"]), "time_travel": bool(job.get("leak_check")),
                      "wall_s": round(wall, 3), "elapsed_s": round(wall, 3)}
            if not ok:
                return _failed(job["grid"], err), metric
            return decode_result((p / "result.npz").read_bytes()), metric

    def run(self, code: str, prices: pd.DataFrame, grid: dict, cost_bps: float,
            leak_check: list | None = None) -> ExecutionResult:
        csv = prices.to_csv()
        jobs = jobs_for(grid, self.n_workers, cost_bps, leak_check)
        with ThreadPoolExecutor(len(jobs)) as pool:
            out = list(pool.map(lambda j: self._run_job(code, csv, j), jobs))
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

    def _make_client(self):
        from contree_sdk import Contree
        from contree_sdk.auth import IAMAuth
        from contree_sdk.config import ContreeConfig
        project = os.getenv("NEBIUS_PROJECT_ID") or os.getenv("CONTREE_PROJECT")
        auth = IAMAuth(project_id=project) if project else IAMAuth()
        return Contree(ContreeConfig(auth=auth, operation_timeout=float(self.timeout_s)))

    async def _deps_image(self, client):
        preset = os.getenv("OC_SANDBOX_IMAGE") or (
            self.CACHE.read_text().strip() if self.CACHE.exists() else None)
        if preset:
            return await client.images.use(preset)
        t0 = time.perf_counter()
        py = await client.images.oci(self.BASE_OCI)
        deps = await py.run(shell=f"mkdir -p {WORKDIR} && pip install --no-cache-dir --quiet numpy pandas",
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

    async def _trial_image(self, deps, code: str, prices_csv: str):
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

    async def _run_job(self, trial_img, job: dict, sem):
        async with sem:
            t0 = time.perf_counter()
            metric = {"configs": len(job["grid"]), "time_travel": bool(job.get("leak_check"))}
            try:
                out = await trial_img.run(shell=f"cd {WORKDIR} && python runner.py",
                                          files={f"{WORKDIR}/job.json": json.dumps(job).encode()},
                                          disposable=False, timeout=self.timeout_s)
                metric.update(wall_s=round(time.perf_counter() - t0, 3),
                              elapsed_s=out.result.elapsed_time.total_seconds(),
                              cost_usd=float(out.result.cost or 0.0), image=str(out.uuid))
                if out.exit_code != 0:
                    return _failed(job["grid"], _tail(out.stderr)), metric
                return decode_result(await out.read(f"{WORKDIR}/result.npz")), metric
            except Exception as e:  # network or API failure for this branch only
                metric.update(wall_s=round(time.perf_counter() - t0, 3),
                              error=f"{type(e).__name__}: {e}")
                return _failed(job["grid"], metric["error"]), metric

    async def _run_async(self, code, prices, grid, cost_bps, leak_check) -> ExecutionResult:
        client = self._client or self._make_client()
        deps = await self._deps_image(client)
        trial_img = await self._trial_image(deps, code, prices.to_csv())
        if trial_img.exit_code != 0:
            err = f"strategy failed to import: {_tail(trial_img.stderr)}"
            return ExecutionResult(pd.DataFrame(index=prices.index), {k: err for k in grid}, grid,
                                   [{"step": "trial checkpoint", "error": err}])
        sem = asyncio.Semaphore(self.n_workers + 1)
        jobs = jobs_for(grid, self.n_workers, cost_bps, leak_check)
        out = await asyncio.gather(*(self._run_job(trial_img, j, sem) for j in jobs))
        res = merge([o[0] for o in out], grid, prices.index)
        res.metrics = [o[1] for o in out]
        return res

    def run(self, code, prices, grid, cost_bps, leak_check: list | None = None) -> ExecutionResult:
        return asyncio.run(self._run_async(code, prices, grid, cost_bps, leak_check))


# ------------------------------------------------------------------ in-process
class InProcessExecutor:
    """Runs strategy code inside this Python process. Trusted code only.

    For experiments over the built-in strategy library, where spawning a
    subprocess or sandbox per run would dominate the runtime. Never use it
    for model-written code.
    """
    name = "in-process (trusted code only)"

    def run(self, code: str, prices: pd.DataFrame, grid: dict, cost_bps: float,
            leak_check: list | None = None) -> ExecutionResult:
        t0 = time.perf_counter()
        ns: dict = {}
        exec(compile(code, "strategy.py", "exec"), ns)       # noqa: S102 (trusted library code)
        fn = ns["positions"]
        keys, R, errors = _harness.backtest(fn, prices, grid, cost_bps)
        leak = _harness.time_travel(fn, prices, leak_check) if leak_check else None
        res = merge([(keys, R, errors, leak)], grid, prices.index)
        wall = time.perf_counter() - t0
        res.metrics = [{"configs": len(grid), "time_travel": bool(leak_check),
                        "wall_s": round(wall, 3), "elapsed_s": round(wall, 3)}]
        return res


def get_executor(name: str):
    return NebiusSandboxExecutor() if name == "nebius" else LocalExecutor()
