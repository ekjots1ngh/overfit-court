"""Executors: where strategy code actually runs.

Every executor runs the same harness (`_harness.py`), so lags, costs and the
time-travel test are identical everywhere. They differ only in isolation:

NebiusSandboxExecutor  Token Factory Sandboxes (microVMs, via contree-sdk).
                       The real thing: model-written code runs here.
LocalExecutor          Subprocesses on this machine. For development; it runs
                       untrusted model code, which is exactly why real runs
                       belong in a sandbox.
InProcessExecutor      Inside this Python process. Only for the trusted
                       built-in strategy library, in experiments where
                       spawning a process per run would dominate the runtime.

A run is split into jobs: chunks of the parameter grid, plus (optionally) one
time-travel job. In Token Factory Sandboxes every job is its own branch of a
single checkpoint, so they all run in parallel from the same starting state:

    python:3.12-slim -> + numpy, pandas      (cached across trials)
                        -> + data + strategy (one checkpoint per trial)
                           -> one branch per job
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
from typing import NamedTuple

import numpy as np
import pandas as pd
from contree_sdk import Contree
from contree_sdk.auth import IAMAuth
from contree_sdk.config import ContreeConfig

from . import _harness

RUNNER = Path(_harness.__file__).read_text()  # uploaded verbatim as runner.py
WORKDIR = "/workspace"


class JobOutput(NamedTuple):
    """What one job (one branch) returns."""

    keys: list  # configurations that ran, in column order of `returns`
    returns: np.ndarray  # T x len(keys)
    errors: dict  # configuration key -> error message
    leak: dict | None  # time-travel report, if this job ran the test


@dataclass
class ExecutionResult:
    returns: pd.DataFrame  # T x N daily returns, one column per configuration key
    errors: dict  # configuration key -> error message
    grid: dict  # configuration key -> parameters
    leak: dict | None = None  # time-travel report, if requested
    metrics: list = field(default_factory=list)  # timing and cost, one dict per job
    setup: list = field(default_factory=list)  # sandbox setup steps, if any

    def summary(self) -> dict:
        return {
            "branches": len(self.metrics),
            "wall_s": round(max((m.get("wall_s", 0) for m in self.metrics), default=0.0), 2),
            "compute_s_total": round(sum(m.get("elapsed_s", 0) for m in self.metrics), 2),
            "sandbox_cost_usd": round(sum(m.get("cost_usd", 0) for m in self.metrics), 6),
            "setup_s": round(sum(step.get("wall_s", 0) for step in self.setup), 2),
        }


def make_jobs(grid: dict, n_chunks: int, cost_bps: float, leak_check: list | None) -> list[dict]:
    """Split the grid into `n_chunks` jobs, plus one time-travel job if requested."""
    keys = list(grid)
    n = max(1, min(n_chunks, len(keys))) if keys else 0
    jobs = [{"grid": {k: grid[k] for k in keys[i::n]}, "cost_bps": cost_bps} for i in range(n)]
    if leak_check:
        jobs.append({"grid": {}, "cost_bps": cost_bps, "leak_check": leak_check})
    return jobs


def decode(blob: bytes) -> JobOutput:
    """Read the result.npz a job writes."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return JobOutput(meta["keys"], z["R"], meta["errors"], meta.get("leak"))


def failed(job: dict, error: str) -> JobOutput:
    """A job that could not run at all: every configuration in it gets the error."""
    return JobOutput([], np.zeros((0, 0)), dict.fromkeys(job["grid"], error), None)


def merge(outputs: list[JobOutput], grid: dict, index: pd.Index) -> ExecutionResult:
    """Combine job outputs into one result, with columns in grid order."""
    columns, errors, leak = {}, {}, None
    for out in outputs:
        errors.update(out.errors)
        leak = out.leak or leak
        for j, key in enumerate(out.keys):
            columns[key] = out.returns[:, j]
    ordered = [k for k in grid if k in columns]
    returns = pd.DataFrame({k: columns[k] for k in ordered}, index=index)
    return ExecutionResult(returns, errors, grid, leak=leak)


def _tail(text, n: int = 400) -> str:
    """The end of a process's output, which is where the error usually is."""
    text = text.decode(errors="replace") if isinstance(text, bytes) else str(text or "")
    return text.strip()[-n:] or "no output"


# --- Token Factory Sandboxes ---
class NebiusSandboxExecutor:
    """Runs every job as a branch of one Token Factory Sandboxes checkpoint.

    Needs NEBIUS_API_KEY and a project with sandbox access (NEBIUS_PROJECT_ID
    or CONTREE_PROJECT), or a profile written by `contree auth`. Set
    OC_SANDBOX_IMAGE to an image UUID that already has numpy and pandas to
    skip the one-off setup (useful if your sandboxes have no outbound network).
    """

    name = "nebius-sandboxes"
    BASE_IMAGE = "docker.io/library/python:3.12-slim"
    IMAGE_CACHE = Path(os.getenv("OC_SANDBOX_CACHE", ".overfit_court_sandbox_image"))

    def __init__(self, n_chunks: int = 8, timeout_s: int = 900, client=None):
        self.n_chunks = n_chunks
        self.timeout_s = timeout_s
        self._client = client  # injectable for tests

    def run(self, code, prices, grid, cost_bps, leak_check: list | None = None) -> ExecutionResult:
        return asyncio.run(self._run(code, prices, grid, cost_bps, leak_check))

    async def _run(self, code, prices, grid, cost_bps, leak_check) -> ExecutionResult:
        client = self._client or self._make_client()
        setup: list[dict] = []
        deps = await self._dependencies_image(client, setup)
        checkpoint = await self._trial_checkpoint(deps, code, prices.to_csv(), setup)
        if checkpoint.exit_code != 0:
            error = f"strategy failed to import: {_tail(checkpoint.stderr)}"
            return ExecutionResult(
                pd.DataFrame(index=prices.index), dict.fromkeys(grid, error), grid, setup=setup
            )

        # At most n_chunks + 1 branches, well inside the service's concurrency limit.
        jobs = make_jobs(grid, self.n_chunks, cost_bps, leak_check)
        done = await asyncio.gather(*(self._branch(checkpoint, job) for job in jobs))
        result = merge([out for out, _ in done], grid, prices.index)
        result.metrics = [metric for _, metric in done]
        result.setup = setup
        return result

    def _make_client(self) -> Contree:
        project = os.getenv("NEBIUS_PROJECT_ID") or os.getenv("CONTREE_PROJECT")
        auth = IAMAuth(project_id=project) if project else IAMAuth()
        return Contree(ContreeConfig(auth=auth, operation_timeout=float(self.timeout_s)))

    async def _dependencies_image(self, client, setup: list):
        """python + numpy + pandas, built once and cached by UUID for later trials."""
        cached = os.getenv("OC_SANDBOX_IMAGE") or (
            self.IMAGE_CACHE.read_text().strip() if self.IMAGE_CACHE.exists() else None
        )
        if cached:
            return await client.images.use(cached)

        t0 = time.perf_counter()
        base = await client.images.oci(self.BASE_IMAGE)
        deps = await base.run(
            shell=f"mkdir -p {WORKDIR} && pip install --no-cache-dir --quiet numpy pandas",
            disposable=False,
            timeout=self.timeout_s,
        )
        if deps.exit_code != 0:
            raise RuntimeError(
                "pip install failed inside the sandbox (outbound network may be off). Build "
                "an image with numpy and pandas, then set OC_SANDBOX_IMAGE. "
                f"stderr: {_tail(deps.stderr)}"
            )
        self.IMAGE_CACHE.write_text(str(deps.uuid))
        setup.append({"step": "dependencies image", "wall_s": round(time.perf_counter() - t0, 2)})
        return deps

    async def _trial_checkpoint(self, deps, code: str, prices_csv: str, setup: list):
        """Upload the harness, the strategy and the data once; every job branches from here."""
        t0 = time.perf_counter()
        checkpoint = await deps.run(
            shell=f"cd {WORKDIR} && python -c 'import numpy, pandas, strategy'",
            files={
                f"{WORKDIR}/runner.py": RUNNER.encode(),
                f"{WORKDIR}/strategy.py": code.encode(),
                f"{WORKDIR}/prices.csv": prices_csv.encode(),
            },
            disposable=False,
            timeout=300,
        )
        setup.append({"step": "trial checkpoint", "wall_s": round(time.perf_counter() - t0, 2)})
        return checkpoint

    async def _branch(self, checkpoint, job: dict) -> tuple[JobOutput, dict]:
        """Run one job in its own branch and read its result file back."""
        t0 = time.perf_counter()
        metric = {"configs": len(job["grid"]), "time_travel": bool(job.get("leak_check"))}
        try:
            out = await checkpoint.run(
                shell=f"cd {WORKDIR} && python runner.py",
                files={f"{WORKDIR}/job.json": json.dumps(job).encode()},
                disposable=False,  # keep the image so result.npz can be read back
                timeout=self.timeout_s,
            )
            metric.update(
                wall_s=round(time.perf_counter() - t0, 3),
                elapsed_s=out.result.elapsed_time.total_seconds(),
                cost_usd=float(out.result.cost or 0.0),
            )
            if out.exit_code != 0:
                return failed(job, _tail(out.stderr)), metric
            return decode(await out.read(f"{WORKDIR}/result.npz")), metric
        except Exception as e:  # a network or API failure loses this branch only
            metric.update(wall_s=round(time.perf_counter() - t0, 3), error=str(e)[:300])
            return failed(job, f"{type(e).__name__}: {e}"), metric


# --- local subprocesses ---
class LocalExecutor:
    """Runs every job in its own subprocess on this machine (development only)."""

    name = "local-subprocess"

    def __init__(self, n_chunks: int = 4, timeout_s: int = 300):
        self.n_chunks = n_chunks
        self.timeout_s = timeout_s

    def run(self, code, prices, grid, cost_bps, leak_check: list | None = None) -> ExecutionResult:
        prices_csv = prices.to_csv()
        jobs = make_jobs(grid, self.n_chunks, cost_bps, leak_check)
        with ThreadPoolExecutor(len(jobs)) as pool:
            done = list(pool.map(lambda job: self._run_job(code, prices_csv, job), jobs))
        result = merge([out for out, _ in done], grid, prices.index)
        result.metrics = [metric for _, metric in done]
        return result

    def _run_job(self, code: str, prices_csv: str, job: dict) -> tuple[JobOutput, dict]:
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            (work / "runner.py").write_text(RUNNER)
            (work / "strategy.py").write_text(code)
            (work / "prices.csv").write_text(prices_csv)
            (work / "job.json").write_text(json.dumps(job))
            try:
                proc = subprocess.run(
                    [sys.executable, "runner.py"],
                    cwd=folder,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
                error = None
                if proc.returncode != 0 or not (work / "result.npz").exists():
                    error = _tail(proc.stderr)
            except subprocess.TimeoutExpired:
                error = f"timed out after {self.timeout_s}s"
            wall = round(time.perf_counter() - t0, 3)
            metric = {
                "configs": len(job["grid"]),
                "time_travel": bool(job.get("leak_check")),
                "wall_s": wall,
                "elapsed_s": wall,
            }
            if error:
                return failed(job, error), metric
            return decode((work / "result.npz").read_bytes()), metric


# --- in-process ---
class InProcessExecutor:
    """Runs the harness in this process. Trusted library code only, never model output."""

    name = "in-process (trusted code only)"

    def run(self, code, prices, grid, cost_bps, leak_check: list | None = None) -> ExecutionResult:
        t0 = time.perf_counter()
        namespace: dict = {}
        exec(compile(code, "strategy.py", "exec"), namespace)  # trusted library code only
        positions = namespace["positions"]
        keys, R, errors = _harness.backtest(positions, prices, grid, cost_bps)
        leak = _harness.time_travel(positions, prices, leak_check) if leak_check else None
        result = merge([JobOutput(keys, R, errors, leak)], grid, prices.index)
        wall = round(time.perf_counter() - t0, 3)
        result.metrics = [
            {
                "configs": len(grid),
                "time_travel": bool(leak_check),
                "wall_s": wall,
                "elapsed_s": wall,
            }
        ]
        return result


def get_executor(name: str):
    """'nebius' for Token Factory Sandboxes; anything else runs locally."""
    return NebiusSandboxExecutor() if name == "nebius" else LocalExecutor()
