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

A run is split into jobs, one per chunk of the parameter grid. Each job
backtests its configurations and time-travel tests every one of them. The
evidence room has one configuration but hundreds of cuts, so its cuts are
shared out between the jobs instead. A job that produces no results fails the
time-travel test, whatever the reason. In sandboxes, a branch lost to a
network or API error is retried once, and if it is lost again its
configurations are counted as lost: the trial then ends in a mistrial rather
than blaming the code. A job that fails because of the code (it isn't valid
Python, crashes or runs out of time) is the code's problem and goes back to
the engineer. In Token Factory Sandboxes every job is its own branch of a
single checkpoint, so they all run in parallel from the same starting state:

    python:3.12-slim -> + numpy, pandas      (cached across trials)
                        -> + data + strategy (one checkpoint per trial; nothing runs yet)
                           -> one branch per job
"""

from __future__ import annotations

import asyncio
import contextlib
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
from contree_sdk.sdk.exceptions import ContreeError, OperationTimedOutError

from . import _harness

RUNNER = Path(_harness.__file__).read_text()  # uploaded verbatim as runner.py
WORKDIR = "/workspace"
# Check the strategy is valid Python without running any of it.
CHECK_SYNTAX = "python -c \"import ast; ast.parse(open('strategy.py').read())\""
# Run one job. A runner that exits cleanly without writing its results (code
# that calls os._exit, say) has failed, and says so.
RUN_JOB = (
    "python runner.py && { test -f result.npz || "
    "{ echo 'the runner exited without writing its results' >&2; exit 1; }; }"
)
# The sandbox gets exactly the numpy and pandas versions this machine has, so a
# sandbox run and a local run of the same job give the same numbers.
REQUIREMENTS = f"numpy=={np.__version__} pandas=={pd.__version__}"


class JobOutput(NamedTuple):
    """What one job (one branch) returns."""

    keys: list  # configurations that ran, in column order of `returns`
    returns: np.ndarray  # T x len(keys)
    errors: dict  # configuration key -> error message
    time_travel: dict | None  # this job's time-travel report, if it ran the test


@dataclass
class ExecutionResult:
    returns: pd.DataFrame  # T x N daily returns, one column per configuration key
    errors: dict  # configuration key -> error message
    grid: dict  # configuration key -> parameters
    time_travel: dict | None = None  # the time-travel report over every job, if requested
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


def make_jobs(grid: dict, n_chunks: int, cost_bps: float, time_travel: bool, cuts) -> list[dict]:
    """Split a run into at most `n_chunks` jobs.

    Each job usually takes every n-th configuration of the grid and checks it
    at every cut (history length, in days). A grid of one configuration, the
    evidence room's, is split by cut instead: each job checks it at every n-th
    cut, so its hundreds of checks run in parallel. Either way every
    configuration is checked at every cut exactly once.
    """
    if not grid:
        raise ValueError("The parameter grid is empty: there is nothing to run.")
    cuts = sorted(int(k) for k in cuts)
    job = {"cost_bps": cost_bps, "time_travel": time_travel}
    if len(grid) == 1 and time_travel and len(cuts) > n_chunks:
        return [{**job, "grid": dict(grid), "cuts": cuts[i::n_chunks]} for i in range(n_chunks)]
    keys = list(grid)
    n = min(n_chunks, len(keys))
    return [{**job, "grid": {k: grid[k] for k in keys[i::n]}, "cuts": cuts} for i in range(n)]


def decode(blob: bytes) -> JobOutput:
    """Read the result.npz a job writes."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return JobOutput(meta["keys"], z["R"], meta["errors"], meta["time_travel"])


def failed(job: dict, error: str, *, lost: bool = False) -> JobOutput:
    """A job that produced no results: every configuration in it gets the error.

    `lost` means the sandbox service lost it; otherwise the strategy failed.
    Either way none of its configurations was verified, so the run can't pass
    the time-travel test.
    """
    n = len(job["grid"])
    if lost:
        error = f"the results of {n} configurations were lost ({error})"
    report = None
    if job.get("time_travel"):
        report = _harness.unverified_report(error, lost_configs=n if lost else 0)
    return JobOutput([], np.zeros((0, 0)), dict.fromkeys(job["grid"], error), report)


def merge(
    outputs: list[JobOutput], grid: dict, index: pd.Index, time_travel: bool
) -> ExecutionResult:
    """Combine job outputs into one result, with columns in grid order.

    When jobs share a configuration (the evidence room's), every job's returns
    for it must agree, and if it failed in any job it failed.
    """
    columns, errors, disagree = {}, {}, []
    for out in outputs:
        errors.update(out.errors)
        for j, key in enumerate(out.keys):
            column = out.returns[:, j]
            earlier = columns.setdefault(key, column)  # the first job's copy, if there was one
            if not np.allclose(earlier, column, rtol=0, atol=_harness.TOLERANCE):
                disagree.append(key)
    ordered = [k for k in grid if k in columns and k not in errors]
    returns = pd.DataFrame({k: columns[k] for k in ordered}, index=index)
    report = None
    if time_travel:
        report = _harness.merge_reports([out.time_travel for out in outputs if out.time_travel])
        if disagree and report["error"] is None:
            report["error"] = (
                f"configuration {disagree[0]} gave different returns in different runs, so it "
                "isn't deterministic"
            )
            _harness.finish(report)
    return ExecutionResult(returns, errors, grid, time_travel=report)


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

    def run(
        self, code, prices, grid, cost_bps, time_travel: bool = True, extra_cuts=()
    ) -> ExecutionResult:
        cuts = _harness.cut_points(len(prices), extra_cuts)
        jobs = make_jobs(grid, self.n_chunks, cost_bps, time_travel, cuts)
        return asyncio.run(self._run(code, prices, grid, jobs, time_travel))

    async def _run(self, code, prices, grid, jobs, time_travel) -> ExecutionResult:
        client = self._client or self._make_client()
        setup: list[dict] = []
        deps, from_cache = await self._dependencies_image(client, setup)
        try:
            checkpoint = await self._trial_checkpoint(deps, code, prices.to_csv(), setup)
        except OperationTimedOutError:
            raise
        except ContreeError:
            if not from_cache:
                raise
            # The service may have cleaned up the image cached by an earlier trial
            # (an app can stay up for weeks): build it again, once.
            deps, _ = await self._dependencies_image(client, setup, use_cache=False)
            checkpoint = await self._trial_checkpoint(deps, code, prices.to_csv(), setup)
        if checkpoint.exit_code != 0:
            error = f"the strategy is not valid Python: {_tail(checkpoint.stderr)}"
            result = merge([failed(job, error) for job in jobs], grid, prices.index, time_travel)
            result.setup = setup
            return result

        # At most n_chunks branches, well inside the service's concurrency limit.
        done = await asyncio.gather(*(self._branch(checkpoint, job) for job in jobs))
        result = merge([out for out, _ in done], grid, prices.index, time_travel)
        result.metrics = [metric for _, metric in done]
        result.setup = setup
        return result

    def _make_client(self) -> Contree:
        project = os.getenv("NEBIUS_PROJECT_ID") or os.getenv("CONTREE_PROJECT")
        auth = IAMAuth(project_id=project) if project else IAMAuth()
        return Contree(ContreeConfig(auth=auth, operation_timeout=float(self.timeout_s)))

    def _cached_image(self) -> str | None:
        """The UUID of the image built earlier, if it has the same numpy and pandas versions."""
        try:
            cache = json.loads(self.IMAGE_CACHE.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(cache, dict) or cache.get("requirements") != REQUIREMENTS:
            return None
        return cache.get("image")

    async def _dependencies_image(self, client, setup: list, use_cache: bool = True):
        """python + numpy + pandas, built once and cached by UUID for later trials.

        Returns the image, and whether it came from this machine's cache file.
        """
        if os.getenv("OC_SANDBOX_IMAGE"):  # chosen by the user: never rebuilt behind their back
            return await client.images.use(os.environ["OC_SANDBOX_IMAGE"]), False
        cached = self._cached_image() if use_cache else None
        if cached:
            return await client.images.use(cached), True

        t0 = time.perf_counter()
        base = await client.images.oci(self.BASE_IMAGE)
        deps = await base.run(
            shell=f"mkdir -p {WORKDIR} && pip install --no-cache-dir --quiet {REQUIREMENTS}",
            disposable=False,
            timeout=self.timeout_s,
        )
        if deps.exit_code != 0:
            raise RuntimeError(
                "pip install failed inside the sandbox (outbound network may be off). Build "
                f"an image with {REQUIREMENTS}, then set OC_SANDBOX_IMAGE. "
                f"stderr: {_tail(deps.stderr)}"
            )
        # A read-only disk only means the next trial builds the image again.
        with contextlib.suppress(OSError):
            self.IMAGE_CACHE.write_text(
                json.dumps({"image": str(deps.uuid), "requirements": REQUIREMENTS})
            )
        setup.append({"step": "dependencies image", "wall_s": round(time.perf_counter() - t0, 2)})
        return deps, False

    async def _trial_checkpoint(self, deps, code: str, prices_csv: str, setup: list):
        """Upload the harness, the strategy and the data once; every job branches from here.

        The strategy is only parsed here, never run: no strategy code runs where
        the data file is. In each branch the runner deletes the file first.
        """
        t0 = time.perf_counter()
        checkpoint = await deps.run(
            shell=f"cd {WORKDIR} && {CHECK_SYNTAX}",
            files={
                f"{WORKDIR}/runner.py": RUNNER.encode(),
                f"{WORKDIR}/strategy.py": code.encode(),
                f"{WORKDIR}/prices.csv": prices_csv.encode(),
            },
            disposable=False,
            timeout=self.timeout_s,
        )
        setup.append({"step": "trial checkpoint", "wall_s": round(time.perf_counter() - t0, 2)})
        return checkpoint

    async def _branch(self, checkpoint, job: dict, tries: int = 2) -> tuple[JobOutput, dict]:
        """Run one job in its own branch and read its result file back.

        A network or API failure is retried once from the same checkpoint. A
        failure of the code itself is not, because it would fail again: a
        non-zero exit, or running past the time limit.
        """
        metric = {"configs": len(job["grid"]), "tries": 0}
        too_slow = f"timed out after {self.timeout_s}s"
        lost = "never ran"
        started = time.perf_counter()
        for attempt in range(1, tries + 1):
            metric["tries"] = attempt
            try:
                out = await checkpoint.run(
                    shell=f"cd {WORKDIR} && {RUN_JOB}",
                    files={f"{WORKDIR}/job.json": json.dumps(job).encode()},
                    disposable=False,  # keep the image so result.npz can be read back
                    timeout=self.timeout_s,
                )
                elapsed = out.result.elapsed_time.total_seconds()
                metric.update(
                    wall_s=round(time.perf_counter() - started, 3),
                    elapsed_s=elapsed,
                    cost_usd=float(out.result.cost or 0.0),
                )
                if out.exit_code != 0:
                    timed_out = elapsed >= 0.99 * self.timeout_s  # killed at the time limit
                    return failed(job, too_slow if timed_out else _tail(out.stderr)), metric
                return decode(await out.read(f"{WORKDIR}/result.npz")), metric
            except OperationTimedOutError:  # the code ran too long: retrying won't help
                metric["wall_s"] = round(time.perf_counter() - started, 3)
                return failed(job, too_slow), metric
            except Exception as e:  # the branch was lost, not the code's fault
                lost = f"{type(e).__name__}: {e}"
                metric["wall_s"] = round(time.perf_counter() - started, 3)
                metric.setdefault("lost_tries", []).append(lost[:300])
        return failed(job, lost, lost=True), metric


# --- local subprocesses ---
SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
SERVICE_PREFIXES = ("NEBIUS_", "TAVILY_", "CONTREE_")


def untrusted_environment() -> dict[str, str]:
    """This process's environment variables minus anything that looks like a credential.

    Model-written code may be told, by a prompt it was given, to look for keys.
    """
    return {
        name: value
        for name, value in os.environ.items()
        if not any(hint in name.upper() for hint in SECRET_HINTS)
        and not name.upper().startswith(SERVICE_PREFIXES)
    }


class LocalExecutor:
    """Runs every job in its own subprocess on this machine (development only).

    The subprocesses get no credentials from this process's environment.
    """

    name = "local-subprocess"

    def __init__(self, n_chunks: int = 4, timeout_s: int = 900):
        self.n_chunks = n_chunks
        self.timeout_s = timeout_s

    def run(
        self, code, prices, grid, cost_bps, time_travel: bool = True, extra_cuts=()
    ) -> ExecutionResult:
        prices_csv = prices.to_csv()
        cuts = _harness.cut_points(len(prices), extra_cuts)
        jobs = make_jobs(grid, self.n_chunks, cost_bps, time_travel, cuts)
        with ThreadPoolExecutor(len(jobs)) as pool:
            done = list(pool.map(lambda job: self._run_job(code, prices_csv, job), jobs))
        result = merge([out for out, _ in done], grid, prices.index, time_travel)
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
                    env=untrusted_environment(),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
                error = None
                if proc.returncode != 0:
                    error = _tail(proc.stderr)
                elif not (work / "result.npz").exists():
                    error = "the runner exited without writing its results"
            except subprocess.TimeoutExpired:
                error = f"timed out after {self.timeout_s}s"
            wall = round(time.perf_counter() - t0, 3)
            metric = {"configs": len(job["grid"]), "wall_s": wall, "elapsed_s": wall}
            if error:
                return failed(job, error), metric
            return decode((work / "result.npz").read_bytes()), metric


# --- in-process ---
class InProcessExecutor:
    """Runs the harness in this process. Trusted library code only, never model output."""

    name = "in-process (trusted code only)"

    def run(
        self, code, prices, grid, cost_bps, time_travel: bool = True, extra_cuts=()
    ) -> ExecutionResult:
        t0 = time.perf_counter()
        load_strategy = _harness.strategy_loader(code)  # trusted library code only
        cuts = _harness.cut_points(len(prices), extra_cuts)
        output = JobOutput(
            *_harness.evaluate(load_strategy, prices, grid, cost_bps, time_travel, cuts)
        )
        result = merge([output], grid, prices.index, time_travel)
        wall = round(time.perf_counter() - t0, 3)
        result.metrics = [{"configs": len(grid), "wall_s": wall, "elapsed_s": wall}]
        return result


EXECUTORS = {"nebius": NebiusSandboxExecutor, "local": LocalExecutor}  # see config.EXECUTOR_NAMES


def get_executor(name: str):
    """The executor named by OC_EXECUTOR: "nebius" (Token Factory Sandboxes) or "local"."""
    try:
        return EXECUTORS[name.strip().lower()]()
    except KeyError:
        raise ValueError(
            f"Unknown executor {name!r}: set OC_EXECUTOR to 'nebius' (Token Factory Sandboxes) "
            "or 'local' (subprocesses on this machine, for development)."
        ) from None
