"""The Token Factory Sandboxes executor, against a stand-in for contree-sdk."""

import json
import uuid

import numpy as np
import pytest
from contree_sdk.sdk.exceptions import ApiStatusCodeError, FailedOperationError
from fake_contree import FakeContree

from overfit_court import _harness
from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.pipeline import expand_grid
from overfit_court.sandbox import (
    REQUIREMENTS,
    JobOutput,
    LocalExecutor,
    NebiusSandboxExecutor,
    get_executor,
    merge,
)

MOMENTUM = FAMILIES["momentum"].code
GRID = expand_grid({"lookback": [20, 60, 120], "skip": [0, 5], "top_frac": [0.2, 0.3]}, 100)


def test_sandbox_results_match_a_local_run():
    prices = synthetic_prices(n_days=600, seed=1)
    local = LocalExecutor(n_chunks=2).run(MOMENTUM, prices, GRID, 5.0)
    fake = FakeContree()
    remote = NebiusSandboxExecutor(n_chunks=4, client=fake).run(MOMENTUM, prices, GRID, 5.0)
    assert remote.returns.shape == local.returns.shape == (600, 12)
    assert np.allclose(remote.returns.to_numpy(), local.returns.to_numpy())
    assert not remote.errors
    assert remote.time_travel == local.time_travel and remote.time_travel["status"] == "passed"
    # base image -> dependencies checkpoint -> trial checkpoint -> 4 parallel branches
    assert len(fake.created) == 1 + 1 + 1 + 4
    assert remote.summary()["branches"] == 4
    assert [step["step"] for step in remote.setup] == ["dependencies image", "trial checkpoint"]


def test_the_dependencies_image_is_reused_only_with_the_same_versions():
    prices = synthetic_prices(n_days=300, seed=1)
    cache = NebiusSandboxExecutor.IMAGE_CACHE
    NebiusSandboxExecutor(client=FakeContree()).run(MOMENTUM, prices, GRID, 5.0)
    assert json.loads(cache.read_text())["requirements"] == REQUIREMENTS

    reused = NebiusSandboxExecutor(client=FakeContree()).run(MOMENTUM, prices, GRID, 5.0)
    assert [step["step"] for step in reused.setup] == ["trial checkpoint"]

    cache.write_text(json.dumps({"image": "old", "requirements": "numpy==1.0 pandas==1.0"}))
    rebuilt = NebiusSandboxExecutor(client=FakeContree()).run(MOMENTUM, prices, GRID, 5.0)
    assert rebuilt.setup[0]["step"] == "dependencies image"


def test_a_cached_image_the_service_no_longer_has_is_built_again():
    prices = synthetic_prices(n_days=300, seed=1)
    cache = NebiusSandboxExecutor.IMAGE_CACHE
    cache.write_text(json.dumps({"image": "gone", "requirements": REQUIREMENTS}))
    fake = FakeContree(forgotten={"gone"})
    result = NebiusSandboxExecutor(client=fake).run(MOMENTUM, prices, GRID, 5.0)
    assert not result.errors and result.time_travel["status"] == "passed"
    assert [step["step"] for step in result.setup] == ["dependencies image", "trial checkpoint"]
    assert json.loads(cache.read_text())["image"] != "gone"


def test_a_read_only_disk_only_costs_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(NebiusSandboxExecutor, "IMAGE_CACHE", tmp_path / "missing" / "cache")
    prices = synthetic_prices(n_days=300, seed=1)
    result = NebiusSandboxExecutor(client=FakeContree()).run(MOMENTUM, prices, GRID, 5.0)
    assert not result.errors and result.time_travel["status"] == "passed"


def test_local_runs_never_see_credentials(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "v1.secret")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "secret")
    snoops = (
        "import os\n\n"
        "def positions(prices, **params):\n"
        "    found = [k for k in os.environ if 'KEY' in k or 'TOKEN' in k or 'NEBIUS' in k]\n"
        "    if found:\n"
        "        raise RuntimeError(f'saw {found}')\n"
        "    return prices * 0\n"
    )
    prices = synthetic_prices(n_days=300, seed=2)
    result = LocalExecutor(n_chunks=1).run(snoops, prices, GRID, 5.0)
    assert not result.errors


def test_a_crashing_strategy_reports_every_configuration():
    prices = synthetic_prices(n_days=300, seed=2)
    crashing = "def positions(prices, **params):\n    raise KeyError('nope')\n"
    result = NebiusSandboxExecutor(client=FakeContree()).run(crashing, prices, GRID, 5.0)
    assert result.returns.shape[1] == 0
    assert len(result.errors) == len(GRID)
    assert "KeyError" in next(iter(result.errors.values()))
    assert result.time_travel["status"] == "unverified"  # nothing ran, so nothing passed


def test_code_that_is_not_python_fails_before_any_branch():
    prices = synthetic_prices(n_days=300, seed=2)
    fake = FakeContree()
    result = NebiusSandboxExecutor(client=fake).run("def positions(:\n", prices, GRID, 5.0)
    assert result.returns.shape[1] == 0
    assert "not valid Python" in next(iter(result.errors.values()))
    assert result.time_travel["status"] == "unverified"
    assert len(fake.created) == 3  # base, dependencies, checkpoint: no branch ever ran


def test_strategy_code_never_runs_where_the_data_file_is():
    """Module-level code that copies the data file finds nothing: the checkpoint only parses
    the code, and each branch deletes the file before loading the strategy."""
    copies_the_data = (
        "import shutil\n"
        "shutil.copy('prices.csv', 'stash.csv')\n\n"
        "def positions(prices, **params):\n"
        "    return prices * 0\n"
    )
    prices = synthetic_prices(n_days=300, seed=2)
    fake = FakeContree()
    result = NebiusSandboxExecutor(n_chunks=2, client=fake).run(copies_the_data, prices, GRID, 5.0)
    assert all("FileNotFoundError" in error for error in result.errors.values())
    assert not any("/workspace/stash.csv" in image.files for image in fake.created)


def test_the_evidence_room_shares_its_checks_across_branches():
    """One configuration, hundreds of cuts: each branch checks every n-th cut."""
    prices = synthetic_prices(n_days=600, seed=1)
    sealed_days = list(range(450, 600))
    fake = FakeContree()
    one = {"committed": {"lookback": 60, "skip": 5, "top_frac": 0.3}}
    result = NebiusSandboxExecutor(n_chunks=4, client=fake).run(
        MOMENTUM, prices, one, 5.0, extra_cuts=sealed_days
    )
    cuts = _harness.cut_points(600, sealed_days)
    assert len(fake.created) == 3 + 4  # four branches for the one configuration
    report = result.time_travel
    assert report["status"] == "passed" and report["configs"] == 1
    assert report["checks"] == report["expected_checks"] == len(cuts)
    assert list(result.returns.columns) == ["committed"]


def test_copies_of_one_configuration_must_agree():
    prices = synthetic_prices(n_days=300, seed=1)
    index = prices.index
    agree = np.ones((300, 1))
    report = _harness.finish(_harness.new_report())
    outputs = [JobOutput(["c"], agree, {}, report), JobOutput(["c"], agree * 2, {}, report)]
    merged = merge(outputs, {"c": {}}, index, time_travel=True)
    assert merged.time_travel["status"] == "unverified"
    assert "isn't deterministic" in merged.time_travel["error"]


def test_a_runner_that_exits_without_results_is_the_codes_fault():
    exits = "import os\n\ndef positions(prices, **params):\n    os._exit(0)\n"
    prices = synthetic_prices(n_days=300, seed=2)
    result = LocalExecutor(n_chunks=1).run(exits, prices, GRID, 5.0)
    assert all("without writing its results" in e for e in result.errors.values())
    assert result.time_travel["lost_configs"] == 0


def test_executor_names_are_checked():
    assert isinstance(get_executor(" Local "), LocalExecutor)
    assert isinstance(get_executor("nebius"), NebiusSandboxExecutor)
    with pytest.raises(ValueError, match="OC_EXECUTOR"):
        get_executor("nebuis")  # a typo must not quietly run model code on this machine


class _AccountInfo:
    """A FakeContree that also answers whoami, as the real client does."""

    def __init__(self, limits=None, status=None):
        self.limits, self.status = limits or {}, status

    async def __call__(self, refresh=False):
        if self.status:
            raise ApiStatusCodeError(status=self.status, error="no")
        return type("WhoAmI", (), {"limits": self.limits})()


def test_the_time_limit_fits_the_accounts_and_bad_credentials_fail_first():
    prices = synthetic_prices(n_days=300, seed=1)
    fake = FakeContree()
    fake.get_token_info = _AccountInfo({"instance_max_timeout": 600})
    executor = NebiusSandboxExecutor(client=fake)
    assert not executor.run(MOMENTUM, prices, GRID, 5.0).errors
    assert executor.timeout_s == 600
    refused = FakeContree()
    refused.get_token_info = _AccountInfo(status=401)
    with pytest.raises(RuntimeError, match="refused the credentials"):
        NebiusSandboxExecutor(client=refused).run(MOMENTUM, prices, GRID, 5.0)
    assert not refused.created  # nothing was uploaded or run


def test_a_run_the_service_stops_at_its_time_limit_is_the_codes_fault(monkeypatch):
    def stopped(self, shell, uploads):
        if "runner.py" in shell:
            raise FailedOperationError(operation_uuid=uuid.uuid4(), error="instance timed out")

    monkeypatch.setattr(FakeContree, "lose_if_unlucky", stopped)
    prices = synthetic_prices(n_days=300, seed=1)
    result = NebiusSandboxExecutor(n_chunks=2, client=FakeContree()).run(
        MOMENTUM, prices, GRID, 5.0
    )
    assert all("timed out" in e for e in result.errors.values())
    assert result.time_travel["lost_configs"] == 0  # not a mistrial: it goes back for repair
