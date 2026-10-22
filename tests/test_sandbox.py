"""The Token Factory Sandboxes executor, against a stand-in for contree-sdk."""

import json

import numpy as np
import pytest
from fake_contree import FakeContree

from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.pipeline import expand_grid
from overfit_court.sandbox import (
    REQUIREMENTS,
    LocalExecutor,
    NebiusSandboxExecutor,
    get_executor,
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


def test_a_crashing_strategy_reports_every_configuration():
    prices = synthetic_prices(n_days=300, seed=2)
    crashing = "def positions(prices, **params):\n    raise KeyError('nope')\n"
    result = NebiusSandboxExecutor(client=FakeContree()).run(crashing, prices, GRID, 5.0)
    assert result.returns.shape[1] == 0
    assert len(result.errors) == len(GRID)
    assert "KeyError" in next(iter(result.errors.values()))
    assert result.time_travel["status"] == "unverified"  # nothing ran, so nothing passed


def test_code_that_does_not_import_fails_before_any_branch():
    prices = synthetic_prices(n_days=300, seed=2)
    result = NebiusSandboxExecutor(client=FakeContree()).run("not python", prices, GRID, 5.0)
    assert result.returns.shape[1] == 0
    assert "failed to import" in next(iter(result.errors.values()))
    assert result.time_travel["status"] == "unverified"


def test_executor_names_are_checked():
    assert isinstance(get_executor(" Local "), LocalExecutor)
    assert isinstance(get_executor("nebius"), NebiusSandboxExecutor)
    with pytest.raises(ValueError, match="OC_EXECUTOR"):
        get_executor("nebuis")  # a typo must not quietly run model code on this machine
