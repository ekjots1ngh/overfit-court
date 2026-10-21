import numpy as np

from overfit_court.agents import MOCK_CODE
from overfit_court.data import synthetic_prices
from overfit_court.pipeline import expand_grid
from overfit_court.sandbox import LocalExecutor, NebiusSandboxExecutor

from fake_contree import FakeContree

GRID = expand_grid({"lookback": [20, 60, 120], "skip": [0, 5], "top_frac": [0.2, 0.3]}, 100)


def test_sandbox_executor_matches_local(tmp_path, monkeypatch):
    monkeypatch.setattr(NebiusSandboxExecutor, "CACHE", tmp_path / "img")
    monkeypatch.delenv("OC_SANDBOX_IMAGE", raising=False)
    prices = synthetic_prices(n_days=600, seed=1)
    local = LocalExecutor(n_workers=2).run(MOCK_CODE, prices, GRID, 5.0)
    fake = FakeContree()
    remote = NebiusSandboxExecutor(n_workers=4, client=fake).run(MOCK_CODE, prices, GRID, 5.0)
    assert remote.returns.shape == local.returns.shape == (600, 12)
    assert np.allclose(remote.returns.values, local.returns.values)
    assert not remote.errors
    # base -> deps checkpoint -> trial checkpoint -> 4 branches
    assert len(fake.created) == 1 + 1 + 1 + 4
    assert remote.summary()["branches"] == 4
    assert (tmp_path / "img").exists()      # deps image cached for next time


def test_sandbox_executor_reports_broken_code(tmp_path, monkeypatch):
    monkeypatch.setattr(NebiusSandboxExecutor, "CACHE", tmp_path / "img")
    prices = synthetic_prices(n_days=300, seed=2)
    bad = "def positions(prices, **p):\n    raise KeyError('nope')\n"
    res = NebiusSandboxExecutor(client=FakeContree()).run(bad, prices, GRID, 5.0)
    assert res.returns.shape[1] == 0
    assert len(res.errors) == len(GRID)
    assert "KeyError" in next(iter(res.errors.values()))


def test_sandbox_executor_reports_import_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(NebiusSandboxExecutor, "CACHE", tmp_path / "img")
    prices = synthetic_prices(n_days=300, seed=2)
    res = NebiusSandboxExecutor(client=FakeContree()).run("this is not python", prices, GRID, 5.0)
    assert res.returns.shape[1] == 0
    assert "failed to import" in next(iter(res.errors.values()))
