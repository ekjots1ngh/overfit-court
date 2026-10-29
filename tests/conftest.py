"""Shared fixtures. Every test runs offline: no API keys, no network, no files in the repo."""

import pytest

from overfit_court import llm
from overfit_court.config import get_settings
from overfit_court.sandbox import NebiusSandboxExecutor

LIVE_VARIABLES = (
    "NEBIUS_API_KEY",
    "TAVILY_API_KEY",
    "NEBIUS_PROJECT_ID",
    "CONTREE_PROJECT",
    "OC_EXECUTOR",
    "OC_SANDBOX_IMAGE",
    "OC_DAILY_BUDGET_USD",
    "OC_ALLOW_LOCAL",
)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Strip live credentials (even ones loaded from a developer's .env) and keep caches in tmp."""
    for name in LIVE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(NebiusSandboxExecutor, "IMAGE_CACHE", tmp_path / "sandbox_image")
    monkeypatch.setattr(llm, "SPEND", llm.Spend())  # each test starts with nothing spent


@pytest.fixture
def settings():
    """Demo-mode settings: library strategies stand in for model output."""
    s = get_settings().offline()
    s.executor = "local"
    return s


def _discard(*_args):
    pass


@pytest.fixture
def silent():
    """A log function that discards progress messages."""
    return _discard
