"""Settings: a value that is set but can't be used is an error, never a silent default."""

import pytest

from overfit_court.config import EXECUTOR_NAMES, get_settings
from overfit_court.sandbox import EXECUTORS


def test_the_executor_defaults_to_sandboxes_once_model_code_can_run(monkeypatch):
    assert get_settings().executor == "local"  # demo mode: only library code runs
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    assert get_settings().executor == "nebius"
    monkeypatch.setenv("OC_EXECUTOR", " Local ")
    assert get_settings().executor == "local"


@pytest.mark.parametrize(
    ("variable", "value", "complaint"),
    [
        ("OC_EXECUTOR", "nebuis", "OC_EXECUTOR"),
        ("OC_MAX_CONFIGS", "1", "at least 2"),
        ("OC_MAX_CONFIGS", "lots", "not a valid int"),
        ("OC_COST_BPS", "5bps", "not a valid float"),
        ("OC_COST_BPS", "-5", "at least 0"),  # a negative cost would pay strategies to trade
        ("OC_COST_BPS", "nan", "finite"),
        ("PRICE_SUPER_OUT", "inf", "finite"),
    ],
)
def test_unusable_settings_are_errors(monkeypatch, variable, value, complaint):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=complaint):
        get_settings()


def test_every_executor_name_has_an_executor():
    assert set(EXECUTOR_NAMES) == set(EXECUTORS)
