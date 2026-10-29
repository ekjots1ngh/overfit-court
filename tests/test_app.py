"""The deployed app is what judges see: it must load, and every demo act must run."""

import importlib
import pkgutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

import overfit_court
from overfit_court import llm

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_every_module_imports():
    for module in pkgutil.iter_modules(overfit_court.__path__):
        importlib.import_module(f"overfit_court.{module.name}")


def test_the_start_here_tab_runs_all_three_acts():
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    assert not app.exception
    for key in ("act1", "act2", "act3"):
        app.button(key=key).click().run()
        assert not app.exception, key
    text = " ".join(block.value for block in app.markdown)
    assert "class='bench'" in text  # act 1's four judges
    # Act 2: statistics alone acquit the leaky draft (stamped as fooled); the court convicts.
    assert "<h4>Statistics alone</h4>" in text and ">ACQUITTED</span>" in text
    assert ">Fooled</span>" in text
    assert "<h4>With the time-travel test</h4>" in text and ">CONVICTED</span>" in text
    assert "Sealed evidence: CONFIRMED" in text  # act 3
    assert any(b.label == "Download the court record" for b in app.get("download_button"))


def test_with_a_key_model_code_only_runs_in_sandboxes(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test")  # nothing is called until a button is clicked
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception
    where = next(r for r in app.radio if r.label == "Run backtests in")
    assert where.value == "Token Factory Sandboxes" and where.disabled
    assert not any("Demo mode" in block.value for block in app.info)


def test_past_the_days_budget_the_public_app_is_in_demo_mode(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test")
    monkeypatch.setenv("OC_DAILY_BUDGET_USD", "2")
    llm.SPEND.add(2.01)
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception
    assert any("budget for live model calls" in block.value for block in app.info)
    where = next(r for r in app.radio if r.label == "Run backtests in")
    assert where.disabled  # demo mode: only library code runs, locally


def test_the_trial_tab_and_the_evidence_room():
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    next(b for b in app.button if b.label == "Begin trial").click().run()
    assert not app.exception
    next(b for b in app.button if b.label.startswith("Commit")).click().run()
    assert not app.exception
    assert any("Sealed evidence:" in block.value for block in app.markdown)
