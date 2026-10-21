"""The deployed app is what judges see: it must load and every act must run."""
import importlib
import pkgutil
from pathlib import Path

import overfit_court
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_every_module_imports():
    for m in pkgutil.iter_modules(overfit_court.__path__):
        importlib.import_module(f"overfit_court.{m.name}")


def test_start_here_tab_runs_all_three_acts(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    for key in ("act1", "act2", "act3"):
        at.button(key=key).click().run()
        assert not at.exception, key
    text = " ".join(m.value for m in at.markdown)
    assert "Statistics alone: ACQUITTED" in text
    assert "With the time-travel test: CONVICTED" in text
    assert "Sealed evidence: CONFIRMED" in text
    assert any(d.label == "Download the court record" for d in at.get("download_button"))
