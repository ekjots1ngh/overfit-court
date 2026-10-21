from overfit_court.agents import audit
from overfit_court.config import get_settings
from overfit_court.data import synthetic_prices
from overfit_court.library import FAMILIES
from overfit_court.llm import LLM
from overfit_court.pipeline import expand_grid, run_trial
from overfit_court.sandbox import InProcessExecutor, LocalExecutor


def _settings():
    s = get_settings()
    s.nebius_api_key = None
    s.tavily_api_key = None
    s.executor = "local"
    return s


def test_expand_grid_caps_size():
    assert len(expand_grid({"a": list(range(20)), "b": list(range(20))}, 50)) == 50


def test_every_library_family_runs_and_passes_audit():
    px = synthetic_prices(n_days=400, seed=1)
    llm = LLM(_settings())
    for f in FAMILIES.values():
        res = InProcessExecutor().run(f.code, px, expand_grid(f.param_grid, 200), 5.0)
        assert not res.errors, f.name
        assert audit(llm, f.code)["risk"] == "low", f.name


def test_noise_is_not_acquitted_end_to_end():
    t = run_trial("momentum", synthetic_prices(seed=3), _settings(), log=lambda *a: None)
    assert t.verdict != "ACQUITTED"
    assert "Distinct bets" in t.to_markdown()


def test_strong_edge_is_acquitted_through_the_subprocess_harness():
    t = run_trial("momentum", synthetic_prices(seed=7, signal=0.0008), _settings(),
                  executor=LocalExecutor(), log=lambda *a: None)
    assert t.verdict == "ACQUITTED"
