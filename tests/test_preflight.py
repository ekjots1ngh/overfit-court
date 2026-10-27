"""The pre-flight check, run against the fakes: it must work before the first live run."""

import pytest
from fake_contree import FakeContree
from fake_token_factory import FakeTokenFactory, bad_request, response

from overfit_court import preflight
from overfit_court.llm import LLM
from overfit_court.sandbox import NebiusSandboxExecutor


def test_with_no_keys_everything_is_skipped(capsys):
    preflight.main()
    out = capsys.readouterr().out
    assert "[SKIP] Token Factory and Sandboxes" in out
    assert "[SKIP] Tavily" in out
    assert "[FAIL]" not in out


def test_models_calls_and_refused_request_parts(settings, capsys):
    settings.nebius_api_key = "test"
    models = list(settings.models.values())

    def token_factory(request):  # Super refuses JSON mode; the others accept everything
        if request["model"] == settings.models["super"] and "response_format" in request:
            return bad_request()
        return response('{"ready": true}')

    llm = LLM(settings, client=FakeTokenFactory(token_factory, models=models))
    assert preflight.check_models(settings, llm)
    preflight.check_calls(llm)
    out = capsys.readouterr().out
    assert out.count("[PASS]") == 1 + 3 + 3  # key, three ids, three calls
    assert "[SKIP] super refused JSON mode" in out and "[FAIL]" not in out
    assert "nano call, thinking off" in out and "ultra call, thinking on" in out


def test_closing_advice(monkeypatch, capsys):
    preflight.main()
    assert "Ready. Next: streamlit run app.py" in capsys.readouterr().out
    monkeypatch.setattr(preflight, "check_tavily", lambda s: preflight.report("FAIL", "Tavily"))
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    preflight.main()
    assert "1 check(s) failed" in capsys.readouterr().out


def test_a_missing_model_id_names_the_ones_available(settings, capsys):
    settings.nebius_api_key = "test"
    llm = LLM(settings, client=FakeTokenFactory([], models=["nvidia/nemotron-other"]))
    assert not preflight.check_models(settings, llm)
    assert "nvidia/nemotron-other" in capsys.readouterr().out


@pytest.mark.parametrize("project", [None, "test-project"])
def test_sandboxes_run_the_real_code_path(monkeypatch, capsys, project):
    if project:
        monkeypatch.setenv("NEBIUS_PROJECT_ID", project)
    monkeypatch.setattr(NebiusSandboxExecutor, "_make_client", lambda self: FakeContree())
    preflight.check_sandboxes()
    out = capsys.readouterr().out
    if project is None:
        assert "[FAIL] Token Factory Sandboxes" in out and "NEBIUS_PROJECT_ID" in out
    else:
        assert "[FAIL]" not in out
        assert "branches in parallel, results identical to a local run" in out
        assert "time-travel test ran in every branch" in out
