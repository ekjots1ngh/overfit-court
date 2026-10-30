"""Court records, and the command line that writes them."""

from overfit_court import __main__ as cli
from overfit_court.data import PLANTED_SIGNAL, synthetic_prices
from overfit_court.pipeline import run_trial, unseal
from overfit_court.report import (
    court_record_html,
    court_record_markdown,
    literature_sentence,
    prior_art_points,
)
from overfit_court.sandbox import InProcessExecutor


def test_records_contain_the_verdict_and_escape_model_output(settings, silent):
    trial = run_trial(
        "momentum",
        synthetic_prices(seed=7, signal=PLANTED_SIGNAL),
        settings=settings,
        executor=InProcessExecutor(),
        log=silent,
    )
    unseal(trial, executor=InProcessExecutor(), log=silent)
    trial.opinion["headline"] = "<script>alert('model output')</script>"

    page = court_record_html(trial)
    assert trial.verdict in page and trial.sealed.commitment in page
    assert "<script>" not in page and "&lt;script&gt;" in page

    markdown = court_record_markdown(trial)
    assert markdown.startswith(f"# {trial.verdict}")
    assert "Sealed evidence: CONFIRMED" in markdown
    assert "5 bps per unit of turnover" in markdown


def test_the_command_line_writes_its_report_even_if_the_evidence_room_fails(
    tmp_path, monkeypatch, capsys
):
    def lost_connection(trial, **kwargs):
        raise ConnectionError("the sandbox service went away")

    monkeypatch.setattr(cli, "unseal", lost_connection)
    monkeypatch.setenv("OC_MAX_CONFIGS", "8")  # a small grid, to keep the test quick
    monkeypatch.chdir(tmp_path)
    cli.trial_command(["momentum", "--unseal", "--ledger", str(tmp_path / "record")])
    assert (tmp_path / "report.md").exists() and (tmp_path / "report.json").exists()
    assert "could not be opened (ConnectionError" in capsys.readouterr().out


def test_prior_art_is_written_in_words_never_as_a_dict():
    single = {
        "published": {"lookback": 252, "skip": 21, "top_frac": 0.3},
        "params": {"lookback": 252, "skip": 21, "top_frac": 0.3},
        "n_configs": 1,
        "sharpe_annual": 1.55,
        "p_value": 0.0004,
    }
    sentence = literature_sentence(single)
    assert "(lookback 252, skip 21, top_frac 0.3), tested on its own" in sentence
    assert "p < 0.001" in sentence and "{" not in sentence
    partial = {**single, "published": {"lookback": 252, "skip": 21}, "n_configs": 3}
    partial["free"] = ["top_frac"]
    sentence = literature_sentence(partial)
    assert "fix lookback 252, skip 21" in sentence and "the best is top_frac 0.3" in sentence
    assert "Reality Check over those 3 alone" in sentence and "{" not in sentence

    prior_art = {
        "summary": "Well known.",
        "citations": [{"title": "A", "url": "https://ex.org/a", "kind": "papers"}],
        "search": {"searches": 4, "failed": 1, "cached": 2, "credits": 4, "recorded": "2026-10-30"},
        "literature_note": "The sources name lookback 12, outside the grid, so it was not tested.",
    }
    summary, sources, third = prior_art_points(prior_art)
    assert summary == (
        "Well known. (4 Tavily searches of papers and practitioner research, 4 credits, "
        "2 from today's cache, 1 failed, recorded live on 2026-10-30)"
    )
    assert sources == prior_art["citations"] and third == prior_art["literature_note"]
    assert prior_art_points({"summary": "Nothing was found."}) == ("Nothing was found.", [], None)
