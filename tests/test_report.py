"""Court records, and the command line that writes them."""

from overfit_court import __main__ as cli
from overfit_court.data import PLANTED_SIGNAL, synthetic_prices
from overfit_court.pipeline import run_trial, unseal
from overfit_court.report import court_record_html, court_record_markdown
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
