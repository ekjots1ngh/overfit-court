"""Court records."""

from overfit_court.data import synthetic_prices
from overfit_court.pipeline import run_trial, unseal
from overfit_court.report import court_record_html, court_record_markdown


def test_records_contain_the_verdict_and_escape_model_output(settings, silent):
    trial = run_trial("momentum", synthetic_prices(seed=7, signal=0.0008), settings, log=silent)
    unseal(trial, settings=settings, log=silent)
    trial.opinion["headline"] = "<script>alert('model output')</script>"

    page = court_record_html(trial)
    assert trial.verdict in page and trial.sealed.commitment in page
    assert "<script>" not in page and "&lt;script&gt;" in page

    markdown = court_record_markdown(trial)
    assert markdown.startswith(f"# {trial.verdict}")
    assert "Sealed evidence: CONFIRMED" in markdown
