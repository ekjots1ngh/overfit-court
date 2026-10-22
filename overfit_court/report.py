"""Court records: a trial written up as a shareable HTML page, or as Markdown for the CLI.

The HTML record is meant to be attached to a strategy write-up or a job
application: the verdict, the numbers behind it, how many configurations were
tried, the time-travel test, and the sealed-evidence commitment hash.
"""

from __future__ import annotations

import hashlib
import html
from datetime import datetime, timezone

from .stats import (
    ACQUITTED,
    CONFIRMED,
    CONVICTED,
    HUNG_JURY,
    INCONCLUSIVE,
    REFUTED,
    fmt_p,
    fmt_years,
)

VERDICT_COLOURS = {
    CONVICTED: "#b91c1c",
    HUNG_JURY: "#b45309",
    ACQUITTED: "#15803d",
    REFUTED: "#b91c1c",
    INCONCLUSIVE: "#b45309",
    CONFIRMED: "#15803d",
}


def _summary_rows(trial) -> list[tuple[str, str]]:
    """The headline numbers, shared by the HTML and Markdown records."""
    e = trial.evidence()
    years = e["years_of_live_trading_to_prove_it"]
    return [
        ("Idea", trial.idea),
        ("Strategy", trial.spec.get("name", "")),
        ("Data", trial.data_source),
        ("Trial number on this data", str(e["trial_number_on_this_data"])),
        (
            "Configurations: this trial / distinct on this data",
            f"{e['n_configs']} / {e['n_distinct_configs_on_this_data']}",
        ),
        (
            "Distinct bets, random matrix theory (this trial / whole record)",
            f"{e['distinct_bets_this_trial']:.1f} / {e['distinct_bets_total']:.1f}",
        ),
        ("Best backtest Sharpe (open evidence)", f"{e['best_sharpe_annual']:.2f}"),
        ("Best Sharpe luck alone would give", f"{e['luck_sharpe_annual']:.2f}"),
        ("Reality Check against everything tried", fmt_p(e["p_value"])),
        ("In-sample winner loses out of sample", f"{e['prob_oos_loss']:.0%} of splits"),
        ("Walk-forward Sharpe", f"{e['walk_forward_sharpe']:.2f}"),
        (
            "Years of live trading to prove it (at the walk-forward Sharpe)",
            "never: it loses money walk-forward" if years is None else fmt_years(years),
        ),
        ("Trading cost", f"{trial.cost_bps:g} bps per unit of turnover"),
        ("Time-travel test", e["time_travel"]),
        ("Failed checks", "; ".join(e["failed_checks"]) or "none"),
        ("Best parameters", str(e["best_params"])),
        ("Strategy code (SHA-256)", hashlib.sha256(trial.code.encode()).hexdigest()),
    ]


# An opening counts even when it gives no valid result, so the record says so.
OPENED_NO_RESULT = "Opened, with no valid result (the opening still counts)."


def _sealed_rows(sealed) -> list[tuple[str, str]]:
    return [
        ("Commitment (SHA-256 of code, parameters, data, costs, harness)", sealed.commitment),
        ("Sealed period", f"{sealed.sealed_days} days from {sealed.sealed_from}"),
        ("Sharpe on open evidence", f"{sealed.open_sharpe_annual:.2f}"),
        ("Sharpe on sealed evidence", f"{sealed.sealed_sharpe_annual:.2f}"),
        ("Test of Sharpe above zero (corrected for earlier openings)", fmt_p(sealed.p_value)),
        ("Earlier openings of this seal", str(sealed.times_opened_before)),
    ]


# --- Markdown ---
def court_record_markdown(trial) -> str:
    """The record as Markdown, as printed by the command line."""
    lines = [
        f"# {trial.verdict}: {trial.spec.get('name', 'strategy')}",
        f"_{trial.opinion.get('headline', '')}_",
        "",
        "| | |",
        "|---|---|",
        *[f"| {label} | {value} |" for label, value in _summary_rows(trial)],
        f"| Model spend | ${sum(c['cost_usd'] for c in trial.cost_table):.4f} |",
    ]
    if trial.sealed:
        lines += ["", f"**Sealed evidence: {trial.sealed.verdict}**", "", "| | |", "|---|---|"]
        lines += [f"| {label} | {value} |" for label, value in _sealed_rows(trial.sealed)]
    elif trial.failed_opening:
        lines += ["", f"**Sealed evidence:** {OPENED_NO_RESULT} {trial.failed_opening}"]
    lines += ["", "**Reasoning**", *[f"- {r}" for r in trial.opinion.get("reasoning", [])]]
    lines += [
        "",
        "**What would change the verdict**",
        *[f"- {r}" for r in trial.opinion.get("what_would_change_my_mind", [])],
    ]
    if trial.opinion.get("note"):
        lines += ["", f"_{trial.opinion['note']}_"]
    if trial.prior_art.get("summary"):
        lines += ["", f"**Prior art:** {trial.prior_art['summary']}"]
    return "\n".join(lines)


# --- HTML ---
def _table(rows: list[tuple[str, str]]) -> str:
    cells = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return f"<table>{cells}</table>"


_STYLE = """
:root { --fg: #111827; --muted: #6b7280; --bg: #ffffff; --line: #e5e7eb; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #f3f4f6; --muted: #9ca3af; --bg: #111827; --line: #374151; }
}
body { font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg);
       background: var(--bg); max-width: 760px; margin: 2rem auto; padding: 0 16px; }
h1 { margin: 0 0 .25rem; }
.sub { color: var(--muted); margin: 0 0 1.5rem; }
table { width: 100%; border-collapse: collapse; margin: .5rem 0 1.5rem; }
th, td { text-align: left; padding: .45rem .5rem; border-bottom: 1px solid var(--line);
         vertical-align: top; overflow-wrap: anywhere; }
th { font-weight: 500; color: var(--muted); width: 45%; }
footer { color: var(--muted); font-size: .85rem; margin-top: 2rem; }
"""


def court_record_html(trial) -> str:
    """A self-contained HTML page (no external assets) recording the trial."""
    e = trial.evidence()
    verdict = trial.verdict
    colour = VERDICT_COLOURS.get(verdict, "inherit")
    if trial.sealed:
        sealed_colour = VERDICT_COLOURS[trial.sealed.verdict]
        sealed = (
            f"<h2>Sealed evidence: <span style='color:{sealed_colour}'>"
            f"{trial.sealed.verdict}</span></h2>{_table(_sealed_rows(trial.sealed))}"
        )
    else:
        sealed = "<h2>Sealed evidence</h2><p>Still sealed: no strategy has been tested on it.</p>"
        if trial.failed_opening:
            why = html.escape(trial.failed_opening)
            sealed = f"<h2>Sealed evidence</h2><p>{OPENED_NO_RESULT} {why}</p>"
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in trial.opinion.get("reasoning", []))
    note = trial.opinion.get("note")  # set when the court's own opinion replaced the judge's
    note = f"<p class='sub'>{html.escape(note)}</p>" if note else ""
    issued = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Overfit Court record: {html.escape(verdict)}</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>⚖️ <span style="color:{colour}; letter-spacing:.04em">{html.escape(verdict)}</span></h1>
<p class="sub">{html.escape(trial.opinion.get("headline", ""))}</p>
{_table(_summary_rows(trial))}
<h2>Reasoning</h2>
<ul>{reasons}</ul>
{note}
{sealed}
<footer>Issued by Overfit Court on {issued}. Data fingerprint
{html.escape(trial.record["dataset"])}, trial {e["trial_number_on_this_data"]} on this data.
The verdict is decided by the statistics; the language model only explains it.</footer>
</body>
</html>"""
