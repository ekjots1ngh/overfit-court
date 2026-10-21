"""A shareable court record: one self-contained HTML page per trial.

Something a student can attach to a strategy write-up or a job application:
the verdict, the numbers behind it, how many configurations were tried, the
time-travel test, and the sealed-evidence commitment hash.
"""
from __future__ import annotations

import hashlib
import html
from datetime import datetime, timezone

from .stats import fmt_prob

COLOURS = {"CONVICTED": "#b91c1c", "HUNG JURY": "#b45309", "ACQUITTED": "#15803d",
           "REFUTED": "#b91c1c", "INCONCLUSIVE": "#b45309", "CONFIRMED": "#15803d"}


def _row(k, v):
    return f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"


def court_record_html(trial) -> str:
    e = trial.evidence()
    v = trial.verdict
    yrs = e["years_of_live_trading_to_prove_it"]
    code_hash = hashlib.sha256(trial.code.encode()).hexdigest()[:16]
    rows = [
        ("Idea", trial.idea),
        ("Strategy", trial.spec.get("name", "")),
        ("Configurations tried (this trial / whole record)",
         f"{e['n_configs']} / {e['n_configs_total_on_this_data']}"),
        ("Distinct bets (random matrix theory)", f"{e['distinct_bets_total']:.1f}"),
        ("Best backtest Sharpe (open evidence)", f"{e['best_sharpe_annual']:.2f}"),
        ("Best Sharpe luck alone would give", f"{e['luck_sharpe_annual']:.2f}"),
        ("Probability the edge is real", fmt_prob(e['p_real'])),
        ("In-sample winner loses out of sample", f"{e['prob_oos_loss']:.0%} of splits"),
        ("Walk-forward Sharpe", f"{e['walk_forward_sharpe']:.2f}"),
        ("Years of live trading to prove it", "never at this Sharpe" if yrs is None else yrs),
        ("Time-travel test", e["time_travel"]),
        ("Failed checks", "; ".join(e["failed_checks"]) or "none"),
        ("Best parameters", e["best_params"]),
        ("Strategy code hash", code_hash),
    ]
    sealed = ""
    if trial.sealed:
        s = trial.sealed
        sealed = (f"<h2>Sealed evidence: <span style='color:{COLOURS[s.verdict]}'>{s.verdict}"
                  f"</span></h2><table>" +
                  _row("Commitment (hash of code + parameters + data)", s.commitment) +
                  _row("Sealed period", f"{s.sealed_days} days from {s.sealed_from}") +
                  _row("Sharpe on open evidence", f"{s.open_sharpe_annual:.2f}") +
                  _row("Sharpe on sealed evidence", f"{s.sealed_sharpe_annual:.2f}") +
                  _row("Probability real (corrected for earlier openings)", fmt_prob(s.p_real)) +
                  _row("Earlier openings of this seal", s.times_opened_before) + "</table>")
    else:
        sealed = "<h2>Sealed evidence</h2><p>Still sealed: no strategy has been tested on it.</p>"
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in trial.opinion.get("reasoning", []))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Overfit Court record: {html.escape(v)}</title>
<style>
:root {{ --fg:#111827; --muted:#6b7280; --bg:#ffffff; --line:#e5e7eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#f3f4f6; --muted:#9ca3af; --bg:#111827; --line:#374151; }} }}
body {{ font: 16px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; color: var(--fg);
       background: var(--bg); max-width: 760px; margin: 2rem auto; padding: 0 16px; }}
h1 {{ margin: 0 0 .25rem; }} .verdict {{ color: {COLOURS.get(v, '#111')}; letter-spacing: .04em; }}
.sub {{ color: var(--muted); margin: 0 0 1.5rem; }}
table {{ width: 100%; border-collapse: collapse; margin: .5rem 0 1.5rem; }}
th, td {{ text-align: left; padding: .45rem .5rem; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-weight: 500; color: var(--muted); width: 45%; }}
footer {{ color: var(--muted); font-size: .85rem; margin-top: 2rem; }}
</style></head><body>
<h1>⚖️ <span class="verdict">{html.escape(v)}</span></h1>
<p class="sub">{html.escape(trial.opinion.get('headline', ''))}</p>
<table>{''.join(_row(k, val) for k, val in rows)}</table>
<h2>Reasoning</h2><ul>{reasons}</ul>
{sealed}
<footer>Issued by Overfit Court on {stamp}. Data fingerprint {html.escape(trial.record['dataset'])},
trial {e['trial_number_on_this_data']} on this data. The verdict is decided by the statistics;
the language model only explains it.</footer>
</body></html>"""
