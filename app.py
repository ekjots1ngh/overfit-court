"""Overfit Court, the web app:  streamlit run app.py

Layout
    Sidebar                 the evidence (which prices), the procedure (grid size,
                            costs, where backtests run) and the court record
    Start here              three one-click demo acts (overfit_court/stories.py)
    Put an idea on trial    your idea, on your data
    Fishing expedition      send a fisherman out on your data
    How it works

The look: ledger paper and banker's-lamp green (.streamlit/config.toml), with
one bold element, the verdict stamp, which lands when a ruling is made. The
only other motion answers a click: the fisherman's search replays on the chart,
and each sandbox branch fills in with its configurations. Both respect the
visitor's reduced-motion setting.

Each visitor gets their own court record in st.session_state. Text that comes
from a model or a user is HTML-escaped before it is rendered (see `bullets`,
`paragraph`, `ruling` and `stamp`), and every action shows a readable error
instead of a stack trace if it fails.
"""

from __future__ import annotations

import difflib
import html
import math
import re
import time
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from overfit_court import llm, stories
from overfit_court.config import Settings, get_settings
from overfit_court.data import PLANTED_SIGNAL, load_csv, synthetic_prices
from overfit_court.fishing import NemotronFisherman, ScriptedFisherman, run_expedition
from overfit_court.ledger import Ledger, record_key
from overfit_court.llm import LLM
from overfit_court.pipeline import SEALED_FRACTION, run_trial, unseal
from overfit_court.report import OPENED_NO_RESULT, court_record_html, prior_art_points
from overfit_court.sandbox import InProcessExecutor, LocalExecutor, NebiusSandboxExecutor
from overfit_court.stats import (
    ACQUITTED,
    CONFIRMED,
    CONVICTED,
    HUNG_JURY,
    INCONCLUSIVE,
    P_VALUE_TO_ACQUIT,
    REFUTED,
    fmt_p,
    fmt_years,
)

DEFAULT_SEED = 22  # a pure-noise market on which a quick trial and a fishing trip both mislead
INK, MUTED, GREEN, RED, OCHRE, BLUE, LUCK = (
    "#1F2A24",
    "#5C6B63",
    "#1F5940",
    "#B42318",
    "#9A6700",
    "#2E4A7D",
    "#8C9A92",
)

# Rubber-stamp ink: fractal noise turned into a mask that is mostly solid, with
# the speckled gaps a real stamp leaves.
INK_MASK = (
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' "
    "height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' "
    "baseFrequency='.85' numOctaves='2' seed='4'/%3E%3CfeColorMatrix values='0 0 0 0 0 "
    "0 0 0 0 0 0 0 0 0 0 -4 0 0 0 3.3'/%3E%3C/filter%3E%3Crect width='100%25' "
    "height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")"
)

CSS = f"""
<style>
:root {{
  --ink: {INK}; --muted: {MUTED}; --green: {GREEN}; --red: {RED}; --ochre: {OCHRE};
  --luck: {LUCK};
}}
.hero h1 {{ font-size: 3.1rem; letter-spacing: -.025em; line-height: 1.05; margin: 0 0 .5rem; }}
.hero p {{ font-size: 1.12rem; max-width: 42rem; margin: 0 0 .4rem; }}
.hero .define {{ font-size: .92rem; color: var(--muted); max-width: 46rem; }}

/* The verdict stamp: the one bold thing on the page. */
.stamp {{
  --tilt: -5deg; position: relative; display: inline-block; color: var(--tone);
  font-weight: 800; text-transform: uppercase; letter-spacing: .09em; line-height: 1;
  white-space: nowrap; padding: .42em .7em .36em; border: .2em double currentColor;
  border-radius: .28em; transform: rotate(var(--tilt));
  -webkit-mask-image: {INK_MASK}; mask-image: {INK_MASK};
  -webkit-mask-size: 240px; mask-size: 240px;
  animation: stamp-down .5s cubic-bezier(.2, .8, .3, 1.25) both;
}}
.stamp.big {{ font-size: 2rem; }}
.stamp.small {{ font-size: .8rem; border-width: .26em; }}
.stamp.good {{ --tone: var(--green); }}
.stamp.bad {{ --tone: var(--red); }}
.stamp.unsure {{ --tone: var(--ochre); }}
.stamp.void {{ --tone: var(--luck); }}
.stamps {{ position: relative; display: inline-block; }}
.stamps .over {{ position: absolute; left: 44%; top: 38%; --tilt: 7deg; animation-delay: .7s; }}
@keyframes stamp-down {{
  0% {{ opacity: 0; transform: rotate(var(--tilt)) scale(2.4); }}
  55% {{ opacity: 1; transform: rotate(var(--tilt)) scale(.93); }}
  100% {{ opacity: 1; transform: rotate(var(--tilt)) scale(1); }}
}}

.ruling {{ display: flex; align-items: center; gap: 1.6rem; flex-wrap: wrap;
           margin: .9rem 0 1.2rem; }}
.ruling .says {{ flex: 1 1 18rem; min-width: 0; }}
.ruling .says h4 {{ margin: 0 0 .25rem; padding: 0; font-size: 1.05rem; font-weight: 700; }}
.ruling .says p {{ margin: 0; max-width: 44rem; }}

/* Four judges of one claim, side by side. */
.bench {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
          gap: 1.4rem; margin: 1rem 0 .6rem; }}
.bench > div {{ border-top: 3px solid var(--tone); padding-top: .65rem; }}
.bench .who {{ font-weight: 700; }}
.bench .stamp {{ margin: .65rem 0 .6rem .1rem; }}
.bench .why {{ color: var(--muted); font-size: .86rem; margin-top: .3rem; }}
.bench .good {{ --tone: var(--green); }}
.bench .bad {{ --tone: var(--red); }}

/* Sandbox branches: one square per configuration. */
.run {{ display: flex; flex-wrap: wrap; gap: .9rem 1.4rem; margin: .5rem 0 .1rem; }}
.run .cells {{ display: grid; grid-template-columns: repeat(var(--cols), 11px); gap: 3px; }}
.run i, .key i {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px;
                  background: var(--green); }}
.run i {{ animation: cell-in .22s ease-out both; animation-delay: calc(var(--i) * 28ms); }}
.run i.leak, .key i.leak {{ background: var(--red); }}
.run i.crash, .key i.crash {{ background: transparent; box-shadow: inset 0 0 0 1px var(--luck); }}
.cap {{ font-size: .8rem; color: var(--muted); margin: .35rem 0 0; line-height: 1.3; }}
.key {{ font-size: .8rem; color: var(--muted); margin: .6rem 0 .9rem; }}
.key i {{ vertical-align: -1px; margin: 0 .3rem 0 .9rem; }}
.key i:first-child {{ margin-left: 0; }}
@keyframes cell-in {{ from {{ opacity: 0; transform: scale(.2); }} to {{ opacity: 1; }} }}

.prior {{ max-width: 52rem; }}
.prior .stamp {{ margin: .3rem 0 .8rem .1rem; }}
.prior .kind {{ color: var(--muted); font-size: .82rem; }}
.hood {{ font-size: .84rem; color: var(--muted); max-width: 52rem; margin: .5rem 0 1.6rem; }}
.record {{ font-size: .9rem; }}
@media (prefers-reduced-motion: reduce) {{
  .stamp, .run i {{ animation: none; }}
}}
</style>
"""

# How each verdict is coloured.
TONE = {
    ACQUITTED: "good",
    CONFIRMED: "good",
    HUNG_JURY: "unsure",
    INCONCLUSIVE: "unsure",
    CONVICTED: "bad",
    REFUTED: "bad",
}

SHARPE = (
    "A Sharpe ratio is a strategy's yearly return divided by how much that return swings. "
    "Above 1 is what most funds hope for. Try enough strategies on random prices, though, "
    "and luck alone gets there too."
)

HOW_IT_WORKS = """
**1. The model plans and codes; the harness keeps it honest.** Nemotron 3 Ultra turns an idea into
a parameter grid, Nemotron 3 Super writes the strategy, Nemotron 3 Nano audits it. The harness
applies the one-day lag and trading costs itself, so generated code can't trade on today's close.
Tavily searches papers and practitioner research for the idea, and if they state its parameters,
that one configuration is also tested on its own: parameters published before your data existed
are a pre-registration.

**2. The time-travel test.** Every configuration is also re-run on history cut short at up to
fourteen points, from the first two days to the first 95%, and every past position must stay
exactly the same. If a position on a past date moves when later data is added, the code read the
future. That catches what code review and statistics both miss: means or standard deviations over
the whole series, negative shifts, back-fills, centred windows. A leak goes back to Nemotron Super
with the exact dates, for repair, and if a repair doesn't fix it, the last try goes to Nemotron
Ultra. The test fails closed: code that crashes on shorter history, or answers differently when
run twice on the same data, can't be verified, and is treated like a leak.

**3. Research happens in sandboxes that can't see the future.** Only the open 75% of the data is
uploaded to Token Factory Sandboxes. Every chunk of the grid is its own branch of one checkpoint.

**4. The court remembers.** Every configuration ever tried on this data is kept. The Reality Check
asks how often zero-edge versions of all of them would do as well as your pick, using a block
bootstrap that keeps fat tails, volatility clustering and the correlation between strategies.
That share is its p-value. Re-running the same grid adds nothing; genuinely new ideas raise the bar.

**5. Random matrix theory counts your real bets.** The eigenvalues of the correlation matrix of
everything tried give the number of independent directions the strategies move in. Unrelated
strategies look a little correlated by chance, more so when they share volatility regimes, so that
part is measured from the data and removed first.

**6. The statistics rule.** Convicted if it reads the future (or can't be shown not to), if luck
explains the result, if the choice loses money when made on past data and traded on the next
period, or if the in-sample winner loses money on unseen data in most splits. Acquitted only when
everything clearly passes. The judge model explains the ruling and can't change it.

**7. The evidence room.** Commit to one strategy and it gets one test on the sealed 25%, in a fresh
run. The commitment, a SHA-256 hash of the code, parameters, data, trading cost and harness, is
recorded before the seal is opened. Every opening is counted, and repeat openings are corrected for.
The time-travel test runs again there with a cut at every sealed day, so even code that cheats
for a few days, once it can see sealed dates, is caught.

**What it can't do.** A model may have memorised what real markets did before its training
cutoff, so on real data the sealed test is only truly blind for dates after that. And the app's
court record lasts for your session; the command line keeps a permanent one.
"""


# --- small pieces ---
def _silent(*_args) -> None:
    """A log function that discards progress messages."""


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def stamp(word: str, tone: str, size: str = "big", extra: str = "") -> str:
    """A rubber stamp reading `word` (escaped), in the colour of `tone`."""
    return f"<span class='stamp {size} {tone} {extra}'>{html.escape(word)}</span>"


def ruling(verdict: str, heading: str, text: str) -> None:
    """A verdict stamp, with what it rules on and why."""
    html_block(
        f"<div class='ruling'>{stamp(verdict, TONE.get(verdict, 'unsure'))}"
        f"<div class='says'><h4>{html.escape(heading)}</h4><p>{html.escape(text)}</p></div></div>"
    )


def paragraph(label: str, text: str) -> None:
    """A bold label followed by text that may come from a model."""
    html_block(f"<p><b>{html.escape(label)}</b> {html.escape(text)}</p>")


def bullets(items) -> None:
    """A bulleted list of points that may come from a model."""
    html_block("<ul>" + "".join(f"<li>{html.escape(str(item))}</li>" for item in items) + "</ul>")


def hood(text: str) -> None:
    """One plain line on what ran underneath: models, branches, checks, cost."""
    html_block(f"<p class='hood'>{html.escape(text)}</p>")


def plain(text) -> str:
    """Text made safe for Streamlit's Markdown, so it shows exactly as written.

    Progress lines and error messages can quote model output, which must never
    turn into links, images, formatting, maths ($...$) or emoji (:name:). Every
    character Markdown or Streamlit treats specially is escaped.
    """
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$:])", r"\\\1", str(text))


def say(line) -> None:
    """A progress line, written plainly."""
    st.write(plain(line))


def run_safely(what: str, action):
    """Run `action`; if it fails, show what went wrong instead of a stack trace."""
    try:
        return action()
    except Exception as e:
        st.error(plain(f"{what} failed. {type(e).__name__}: {e}"))
        return None


def plural(n: int, word: str) -> str:
    """'1 branch', '2 branches', '3 configurations'."""
    ending = "" if n == 1 else "es" if word.endswith(("s", "ch")) else "s"
    return f"{n:,} {word}{ending}"


def share(p: float) -> str:
    """A probability as a plain percentage, keeping small ones visible."""
    if p >= 0.005:
        return f"{p:.0%}"
    return f"{p:.1%}" if p >= 0.001 else "under 0.1%"


# --- what ran underneath ---
def model_calls(cost_table: list) -> str:
    """Which Nemotron tiers were asked, how often, at what cost (or that replies were canned)."""
    if not cost_table:
        return "no model calls"
    if all(call["model"].endswith("(mock)") for call in cost_table):
        return "model replies are canned in demo mode"
    tiers = Counter(call["tier"] for call in cost_table)
    asked = ", ".join(f"{tiers[t]} to {t.title()}" for t in ("ultra", "super", "nano") if tiers[t])
    cost = sum(call["cost_usd"] for call in cost_table)
    return f"Nemotron 3 calls: {asked}, ${cost:.4f} in all"


def where_it_ran(executor: str, branches: int) -> str:
    """Where the backtests ran, in words."""
    if executor.startswith("nebius"):
        return plural(branches, "Token Factory Sandbox branch")
    if executor.startswith("local"):
        return plural(branches, "local process")
    return "this app's own process (library code only)"


def trial_hood(trial) -> str:
    """The under-the-hood line for a trial: models, versions, checks, branches, time, search."""
    report = trial.time_travel or {}
    run = trial.execution
    versions = len(trial.code_history)
    return (
        f"Under the hood: {model_calls(trial.cost_table)}. "
        f"{plural(versions, 'version')} of the code; the one judged ran "
        f"{plural(report.get('configs', 0), 'configuration')} with "
        f"{report.get('checks', 0):,} time-travel checks in "
        f"{where_it_ran(trial.executor, run.get('branches', 1))}, "
        f"{run.get('wall_s', 0):.1f}s from start to finish. {searches(trial.prior_art)}"
    )


def searches(prior_art: dict) -> str:
    """How many Tavily searches the trial ran, and their credits."""
    done = prior_art.get("search") or {}
    if not done.get("searches"):
        return "No live Tavily search."
    credits = f", {done['credits']} credits" if done.get("credits") is not None else ""
    return f"Tavily: {plural(done['searches'], 'search')} of papers and practice{credits}."


def repaired_by(trial) -> str:
    """Who repaired the code: the tier of the last repair call, or the demo's stand-in."""
    repairs = [call for call in trial.cost_table if call["role"] == "coder-repair"]
    if not repairs or repairs[-1]["model"].endswith("(mock)"):
        return "Repaired by the library's fixed version (demo mode stands in for Nemotron)"
    return f"Repaired by Nemotron 3 {repairs[-1]['tier'].title()}"


# --- the bench: four judges of a fisherman's claim ---
def four_judges(expedition) -> None:
    court, sealed = expedition.court, expedition.sealed
    if court.reasons:
        court_finding = f"{court.verdict}: {'; '.join(court.reasons)}"
    else:
        court_finding = f"{court.verdict}, {fmt_p(court.reality_check.p_value)}"
    sealed_finding = (
        "Not opened"
        if sealed is None
        else f"{sealed.verdict}, Sharpe {sealed.sealed_sharpe_annual:.2f} on sealed data"
    )
    judges = [
        (
            "A standard backtest test",
            expedition.naive_p_value <= P_VALUE_TO_ACQUIT,
            fmt_p(expedition.naive_p_value),
            "Tests the claimed strategy as if it were the only one tried",
        ),
        (
            "A check of the claim's own grid",
            expedition.claim_grid_p_value <= P_VALUE_TO_ACQUIT,
            fmt_p(expedition.claim_grid_p_value),
            "What reporting only the last attempt would show",
        ),
        (
            "Overfit Court",
            court.verdict == ACQUITTED,
            court_finding,
            f"Counts all {expedition.n_configs} branches, about "
            f"{expedition.distinct_bets:.0f} independent directions",
        ),
        (
            "The sealed evidence",
            sealed is not None and sealed.verdict == CONFIRMED,
            sealed_finding,
            "Data no branch ever saw",
        ),
    ]
    seats = "".join(
        f"<div class='{'good' if convinced else 'bad'}'><div class='who'>{html.escape(who)}</div>"
        + stamp(
            "Convinced" if convinced else "Not convinced",
            "good" if convinced else "bad",
            "small",
        )
        + f"<div>{html.escape(finding)}</div><div class='why'>{html.escape(basis)}</div></div>"
        for who, convinced, finding, basis in judges
    )
    html_block(f"<div class='bench'>{seats}</div>")


# --- charts ---
def luck_chart(table: pd.DataFrame, target: float, n_attempts: int, y_range, height: int = 300):
    """What luck alone reaches over every branch so far (the shaded area), each attempt's own
    best Sharpe (dots), the fisherman's best so far (the step line) and the target (dashed).

    The axes are fixed by the caller (`n_attempts`, `y_range`), so the frames of a replay,
    which show only the first rows of the table, don't jump.
    """
    x = alt.X(
        "k:Q",
        title="attempt",
        scale=alt.Scale(domain=[1, max(2, n_attempts)]),
        axis=alt.Axis(tickMinStep=1, format="d", grid=False),
    )
    scale = alt.Scale(domain=list(y_range), nice=False)

    def y(field: str):  # one title on every layer, so the merged axis keeps it
        return alt.Y(f"{field}:Q", scale=scale, title="Sharpe ratio, per year")

    base = alt.Chart(table)
    has_luck = table["luck_annual"].notna().any()
    goal = alt.Chart(pd.DataFrame({"y": [target], "k": [1], "label": [f"target {target:.1f}"]}))
    tip = table.tail(1).assign(best="fisherman's best", luck="luck alone")
    layers = [
        goal.mark_rule(strokeDash=[5, 4], color=OCHRE, strokeWidth=1.5).encode(y=y("y")),
        goal.mark_text(align="left", dx=2, dy=-9, color=OCHRE, fontSize=12).encode(
            x=x, y=y("y"), text="label:N"
        ),
    ]
    if has_luck:
        layers += [
            base.mark_area(color=LUCK, opacity=0.22).encode(
                x=x, y=y("luck_annual"), y2=alt.datum(0)
            ),
            base.mark_line(color=LUCK, strokeWidth=1.5).encode(x=x, y=y("luck_annual")),
            alt.Chart(tip)
            .mark_text(align="right", dx=-4, dy=14, color=MUTED, fontSize=12)
            .encode(x=x, y=y("luck_annual"), text="luck:N"),
        ]
    layers += [
        base.mark_circle(color=BLUE, opacity=0.35, size=34).encode(
            x=x,
            y=y("best_sharpe_annual"),
            tooltip=[
                alt.Tooltip("k:Q", title="attempt"),
                alt.Tooltip("idea:N", title="tried"),
                alt.Tooltip("best_sharpe_annual:Q", title="its best Sharpe", format=".2f"),
            ],
        ),
        base.mark_line(color=BLUE, strokeWidth=2.6, interpolate="step-after").encode(
            x=x, y=y("running_best_annual")
        ),
        alt.Chart(tip)
        .mark_text(align="right", dx=-4, dy=-10, color=BLUE, fontSize=12, fontWeight=600)
        .encode(x=x, y=y("running_best_annual"), text="best:N"),
    ]
    return alt.layer(*layers).properties(height=height).configure_view(strokeWidth=0)


def draw_search(expedition, animate: bool, height: int = 300) -> None:
    """The luck chart, replayed attempt by attempt when `animate` (just after a click)."""
    table = expedition.attempts_table()
    values = table[["best_sharpe_annual", "running_best_annual", "luck_annual"]].to_numpy()
    y_range = (
        min(0.0, float(np.nanmin(values))),
        max(expedition.target, float(np.nanmax(values))) * 1.15,
    )
    n = len(table)
    slot = st.empty()
    for k in range(1, n + 1) if animate and n > 1 else [n]:
        chart = luck_chart(table.iloc[:k], expedition.target, n, y_range, height)
        slot.altair_chart(chart, width="stretch")
        if k < n:
            time.sleep(min(0.25, 2.4 / n))  # about two seconds, however many attempts


def spectrum_chart(spectrum, max_bars: int = 40, floor: float = 1e-2):
    """Eigenvalues of everything tried, against the largest one luck alone produces.

    A log scale has no zero, so bars start at `floor` (tiny eigenvalues sit on it),
    and both layers share one log scale: a linear layer would blank the chart.
    """
    eigenvalues = np.array(spectrum.eigenvalues[:max_bars])
    bars = pd.DataFrame(
        {
            "rank": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": np.maximum(eigenvalues, floor),
            "floor": floor,
            "kind": np.where(
                eigenvalues > spectrum.noise_edge, "a real common direction", "within noise"
            ),
        }
    )
    log_scale = alt.Scale(type="log", domainMin=floor)
    columns = (
        alt.Chart(bars)
        .mark_bar()
        .encode(
            x=alt.X("rank:O", title="eigenvalue rank", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("eigenvalue:Q", scale=log_scale, title="eigenvalue (log scale)"),
            y2="floor:Q",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["a real common direction", "within noise"], range=[GREEN, LUCK]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    edge = (
        alt.Chart(pd.DataFrame({"y": [spectrum.noise_edge]}))
        .mark_rule(strokeDash=[6, 4], color=OCHRE)
        .encode(y=alt.Y("y:Q", scale=log_scale))
    )
    return (columns + edge).properties(height=300).configure_view(strokeWidth=0)


def equity_chart(returns: pd.DataFrame, best: str | None = None, every: int = 5, height=300):
    """Growth of 1 for every column of daily returns: all in grey, `best` in green on top."""
    growth = (1 + returns).cumprod().iloc[::every]
    long = growth.reset_index(names="date").melt(
        id_vars="date", var_name="configuration", value_name="growth"
    )
    x = alt.X("date:T", title=None)
    y = alt.Y("growth:Q", title="growth of 1", scale=alt.Scale(zero=False))
    fan = (
        alt.Chart(long[long["configuration"] != best])
        .mark_line(strokeWidth=1, opacity=0.35, color=LUCK)
        .encode(x=x, y=y, detail="configuration:N")
    )
    layers = [fan]
    if best in growth.columns:
        layers.append(
            alt.Chart(long[long["configuration"] == best])
            .mark_line(strokeWidth=2.4, color=GREEN)
            .encode(x=x, y=y)
        )
    return alt.layer(*layers).properties(height=height).configure_view(strokeWidth=0)


# --- sandbox branches ---
def branch_grid(
    keys: list, report: dict, branches: int, per_branch: list | None = None, key: bool = True
) -> None:
    """One square per configuration, grouped by the branch that ran it.

    Green: no time travel. Red: a past position moved when later data arrived.
    Outlined: it crashed or never traded. Configurations are dealt to branches
    the way the executors deal them (every n-th), and the squares of every
    branch fill in at once, as the branches ran. One line underneath says how
    many ran where, and how long the slowest branch took.
    """
    leaky, tested = set(report.get("leaky", [])), set(report.get("tested", []))
    n = max(1, min(branches, len(keys)))
    columns = []
    for b in range(n):
        dealt = keys[b::n]
        cells = "".join(
            f"<i class='{'leak' if k in leaky else 'ok' if k in tested else 'crash'}' "
            f"style='--i:{j}' title='{html.escape(str(k))}'></i>"
            for j, k in enumerate(dealt)
        )
        cols = max(3, min(8, math.ceil(math.sqrt(len(dealt)))))
        columns.append(
            f"<div><div class='cells' style='--cols:{cols}'>{cells}</div>"
            f"<div class='cap'>Branch {b + 1}</div></div>"
        )
    html_block(f"<div class='run'>{''.join(columns)}</div>")
    walls = [m["wall_s"] for m in per_branch or [] if "wall_s" in m]
    slowest = f", the slowest in {max(walls):.1f}s" if walls else ""
    html_block(
        f"<div class='cap'>{plural(len(keys), 'configuration')} in {plural(n, 'branch')} "
        f"running at once{slowest}.</div>"
    )
    if key:
        branch_key()


def branch_key() -> None:
    html_block(
        "<div class='key'><i></i>never read the future<i class='leak'></i>read the future"
        "<i class='crash'></i>crashed or never traded</div>"
    )


def time_travel_panel(trial) -> None:
    """What the time-travel test found, on every version of the code that was judged or caught."""
    report = trial.time_travel or {}
    keys = list(trial.grid)
    branches = trial.execution.get("branches", 1)
    if not trial.passed_time_travel:
        heading = "Time travel" if trial.time_traveller else "Time travel: could not be verified"
        paragraph(f"{heading}.", trial.evidence()["time_travel_detail"])
        branch_grid(keys, report, branches, trial.execution.get("per_branch"))
        return
    if trial.caught_and_repaired:
        caught = trial.first_leak
        repairs = trial.judged_attempt - caught["attempt"]
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Version {caught['attempt']}, as first written**")
            branch_grid(keys, caught["time_travel"] or {}, branches, key=False)
        with right:
            st.markdown(f"**Version {trial.judged_attempt}, after repair**")
            branch_grid(keys, report, branches, trial.execution.get("per_branch"), key=False)
        branch_key()
        paragraph(f"Caught in version {caught['attempt']}:", caught["problem"])
        st.markdown(
            f"**{repaired_by(trial)}** in {plural(repairs, 'attempt')}, then re-tested: "
            "no time travel."
        )
        with st.expander("What changed in the code"):
            diff = difflib.unified_diff(
                caught["code"].splitlines(True),
                trial.code.splitlines(True),
                f"version {caught['attempt']}",
                "after repair",
                n=2,
            )
            st.code("".join(diff), language="diff")
        return
    cuts = report["expected_checks"] // max(1, report["configs"])
    st.markdown(
        f"**Time-travel test: passed.** All {report['configs']} configurations were re-run on "
        f"history cut short at {cuts} points, and every past position stayed exactly the same."
    )
    branch_grid(keys, report, branches, trial.execution.get("per_branch"))


def prior_art_card(trial) -> None:
    """What Tavily found: has this been published, by whom, and how the literature's own
    parameters do when tested on their own."""
    summary, sources, literature = prior_art_points(trial.prior_art)
    known = trial.prior_art.get("already_known")
    mark = stamp("Already published", "unsure", "small") if known else ""
    kinds = {"papers": "paper", "practice": "practitioner research"}
    items = "".join(
        f"<li><a href='{html.escape(s['url'])}' target='_blank' rel='noopener'>"
        f"{html.escape(s['title'])}</a> <span class='kind'>{kinds.get(s.get('kind'), 'web')}"
        "</span></li>"
        for s in sources
        if s["url"].startswith(("https://", "http://"))
    )
    html_block(
        f"<div class='prior'>{mark}<p>{html.escape(summary)}</p>"
        + (f"<ul>{items}</ul>" if items else "")
        + (f"<p>{html.escape(literature)}</p>" if literature else "")
        + "</div>"
    )


def sealed_ruling(sealed) -> None:
    ruling(
        sealed.verdict,
        f"Sealed evidence: {sealed.verdict}",
        f"Sharpe {sealed.open_sharpe_annual:.2f} on the open evidence, "
        f"{sealed.sealed_sharpe_annual:.2f} on {sealed.sealed_days} sealed days it never saw "
        f"({fmt_p(sealed.p_value)}). Commitment {sealed.commitment[:16]}…",
    )


# --- sidebar ---
def sidebar(settings: Settings):
    """The evidence and the procedure.

    Returns (prices, seed, use_sandboxes, record_box); `prices` is None until a
    usable CSV is uploaded, and `record_box` is filled in last by render_court_record.
    """
    with st.sidebar:
        st.header("Evidence")
        source = st.radio(
            "Price data", ["Synthetic: pure noise", "Synthetic: planted edge", "Upload CSV"]
        )
        upload = (
            st.file_uploader("Daily closes, one column per asset", type="csv")
            if source == "Upload CSV"
            else None
        )
        seed = int(st.number_input("Seed", 0, 10_000, DEFAULT_SEED))
        st.caption(
            f"The last {SEALED_FRACTION:.0%} of dates is sealed: it is never uploaded to the "
            "research sandboxes."
        )
        prices = None
        if source != "Upload CSV":
            signal = PLANTED_SIGNAL if "planted" in source else 0.0
            prices = synthetic_prices(seed=seed, signal=signal)
        elif upload is not None:
            prices = run_safely("Reading the CSV", lambda: load_csv(upload))

        st.header("Procedure")
        settings.max_configs = st.slider(
            "Max configurations per trial", 20, 200, settings.max_configs
        )
        settings.cost_bps = st.slider("Cost per unit turnover (bps)", 0.0, 20.0, settings.cost_bps)
        can_use_sandboxes = not settings.mock_llm
        # With a key set, the code is written by a model from a visitor's idea, so it
        # only runs in sandboxes unless the person deploying the app allows otherwise.
        sandboxes_only = can_use_sandboxes and not settings.allow_local
        sandboxes_first = sandboxes_only or (can_use_sandboxes and settings.executor == "nebius")
        where = st.radio(
            "Run backtests in",
            ["Token Factory Sandboxes", "Local (dev only)"],
            index=0 if sandboxes_first else 1,
            disabled=not can_use_sandboxes or sandboxes_only,
            help="Sandboxes need NEBIUS_API_KEY and a project with sandbox access. With a key "
            "set, model-written code only runs locally if OC_ALLOW_LOCAL=1.",
        )
        st.header("Court record")
        record_box = st.container()
    return prices, seed, can_use_sandboxes and where.startswith("Token"), record_box


def render_court_record(box, prices, ledger: Ledger) -> None:
    """Every trial on the current data, drawn last so it includes anything run this time."""
    with box:
        if prices is None:
            return
        key = record_key(prices, SEALED_FRACTION)
        history, seals = ledger.history(key), ledger.seals(key)
        if not history:
            st.caption("No trials on this data yet. Every attempt will count against the next.")
            return
        st.markdown(
            f"**{plural(len(history), 'trial')} on this data, "
            f"{plural(sum(h['n'] for h in history), 'configuration')}, "
            f"{plural(len(seals), 'seal opening')}.**"
        )
        first_shown = max(1, len(history) - 7)
        for i, item in enumerate(history[-8:], first_shown):
            idea, verdict = item["idea"], item["verdict"]
            if verdict == "expedition":  # one attempt of a fishing expedition
                idea, verdict = idea.removeprefix("[expedition] "), "Fishing"
            idea = idea if len(idea) <= 55 else idea[:54].rstrip() + "…"
            verdict, idea = html.escape(verdict), html.escape(idea)
            html_block(f"<div class='record'>{i}. {verdict}: {idea}</div>")
        if st.button("Wipe the record for this data"):
            ledger.clear(key)
            for name in ("trial", "expedition"):
                st.session_state.pop(name, None)
            st.rerun()


# --- tabs ---
def render_story(settings: Settings, ledger: Ledger, executor) -> None:
    st.markdown(
        "Three short acts, one click each. Each uses its own fixed market, picked to show its "
        "story clearly, so everyone sees the same thing. The README's results measure the court "
        "over many markets."
    )
    story = st.session_state.story

    st.subheader("Act 1. An AI is told to find alpha in pure noise")
    st.markdown(
        "Ten years of prices generated from noise: there is nothing to find. A fisherman agent "
        "tries strategy after strategy, each in a new branch, until one reaches a Sharpe of 1."
    )
    just_fished = st.button("Send the fisherman", key="act1")
    if just_fished:
        with st.spinner("Fishing..."):
            act = run_safely("Act 1", lambda: stories.act_fishing(settings, ledger, log=_silent))
        if act is not None:
            story["fish"] = act
        just_fished = act is not None
    if "fish" in story:
        expedition = story["fish"]
        st.markdown(
            f"After **{plural(len(expedition.attempts), 'attempt')}** and "
            f"**{expedition.n_configs} branches** it claims **{expedition.claimed_family} "
            f"{expedition.claimed_params}** with a Sharpe of "
            f"**{expedition.claimed_sharpe_annual:.2f}**. The market is pure noise, so the right "
            "answer from every judge is *not convinced*."
        )
        draw_search(expedition, animate=just_fished, height=260)
        four_judges(expedition)
        hood(
            f"Under the hood: a scripted fisherman, so the story is the same for everyone (send "
            f"Nemotron 3 Super fishing in the Fishing expedition tab). After every attempt the "
            f"court re-measures luck over all {expedition.n_configs} branches so far, with a "
            "block bootstrap."
        )

    st.subheader("Act 2. A strategy that reads the future")
    st.markdown(
        f"*“{stories.ANCHOR_IDEA}”* Its natural first draft measures the average over "
        "the whole price history, including dates that haven't happened yet. A code review "
        "usually misses it. Statistics can't see it at all."
    )
    if st.button("Put it on trial", key="act2"):
        with st.spinner("Running the trial and the time-travel test..."):
            act = run_safely(
                "Act 2",
                lambda: stories.act_time_travel(settings, ledger, executor=executor, log=_silent),
            )
        if act is not None:
            story["time_travel"] = act
    if "time_travel" in story:
        act = story["time_travel"]
        alone, trial = act.statistics_alone, act.trial
        evidence = trial.evidence()
        numbers = (
            f"Backtest Sharpe {act.leaky_sharpe_annual:.2f} against "
            f"{alone.reality_check.null_median_annual:.2f} from luck "
            f"({fmt_p(alone.reality_check.p_value)})."
        )
        left, right = st.columns(2)
        with left:
            if alone.verdict == ACQUITTED:
                html_block(
                    "<div class='ruling'><span class='stamps'>"
                    + stamp(ACQUITTED, "void")
                    + stamp("Fooled", "bad", "big", "over")
                    + "</span><div class='says'><h4>Statistics alone</h4>"
                    f"<p>{html.escape(numbers)} Every multiple-testing check passes, on pure "
                    "noise, because the code is cheating.</p></div></div>"
                )
            else:
                ruling(
                    alone.verdict,
                    "Statistics alone",
                    f"{numbers} With these settings the statistics happen not to acquit it, but "
                    "they can't see that it reads the future.",
                )
        with right:
            if trial.caught_and_repaired:
                outcome = (
                    f"Caught and repaired. The honest version has Sharpe "
                    f"{evidence['best_sharpe_annual']:.2f} against "
                    f"{evidence['luck_sharpe_annual']:.2f} from luck."
                )
            else:
                outcome = f"Caught. {evidence['time_travel_detail']}"
            ruling(trial.verdict, "With the time-travel test", outcome)
        time_travel_panel(trial)
        hood(trial_hood(trial))

    st.subheader("Act 3. A real edge survives")
    st.markdown(
        f"Now a market with a genuine, persistent edge planted in it. *“{stories.EDGE_IDEA}”*"
    )
    if st.button("Put it on trial and open the evidence room", key="act3"):
        with st.spinner("Trial, then one test on the sealed evidence..."):
            act = run_safely(
                "Act 3",
                lambda: stories.act_real_edge(settings, ledger, executor=executor, log=_silent),
            )
        if act is not None:
            story["edge"] = act
    if "edge" in story:
        trial = story["edge"]
        evidence = trial.evidence()
        ruling(
            trial.verdict,
            "The court",
            f"Sharpe {evidence['best_sharpe_annual']:.2f} against "
            f"{evidence['luck_sharpe_annual']:.2f} from luck ({fmt_p(evidence['p_value'])}). "
            f"Time-travel test {evidence['time_travel']}.",
        )
        if trial.sealed:
            sealed_ruling(trial.sealed)
        st.markdown("**Is it already published?**")
        prior_art_card(trial)
        hood(trial_hood(trial))
        st.download_button(
            "Download the court record",
            court_record_html(trial),
            "overfit_court_record.html",
            "text/html",
            key="act3_download",
        )


def render_trial(settings: Settings, ledger: Ledger, prices, executor) -> None:
    idea = st.text_area(
        "Your idea",
        "Buy the assets that went up most over the last few months and short the ones that "
        "went down most.",
        height=80,
    )
    if st.button("Begin trial", type="primary", disabled=prices is None):
        with st.status("Court in session...", expanded=True) as status:
            try:
                st.session_state.trial = run_trial(
                    idea, prices, settings=settings, ledger=ledger, executor=executor, log=say
                )
                status.update(label="Verdict reached", state="complete", expanded=False)
            except Exception as e:  # show a mistrial instead of a stack trace
                status.update(label="Mistrial", state="error")
                st.error(plain(f"{type(e).__name__}: {e}"))

    trial = st.session_state.get("trial")
    if not trial:
        return
    e = trial.evidence()
    ruling(trial.verdict, "The court", trial.opinion.get("headline", ""))
    if e["failed_checks"]:
        paragraph("Failed:", "; ".join(e["failed_checks"]) + ".")

    metrics = st.columns(6)
    metrics[0].metric(
        "Best Sharpe",
        f"{e['best_sharpe_annual']:.2f}",
        help="The best configuration's yearly Sharpe ratio, on the open evidence only",
    )
    raised = e["luck_sharpe_annual"] - e["luck_sharpe_annual_this_trial_only"]
    metrics[1].metric(
        "Luck reaches",
        f"{e['luck_sharpe_annual']:.2f}",
        delta=f"+{raised:.2f} from your record" if raised > 0.005 else None,
        delta_color="inverse",
        help="The best Sharpe that zero-edge versions of everything tried on this data "
        "typically reach",
    )
    metrics[2].metric(
        "Luck matches it",
        share(e["p_value"]),
        help=f"How often zero-edge versions of everything tried on this data do as well as "
        f"your pick, in bootstrap resamples: the Reality Check's p-value ({fmt_p(e['p_value'])})."
        " Small is good. It is not the probability that the edge is real.",
    )
    metrics[3].metric(
        "Luck's real tries",
        f"{e['luck_equivalent_tries']:,}",
        help=f"Luck reaches that Sharpe as easily as the best of this many independent "
        f"strategies would. {e['n_distinct_configs_on_this_data']} distinct configurations "
        f"have been tried on this data, moving in about {e['distinct_bets_total']:.0f} "
        "independent directions.",
    )
    metrics[4].metric(
        "Fails on new data",
        f"{e['prob_oos_loss']:.0%}",
        help="How often the best configuration on one half of the history loses money on the "
        "other half (CSCV splits)",
    )
    metrics[5].metric(
        "Years to prove",
        fmt_years(e["years_of_live_trading_to_prove_it"]),
        help="How long a live record at the walk-forward Sharpe would need to run to show, "
        "with 95% confidence, that the Sharpe is above zero",
    )

    time_travel_panel(trial)

    left, right = st.columns(2)
    with left:
        st.subheader("Every backtest you could have cherry-picked")
        st.altair_chart(equity_chart(trial.returns, trial.findings.best_key), width="stretch")
        st.caption(
            f"Open evidence only. In green, the best: `{e['best_params']}`, the one a backtest "
            "report would show you."
        )
    with right:
        st.subheader("How many different bets was that?")
        st.altair_chart(spectrum_chart(trial.findings.spectrum), width="stretch")
        st.caption(
            f"{e['n_distinct_configs_on_this_data']} distinct configurations, moving in about "
            f"{e['distinct_bets_total']:.0f} independent directions. Bars above the dashed line "
            "are real common directions; below it is what unrelated strategies with the same "
            "volatility would show by chance."
        )

    st.subheader("The judgment")
    bullets(trial.opinion.get("reasoning", []))
    st.markdown("**What would change the verdict**")
    bullets(trial.opinion.get("what_would_change_my_mind", []))
    if trial.opinion.get("note"):
        st.caption(plain(trial.opinion["note"]))

    st.subheader("The evidence room")
    render_evidence_room(trial, ledger, executor)

    st.subheader("Is it already published?")
    prior_art_card(trial)

    with st.expander(f"Look-ahead audit: {trial.audit['risk']} risk"):
        bullets(trial.audit["issues"] or ["No issues found."])
    with st.expander("Strategy code"):
        st.code(trial.code, language="python")
    with st.expander("Model routing, sandboxes and spend"):
        st.dataframe(pd.DataFrame(trial.cost_table), width="stretch")
        st.write({"executor": trial.executor, **trial.execution})
    hood(trial_hood(trial))
    left, right = st.columns(2)
    left.download_button(
        "Download the court record (HTML)",
        court_record_html(trial),
        "overfit_court_record.html",
        "text/html",
    )
    right.download_button(
        "Download the full report (JSON)", trial.to_json(), "overfit_court_report.json"
    )


def render_evidence_room(trial, ledger: Ledger, executor) -> None:
    if trial.sealed is not None:
        sealed_ruling(trial.sealed)
        dates = trial.prices_full.index[trial.prices_full.index >= trial.cut]
        sealed = pd.DataFrame({"committed": trial.sealed.sealed_returns}, index=dates)
        st.altair_chart(equity_chart(sealed, "committed", every=1, height=200), width="stretch")
        if trial.sealed.times_opened_before:
            st.caption(
                f"Corrected for {plural(trial.sealed.times_opened_before, 'earlier opening')}."
            )
        return

    e = trial.evidence()
    st.markdown(
        f"The last {trial.sealed_fraction:.0%} of the data (from {e['sealed_from']}) has never "
        "been inside a research sandbox. Commit to the winning configuration and it gets exactly "
        "one test there."
    )
    openings = len(ledger.seals(trial.record["dataset"]))
    if openings:
        st.warning(
            f"The seal on this data has been opened {plural(openings, 'time')} already. "
            "The court will correct for that."
        )
    if trial.failed_opening:
        st.caption(plain(f"This trial's last opening: {OPENED_NO_RESULT} {trial.failed_opening}"))
    if not trial.passed_time_travel:
        st.caption("Shut: a strategy that may read the future could read the sealed evidence too.")
        return
    if trial.audit.get("escapes"):
        st.caption(
            "Shut: the code reaches outside the prices it is given, so the time-travel test "
            "can't vouch for it."
        )
        return
    if st.button("Commit and open the evidence room"):
        with st.spinner("Opening a fresh sandbox with the sealed evidence..."):
            sealed = run_safely(
                "Opening the evidence room", lambda: unseal(trial, executor=executor, log=_silent)
            )
        if sealed is not None:
            st.rerun()  # redraw with the result (an error stays on screen instead)


def render_fishing(settings: Settings, ledger: Ledger, prices, seed: int, use_sandboxes) -> None:
    st.markdown(
        "Sandboxes that branch like git make it nearly free for an agent to try path after path "
        "until something works. Send a fisherman out with a target and see what four judges make "
        "of what it brings back."
    )
    target_column, attempts_column, fisherman_column = st.columns(3)
    target = target_column.slider("Target Sharpe", 0.6, 1.5, 1.0, 0.05)
    max_attempts = attempts_column.slider("Attempts allowed", 5, 60, 40)
    who = fisherman_column.radio(
        "Fisherman",
        ["Scripted (explore, then hill-climb)", "Nemotron 3 Super"],
        disabled=settings.mock_llm,
        index=0 if settings.mock_llm else 1,
        help="Nemotron needs NEBIUS_API_KEY. The scripted fisherman makes the moves any "
        "search agent makes, and its results reproduce.",
    )
    just_fished = st.button("Send the fisherman out", type="primary", disabled=prices is None)
    if just_fished:
        if who.startswith("Nemotron") and not settings.mock_llm:
            fisherman = NemotronFisherman(LLM(settings), fallback_seed=seed)
        else:
            fisherman = ScriptedFisherman(seed)
        # Library code only, so in-process is safe when sandboxes are not in use.
        executor = NebiusSandboxExecutor() if use_sandboxes else InProcessExecutor()
        with st.status("Fishing...", expanded=False) as status:
            try:
                st.session_state.expedition = run_expedition(
                    prices,
                    target=target,
                    max_attempts=max_attempts,
                    fisherman=fisherman,
                    settings=settings,
                    ledger=ledger,
                    executor=executor,
                    track_luck=True,
                    seed=seed,
                    log=say,
                )
                status.update(label="The fisherman is back", state="complete")
            except Exception as e:
                status.update(label="The expedition failed", state="error")
                st.error(plain(f"{type(e).__name__}: {e}"))
                just_fished = False

    expedition = st.session_state.get("expedition")
    if not expedition:
        return
    summary = expedition.summary()
    st.subheader(
        f"{plural(len(expedition.attempts), 'attempt')}, {expedition.n_configs} branches, "
        f"about {expedition.distinct_bets:.0f} independent directions"
    )
    draw_search(expedition, animate=just_fished)
    st.caption(
        "Dots are each attempt's best; the line is the best so far; the shaded area is what "
        "luck alone reaches over every branch so far. When the best only keeps pace with luck, "
        "it has found nothing."
    )
    if expedition.fallback_turns:
        st.caption(
            f"Nemotron's proposal was unusable on {expedition.fallback_turns} of "
            f"{len(expedition.attempts)} turns; the scripted fisherman took those turns."
        )
    paragraph(
        "The claim:",
        f"{summary['claimed']} with Sharpe {summary['claimed_sharpe_annual']:.2f} on the open "
        "evidence.",
    )
    four_judges(expedition)
    with st.expander("Every attempt"):
        st.dataframe(expedition.attempts_table(), width="stretch")


# --- page ---
def main() -> None:
    st.set_page_config(page_title="Overfit Court", page_icon="⚖️", layout="wide")
    html_block(CSS)
    settings = get_settings()
    # A public app caps live model spend per day; past the cap it runs in demo mode.
    budget = settings.daily_budget_usd
    over_budget = not settings.mock_llm and budget > 0 and llm.SPEND.today() >= budget
    if over_budget:
        settings = settings.offline()
    st.session_state.setdefault("ledger", Ledger())
    st.session_state.setdefault("story", {})
    ledger: Ledger = st.session_state.ledger

    prices, seed, use_sandboxes, record_box = sidebar(settings)
    # Model-written code never runs in-process: sandboxes when available, else subprocesses.
    executor = NebiusSandboxExecutor() if use_sandboxes else LocalExecutor()

    html_block(
        "<div class='hero'><h1>Overfit Court</h1>"
        "<p>AI agents can now try a thousand trading ideas before lunch. This court counts every "
        "one of them, checks that none of them read the future, keeps a slice of history they "
        "never see, and lets the statistics, not the model, decide.</p>"
        f"<p class='define'>{html.escape(SHARPE)}</p></div>"
    )
    if over_budget:
        st.info(
            f"Demo mode until midnight UTC: today's ${budget:.2f} budget for live model calls on "
            "this public app is used up. Model replies are canned and ideas are matched to a "
            "built-in strategy library; the backtests, the time-travel test and the statistics "
            "are all real."
        )
    elif settings.mock_llm:
        st.info(
            "Demo mode: no NEBIUS_API_KEY is set, so model replies are canned and ideas are "
            "matched to a built-in strategy library. The backtests, the time-travel test and "
            "the statistics are all real."
        )
    elif budget > 0:
        st.caption(
            f"Live: Nemotron 3 on Nebius Token Factory. ${llm.SPEND.today():.2f} of today's "
            f"${budget:.2f} budget for this public app is spent; past it, the app switches to "
            "demo mode until midnight UTC."
        )

    story, trial, fishing, how = st.tabs(
        ["Start here", "Put an idea on trial", "Fishing expedition", "How it works"]
    )
    with story:
        render_story(settings, ledger, executor)
    with trial:
        render_trial(settings, ledger, prices, executor)
    with fishing:
        render_fishing(settings, ledger, prices, seed, use_sandboxes)
    with how:
        st.markdown(HOW_IT_WORKS)
    render_court_record(record_box, prices, ledger)


main()
