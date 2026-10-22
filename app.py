"""Overfit Court, the web app:  streamlit run app.py

Layout
    Sidebar               the evidence (which prices), the procedure (grid size,
                          costs, where backtests run) and the court record
    ▶ Start here          three one-click demo acts (overfit_court/stories.py)
    ⚖️ Put an idea on trial   your idea, on your data
    🎣 Fishing expedition  send a fisherman out on your data
    📜 How it works

Each visitor gets their own court record in st.session_state. Text that comes
from a model or a user is HTML-escaped before it is rendered (see `bullets`,
`paragraph` and `banner`), and every action shows a readable error instead of
a stack trace if it fails.
"""

from __future__ import annotations

import difflib
import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from overfit_court import stories
from overfit_court.config import Settings, get_settings
from overfit_court.data import PLANTED_SIGNAL, load_csv, synthetic_prices
from overfit_court.fishing import NemotronFisherman, ScriptedFisherman, run_expedition
from overfit_court.ledger import Ledger, record_key
from overfit_court.llm import LLM
from overfit_court.pipeline import SEALED_FRACTION, run_trial, unseal
from overfit_court.report import court_record_html
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

CSS = """
<style>
.banner { padding: 1rem 1.3rem; border-radius: 14px; margin: .5rem 0 1rem 0; }
.banner h2, .banner h3 { margin: 0; letter-spacing: .05em; }
.banner p { margin: .35rem 0 0 0; font-size: 1.02rem; }
.good   { background: rgba(22, 163, 74, .12);  border: 1px solid rgba(22, 163, 74, .45); }
.bad    { background: rgba(220, 38, 38, .12);  border: 1px solid rgba(220, 38, 38, .45); }
.unsure { background: rgba(217, 119, 6, .12);  border: 1px solid rgba(217, 119, 6, .45); }
.fooled { background: rgba(107, 114, 128, .10); border: 1px dashed rgba(107, 114, 128, .6); }
.fooled h3 { text-decoration: line-through; text-decoration-color: rgba(220, 38, 38, .8); }
.judge { padding: .8rem 1rem; border-radius: 12px; height: 100%; }
.judge .who { font-size: .85rem; opacity: .75; text-transform: uppercase; letter-spacing: .06em; }
.judge .what { font-size: 1.15rem; font-weight: 600; margin: .25rem 0; }
.judge .how { opacity: .7; font-size: .85rem; }
.act { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; opacity: .7;
       margin-top: 1rem; }
.record { font-size: .9rem; opacity: .85; }
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

HOW_IT_WORKS = """
**1. The model plans and codes; the harness keeps it honest.** Nemotron 3 Ultra turns an idea into
a parameter grid, Nemotron 3 Super writes the strategy, Nemotron 3 Nano audits it. The harness
applies the one-day lag and trading costs itself, so generated code cannot trade on today's close.

**2. The time-travel test.** Every configuration is also re-run on history cut short at eight
points, from the first 1% of the days to the first 80%, and every past position must stay exactly
the same. If a position on a past date moves when later data is added, the code read the future.
That catches what code review and statistics both miss: means or standard deviations over the
whole series, negative shifts, back-fills, centred windows. A leak goes back to Nemotron Super with
the exact dates, for repair. The test fails closed: code that crashes on shorter history can't be
verified, and is treated like a leak.

**3. Research happens in sandboxes that cannot see the future.** Only the open 75% of the data is
uploaded to Token Factory Sandboxes. Every chunk of the grid is its own branch of one checkpoint.

**4. The court remembers.** Every configuration ever tried on this data is kept. The Reality Check
asks how often zero-edge versions of all of them would do as well as your pick, using a block
bootstrap that preserves fat tails, volatility clustering and the correlation between strategies.
That share is its p-value. Re-running the same grid adds nothing; genuinely new ideas raise the bar.

**5. Random matrix theory counts your real bets.** The eigenvalues of the correlation matrix of
everything tried give the number of genuinely independent bets. Unrelated strategies look a little
correlated by chance, more so when they share volatility regimes, so that part is measured from the
data and removed first.

**6. The statistics rule.** Convicted if it reads the future (or can't be shown not to), if luck
explains the result, if the choice fails walk-forward, or if the in-sample winner loses money out
of sample in most splits. Acquitted only when everything clearly passes. The judge model explains
the ruling and cannot change it.

**7. The evidence room.** Commit to one strategy and it gets one test on the sealed 25%, in a fresh
run. The commitment, a SHA-256 hash of the code, parameters, data, trading cost and harness, is
recorded before the seal is opened. Every opening is counted, and repeat openings are corrected for.
"""


# --- components ---
def _silent(*_args) -> None:
    """A log function that discards progress messages."""


def banner(tone: str, title: str, text: str, level: str = "h2") -> None:
    st.markdown(
        f"<div class='banner {tone}'><{level}>{html.escape(title)}</{level}>"
        f"<p>{html.escape(text)}</p></div>",
        unsafe_allow_html=True,
    )


def paragraph(label: str, text: str) -> None:
    """A bold label followed by text that may come from a model."""
    st.markdown(f"<p><b>{html.escape(label)}</b> {html.escape(text)}</p>", unsafe_allow_html=True)


def bullets(items) -> None:
    """A bulleted list of points that may come from a model."""
    points = "".join(f"<li>{html.escape(str(item))}</li>" for item in items)
    st.markdown(f"<ul>{points}</ul>", unsafe_allow_html=True)


def run_safely(what: str, action):
    """Run `action`; if it fails, show what went wrong instead of a stack trace."""
    try:
        return action()
    except Exception as e:
        st.error(f"{what} failed. {type(e).__name__}: {e}")
        return None


def judge_card(column, who: str, says_real: bool, finding: str, basis: str) -> None:
    column.markdown(
        f"<div class='judge {'good' if says_real else 'bad'}'>"
        f"<div class='who'>{html.escape(who)}</div>"
        f"<div class='what'>{'Says it is real' if says_real else 'Not convinced'}</div>"
        f"<div>{html.escape(finding)}</div><div class='how'>{html.escape(basis)}</div></div>",
        unsafe_allow_html=True,
    )


def four_judges(expedition) -> None:
    """The four verdicts on a fisherman's claim, side by side."""
    court, sealed = expedition.court, expedition.sealed
    if court.reasons:
        court_finding = f"{court.verdict}: {'; '.join(court.reasons)}"
    else:
        court_finding = f"{court.verdict}, {fmt_p(court.reality_check.p_value)}"
    if sealed is None:
        sealed_finding = "not opened"
    else:
        sealed_finding = f"{sealed.verdict}, Sharpe {sealed.sealed_sharpe_annual:.2f}"
    columns = st.columns(4)
    judge_card(
        columns[0],
        "Standard backtest test",
        expedition.naive_p_value <= P_VALUE_TO_ACQUIT,
        fmt_p(expedition.naive_p_value),
        "Tests the claimed strategy as if it were the only one",
    )
    judge_card(
        columns[1],
        "Check of the final grid only",
        expedition.final_grid_p_value <= P_VALUE_TO_ACQUIT,
        fmt_p(expedition.final_grid_p_value),
        "What selective reporting of the last attempt shows",
    )
    judge_card(
        columns[2],
        "Overfit Court",
        court.verdict == ACQUITTED,
        court_finding,
        f"Counts all {expedition.n_configs} branches "
        f"({expedition.distinct_bets:.0f} distinct bets)",
    )
    judge_card(
        columns[3],
        "Sealed evidence",
        sealed is not None and sealed.verdict == CONFIRMED,
        sealed_finding,
        "Data no branch ever saw",
    )


def luck_chart(expedition, height: int = 280) -> None:
    """The fisherman's best Sharpe so far against what luck alone would give."""
    table = expedition.attempts_table()
    series = ["running_best_annual"]
    if table["luck_annual"].notna().any():
        series.append("luck_annual")
    long = table.melt(id_vars="k", value_vars=series, var_name="series", value_name="sharpe")
    long["series"] = long["series"].map(
        {"running_best_annual": "fisherman's best so far", "luck_annual": "luck alone"}
    )
    lines = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", title="attempt", axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y("sharpe:Q", title="annualised Sharpe"),
            color=alt.Color("series:N", legend=alt.Legend(orient="top", title=None)),
        )
    )
    target = (
        alt.Chart(pd.DataFrame({"y": [expedition.target]}))
        .mark_rule(strokeDash=[6, 4])
        .encode(y="y:Q")
    )
    st.altair_chart((lines + target).properties(height=height), width="stretch")


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
                eigenvalues > spectrum.noise_edge, "real common structure", "within noise"
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
                    domain=["real common structure", "within noise"], range=["#2563eb", "#94a3b8"]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    edge = (
        alt.Chart(pd.DataFrame({"y": [spectrum.noise_edge]}))
        .mark_rule(strokeDash=[6, 4], color="#f59e0b")
        .encode(y=alt.Y("y:Q", scale=log_scale))
    )
    return (columns + edge).properties(height=300)


def equity_chart(returns: pd.DataFrame, every: int = 5, height: int = 300):
    """Growth of 1 for every column of daily returns (one point in `every` days, no legend)."""
    growth = (1 + returns).cumprod().iloc[::every]
    long = growth.reset_index(names="date").melt(
        id_vars="date", var_name="configuration", value_name="growth"
    )
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=1, opacity=0.6)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("growth:Q", title="growth of 1", scale=alt.Scale(zero=False)),
            color=alt.Color("configuration:N", legend=None),
        )
        .properties(height=height)
    )


def sealed_banner(sealed) -> None:
    banner(
        TONE[sealed.verdict],
        f"Sealed evidence: {sealed.verdict}",
        f"Sharpe {sealed.open_sharpe_annual:.2f} on the open evidence, "
        f"{sealed.sealed_sharpe_annual:.2f} on {sealed.sealed_days} sealed days it never saw "
        f"({fmt_p(sealed.p_value)}). Commitment {sealed.commitment[:16]}…",
        "h3",
    )


def time_travel_panel(trial) -> None:
    report = trial.time_travel or {}
    if not trial.passed_time_travel:
        title = "Time travel" if trial.time_traveller else "Time travel: could not be verified"
        banner("bad", title, trial.evidence()["time_travel_detail"], "h3")
    elif trial.caught_and_repaired:
        first = trial.code_history[0]
        repairs = len(trial.code_history) - 1
        paragraph("⏳ Caught on attempt 1:", first["problem"])
        st.markdown(
            f"**Repaired by Nemotron 3 Super** in {repairs} "
            f"{'attempt' if repairs == 1 else 'attempts'}, then re-tested: no time travel."
        )
        with st.expander("What changed in the code"):
            diff = difflib.unified_diff(
                first["code"].splitlines(True),
                trial.code.splitlines(True),
                "first draft",
                "after repair",
                n=2,
            )
            st.code("".join(diff), language="diff")
    else:
        cuts = report["expected_checks"] // max(1, report["configs"])
        st.markdown(
            f"**⏳ Time-travel test: passed.** All {report['configs']} configurations were re-run "
            f"on history cut short at {cuts} points, and every past position stayed exactly the "
            "same, so the strategy never used future data."
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
        seed = int(st.number_input("Seed", 0, 10_000, 7))
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
        where = st.radio(
            "Run backtests in",
            ["Token Factory Sandboxes", "Local (dev only)"],
            index=0 if can_use_sandboxes and settings.executor == "nebius" else 1,
            disabled=not can_use_sandboxes,
            help="Sandboxes need NEBIUS_API_KEY and a project with sandbox access.",
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
            f"**{len(history)} trial(s) on this data, {sum(h['n'] for h in history)} "
            f"configurations, {len(seals)} seal opening(s).**"
        )
        first_shown = max(1, len(history) - 7)
        for i, item in enumerate(history[-8:], first_shown):
            idea = item["idea"] if len(item["idea"]) <= 55 else item["idea"][:54].rstrip() + "…"
            st.markdown(
                f"<div class='record'>{i}. {html.escape(item['verdict'])}: "
                f"{html.escape(idea)}</div>",
                unsafe_allow_html=True,
            )
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

    st.markdown("<div class='act'>Act 1</div>", unsafe_allow_html=True)
    st.subheader("🎣 An AI is told to find alpha in pure noise")
    st.markdown(
        "Ten years of prices generated from noise: there is nothing to find. A fisherman agent "
        "tries strategy after strategy, each a new branch, until it hits a Sharpe of 1."
    )
    if st.button("Send the fisherman", key="act1"):
        with st.spinner("Fishing..."):
            act = run_safely("Act 1", lambda: stories.act_fishing(settings, ledger, log=_silent))
        if act is not None:
            story["fish"] = act
    if "fish" in story:
        expedition = story["fish"]
        st.markdown(
            f"After **{len(expedition.attempts)} attempts** and **{expedition.n_configs} "
            f"branches** it claims **{expedition.claimed_family} {expedition.claimed_params}** "
            f"with Sharpe **{expedition.claimed_sharpe_annual:.2f}**. The market is pure noise, "
            "so the right answer from every judge is *not convinced*."
        )
        four_judges(expedition)
        luck_chart(expedition, 240)
        st.caption(
            "Scripted fisherman, so the story is reproducible. Send Nemotron 3 Super fishing in "
            "the Fishing expedition tab."
        )

    st.markdown("<div class='act'>Act 2</div>", unsafe_allow_html=True)
    st.subheader("⏳ A strategy that reads the future")
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
        left, right = st.columns(2)
        with left:
            banner(
                "fooled",
                f"Statistics alone: {alone.verdict}",
                f"Fooled. Backtest Sharpe {act.leaky_sharpe_annual:.2f} against "
                f"{alone.reality_check.null_median_annual:.2f} from luck "
                f"({fmt_p(alone.reality_check.p_value)}). Every multiple-testing check passes, "
                "on pure noise, because the code is cheating.",
                "h3",
            )
        with right:
            banner(
                TONE[trial.verdict],
                f"With the time-travel test: {trial.verdict}",
                f"Caught, sent back to Nemotron Super, repaired. The honest version has Sharpe "
                f"{evidence['best_sharpe_annual']:.2f} against "
                f"{evidence['luck_sharpe_annual']:.2f} from luck.",
                "h3",
            )
        time_travel_panel(trial)

    st.markdown("<div class='act'>Act 3</div>", unsafe_allow_html=True)
    st.subheader("📈 A real edge survives")
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
        banner(
            TONE[trial.verdict],
            trial.verdict,
            f"Sharpe {evidence['best_sharpe_annual']:.2f} against "
            f"{evidence['luck_sharpe_annual']:.2f} from luck ({fmt_p(evidence['p_value'])}). "
            f"Time-travel test {evidence['time_travel']}.",
            "h3",
        )
        if trial.sealed:
            sealed_banner(trial.sealed)
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
                    idea, prices, settings=settings, ledger=ledger, executor=executor, log=st.write
                )
                status.update(label="Verdict reached", state="complete")
            except Exception as e:  # show a mistrial instead of a stack trace
                status.update(label="Mistrial", state="error")
                st.error(f"{type(e).__name__}: {e}")

    trial = st.session_state.get("trial")
    if not trial:
        return
    e = trial.evidence()
    banner(TONE[trial.verdict], trial.verdict, trial.opinion.get("headline", ""))
    if e["failed_checks"]:
        paragraph("Failed:", "; ".join(e["failed_checks"]) + ".")

    metrics = st.columns(6)
    metrics[0].metric(
        "Best Sharpe",
        f"{e['best_sharpe_annual']:.2f}",
        help="The best configuration's annualised Sharpe, on the open evidence only",
    )
    raised = e["luck_sharpe_annual"] - e["luck_sharpe_annual_this_trial_only"]
    metrics[1].metric(
        "Luck alone gives",
        f"{e['luck_sharpe_annual']:.2f}",
        delta=f"+{raised:.2f} from your record" if raised > 0.005 else None,
        delta_color="inverse",
        help="Median best Sharpe of zero-edge versions of everything tried on this data",
    )
    metrics[2].metric(
        "Reality Check p",
        fmt_p(e["p_value"], label=False),
        help="How often zero-edge versions of everything tried on this data do as well as "
        "your pick, in bootstrap resamples. Small is good. It is not the probability that "
        "the edge is real.",
    )
    metrics[3].metric(
        "Distinct bets",
        f"{e['distinct_bets_total']:.0f}",
        help=f"from {e['n_distinct_configs_on_this_data']} distinct configurations tried on "
        f"this data ({e['n_configs_tried_on_this_data']} runs, repeats included)",
    )
    metrics[4].metric(
        "Loses out of sample",
        f"{e['prob_oos_loss']:.0%}",
        help="How often the in-sample winner loses money out of sample (CSCV splits)",
    )
    metrics[5].metric(
        "Years to prove it",
        fmt_years(e["years_of_live_trading_to_prove_it"]),
        help="How long a live record at the walk-forward Sharpe would need to run to show, "
        "with 95% confidence, that the Sharpe is above zero",
    )

    time_travel_panel(trial)

    left, right = st.columns(2)
    with left:
        st.subheader("Every backtest you could have cherry-picked")
        st.altair_chart(equity_chart(trial.returns), width="stretch")
        st.caption(f"Open evidence only. Best: `{e['best_params']}`.")
    with right:
        st.subheader("How many different bets was that?")
        st.altair_chart(spectrum_chart(trial.findings.spectrum), width="stretch")
        st.caption(
            f"{e['n_distinct_configs_on_this_data']} distinct configurations, about "
            f"{e['distinct_bets_total']:.0f} distinct bets. Eigenvalues above the dashed line "
            "are real common structure; below it is what unrelated strategies with the same "
            "volatility would show by chance."
        )

    st.subheader("The judgment")
    bullets(trial.opinion.get("reasoning", []))
    st.markdown("**What would change the verdict**")
    bullets(trial.opinion.get("what_would_change_my_mind", []))

    st.subheader("🔒 The evidence room")
    render_evidence_room(trial, ledger, executor)

    with st.expander(f"Look-ahead audit: {trial.audit['risk']} risk"):
        bullets(trial.audit["issues"] or ["No issues found."])
    with st.expander("Prior art (Tavily)"):
        st.text(trial.prior_art.get("summary") or "")
        links = "".join(
            f"<li><a href='{html.escape(c['url'])}' target='_blank' rel='noopener'>"
            f"{html.escape(c['title'])}</a></li>"
            for c in trial.prior_art.get("citations", [])
            if c["url"].startswith(("https://", "http://"))
        )
        if links:
            st.markdown(f"<ul>{links}</ul>", unsafe_allow_html=True)
    with st.expander("Strategy code"):
        st.code(trial.code, language="python")
    with st.expander("Model routing, sandboxes and spend"):
        st.dataframe(pd.DataFrame(trial.cost_table), width="stretch")
        st.write({"executor": trial.executor, **trial.execution})
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
        sealed_banner(trial.sealed)
        dates = trial.prices_full.index[trial.prices_full.index >= trial.cut]
        sealed = pd.DataFrame({"committed": trial.sealed.sealed_returns}, index=dates)
        st.altair_chart(equity_chart(sealed, every=1, height=200), width="stretch")
        if trial.sealed.times_opened_before:
            st.caption(f"Corrected for {trial.sealed.times_opened_before} earlier opening(s).")
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
            f"The seal on this data has been opened {openings} time(s) already. "
            "The court will correct for that."
        )
    if not trial.passed_time_travel:
        st.caption("Shut: a strategy that may read the future could read the sealed evidence too.")
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
        "until something works. Send a fisherman out with a target and watch four judges look at "
        "what it brings back."
    )
    target_column, attempts_column, fisherman_column = st.columns(3)
    target = target_column.slider("Target Sharpe", 0.6, 1.5, 1.0, 0.05)
    max_attempts = attempts_column.slider("Attempts allowed", 5, 60, 40)
    who = fisherman_column.radio(
        "Fisherman",
        ["Scripted (explore + hill-climb)", "Nemotron 3 Super"],
        disabled=settings.mock_llm,
        index=0 if settings.mock_llm else 1,
    )
    if st.button("Send the fisherman out", type="primary", disabled=prices is None):
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
                    log=st.write,
                )
                status.update(label="The fisherman is back", state="complete")
            except Exception as e:
                status.update(label="The expedition failed", state="error")
                st.error(f"{type(e).__name__}: {e}")

    expedition = st.session_state.get("expedition")
    if not expedition:
        return
    summary = expedition.summary()
    st.subheader(
        f"{len(expedition.attempts)} attempts, {expedition.n_configs} branches, "
        f"{expedition.distinct_bets:.0f} distinct bets"
    )
    luck_chart(expedition)
    st.caption(
        "The dashed line is the target. When the fisherman's best only keeps pace with luck, it "
        "has found nothing."
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
    st.markdown(CSS, unsafe_allow_html=True)
    settings = get_settings()
    st.session_state.setdefault("ledger", Ledger())
    st.session_state.setdefault("story", {})
    ledger: Ledger = st.session_state.ledger

    prices, seed, use_sandboxes, record_box = sidebar(settings)
    # Model-written code never runs in-process: sandboxes when available, else subprocesses.
    executor = NebiusSandboxExecutor() if use_sandboxes else LocalExecutor()

    st.title("⚖️ Overfit Court")
    st.caption(
        "AI agents can now try a thousand trading ideas before lunch. Overfit Court counts every "
        "one of them, tests that none of them read the future, keeps a sealed slice of history "
        "they never see, and lets the statistics, not the model, decide."
    )
    if settings.mock_llm:
        st.info(
            "Demo mode: no NEBIUS_API_KEY is set, so model calls use canned responses and ideas "
            "are routed to a built-in strategy library. Everything else is the real court."
        )

    story, trial, fishing, how = st.tabs(
        ["▶ Start here", "⚖️ Put an idea on trial", "🎣 Fishing expedition", "📜 How it works"]
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
