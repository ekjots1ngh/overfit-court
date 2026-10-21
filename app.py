"""Streamlit front end: streamlit run app.py"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from overfit_court.config import get_settings
from overfit_court.data import load_csv, synthetic_prices
from overfit_court.fishing import NemotronFisherman, ScriptedFisherman, run_expedition
from overfit_court.ledger import Ledger
from overfit_court.llm import LLM
from overfit_court.pipeline import SEALED_FRAC, record_key, run_trial, unseal
from overfit_court.sandbox import InProcessExecutor, LocalExecutor, NebiusSandboxExecutor

st.set_page_config(page_title="Overfit Court", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
.banner {padding: 1rem 1.3rem; border-radius: 14px; margin: .5rem 0 1rem 0;}
.banner h2, .banner h3 {margin: 0; letter-spacing: .05em;}
.banner p {margin: .35rem 0 0 0; font-size: 1.02rem;}
.CONVICTED, .REFUTED, .no {background: rgba(220,38,38,.12); border: 1px solid rgba(220,38,38,.45);}
.HUNG, .INCONCLUSIVE {background: rgba(217,119,6,.12); border: 1px solid rgba(217,119,6,.45);}
.ACQUITTED, .CONFIRMED, .yes {background: rgba(22,163,74,.12); border: 1px solid rgba(22,163,74,.45);}
.judge {padding: .8rem 1rem; border-radius: 12px; height: 100%;}
.judge .who {font-size: .85rem; opacity: .75; text-transform: uppercase; letter-spacing: .06em;}
.judge .what {font-size: 1.15rem; font-weight: 600; margin: .25rem 0;}
.record {font-size: .9rem; opacity: .85;}
</style>""", unsafe_allow_html=True)

s = get_settings()
if "ledger" not in st.session_state:
    st.session_state.ledger = Ledger()          # per-session memory in the hosted demo
ledger: Ledger = st.session_state.ledger


def banner(kind: str, title: str, text: str, level: str = "h2"):
    st.markdown(f"<div class='banner {kind}'><{level}>{title}</{level}><p>{text}</p></div>",
                unsafe_allow_html=True)


def spectrum_chart(spec, max_bars: int = 40):
    lam = np.array(spec.eigenvalues[:max_bars])
    df = pd.DataFrame({"rank": np.arange(1, len(lam) + 1), "eigenvalue": np.maximum(lam, 1e-3),
                       "kind": np.where(lam > spec.mp_upper, "structure", "noise band")})
    band = pd.DataFrame({"lo": [max(spec.mp_lower, 1e-3)], "hi": [spec.mp_upper]})
    y = alt.Y("eigenvalue:Q", scale=alt.Scale(type="log"), title="eigenvalue (log)")
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("rank:O", title="eigenvalue rank", axis=alt.Axis(labels=False, ticks=False)),
        y=y, color=alt.Color("kind:N", scale=alt.Scale(domain=["structure", "noise band"],
                                                        range=["#2563eb", "#94a3b8"]),
                             legend=alt.Legend(orient="top", title=None)))
    rect = alt.Chart(band).mark_rect(opacity=.15, color="#f59e0b").encode(y="lo:Q", y2="hi:Q")
    return (rect + bars).properties(height=300)


def load_prices(source, upload, seed):
    if source == "Upload CSV":
        return load_csv(upload) if upload is not None else None
    return synthetic_prices(seed=int(seed), signal=0.0008 if "planted" in source else 0.0)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Evidence")
    source = st.radio("Price data", ["Synthetic: pure noise", "Synthetic: planted edge",
                                     "Upload CSV"])
    upload = st.file_uploader("Daily closes, one column per asset", type="csv") \
        if source == "Upload CSV" else None
    seed = st.number_input("Seed", 0, 10_000, 7)
    st.caption(f"The last {int(SEALED_FRAC * 100)}% of dates is sealed: it is never uploaded "
               "to the research sandboxes.")
    st.header("Procedure")
    s.max_configs = st.slider("Max configurations per trial", 20, 200, s.max_configs)
    s.cost_bps = st.slider("Cost per unit turnover (bps)", 0.0, 20.0, s.cost_bps)
    can_sandbox = bool(s.nebius_api_key)
    run_where = st.radio("Run backtests in", ["Token Factory Sandboxes", "Local (dev only)"],
                         index=0 if can_sandbox and s.executor == "nebius" else 1,
                         disabled=not can_sandbox,
                         help="Sandboxes need NEBIUS_API_KEY and a project with sandbox access.")
    st.header("Court record")
    record_box = st.container()

prices = load_prices(source, upload, seed)
use_sandbox = run_where.startswith("Token") and can_sandbox

st.title("⚖️ Overfit Court")
st.caption("AI agents can now try a thousand trading ideas before lunch. Overfit Court counts "
           "every one of them, keeps a sealed slice of history they never see, and lets the "
           "statistics, not the model, decide whether an edge is real.")
if s.mock_llm:
    st.info("Demo mode: no NEBIUS_API_KEY is set, so model calls use canned responses and ideas "
            "are routed to a built-in strategy library.")

tab_trial, tab_fish, tab_how = st.tabs(["⚖️ Put an idea on trial", "🎣 The fishing expedition",
                                        "📜 How the court works"])

# ------------------------------------------------------------------ trial tab
with tab_trial:
    idea = st.text_area("Your idea", "Buy the assets that went up most over the last few months "
                        "and short the ones that went down most.", height=80)
    if st.button("Begin trial", type="primary", disabled=prices is None):
        ex = NebiusSandboxExecutor() if use_sandbox else LocalExecutor()
        with st.status("Court in session...", expanded=True) as status:
            try:
                st.session_state.trial = run_trial(idea, prices, s, ledger, ex, log=st.write)
                status.update(label="Verdict reached", state="complete")
            except Exception as e:
                status.update(label="Mistrial", state="error")
                st.error(f"{type(e).__name__}: {e}")

    trial = st.session_state.get("trial")
    if trial:
        e = trial.evidence()
        banner(trial.verdict.split()[0], trial.verdict, trial.opinion.get("headline", ""))
        c = st.columns(6)
        c[0].metric("Best backtest Sharpe", f"{e['best_sharpe_annual']:.2f}",
                    help="Annualised, on the open evidence only")
        d = e["luck_sharpe_annual"] - e["luck_sharpe_annual_this_trial_only"]
        c[1].metric("Luck alone would give", f"{e['luck_sharpe_annual']:.2f}",
                    delta=f"+{d:.2f} from your record" if d > 0.005 else None,
                    delta_color="inverse",
                    help="Median best Sharpe of zero-edge versions of everything tried on this data")
        c[2].metric("P(edge is real)", f"{e['p_real']:.0%}", help="Reality Check over the whole record")
        c[3].metric("Distinct bets", f"{e['distinct_bets_total']:.0f}",
                    help=f"of {e['n_configs_total_on_this_data']} configurations tried on this data")
        c[4].metric("Winner loses out of sample", f"{e['prob_oos_loss']:.0%}")
        yrs = e["years_of_live_trading_to_prove_it"]
        c[5].metric("Years to prove it live", "never" if yrs is None else f"{yrs:.1f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Every backtest you could have cherry-picked")
            st.line_chart((1 + trial.returns).cumprod(), height=300)
            st.caption(f"Open evidence only. Best: {e['best_params']}.")
        with right:
            st.subheader("How many different bets was that?")
            st.altair_chart(spectrum_chart(trial.j.spec_all), width="stretch")
            st.caption(f"{e['n_configs_total_on_this_data']} configurations, "
                       f"{e['distinct_bets_total']:.0f} distinct bets. Eigenvalues inside the "
                       "shaded Marchenko-Pastur band are what pure sampling noise produces.")

        st.subheader("The judgment")
        for r in trial.opinion.get("reasoning", []):
            st.markdown(f"- {r}")
        st.markdown("**What would change the verdict:** " +
                    "; ".join(trial.opinion.get("what_would_change_my_mind", [])))

        st.subheader("🔒 The evidence room")
        if trial.sealed is None:
            opened = len(ledger.seals(trial.record["dataset"]))
            st.markdown(f"The last {int(trial.sealed_frac * 100)}% of the data (from "
                        f"{e['sealed_from']}) has never been inside a research sandbox. Commit "
                        "to the winning configuration and it gets exactly one test there.")
            if opened:
                st.warning(f"The seal on this data has been opened {opened} time(s) already. "
                           "The court will correct for that.")
            if st.button("Commit and open the evidence room"):
                ex = NebiusSandboxExecutor() if use_sandbox else LocalExecutor()
                with st.spinner("Opening a fresh sandbox with the sealed evidence..."):
                    unseal(trial, ex, ledger, s, log=lambda *a: None)
                st.rerun()
        else:
            sr = trial.sealed
            banner(sr.verdict, f"Sealed evidence: {sr.verdict}",
                   f"Sharpe {sr.open_sharpe_annual:.2f} on the open evidence, "
                   f"{sr.sealed_sharpe_annual:.2f} on {sr.sealed_days} sealed days it never saw. "
                   f"P(real) {sr.p_real:.0%}. Commitment {sr.commitment}.", "h3")
            st.line_chart(pd.Series(np.cumprod(1 + np.array(sr.sealed_returns)),
                                    name="sealed evidence"), height=200)
            if sr.times_opened_before:
                st.caption(f"Corrected for {sr.times_opened_before} earlier opening(s).")

        with st.expander(f"Look-ahead audit: {trial.audit['risk']} risk"):
            st.write(trial.audit["issues"] or "No issues found.")
        with st.expander("Prior art (Tavily)"):
            st.write(trial.prior_art.get("summary"))
            for cit in trial.prior_art.get("citations", []):
                st.markdown(f"- [{cit.get('title')}]({cit.get('url')})")
        with st.expander("Strategy code written by Nemotron"):
            st.code(trial.code, language="python")
        with st.expander("Model routing, sandboxes and spend"):
            st.dataframe(pd.DataFrame(trial.cost_table), width="stretch")
            st.write({"executor": trial.executor, **trial.exec_summary})
        st.download_button("Download the full report (JSON)", trial.to_json(),
                           "overfit_court_report.json")

# ------------------------------------------------------------------ fishing tab
with tab_fish:
    st.markdown("Sandboxes that branch like git make it nearly free for an agent to try path "
                "after path until something works. Send a fisherman out with a target and "
                "watch four judges look at what it brings back.")
    f1, f2, f3 = st.columns(3)
    target = f1.slider("Target Sharpe", 0.6, 1.5, 1.0, 0.05)
    attempts = f2.slider("Attempts allowed", 5, 60, 40)
    who = f3.radio("Fisherman", ["Scripted (explore + hill-climb)", "Nemotron 3 Super"],
                   disabled=s.mock_llm, index=0 if s.mock_llm else 1)
    if st.button("Send the fisherman out", type="primary", disabled=prices is None):
        fisher = (NemotronFisherman(LLM(s), fallback_seed=int(seed))
                  if who.startswith("Nemotron") and not s.mock_llm else ScriptedFisherman(int(seed)))
        ex = NebiusSandboxExecutor() if use_sandbox else InProcessExecutor()
        with st.status("Fishing...", expanded=False) as status:
            st.session_state.expedition = run_expedition(
                prices, target, attempts, fisher, s, ledger, ex, track_luck=True,
                seed=int(seed), log=st.write)
            status.update(label="The fisherman is back", state="complete")

    exp = st.session_state.get("expedition")
    if exp:
        sm = exp.summary()
        tbl = exp.attempts_table()
        long = tbl.melt(id_vars="k", value_vars=["running_best_annual", "luck_annual"],
                        var_name="series", value_name="sharpe").replace(
            {"running_best_annual": "fisherman's best so far", "luck_annual": "luck alone"})
        chart = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("k:Q", title="attempt"), y=alt.Y("sharpe:Q", title="annualised Sharpe"),
            color=alt.Color("series:N", legend=alt.Legend(orient="top", title=None)))
        rule = alt.Chart(pd.DataFrame({"y": [exp.target]})).mark_rule(strokeDash=[6, 4]).encode(y="y:Q")
        st.subheader(f"{len(exp.attempts)} attempts, {exp.n_configs} branches, "
                     f"{exp.distinct_bets:.0f} distinct bets")
        st.altair_chart((chart + rule).properties(height=300), width="stretch")
        st.caption("The dashed line is the target. When the fisherman's best only keeps pace "
                   "with luck, it has found nothing.")

        st.markdown(f"**The claim:** {sm['claimed']} with Sharpe "
                    f"{sm['claimed_sharpe_annual']:.2f} on the open evidence.")
        j = st.columns(4)
        judges = [
            ("Standard backtest test", exp.naive_p_real >= 0.95,
             f"P(real) {exp.naive_p_real:.0%}", "Tests the claimed strategy as if it were the only one"),
            ("Check of the final grid only", exp.final_grid_p_real >= 0.95,
             f"P(real) {exp.final_grid_p_real:.0%}", "What selective reporting of the last attempt shows"),
            ("Overfit Court", exp.court.verdict == "ACQUITTED",
             f"{exp.court.verdict}, P(real) {exp.court.rc.p_real:.0%}",
             f"Counts all {exp.n_configs} branches"),
            ("Sealed evidence", exp.sealed is not None and exp.sealed.verdict == "CONFIRMED",
             "not opened" if exp.sealed is None else
             f"{exp.sealed.verdict}, Sharpe {exp.sealed.sealed_sharpe_annual:.2f}",
             "Data no branch ever saw"),
        ]
        for col, (who_, real, what, how) in zip(j, judges):
            col.markdown(f"<div class='judge {'yes' if real else 'no'}'><div class='who'>{who_}"
                         f"</div><div class='what'>{'Says it is real' if real else 'Not convinced'}"
                         f"</div><div>{what}</div><div style='opacity:.7;font-size:.85rem'>{how}"
                         f"</div></div>", unsafe_allow_html=True)
        with st.expander("Every attempt"):
            st.dataframe(tbl, width="stretch")

# ------------------------------------------------------------------ how tab
with tab_how:
    st.markdown("""
**1. The model plans and codes; the harness keeps it honest.** Nemotron 3 Ultra turns an idea into
a parameter grid, Nemotron 3 Super writes the strategy, Nemotron 3 Nano audits it for look-ahead
bias. The harness applies the one-day lag and trading costs itself, so generated code cannot cheat.

**2. Research happens in sandboxes that cannot see the future.** Only the open 75% of the data is
uploaded to Token Factory Sandboxes. Every configuration runs in a branch of one checkpoint.

**3. The court remembers.** Every configuration ever tried on this data is kept. The Reality Check
asks how good the best of all of them would look if none had any edge, using a block bootstrap
that preserves fat tails, volatility clustering and the correlation between strategies. Re-running
the same grid adds nothing; genuinely new ideas raise the bar.

**4. Random matrix theory counts your real bets.** The correlation spectrum of everything tried is
compared with the Marchenko-Pastur law. What pure noise explains is removed; what is left is the
number of genuinely independent bets.

**5. The statistics rule.** Convicted if luck explains the result, if the choice fails
walk-forward, or if the in-sample winner loses money out of sample in most splits. Acquitted only
when all three clearly pass. The judge model explains the ruling and cannot change it.

**6. The evidence room.** Commit to one strategy and it gets one test on the sealed 25%, in a fresh
sandbox. Every opening is recorded, and repeat openings are corrected for.
""")

# ------------------------------------------------------------------ court record
with record_box:
    if prices is not None:
        key = record_key(prices, SEALED_FRAC)
        hist, seals = ledger.history(key), ledger.seals(key)
        if hist:
            st.markdown(f"**{len(hist)} trial(s) on this data, "
                        f"{sum(h['n'] for h in hist)} configurations, "
                        f"{len(seals)} seal opening(s).**")
            for i, h in enumerate(hist[-8:], max(1, len(hist) - 7)):
                st.markdown(f"<div class='record'>{i}. {h['verdict']}: {h['idea'][:55]}</div>",
                            unsafe_allow_html=True)
            if st.button("Wipe the record for this data"):
                ledger.clear(key)
                for k in ("trial", "expedition"):
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            st.caption("No trials on this data yet. Every attempt will count against the next.")
