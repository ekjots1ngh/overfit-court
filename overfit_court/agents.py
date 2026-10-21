"""The courtroom. Each role is pinned to the cheapest Nemotron tier that can do it.

  Prosecutor/planner  -> Ultra  (turns an idea into a testable spec + grid)
  Engineer/coder      -> Super  (writes the strategy function, repairs it)
  Auditor             -> Nano   (fast look-ahead bias review of the code)
  Clerk               -> Nano   (triages Tavily prior-art hits)
  Judge               -> Ultra  (explains the statistical verdict in plain English)
"""
from __future__ import annotations

import json
import re

from . import library
from .stats import fmt_prob
from .llm import LLM, extract_code

# ---------------------------------------------------------------- planner
PLANNER_SYS = """You are a quantitative research lead. Turn a trading idea into a
testable spec for a daily-bar, multi-asset backtest. Reply with JSON only:
{"name": str, "hypothesis": str, "signal_description": str,
 "param_grid": {"param_name": [values, ...], ...},
 "search_queries": [str, str],
 "known_risks": [str, ...]}
Rules: 2-4 parameters, total grid size between 30 and 150 combinations, values
are ints or floats. search_queries should find academic papers or practitioner
write-ups of this exact effect."""

MOCK_SPEC = library.spec_for(library.FAMILIES["momentum"])   # demo mode default


class SpecError(ValueError):
    pass


def validate_spec(spec: dict, max_params: int = 4) -> dict:
    """Coerce a model-written spec into something the harness can run."""
    if not isinstance(spec, dict):
        raise SpecError("spec is not a JSON object")
    grid = spec.get("param_grid")
    if not isinstance(grid, dict) or not grid:
        raise SpecError("param_grid must be a non-empty object of lists")
    clean = {}
    for name, values in grid.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise SpecError(f"parameter name {name!r} is not a valid python identifier")
        if not isinstance(values, list):
            values = [values]
        vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        vals = [int(v) if float(v).is_integer() else float(v) for v in vals]  # 60.0 -> 60 for windows
        if not vals:
            raise SpecError(f"parameter {name!r} has no numeric values")
        clean[name] = sorted(set(vals))
    if len(clean) > max_params:
        raise SpecError(f"{len(clean)} parameters; use at most {max_params}")
    queries = spec.get("search_queries") or []
    out = dict(spec)
    out["param_grid"] = clean
    out["search_queries"] = [q for q in queries if isinstance(q, str)][:3]
    out.setdefault("name", "Strategy")
    out.setdefault("hypothesis", "")
    return out


def plan(llm: LLM, idea: str, retries: int = 1) -> dict:
    user = f"Idea:\n{idea}"
    for attempt in range(retries + 1):
        try:
            raw = llm.chat_json("planner", "ultra", PLANNER_SYS, user,
                                mock=lambda: json.dumps(library.spec_for(library.route(idea))))
            return validate_spec(raw)
        except (SpecError, ValueError) as e:
            if attempt == retries:
                raise SpecError(f"planner could not produce a valid spec: {e}") from e
            user = f"Idea:\n{idea}\n\nYour previous reply was invalid: {e}. Reply with valid JSON only."


# ---------------------------------------------------------------- coder
CODER_SYS = """You write trading signal code. Output ONE python code block defining:

def positions(prices: pd.DataFrame, **params) -> pd.DataFrame

- prices: daily close prices, DatetimeIndex rows, one column per asset.
- Return target weights with the same index and columns, values in [-1, 1].
- Row t may only use prices up to and including row t. The harness lags your
  weights by one day and charges costs, so do NOT shift them yourself.
- Only import numpy and pandas. No I/O, no randomness, no network.
- Handle warm-up NaNs by returning 0 weight."""

MOCK_CODE = library.FAMILIES["momentum"].code


def write_code(llm: LLM, spec: dict, error: str | None = None, previous: str | None = None) -> str:
    user = f"Spec:\n{json.dumps(spec, indent=2)}\n\nParameters: {list(spec['param_grid'])}"
    if error:
        user += f"\n\nYour previous code failed with:\n{error}\n\nPrevious code:\n{previous}\nFix it."
    fam = library.FAMILIES.get(spec.get("family", library.DEFAULT), library.FAMILIES[library.DEFAULT])
    mock_code = fam.fixed_code if (error and fam.fixed_code) else fam.code
    text = llm.chat("coder" if not error else "coder-repair", "super", CODER_SYS, user,
                    mock=lambda: "```python\n" + mock_code + "```", max_tokens=3000)
    return extract_code(text)


# ---------------------------------------------------------------- auditor
LOOKAHEAD_PATTERNS = [
    (r"shift\(\s*-\s*\d+", "negative shift reads future rows"),
    (r"\.bfill\(|fillna\([^)]*method\s*=\s*['\"](bfill|backfill)", "back-fill leaks future values"),
    (r"iloc\[[^\]]*\+\s*1", "indexing i+1 may read tomorrow"),
    (r"center\s*=\s*True", "centred rolling window uses future data"),
    (r"import\s+(os|sys|subprocess|requests|socket)", "forbidden import"),
]

AUDITOR_SYS = """You audit trading code for look-ahead bias and data leakage.
Reply JSON only: {"risk": "low"|"medium"|"high", "issues": [str, ...]}.
Full-sample normalisation, future shifts, back-fills and centred windows are high risk."""


def audit(llm: LLM, code: str) -> dict:
    static = [msg for pat, msg in LOOKAHEAD_PATTERNS if re.search(pat, code, re.MULTILINE)]
    review = llm.chat_json("auditor", "nano", AUDITOR_SYS, code,
                           mock=lambda: json.dumps({"risk": "high" if static else "low",
                                                    "issues": static}), max_tokens=800)
    issues = sorted(set(static) | set(review.get("issues", [])))
    risk = review.get("risk", "low")
    if static and risk == "low":
        risk = "medium"
    return {"risk": risk, "issues": issues, "static_hits": static}


# ---------------------------------------------------------------- clerk
CLERK_SYS = """You triage web search results about a trading idea. Reply JSON only:
{"already_known": bool, "summary": str (max 2 sentences),
 "citations": [{"title": str, "url": str}] (max 3, only from the given results)}"""


def triage_prior_art(llm: LLM, spec: dict, hits: list[dict]) -> dict:
    if not hits or not any(h.get("url") for h in hits):
        return {"already_known": None, "summary": "No live prior-art search was run.",
                "citations": []}
    user = f"Idea: {spec.get('hypothesis')}\n\nResults:\n{json.dumps(hits, indent=1)}"
    return llm.chat_json("clerk", "nano", CLERK_SYS, user,
                         mock=lambda: "{}", max_tokens=800)


# ---------------------------------------------------------------- judge
JUDGE_SYS = """You are the judge in a court that tries trading strategies for
overfitting. The verdict has ALREADY been decided by the statistics and you may
not change it. Explain it to a smart non-specialist. Key ideas you may use:
luck_sharpe_annual is the best Sharpe that zero-edge versions of everything
tried would typically reach (a bootstrap Reality Check over the whole court
record, not just this trial); p_real is the probability the edge is real after
that correction; distinct_bets_total is how many genuinely independent bets
the configurations amount to (random matrix theory), which can be far fewer
than the number of configurations; prob_oos_loss is how often the in-sample
winner loses money out of sample; pbo is only about whether the exact
parameters were overfit, and a high pbo with a low prob_oos_loss means the
parameters don't matter but the strategy family works. time_travel reports the
harness test that re-runs the strategy on truncated history: if it FAILED, the
code read the future, the backtest is meaningless whatever its Sharpe, and that
is the headline. Reply JSON only:
{"headline": str, "reasoning": [str, str, str], "what_would_change_my_mind": [str, str]}
Use only numbers that appear in the evidence. Be direct, no hype."""


def judge(llm: LLM, verdict: str, evidence: dict) -> dict:
    def mock():
        e = evidence
        if e.get("time_travel", "").startswith("FAILED"):
            return json.dumps({
                "headline": f"{verdict}: this strategy reads the future. Its backtest Sharpe of "
                            f"{e['best_sharpe_annual']:.2f} is not evidence of anything.",
                "reasoning": [
                    e["time_travel_detail"],
                    "A strategy can only trade on what it knew at the time. When positions on past "
                    "dates change as later data arrives, the backtest is using tomorrow's prices.",
                    f"{e['repairs']} repair attempt(s) did not remove the leak, so the court does not "
                    "weigh the statistics, and the evidence room stays shut: a strategy that reads "
                    "the future would read the sealed evidence too.",
                ],
                "what_would_change_my_mind": ["Rewriting every statistic as a rolling or expanding "
                                              "window over past data only",
                                              "Passing the time-travel test, then the sealed evidence"],
            })
        yrs = e["years_of_live_trading_to_prove_it"]
        memory = ""
        if e["trial_number_on_this_data"] > 1:
            memory = (f" This is trial {e['trial_number_on_this_data']} on this data, and the "
                      f"court counts all {e['n_configs_total_on_this_data']} configurations tried so "
                      f"far (about {e['distinct_bets_total']:.0f} distinct bets).")
        return json.dumps({
            "headline": f"{verdict}: the best backtest Sharpe of {e['best_sharpe_annual']:.2f} "
                        f"against {e['luck_sharpe_annual']:.2f} from luck alone.",
            "reasoning": [
                f"{e['n_configs']} configurations were tried here, but they amount to only about "
                f"{e['distinct_bets_this_trial']:.0f} genuinely different bets. Zero-edge versions "
                f"of them typically reach a best Sharpe of {e['luck_sharpe_annual']:.2f}.{memory}",
                f"After that correction the probability the edge is real is {fmt_prob(e['p_real'])}. "
                + ("No length of live track record would confirm it, because it does not beat luck."
                   if yrs is None else f"Proving it live would take about {yrs} years."),
                f"The in-sample winner lost money out of sample in {e['prob_oos_loss']:.0%} of "
                f"train/test splits, and chosen walk-forward its Sharpe is "
                f"{e['walk_forward_sharpe']:.2f}. (Probability the exact parameters were overfit: "
                f"{e['pbo']:.0%}.)",
            ],
            "what_would_change_my_mind": ["Committing to one configuration and passing the sealed "
                                          "evidence test",
                                          "Similar results on assets or years it has never seen"],
        })
    return llm.chat_json("judge", "ultra", JUDGE_SYS,
                         f"Verdict: {verdict}\nEvidence:\n{json.dumps(evidence, indent=2)}",
                         mock=mock, max_tokens=1200)
