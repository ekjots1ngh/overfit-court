"""The courtroom's model-driven roles, each pinned to the cheapest Nemotron tier that can do it.

    Planner   Nemotron 3 Ultra   turns an idea into a testable spec and grid
    Engineer  Nemotron 3 Super   writes the strategy, and repairs it from errors
    Auditor   Nemotron 3 Nano    reviews the final code for look-ahead bias
    Clerk     Nemotron 3 Nano    triages Tavily prior-art search results
    Judge     Nemotron 3 Ultra   explains the statistical verdict in plain English

None of them decides anything that matters. The verdict comes from `stats`;
look-ahead bias is caught by the harness's time-travel test; the judge may
only explain. Each call also supplies a `mock`: the reply used in demo mode,
when no API key is set, built from the strategy library.
"""

from __future__ import annotations

import json
import re

from . import library
from .llm import LLM, extract_code
from .stats import fmt_prob

# --- planner ---
PLANNER_SYS = """You are a quantitative research lead. Turn a trading idea into a
testable spec for a daily-bar, multi-asset backtest. Reply with JSON only:
{"name": str, "hypothesis": str, "signal_description": str,
 "param_grid": {"param_name": [values, ...], ...},
 "search_queries": [str, str],
 "known_risks": [str, ...]}
Rules: 2-4 parameters, total grid size between 30 and 150 combinations, values
are ints or floats. search_queries should find academic papers or practitioner
write-ups of this exact effect."""


class SpecError(ValueError):
    """The planner's spec can't be turned into a runnable parameter grid."""


def validate_spec(spec: dict, max_params: int = 4) -> dict:
    """Coerce a model-written spec into something the harness can run.

    Non-numeric values are dropped, whole-number floats become ints (so a window
    of 60.0 works with rolling()), and every parameter name must be a valid
    Python identifier because it is passed to positions() as a keyword.
    """
    if not isinstance(spec, dict):
        raise SpecError("spec is not a JSON object")
    grid = spec.get("param_grid")
    if not isinstance(grid, dict) or not grid:
        raise SpecError("param_grid must be a non-empty object of lists")

    clean = {}
    for name, values in grid.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise SpecError(f"parameter name {name!r} is not a valid python identifier")
        values = values if isinstance(values, list) else [values]
        numbers = [v for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
        if not numbers:
            raise SpecError(f"parameter {name!r} has no numeric values")
        clean[name] = sorted({int(v) if float(v).is_integer() else float(v) for v in numbers})
    if len(clean) > max_params:
        raise SpecError(f"{len(clean)} parameters; use at most {max_params}")

    queries = spec.get("search_queries") or []
    return {
        **spec,
        "param_grid": clean,
        "search_queries": [q for q in queries if isinstance(q, str)][:3],
        "name": spec.get("name", "Strategy"),
        "hypothesis": spec.get("hypothesis", ""),
    }


def plan(llm: LLM, idea: str, retries: int = 1) -> dict:
    """Ask the planner for a spec; if it is unusable, say why and ask again."""
    problem = None
    for _ in range(retries + 1):
        user = f"Idea:\n{idea}"
        if problem:
            user += f"\n\nYour previous reply was invalid: {problem}. Reply with valid JSON."
        try:
            reply = llm.chat_json(
                "planner",
                "ultra",
                PLANNER_SYS,
                user,
                mock=lambda: json.dumps(library.spec_for(library.route(idea))),
            )
            return validate_spec(reply)
        except ValueError as e:  # a SpecError, or a reply that isn't JSON at all
            problem = e
    raise SpecError(f"planner could not produce a valid spec: {problem}") from problem


# --- engineer ---
CODER_SYS = """You write trading signal code. Output ONE python code block defining:

def positions(prices: pd.DataFrame, **params) -> pd.DataFrame

- prices: daily close prices, DatetimeIndex rows, one column per asset.
- Return target weights with the same index and columns, values in [-1, 1].
- Row t may only use prices up to and including row t. The harness lags your
  weights by one day and charges costs, so do NOT shift them yourself.
- Only import numpy and pandas. No I/O, no randomness, no network.
- Handle warm-up NaNs by returning 0 weight."""


def write_code(llm: LLM, spec: dict, error: str | None = None, previous: str | None = None) -> str:
    """Write the strategy, or repair `previous` given the `error` it produced.

    `error` is either a crash or the time-travel test's description of a leak,
    which names the parameters and dates involved.
    """
    user = f"Spec:\n{json.dumps(spec, indent=2)}\n\nParameters: {list(spec['param_grid'])}"
    if error:
        user += (
            f"\n\nYour previous code failed with:\n{error}\n\nPrevious code:\n{previous}\nFix it."
        )

    family = library.FAMILIES.get(spec.get("family"), library.FAMILIES[library.DEFAULT])
    demo_code = family.fixed_code if (error and family.fixed_code) else family.code
    reply = llm.chat(
        "coder-repair" if error else "coder",
        "super",
        CODER_SYS,
        user,
        mock=lambda: f"```python\n{demo_code}```",
        max_tokens=3000,
    )
    return extract_code(reply)


# --- auditor ---
# Cheap pattern checks that run before the model review. They are only a first
# line: the time-travel test in the harness is what actually catches leaks.
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
    """Static patterns plus a Nemotron Nano review. A pattern hit is at least medium risk."""
    static = [message for pattern, message in LOOKAHEAD_PATTERNS if re.search(pattern, code)]
    review = llm.chat_json(
        "auditor",
        "nano",
        AUDITOR_SYS,
        code,
        mock=lambda: json.dumps({"risk": "high" if static else "low", "issues": static}),
        max_tokens=800,
    )
    risk = review.get("risk", "low")
    if static and risk == "low":
        risk = "medium"
    return {
        "risk": risk,
        "issues": sorted(set(static) | set(review.get("issues", []))),
        "static_hits": static,
    }


# --- clerk ---
CLERK_SYS = """You triage web search results about a trading idea. Reply JSON only:
{"already_known": bool, "summary": str (max 2 sentences),
 "citations": [{"title": str, "url": str}] (max 3, only from the given results)}"""


def triage_prior_art(llm: LLM, spec: dict, hits: list[dict]) -> dict:
    """Is this effect already published? Summarise the Tavily hits with citations."""
    if not any(hit.get("url") for hit in hits):
        return {
            "already_known": None,
            "summary": "No live prior-art search was run.",
            "citations": [],
        }
    user = f"Idea: {spec.get('hypothesis')}\n\nResults:\n{json.dumps(hits, indent=1)}"
    return llm.chat_json("clerk", "nano", CLERK_SYS, user, mock=lambda: "{}", max_tokens=800)


# --- judge ---
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
    """Nemotron Ultra explains a verdict that the statistics have already decided."""
    return llm.chat_json(
        "judge",
        "ultra",
        JUDGE_SYS,
        f"Verdict: {verdict}\nEvidence:\n{json.dumps(evidence, indent=2)}",
        mock=lambda: json.dumps(demo_judgement(verdict, evidence)),
        max_tokens=1200,
    )


def demo_judgement(verdict: str, e: dict) -> dict:
    """The judge's explanation in demo mode, written from the same evidence the model sees."""
    if e["time_travel"].startswith("FAILED"):
        return {
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
            "what_would_change_my_mind": [
                "Rewriting every statistic as a rolling or expanding window over past data only",
                "Passing the time-travel test, then the sealed evidence",
            ],
        }

    memory = ""
    if e["trial_number_on_this_data"] > 1:
        memory = (
            f" This is trial {e['trial_number_on_this_data']} on this data, and the court counts "
            f"all {e['n_configs_total_on_this_data']} configurations tried so far (about "
            f"{e['distinct_bets_total']:.0f} distinct bets)."
        )
    years = e["years_of_live_trading_to_prove_it"]
    proof = (
        "No length of live track record would confirm it, because it does not beat luck."
        if years is None
        else f"Proving it live would take about {years} years."
    )
    return {
        "headline": f"{verdict}: the best backtest Sharpe of {e['best_sharpe_annual']:.2f} "
        f"against {e['luck_sharpe_annual']:.2f} from luck alone.",
        "reasoning": [
            f"{e['n_configs']} configurations were tried here, but they amount to only about "
            f"{e['distinct_bets_this_trial']:.0f} genuinely different bets. Zero-edge versions of "
            f"them typically reach a best Sharpe of {e['luck_sharpe_annual']:.2f}.{memory}",
            f"After that correction the probability the edge is real is "
            f"{fmt_prob(e['p_real'])}. {proof}",
            f"The in-sample winner lost money out of sample in {e['prob_oos_loss']:.0%} of "
            f"train/test splits, and chosen walk-forward its Sharpe is "
            f"{e['walk_forward_sharpe']:.2f}. (Probability the exact parameters were overfit: "
            f"{e['pbo']:.0%}.)",
        ],
        "what_would_change_my_mind": [
            "Committing to one configuration and passing the sealed evidence test",
            "Similar results on assets or years it has never seen",
        ],
    }
