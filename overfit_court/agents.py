"""The courtroom's model-driven roles, each pinned to the cheapest Nemotron tier that can do it.

    Planner   Nemotron 3 Ultra   turns an idea into a testable spec and grid
    Engineer  Nemotron 3 Super   writes the strategy, and repairs it from errors
                                 (a problem that survives a repair goes to Ultra)
    Auditor   Nemotron 3 Nano    reviews the code judged for look-ahead bias
    Clerk     Nemotron 3 Nano    triages Tavily prior-art search results
    Judge     Nemotron 3 Ultra   explains the statistical verdict in plain English

The thinking roles (planner, engineer, judge) think before they answer; the
auditor and clerk answer straight away, which is all their tasks need.

None of them decides anything that matters. The verdict comes from `stats`;
look-ahead bias is caught by the harness's time-travel test; the judge may
only explain. Every reply is checked and normalised before it is used, because
a live model can return the wrong types, an unknown risk level, citations it
was never given, or a headline that contradicts the verdict. And when the
auditor, clerk or judge can't give a usable reply at all, the court carries on
with its own answer rather than throwing away a trial whose backtests have run.

Each call also supplies a `mock`: the reply used in demo mode, when no API key
is set, built from the strategy library.
"""

from __future__ import annotations

import ast
import json
import keyword
import math
import re

from . import library
from ._harness import LEAKED, PASSED
from .llm import LLM, extract_code
from .stats import ACQUITTED, CONVICTED, HUNG_JURY, fmt_p


def _strings(value, limit: int = 8) -> list[str]:
    """A model's list of points as a list of non-empty strings (a lone string becomes one)."""
    items = value if isinstance(value, list) else [value]
    points = [item if isinstance(item, str) else json.dumps(item) for item in items if item]
    return [point.strip() for point in points if point.strip()][:limit]


# Token budgets. Roles that think first need room for the thinking as well as the
# answer; quick roles (auditing, triage) answer straight away and need little.
THINKING_BUDGET = {"planner": 8000, "coder": 12000, "judge": 6000}
QUICK_BUDGET = 1500


def _advice(llm: LLM, role: str, tier: str, system: str, user: str, *, mock, reasoning: bool):
    """(reply, None) from an advisory role, or (None, why) if no usable reply came back.

    Advice is not evidence: the auditor, clerk and judge never decide a verdict,
    so when their reply can't be had or read, the caller uses the court's own.
    """
    budget = THINKING_BUDGET[role] if reasoning else QUICK_BUDGET
    try:
        reply = llm.chat_json(
            role, tier, system, user, mock=mock, reasoning=reasoning, max_tokens=budget
        )
        return reply, None
    except Exception as e:  # an unreadable or empty reply, or the API failing
        return None, f"{type(e).__name__}: {e}"[:200]


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


def _parameter_value(value) -> int | float | None:
    """A grid value as a plain int or float, or None if it isn't a usable number.

    Whole-number floats become ints, so a window of 60.0 works with rolling().
    Booleans, infinities, NaN and integers too big for a float are not numbers here.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except OverflowError:  # an integer with hundreds of digits
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def validate_spec(spec: dict, max_params: int = 4, max_values: int = 50) -> dict:
    """Coerce a model-written spec into something the harness can run.

    Values that aren't usable numbers are dropped (see _parameter_value). Every
    parameter name is passed to positions() as a keyword, so it must be a valid
    Python identifier, not a keyword, and not "prices", which positions()
    already takes. The grid must hold at least two configurations, because one
    alone can't be tested for luck, and each parameter at most `max_values`.
    """
    if not isinstance(spec, dict):
        raise SpecError("spec is not a JSON object")
    grid = spec.get("param_grid")
    if not isinstance(grid, dict) or not grid:
        raise SpecError("param_grid must be a non-empty object of lists")

    clean = {}
    for name, values in grid.items():
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            raise SpecError(f"parameter name {name!r} is not a valid python identifier")
        if name == "prices":
            raise SpecError("'prices' can't be a parameter name: positions() already takes prices")
        values = values if isinstance(values, list) else [values]
        numbers = {_parameter_value(v) for v in values} - {None}
        if not numbers:
            raise SpecError(f"parameter {name!r} has no finite numeric values")
        if len(numbers) > max_values:
            raise SpecError(
                f"parameter {name!r} has {len(numbers)} values; use at most {max_values}"
            )
        clean[name] = sorted(numbers)
    if len(clean) > max_params:
        raise SpecError(f"{len(clean)} parameters; use at most {max_params}")
    if math.prod(len(values) for values in clean.values()) < 2:
        raise SpecError("the grid has only one configuration; give a parameter several values")

    queries = spec.get("search_queries")
    if isinstance(queries, str):
        queries = [queries]
    elif not isinstance(queries, list):
        queries = []
    family = spec.get("family")  # only the built-in library's own specs name a family
    return {
        **spec,
        "param_grid": clean,
        "search_queries": [q for q in queries if isinstance(q, str)][:3],
        "name": str(spec.get("name") or "Strategy"),
        "hypothesis": str(spec.get("hypothesis") or ""),
        "family": family if isinstance(family, str) and family in library.FAMILIES else None,
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
                reasoning=True,
                max_tokens=THINKING_BUDGET["planner"],
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
- It is also run on shorter histories, down to the first few days, to check
  that no position uses the future. It must work on any length: return 0
  weight wherever there is not enough data yet (including warm-up NaNs).
- Only import numpy and pandas (math and similar pure helpers are fine). No
  I/O, no randomness, no network, and nothing beyond the prices argument: no
  files, no globals(), getattr() or Python internals, no changes to modules."""


def write_code(
    llm: LLM,
    spec: dict,
    error: str | None = None,
    previous: str | None = None,
    tier: str = "super",
) -> str:
    """Write the strategy, or repair `previous` given the `error` it produced.

    `error` is either a crash or the time-travel test's description of the
    problem, which names the parameters and dates involved.
    """
    user = f"Spec:\n{json.dumps(spec, indent=2)}\n\nParameters: {list(spec['param_grid'])}"
    if error:
        user += (
            f"\n\nYour previous code failed with:\n{error}\n\nPrevious code:\n{previous}\nFix it."
        )

    family = library.family_of(spec)
    demo_code = family.fixed_code if (error and family.fixed_code) else family.code
    reply = llm.chat(
        "coder-repair" if error else "coder",
        tier,
        CODER_SYS,
        user,
        mock=lambda: f"```python\n{demo_code}```",
        reasoning=True,
        max_tokens=THINKING_BUDGET["coder"],
    )
    return extract_code(reply)


# --- auditor ---
# Cheap pattern checks that run before the model review. The look-ahead patterns
# are only a first line: the time-travel test in the harness is what actually
# catches leaks.
LOOKAHEAD_PATTERNS = [
    (r"shift\(\s*-\s*\d+", "negative shift reads future rows"),
    (r"\.bfill\(|fillna\([^)]*method\s*=\s*['\"](bfill|backfill)", "back-fill leaks future values"),
    (r"iloc\[[^\]]*\+\s*1", "indexing i+1 may read tomorrow"),
    (r"center\s*=\s*True", "centred rolling window uses future data"),
]
RISK_LEVELS = ("low", "medium", "high")

# What `escapes` allows and looks for. Code that reaches outside the prices it is
# given could find the full history elsewhere (a file, the interpreter's memory,
# state left in a shared module), which the time-travel test can't rule out.
SAFE_MODULES = {  # numpy, pandas and pure standard-library helpers with no I/O
    "numpy",
    "pandas",
    "math",
    "cmath",
    "statistics",
    "decimal",
    "fractions",
    "numbers",
    "typing",
    "dataclasses",
    "enum",
    "abc",
    "functools",
    "itertools",
    "operator",
    "collections",
    "bisect",
    "heapq",
    "copy",
    "re",
    "string",
    "datetime",
    "warnings",
    "__future__",
}
REACHING_BUILTINS = {
    "open",
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "__import__",
    "breakpoint",
    "input",
}
FILE_ATTRIBUTES = {  # besides read_* and save*
    "load",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "memmap",
    "tofile",
    "to_csv",
    "to_pickle",
    "to_parquet",
    "to_json",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_sql",
    "to_html",
    "to_xml",
    "to_stata",
    "to_orc",
    "to_clipboard",
    "to_latex",
    "to_markdown",
}
STRING_EVALUATORS = {"eval", "query"}  # pandas; harmless unless pointed at other frames
FRAME_KEYWORDS = {"level", "local_dict", "global_dict", "resolvers"}
SAFE_DUNDERS = {"__name__", "__init__", "__all__", "__doc__"}


def escapes(code: str) -> list[str]:
    """Ways the code reaches outside the prices it is given, found in its syntax tree.

    Imports beyond numpy, pandas and pure standard-library helpers; built-ins
    that reach into the interpreter or the file system; reading or writing
    files; pandas eval or query pointed at other frames; Python's internals
    (dunder names); and changes to an imported module or class, which outlive
    the run. A trial sends any of these back to the engineer; if they stay,
    the audit risk is high and the evidence room stays shut. A block under
    `if __name__ == "__main__":` never runs, so it is skipped. Code that
    doesn't parse has nothing to report: it can't run either.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    tree.body = [node for node in tree.body if not _is_main_block(node)]
    imported = {
        (alias.asname or alias.name).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    found = {_escape(node) for node in ast.walk(tree)}
    found |= {_shared_change(node, imported) for node in ast.walk(tree)}
    return sorted(found - {None})


def _is_main_block(node: ast.AST) -> bool:
    """Is this `if __name__ == "__main__":`, which never runs when the harness loads the code?"""
    test = getattr(node, "test", None)
    return (
        isinstance(node, ast.If)
        and isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and any(getattr(c, "value", None) == "__main__" for c in test.comparators)
    )


def _shared_change(node: ast.AST, imported: set[str]) -> str | None:
    """An assignment to an attribute or item of something imported: state that outlives a run."""
    if isinstance(node, ast.Assign | ast.Delete):
        targets = node.targets
    elif isinstance(node, ast.AugAssign | ast.AnnAssign):
        targets = [node.target]
    else:
        return None
    for target in targets:
        root = target
        while isinstance(root, ast.Attribute | ast.Subscript):
            root = root.value
        if root is not target and isinstance(root, ast.Name) and root.id in imported:
            return f"changes {root.id}, which is shared with every other run"
    return None


def _escape(node: ast.AST) -> str | None:
    """What, if anything, one node of a syntax tree reaches outside its arguments for."""
    if isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        modules = [node.module or ""]  # a relative import has no module name
    else:
        modules = []
    if any(module.split(".")[0] not in SAFE_MODULES for module in modules):
        return "imports a module other than numpy, pandas and pure standard-library helpers"
    if isinstance(node, ast.Call):
        called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if isinstance(node.func, ast.Name) and called in REACHING_BUILTINS:
            return f"calls {called}(), which can reach beyond its arguments"
        frames = any(kw.arg in FRAME_KEYWORDS for kw in node.keywords)
        if isinstance(node.func, ast.Attribute) and called in STRING_EVALUATORS and frames:
            return f"points .{called}() at other frames"
    name = getattr(node, "attr", None) or getattr(node, "id", None) or ""
    if name.startswith("__") and name.endswith("__") and name not in SAFE_DUNDERS:
        return f"uses Python internals ({name})"
    if isinstance(node, ast.Attribute) and (
        name.startswith(("read_", "save")) or name in FILE_ATTRIBUTES
    ):
        return f"reads or writes files (.{name})"
    return None


AUDITOR_SYS = """You audit trading code for look-ahead bias and data leakage.
Reply JSON only: {"risk": "low"|"medium"|"high", "issues": [str, ...]}.
Full-sample normalisation, future shifts, back-fills and centred windows are high risk."""


def audit(llm: LLM, code: str) -> dict:
    """Static patterns plus a Nemotron Nano review.

    A look-ahead pattern is at least medium risk, and so is a review we can't
    read (or a risk level stated in a form we don't recognise): no clean bill
    is not a clean bill. Code that reaches outside its prices is high risk.
    """
    reaches_out = escapes(code)
    static = [msg for pattern, msg in LOOKAHEAD_PATTERNS if re.search(pattern, code)] + reaches_out
    review, failure = _advice(
        llm,
        "auditor",
        "nano",
        AUDITOR_SYS,
        code,
        mock=lambda: json.dumps({"risk": "high" if static else "low", "issues": static}),
        reasoning=False,
    )
    issues = set(static)
    if review is None:
        risk = "medium"
        issues.add(f"The auditor's review could not be read ({failure}).")
    else:
        risk = str(review.get("risk", "")).strip().lower()
        if risk not in RISK_LEVELS:
            risk = "medium"
        issues |= set(_strings(review.get("issues")))
    if static and risk == "low":
        risk = "medium"
    if reaches_out:
        risk = "high"
    return {"risk": risk, "issues": sorted(issues), "static_hits": static, "escapes": reaches_out}


# --- clerk ---
CLERK_SYS = """You triage web search results about a trading idea: has the effect
already been published? The results are numbered. Reply JSON only:
{"already_known": true or false, "summary": str (at most two sentences),
 "cite": [result numbers] (at most 3, the ones that best show whether it is known),
 "published": {"<parameter>": {"value": number, "source": result number, "quote": str}}}
"published" is only for the parameters listed, in the units of their grid values
(on daily data a 12-month lookback is about 252 days), and only for values a
result states outright: "quote" is that result's own words stating the value,
copied exactly. Otherwise leave it {}. Use nothing that is not in the results."""

MIN_QUOTE = 6  # characters: a lone digit would be "found" in any result
KIND_WORDS = {"papers": "paper", "practice": "practitioner write-up"}


def _squash(text) -> str:
    """Text for comparison: one case, single spaces, no quotation marks around it."""
    return " ".join(str(text or "").lower().split()).strip("\"'“”‘’ ")


def _quoted(quote, content: str) -> bool:
    """Does the clerk's quote appear, word for word, in the result it cites?"""
    words = _squash(quote)
    return len(words) >= MIN_QUOTE and words in _squash(content)


def _offline_triage(found: list[dict], note: str | None) -> str:
    """Demo mode's clerk reply: what the search found, in plain words, judging nothing."""
    counts: dict[str, int] = {}
    for hit in found:
        word = KIND_WORDS.get(hit.get("kind"), "result")
        counts[word] = counts.get(word, 0) + 1
    what = " and ".join(f"{n} {word}{'s' if n != 1 else ''}" for word, n in counts.items())
    summary = (
        f"{note} " if note else ""
    ) + f"The search found {what}; the top results are listed. No model read them (demo mode)."
    return json.dumps(
        {
            "already_known": None,
            "summary": summary,
            "cite": list(range(min(3, len(found)))),
            "published": {},
        }
    )


def _result_number(value, n: int) -> int | None:
    """A result number the clerk gave, if it is one (a whole number in range, not a bool)."""
    number = _parameter_value(value)
    if isinstance(number, int) and 0 <= number < n:
        return number
    return None


def _cited(hit: dict) -> dict:
    """A search result as cited: the search's own title and URL, never the model's."""
    return {"title": hit.get("title") or hit["url"], "url": hit["url"], "kind": hit.get("kind")}


def triage_prior_art(llm: LLM, spec: dict, hits: list[dict], note: str | None = None) -> dict:
    """Is this effect already published? Summarise the Tavily hits, citing them by number.

    Citing by number means the clerk can only point at what the search found.
    It also reports any parameter values the sources state, for the plan's own
    parameters, each with the words that state it: a value is kept only if its
    quote really is in the cited result. The court then tests that
    configuration on its own (see pipeline.literature_test). The top results
    are kept either way, so the record shows what was found even if the triage
    fails. In demo mode the clerk's reply is a plain summary of the hits.
    """
    found = [hit for hit in hits if hit.get("url")]
    top = [_cited(hit) for hit in found[:5]]
    if not found:
        return {"already_known": None, "summary": note or "Nothing was found.", "citations": []}
    grid = spec.get("param_grid") or {}
    numbered = "\n\n".join(
        f"[{i}] ({hit.get('kind', 'web')}) {hit.get('title', '')}\n{hit['url']}\n"
        f"{hit.get('content', '')}"
        for i, hit in enumerate(found)
    )
    user = (
        f"Idea: {spec.get('hypothesis')}\n"
        f"Parameters and their grid values: {json.dumps(grid)}\n\nResults:\n{numbered}"
    )
    reply, failure = _advice(
        llm,
        "clerk",
        "nano",
        CLERK_SYS,
        user,
        mock=lambda: _offline_triage(found, note),
        reasoning=False,
    )
    if reply is None:
        summary = f"The prior-art triage failed ({failure}). The top results are listed."
        return {"already_known": None, "summary": summary, "citations": [], "top_results": top}

    cite = reply.get("cite") if isinstance(reply.get("cite"), list) else []
    numbers = [n for n in (_result_number(c, len(found)) for c in cite) if n is not None]
    citations = [_cited(found[n]) for n in dict.fromkeys(numbers)][:3]

    published = {}
    stated = reply.get("published") if isinstance(reply.get("published"), dict) else {}
    for name, item in stated.items():
        if name not in grid or not isinstance(item, dict):
            continue
        value = _parameter_value(item.get("value"))
        source = _result_number(item.get("source"), len(found))
        if value is None or source is None:
            continue
        if not _quoted(item.get("quote"), found[source].get("content", "")):
            continue  # the result doesn't say that, so the value is the model's, not a paper's
        published[name] = {
            "value": value,
            "source": _cited(found[source]),
            "quote": " ".join(str(item["quote"]).split()),
        }

    known = reply.get("already_known")
    return {
        "already_known": known if isinstance(known, bool) else None,
        "summary": str(reply.get("summary") or ""),
        "citations": citations,
        "published": published,
        "top_results": top,
    }


# --- judge ---
JUDGE_SYS = """You are the judge in a court that tries trading strategies for
overfitting. The verdict has ALREADY been decided by the statistics and you may
not change it. Explain it to a smart non-specialist. Key ideas you may use:
luck_sharpe_annual is the best Sharpe that zero-edge versions of everything
tried would typically reach (a bootstrap Reality Check over the whole court
record, not just this trial). p_value is the share of bootstrap resamples in
which those zero-edge versions do at least as well as the pick: small means
luck struggles to explain the result. It is NOT the probability that the
strategy is real, so never describe it that way. distinct_bets_total is how
many genuinely independent bets the configurations amount to (random matrix
theory), which can be far fewer than the number of configurations, while
luck_equivalent_tries is how many independent tries would give luck the same
best Sharpe, the count that says how many chances luck really had;
prob_oos_loss is the share of train/test splits in which the in-sample winner
loses money out of sample; pbo is the share of those splits in which it ranks
in the bottom half out of sample, which is only about whether the exact
parameters matter: a high pbo with a low prob_oos_loss means the parameters
don't matter but the strategy family works. deflated_sharpe is a confidence
level from an analytic test that assumes independent trials, reported for
reference only; like p_value, it is not the probability that the strategy is
real. years_of_live_trading_to_prove_it is how long a live record at the
walk-forward Sharpe would need to run to show its Sharpe is above zero.
time_travel reports the harness test that re-runs the strategy on truncated
history: if it FAILED, the code read the future or could not be shown not to,
the backtest is meaningless whatever its Sharpe, and that is the headline.
prior_art_known says whether a web search found this effect already
published (prior_art_citations names the sources): if true, say so and expect
the edge to have shrunk since publication, as published anomalies usually do.
literature_test, when present, is one configuration fixed by a paper before
this data was seen and tested alone: its p_value is a single pre-registered
test with no correction for the search, context for your explanation and
never a reason to change the verdict.
Reply JSON only:
{"headline": str, "reasoning": [str, str, str], "what_would_change_my_mind": [str, str]}
Use only numbers that appear in the evidence. Be direct, no hype."""

VERDICT_WORDS = {
    CONVICTED: re.compile(r"\bconvict", re.I),
    HUNG_JURY: re.compile(r"\bhung\b", re.I),
    ACQUITTED: re.compile(r"\bacquit", re.I),
}


def judge(llm: LLM, verdict: str, evidence: dict) -> dict:
    """Nemotron Ultra explains a verdict that the statistics have already decided.

    The reply is normalised. Any part of it that can't be used (all of it, if
    no reply came back) is replaced with the court's own explanation, and the
    opinion's "note" says which: a headline naming a different verdict, or an
    empty headline, reasoning or list of what would change the verdict.
    """
    fallback = court_opinion(verdict, evidence)
    reply, failure = _advice(
        llm,
        "judge",
        "ultra",
        JUDGE_SYS,
        f"Verdict: {verdict}\nEvidence:\n{json.dumps(evidence, indent=2)}",
        mock=lambda: json.dumps(fallback),
        reasoning=True,
    )
    if reply is None:
        return {
            **fallback,
            "note": f"The court's own opinion: the judge's reply failed ({failure}).",
        }
    headline = str(reply.get("headline") or "").strip()
    if any(
        pattern.search(headline) for other, pattern in VERDICT_WORDS.items() if other != verdict
    ):
        headline = ""  # it names a different verdict: never shown
    opinion = {
        "headline": headline,
        "reasoning": _strings(reply.get("reasoning"), limit=5),
        "what_would_change_my_mind": _strings(reply.get("what_would_change_my_mind"), limit=4),
    }
    replaced = [part for part, value in opinion.items() if not value]
    for part in replaced:
        opinion[part] = fallback[part]
    if replaced:
        parts = ", ".join(part.replace("_", " ") for part in replaced)
        opinion["note"] = f"The court's own {parts}: the judge's could not be used."
    return opinion


def court_opinion(verdict: str, e: dict) -> dict:
    """The court's own explanation, written from the same evidence the model sees.

    Used in demo mode, and as the fallback when a live reply can't be used.
    """
    if e["time_travel_status"] != PASSED:
        if e["time_travel_status"] == LEAKED:
            what = "reads the future"
            why = (
                "A strategy can only trade on what it knew at the time. When positions on past "
                "dates change as later data arrives, the backtest is using tomorrow's prices."
            )
        else:
            what = "could not be shown not to read the future"
            why = (
                "The only way to check that a strategy never uses the future is to re-run it on "
                "shorter history. This one could not be re-run that way, so the court cannot "
                "rule it out."
            )
        repairs = e["repairs"]
        if repairs:
            unfixed = f"{repairs} repair{'s' if repairs > 1 else ''} did not fix it"
        else:
            unfixed = "It was not repaired"
        return {
            "headline": f"{verdict}: this strategy {what}. Its backtest Sharpe of "
            f"{e['best_sharpe_annual']:.2f} is not evidence of anything.",
            "reasoning": [
                e["time_travel_detail"],
                why,
                f"{unfixed}, so the court does not weigh the statistics, and the evidence room "
                "stays shut: a strategy that may read the future could read the sealed evidence "
                "too.",
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
            f"all {e['n_distinct_configs_on_this_data']} distinct configurations tried so far."
        )
    years = e["years_of_live_trading_to_prove_it"]
    if years is None:
        proof = (
            "Chosen on past data and traded on the next period, it loses money, so no live "
            "track record would confirm it."
        )
    elif years > 100:
        proof = "At the Sharpe it earned on unseen data, proving it live would take over a century."
    else:
        proof = (
            f"At the Sharpe it earned on unseen data, proving it live would take about {years} "
            "years."
        )
    reasoning = [
        f"{e['n_configs']} configurations were tried here. They move in only about "
        f"{e['distinct_bets_this_trial']:.0f} independent directions, yet luck had about as "
        f"many chances as {e['luck_equivalent_tries']} independent tries: zero-edge versions "
        f"typically reach a best Sharpe of {e['luck_sharpe_annual']:.2f}.{memory}",
        f"Luck alone does as well as the winner in {e['p_value']:.1%} of bootstrap "
        f"resamples (the Reality Check's {fmt_p(e['p_value'])}). {proof}",
        f"The best choice on past data lost money on unseen data in "
        f"{e['prob_oos_loss']:.0%} of train/test splits, and chosen on past data and traded "
        f"on the next period its Sharpe is {e['walk_forward_sharpe']:.2f}. (It ranked in the "
        f"bottom half on unseen data in {e['pbo']:.0%} of splits, a measure of how much the "
        "exact parameter choice was luck.)",
    ]
    if e.get("literature_test"):
        reasoning.append(literature_line(e["literature_test"], e.get("prior_art_known")))
    return {
        "headline": f"{verdict}: the best backtest Sharpe of {e['best_sharpe_annual']:.2f} "
        f"against {e['luck_sharpe_annual']:.2f} from luck alone.",
        "reasoning": reasoning,
        "what_would_change_my_mind": [
            "Committing to one configuration and passing the sealed evidence test",
            "Similar results on assets or years it has never seen",
        ],
    }


def literature_line(test: dict, known: bool | None) -> str:
    """The literature's own configuration, tested alone, in one sentence of the opinion."""
    named = ", ".join(f"{name} {value:g}" for name, value in test["published"].items())
    if test.get("n_configs", 1) == 1:
        how = "tested on its own as a single test fixed before this data"
    else:
        how = (
            f"the best of the {test['n_configs']} configurations consistent with it, judged by "
            "a Reality Check over those alone"
        )
    shrunk = " The effect is already published, so expect its edge to have shrunk." if known else ""
    return (
        f"The sources name {named} ({test['source']}); {how}, it earned a Sharpe of "
        f"{test['sharpe_annual']:.2f} ({fmt_p(test['p_value'])}).{shrunk}"
    )
