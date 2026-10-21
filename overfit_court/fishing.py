"""The fishing expedition: what happens when an AI quant is told to find alpha.

Sandboxes that branch like git make it nearly free for an agent to try path
after path until something works: the garden of forking paths, at machine
speed. Here a "fisherman" is given a target Sharpe and a budget of attempts.
Each attempt picks a strategy family, parameter values and an asset universe,
and runs them as new branches. The fisherman stops the moment it hits the
target, exactly as a real p-hacker would. Its claim is then judged four ways:

1. a standard backtest significance test of the claimed strategy alone;
2. an honest-looking check of only the final grid it reports;
3. Overfit Court, counting every branch the fisherman walked;
4. the sealed evidence, which no branch ever had access to.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as st

from . import stats
from .config import Settings, get_settings
from .ledger import Ledger, record_key, record_seed
from .library import FISHABLE, UNIVERSES, with_universe
from .llm import LLM
from .pipeline import (
    SEALED_FRACTION,
    Judgement,
    Log,
    SealedResult,
    drop_flat,
    judge_statistics,
    split_sealed,
    unseal_strategy,
)
from .sandbox import get_executor


# --- the search space ---
def fishing_grid(family: str) -> dict:
    """A family's own parameters plus the universe-selection fork."""
    return {**FISHABLE[family].param_grid, "universe": UNIVERSES}


def _combinations(grid: dict) -> list[dict]:
    names = list(grid)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*grid.values())]


def _identity(family: str, params: dict) -> str:
    """A hashable name for one (family, parameters) branch, to avoid trying it twice."""
    return family + json.dumps(params, sort_keys=True)


# --- fishermen ---
class ScriptedFisherman:
    """Half the time explore a new family at random; otherwise hill-climb around the best so far.

    Deterministic for a given seed, which makes experiments reproducible.
    """

    name = "scripted fisherman (explore + hill-climb)"

    def __init__(self, seed: int = 0, explore: float = 0.5, branches_per_attempt: int = 12):
        self.rng = random.Random(seed)
        self.explore = explore
        self.branches_per_attempt = branches_per_attempt

    def propose(self, history: list, tried: set, target: float, attempts_left: int = 0):
        """Return (family, list of parameter dicts, one-line description), or (None, [], ...)."""
        if not history or self.rng.random() < self.explore:
            family, pool, idea = self._explore(history, tried)
        else:
            family, pool, idea = self._refine(history, tried)
            if not pool:  # the neighbourhood is exhausted: go somewhere new
                family, pool, idea = self._explore(history, tried)
        self.rng.shuffle(pool)
        return family, pool[: self.branches_per_attempt], idea

    def _explore(self, history: list, tried: set):
        """Pick a family with untried branches, preferring ones not visited yet."""
        open_families = [
            f
            for f in FISHABLE
            if any(_identity(f, c) not in tried for c in _combinations(fishing_grid(f)))
        ]
        if not open_families:
            return None, [], "nothing left to try"
        unvisited = [f for f in open_families if not any(a.family == f for a in history)]
        family = self.rng.choice(unvisited or open_families)
        pool = [c for c in _combinations(fishing_grid(family)) if _identity(family, c) not in tried]
        return family, pool, f"explore {FISHABLE[family].title.lower()}"

    @staticmethod
    def _refine(history: list, tried: set):
        """Try the untried neighbours of the best result: one step either way in each parameter."""
        best = max(history, key=lambda a: a.best_sharpe_annual)
        grid = fishing_grid(best.family)
        neighbourhood = {}
        for param, value in best.best_params.items():
            values = list(grid[param])
            i = values.index(value) if value in values else 0
            neighbourhood[param] = values[max(0, i - 1) : i + 2]
        pool = [c for c in _combinations(neighbourhood) if _identity(best.family, c) not in tried]
        idea = f"refine {FISHABLE[best.family].title.lower()} around {best.best_params}"
        return best.family, pool, idea


FISHERMAN_SYS = """You are an ambitious quant researcher. Your manager wants a strategy
with an annualised Sharpe ratio of at least {target} on this dataset, and you have
{left} attempts left. Choose the next strategy family and parameter values to try,
learning from your previous results. Families and allowed parameter values:
{catalogue}
Reply JSON only: {{"family": str, "idea": str (one sentence),
"params": [{{param: value, ...}}, ...] (between 4 and 16 combinations)}}"""


class NemotronFisherman:
    """Nemotron 3 Super decides where to fish next, from the full history of attempts.

    Anything it proposes outside the allowed families and values is discarded;
    if nothing usable remains, the scripted fisherman takes that turn.
    """

    name = "Nemotron fisherman"

    def __init__(self, llm: LLM, max_branches: int = 16, fallback_seed: int = 0):
        self.llm = llm
        self.max_branches = max_branches
        self.fallback = ScriptedFisherman(fallback_seed)
        families = "\n".join(
            f"- {f.name} ({f.title}): "
            + ", ".join(f"{p} in {values}" for p, values in f.param_grid.items())
            for f in FISHABLE.values()
        )
        self.catalogue = (
            families + "\nEvery family also takes universe in 0..24: 0 trades every asset, "
            "1..24 each trade a fixed half of the assets."
        )

    def propose(self, history: list, tried: set, target: float, attempts_left: int = 0):
        past = [
            f"attempt {a.k}: {a.family} {a.idea!r}: best Sharpe {a.best_sharpe_annual:.2f} "
            f"with {a.best_params} over {a.n} configurations"
            for a in history
        ]
        system = FISHERMAN_SYS.format(
            target=target, left=attempts_left or "several", catalogue=self.catalogue
        )
        user = "Previous attempts:\n" + ("\n".join(past) if past else "none yet")
        reply = self.llm.chat_json(
            "fisherman", "super", system, user, mock=lambda: "{}", max_tokens=1500
        )
        family = reply.get("family")
        if family not in FISHABLE:
            return self.fallback.propose(history, tried, target, attempts_left)

        grid = fishing_grid(family)
        allowed = []
        for params in reply.get("params", [])[: self.max_branches]:
            if (
                isinstance(params, dict)
                and set(params) == set(grid)
                and all(params[p] in grid[p] for p in grid)
                and _identity(family, params) not in tried
            ):
                allowed.append({p: params[p] for p in grid})
        if not allowed:
            return self.fallback.propose(history, tried, target, attempts_left)
        return family, allowed, str(reply.get("idea", ""))[:160]


# --- the expedition ---
@dataclass
class Attempt:
    k: int
    family: str
    idea: str
    n: int  # branches run in this attempt
    best_key: str
    best_params: dict
    best_sharpe_annual: float
    running_best_annual: float  # best Sharpe of any attempt so far
    luck_annual: float | None = None  # what luck alone gives over every branch so far


@dataclass
class Expedition:
    fisherman: str
    target: float
    attempts: list
    hit_target: bool
    claimed_family: str
    claimed_params: dict
    claimed_sharpe_annual: float
    naive_p_real: float  # judge 1: a standard significance test of the claim alone
    final_grid_p_real: float  # judge 2: a Reality Check over only the final attempt's grid
    court: Judgement  # judge 3: Overfit Court over every branch walked
    sealed: SealedResult | None  # judge 4: the sealed evidence
    n_configs: int
    distinct_bets: float
    returns: pd.DataFrame = field(repr=False, default=None)

    def summary(self) -> dict:
        return {
            "fisherman": self.fisherman,
            "target": self.target,
            "attempts": len(self.attempts),
            "hit_target": self.hit_target,
            "claimed": f"{self.claimed_family} {self.claimed_params}",
            "claimed_sharpe_annual": round(self.claimed_sharpe_annual, 2),
            "naive_p_real": round(self.naive_p_real, 3),
            "final_grid_p_real": round(self.final_grid_p_real, 3),
            "court_verdict": self.court.verdict,
            "court_p_real": round(self.court.reality_check.p_real, 3),
            "luck_sharpe_annual": round(self.court.reality_check.null_median_annual, 2),
            "configs": self.n_configs,
            "distinct_bets": round(self.distinct_bets, 1),
            "sealed": None if self.sealed is None else self.sealed.verdict,
            "sealed_sharpe_annual": None
            if self.sealed is None
            else self.sealed.sealed_sharpe_annual,
        }

    def attempts_table(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(a) for a in self.attempts])


def run_expedition(
    prices: pd.DataFrame,
    target: float = 1.0,
    max_attempts: int = 40,
    *,
    fisherman=None,
    settings: Settings | None = None,
    ledger: Ledger | None = None,
    executor=None,
    sealed_fraction: float = SEALED_FRACTION,
    open_seal: bool = True,
    track_luck: bool = False,
    n_boot: int = 500,
    seed: int = 0,
    log: Log = print,
) -> Expedition:
    """Send a fisherman out until it hits `target` or runs out of attempts, then judge the claim.

    Every branch joins the court record. `track_luck` also computes, after each
    attempt, what luck alone would give over every branch so far (for the chart).
    """
    settings = settings or get_settings()
    ledger = ledger if ledger is not None else Ledger()
    executor = executor or get_executor(settings.executor)
    if fisherman is None:
        fisherman = (
            ScriptedFisherman(seed)
            if settings.mock_llm
            else NemotronFisherman(LLM(settings), fallback_seed=seed)
        )
    key = record_key(prices, sealed_fraction)
    open_prices, cut = split_sealed(prices, sealed_fraction)
    prior = ledger.prior_returns(key)

    attempts: list[Attempt] = []
    tried: set[str] = set()
    results: list[pd.DataFrame] = []  # one per attempt
    params_of: dict[str, dict] = {}  # branch key -> parameters
    family_of: dict[str, str] = {}  # branch key -> family
    running_best = -np.inf

    for k in range(1, max_attempts + 1):
        family, branches, idea = fisherman.propose(attempts, tried, target, max_attempts - k + 1)
        if family is None or not branches:
            log("The fisherman has run out of new places to fish.")
            break
        grid = {f"a{k:02d}_{i:02d}": params for i, params in enumerate(branches)}
        tried.update(_identity(family, params) for params in branches)

        result = executor.run(
            with_universe(FISHABLE[family].code), open_prices, grid, settings.cost_bps
        )
        returns = drop_flat(result.returns)
        if returns.shape[1] == 0:
            continue
        sharpes = returns.apply(stats.sharpe)
        best_key = str(sharpes.idxmax())
        best_annual = stats.annualise(float(sharpes.max()))
        running_best = max(running_best, best_annual)
        results.append(returns)
        params_of.update(result.grid)
        family_of.update(dict.fromkeys(returns.columns, family))

        luck = None
        if track_luck:
            so_far = np.column_stack([*prior, *[r.to_numpy() for r in results]])
            luck = stats.reality_check(so_far, float(sharpes.max()), n_boot=200, seed=k)
        attempts.append(
            Attempt(
                k=k,
                family=family,
                idea=idea,
                n=returns.shape[1],
                best_key=best_key,
                best_params=result.grid[best_key],
                best_sharpe_annual=round(best_annual, 3),
                running_best_annual=round(running_best, 3),
                luck_annual=None if luck is None else round(luck.null_median_annual, 3),
            )
        )
        ledger.record(key, f"[expedition] {idea}", returns.to_numpy(), "expedition", family)
        log(
            f"Attempt {k}: {idea} ({len(branches)} branches): best Sharpe {best_annual:.2f}, "
            f"running best {running_best:.2f}"
        )
        if best_annual >= target:
            log(f"The fisherman claims a discovery: Sharpe {best_annual:.2f} >= {target}.")
            break

    # The claim: the best branch of the whole expedition.
    everything = pd.concat(results, axis=1)
    sharpes = everything.apply(stats.sharpe)
    claim = str(sharpes.idxmax())
    claim_sharpe = float(sharpes.max())
    family, params = family_of[claim], params_of[claim]
    r = everything[claim].to_numpy()

    # Judge 1: the claimed strategy tested as if it were the only one ever tried.
    naive = stats.probabilistic_sharpe_ratio(
        claim_sharpe, 0.0, len(r), float(st.skew(r)), float(st.kurtosis(r, fisher=False))
    )
    # Judge 2: a Reality Check over only the grid the claim came from (selective reporting).
    final_grid = next(res for res in results if claim in res.columns)
    reported = stats.reality_check(final_grid.to_numpy(), claim_sharpe, n_boot=n_boot, seed=seed)
    # Judge 3: Overfit Court, over every branch walked (and anything tried on this data before).
    log("Judging the claim against every branch walked...")
    court = judge_statistics(everything, params_of, prior, record_seed(key), n_boot)
    # Judge 4: one test on the sealed evidence.
    sealed = None
    if open_seal and cut is not None:
        sealed = unseal_strategy(
            with_universe(FISHABLE[family].code),
            params,
            open_sharpe=claim_sharpe,
            key=key,
            cut=cut,
            prices=prices,
            executor=executor,
            ledger=ledger,
            settings=settings,
            log=log,
        )

    return Expedition(
        fisherman=fisherman.name,
        target=target,
        attempts=attempts,
        hit_target=running_best >= target,
        claimed_family=family,
        claimed_params=params,
        claimed_sharpe_annual=stats.annualise(claim_sharpe),
        naive_p_real=naive,
        final_grid_p_real=reported.p_real,
        court=court,
        sealed=sealed,
        n_configs=everything.shape[1],
        distinct_bets=court.spectrum.distinct_bets,
        returns=everything,
    )
