"""The Fishing Expedition: what happens when an AI quant is told to find alpha.

Sandboxes that branch like git make it trivially cheap for an agent to try
path after path until something works: the garden of forking paths, at
machine speed. Here a fisherman agent is given a target Sharpe and a budget
of attempts. Every attempt is a new branch; it stops the moment it hits the
target, exactly as a real p-hacker does. Its claim is then judged four ways:

  1. a standard backtest significance test on the claimed strategy alone
  2. an honest-looking check of only the final grid it reports
  3. Overfit Court, counting every branch the fisherman walked
  4. the sealed evidence, which no branch ever had access to
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
from .ledger import Ledger
from .library import FISHABLE as FAMILIES, UNIVERSES, with_universe
from .llm import LLM
from .pipeline import (SEALED_FRAC, Judgement, SealedResult, drop_flat, judge_statistics,
                       record_key, split_sealed, unseal_strategy)
from .sandbox import get_executor


# ------------------------------------------------------------------ fishermen
def _combos(grid: dict) -> list[dict]:
    names = list(grid)
    return [dict(zip(names, v)) for v in itertools.product(*grid.values())]


def _key(family: str, params: dict) -> str:
    return family + json.dumps(params, sort_keys=True)


def fishing_grid(family: str) -> dict:
    """A family's parameters plus the universe-selection degree of freedom."""
    return {**FAMILIES[family].param_grid, "universe": UNIVERSES}


class ScriptedFisherman:
    """Random exploration plus hill-climbing around the best result so far."""
    name = "scripted fisherman (explore + hill-climb)"

    def __init__(self, seed: int = 0, explore: float = 0.5, grid_size: int = 12):
        self.rng = random.Random(seed)
        self.explore = explore
        self.grid_size = grid_size

    def _open_families(self, tried: set) -> list[str]:
        return [f for f in FAMILIES
                if any(_key(f, c) not in tried for c in _combos(fishing_grid(f)))]

    def _explore(self, history, tried):
        open_fams = self._open_families(tried)
        if not open_fams:
            return None, [], "nothing left to try"
        fresh = [f for f in open_fams if not any(a.family == f for a in history)]
        fam = self.rng.choice(fresh or open_fams)
        pool = [c for c in _combos(fishing_grid(fam)) if _key(fam, c) not in tried]
        return fam, pool, f"explore {FAMILIES[fam].title.lower()}"

    def propose(self, history: list, tried: set, target: float, left: int = 0):
        if not history or self.rng.random() < self.explore:
            fam, pool, why = self._explore(history, tried)
        else:
            best = max(history, key=lambda a: a.best_sharpe_annual)
            fam, centre = best.family, best.best_params
            grid = fishing_grid(fam)
            local = {}
            for p, v in centre.items():
                vals = list(grid[p])
                i = vals.index(v) if v in vals else 0
                local[p] = vals[max(0, i - 1): i + 2]
            pool = [c for c in _combos(local) if _key(fam, c) not in tried]
            why = f"refine {FAMILIES[fam].title.lower()} around {centre}"
            if not pool:
                fam, pool, why = self._explore(history, tried)
        self.rng.shuffle(pool)
        return fam, pool[: self.grid_size], why


FISHER_SYS = """You are an ambitious quant researcher. Your manager wants a strategy
with an annualised Sharpe ratio of at least {target} on this dataset, and you have
{left} attempts left. Choose the next strategy family and parameter values to try,
learning from your previous results. Families and allowed parameter values:
{catalogue}
Reply JSON only: {{"family": str, "idea": str (one sentence),
"params": [{{param: value, ...}}, ...] (between 4 and 16 combinations)}}"""


class NemotronFisherman:
    """Nemotron Super decides where to fish next, from the full history of attempts."""
    name = "Nemotron fisherman"

    def __init__(self, llm: LLM, max_combos: int = 16, fallback_seed: int = 0):
        self.llm = llm
        self.max_combos = max_combos
        self.fallback = ScriptedFisherman(fallback_seed)
        self.catalogue = "\n".join(
            f"- {f.name} ({f.title}): " + ", ".join(f"{p} in {v}" for p, v in f.param_grid.items())
            for f in FAMILIES.values()) + (
            "\nEvery family also takes universe in 0..24: 0 trades every asset, 1..24 each "
            "trade a fixed half of the assets.")

    def propose(self, history, tried, target, left: int = 0):
        rows = [f"attempt {a.k}: {a.family} {a.idea!r}: best Sharpe {a.best_sharpe_annual:.2f} "
                f"with {a.best_params} over {a.n} configurations" for a in history]
        system = FISHER_SYS.format(target=target, left=left or "several", catalogue=self.catalogue)
        user = "Previous attempts:\n" + ("\n".join(rows) if rows else "none yet")
        out = self.llm.chat_json("fisherman", "super", system, user,
                                 mock=lambda: "{}", max_tokens=1500)
        fam = out.get("family")
        if fam not in FAMILIES:
            return self.fallback.propose(history, tried, target, left)
        grid = fishing_grid(fam)
        combos = []
        for c in out.get("params", [])[: self.max_combos]:
            if isinstance(c, dict) and set(c) == set(grid) and all(c[p] in grid[p] for p in grid):
                if _key(fam, c) not in tried:
                    combos.append({p: c[p] for p in grid})
        if not combos:
            return self.fallback.propose(history, tried, target, left)
        return fam, combos, str(out.get("idea", ""))[:160]


# ------------------------------------------------------------------ expedition
@dataclass
class Attempt:
    k: int
    family: str
    idea: str
    n: int
    best_key: str
    best_params: dict
    best_sharpe_annual: float
    running_best_annual: float
    luck_annual: float | None = None     # Reality Check null median over everything so far


@dataclass
class Expedition:
    fisherman: str
    target: float
    attempts: list
    hit_target: bool
    claimed_family: str
    claimed_params: dict
    claimed_sharpe_annual: float
    naive_p_real: float           # standard test of the claimed strategy alone (PSR vs 0)
    final_grid_p_real: float      # honest-looking check on the reported grid only
    court: Judgement
    sealed: SealedResult | None
    n_configs: int
    distinct_bets: float
    returns: pd.DataFrame = field(repr=False, default=None)

    @property
    def naive_says_real(self) -> bool:
        return self.naive_p_real >= 0.95

    def summary(self) -> dict:
        return {
            "fisherman": self.fisherman, "target": self.target, "attempts": len(self.attempts),
            "hit_target": self.hit_target, "claimed": f"{self.claimed_family} {self.claimed_params}",
            "claimed_sharpe_annual": round(self.claimed_sharpe_annual, 2),
            "naive_p_real": round(self.naive_p_real, 3),
            "final_grid_p_real": round(self.final_grid_p_real, 3),
            "court_verdict": self.court.verdict, "court_p_real": round(self.court.rc.p_real, 3),
            "luck_sharpe_annual": round(self.court.rc.null_median_annual, 2),
            "configs": self.n_configs, "distinct_bets": round(self.distinct_bets, 1),
            "sealed": None if self.sealed is None else self.sealed.verdict,
            "sealed_sharpe_annual": None if self.sealed is None else self.sealed.sealed_sharpe_annual,
        }

    def attempts_table(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(a) for a in self.attempts])


def run_expedition(prices: pd.DataFrame, target: float = 1.0, max_attempts: int = 40,
                   fisherman=None, settings: Settings | None = None, ledger: Ledger | None = None,
                   executor=None, sealed_frac: float = SEALED_FRAC, open_seal: bool = True,
                   track_luck: bool = False, n_boot: int = 500, seed: int = 0,
                   log=print) -> Expedition:
    s = settings or get_settings()
    ledger = ledger if ledger is not None else Ledger()
    executor = executor or get_executor(s.executor)
    if fisherman is None:
        fisherman = ScriptedFisherman(seed) if s.mock_llm else NemotronFisherman(LLM(s), fallback_seed=seed)
    key = record_key(prices, sealed_frac)
    open_px, cut = split_sealed(prices, sealed_frac)
    prior = ledger.prior_returns(key)

    attempts, tried, mats, grid_all, fam_of = [], set(), [], {}, {}
    running = -np.inf
    for k in range(1, max_attempts + 1):
        fam, combos, idea = fisherman.propose(attempts, tried, target, max_attempts - k + 1)
        if fam is None or not combos:
            log("The fisherman has run out of new places to fish.")
            break
        grid = {f"a{k:02d}_{i:02d}": c for i, c in enumerate(combos)}
        for c in combos:
            tried.add(_key(fam, c))
        res = executor.run(with_universe(FAMILIES[fam].code), open_px, grid, s.cost_bps)
        res.returns = drop_flat(res.returns)
        if res.returns.shape[1] == 0:
            continue
        srs = res.returns.apply(stats.sharpe)
        bk = str(srs.idxmax())
        best_ann = stats.annualise(float(srs.max()))
        running = max(running, best_ann)
        mats.append(res.returns)
        grid_all.update(res.grid)
        fam_of.update({c: fam for c in res.returns.columns})
        luck = None
        if track_luck:
            M_sofar = np.column_stack([*prior, *[m.values for m in mats]])
            luck = stats.reality_check(M_sofar, float(srs.max()), n_boot=200,
                                       seed=k).null_median_annual
        attempts.append(Attempt(k, fam, idea, res.returns.shape[1], bk, res.grid[bk],
                                round(best_ann, 3), round(running, 3),
                                None if luck is None else round(luck, 3)))
        ledger.record(key, f"[expedition] {idea}", res.returns.values, srs.values,
                      "expedition", fam)
        log(f"Attempt {k}: {idea} ({len(combos)} branches): best Sharpe {best_ann:.2f}, "
            f"running best {running:.2f}")
        if best_ann >= target:
            log(f"The fisherman claims a discovery: Sharpe {best_ann:.2f} >= {target}.")
            break

    M = pd.concat(mats, axis=1)
    srs_all = M.apply(stats.sharpe)
    best = str(srs_all.idxmax())
    fam, params = fam_of[best], grid_all[best]
    r = M[best].values
    naive = stats.probabilistic_sharpe_ratio(float(srs_all.max()), 0.0, len(r),
                                             float(st.skew(r)), float(st.kurtosis(r, fisher=False)))
    last = next(m for m in mats if best in m.columns)
    final_grid = stats.reality_check(last.values, float(srs_all.max()), n_boot=n_boot, seed=seed)
    log("Judging the claim against every branch walked...")
    court = judge_statistics(M, grid_all, prior, key, "low", n_boot)
    sealed = None
    if open_seal and cut is not None:
        sealed = unseal_strategy(with_universe(FAMILIES[fam].code), params, float(srs_all.max()),
                                 key, cut, prices, executor, ledger, s, log)
    return Expedition(fisherman.name, target, attempts, running >= target, fam, params,
                      stats.annualise(float(srs_all.max())), naive, final_grid.p_real, court,
                      sealed, M.shape[1], court.spec_all.distinct_bets, M)
