"""The court remembers.

Every configuration ever tried on a dataset is kept (its return stream on the
open evidence), so the next trial is judged against all of it: the Reality
Check's luck benchmark and the distinct-bets count both use the full record.
Openings of the sealed evidence are recorded too, because evidence that has
been looked at is no longer sealed.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def fingerprint(prices: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, prices.columns)).encode())
    h.update(pd.util.hash_pandas_object(prices, index=True).values.tobytes())
    return h.hexdigest()[:16]


class Ledger:
    """In memory by default; pass a directory to persist across sessions."""

    def __init__(self, path: str | Path | None = None):
        self.dir = Path(path).expanduser() if path else None
        self.index: dict[str, dict] = {}
        self._returns: dict[str, list[np.ndarray]] = {}
        if self.dir and (self.dir / "index.json").exists():
            try:
                self.index = json.loads((self.dir / "index.json").read_text())
            except json.JSONDecodeError:
                self.index = {}

    # ------------------------------------------------------------ reading
    def _entry(self, fp: str) -> dict:
        return self.index.setdefault(fp, {"trials": [], "seals": []})

    def history(self, fp: str) -> list[dict]:
        return self.index.get(fp, {}).get("trials", [])

    def seals(self, fp: str) -> list[dict]:
        return self.index.get(fp, {}).get("seals", [])

    def prior_returns(self, fp: str) -> list[np.ndarray]:
        if fp not in self._returns:
            mats = []
            for t in self.history(fp):
                f = t.get("file")
                if f and self.dir and (self.dir / f).exists():
                    mats.append(np.load(self.dir / f).astype(float))
            self._returns[fp] = mats
        return self._returns[fp]

    def prior_sharpes(self, fp: str) -> np.ndarray:
        return np.array([s for t in self.history(fp) for s in t["sharpes"]], dtype=float)

    # ------------------------------------------------------------ writing
    def record(self, fp: str, idea: str, returns: np.ndarray, sharpes, verdict: str,
               family: str | None = None) -> None:
        returns = np.asarray(returns, dtype=float)
        self.prior_returns(fp)                       # make sure the cache is loaded
        entry = self._entry(fp)
        k = len(entry["trials"])
        item = {"ts": round(time.time()), "idea": idea[:160], "family": family,
                "n": int(returns.shape[1]), "sharpes": [round(float(x), 6) for x in sharpes],
                "verdict": verdict}
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            item["file"] = f"{fp}_{k:04d}.npy"
            np.save(self.dir / item["file"], returns.astype(np.float32))
        entry["trials"].append(item)
        self._returns.setdefault(fp, []).append(returns)
        self._save()

    def record_seal(self, fp: str, commitment: str, summary: dict) -> None:
        self._entry(fp)["seals"].append({"ts": round(time.time()), "commitment": commitment,
                                         **summary})
        self._save()

    def clear(self, fp: str | None = None) -> None:
        fps = list(self.index) if fp is None else [fp]
        for f in fps:
            for t in self.index.get(f, {}).get("trials", []):
                if self.dir and t.get("file") and (self.dir / t["file"]).exists():
                    (self.dir / t["file"]).unlink()
            self.index.pop(f, None)
            self._returns.pop(f, None)
        self._save()

    def _save(self) -> None:
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "index.json").write_text(json.dumps(self.index))
