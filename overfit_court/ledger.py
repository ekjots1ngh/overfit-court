"""The court remembers.

Every configuration ever tried on a dataset is kept: its daily returns on the
open evidence. The next trial on the same data is judged against all of them,
so the Reality Check's bar for luck and the distinct-bet count both grow as
genuinely new ideas are tried. Each opening of the sealed evidence is recorded
too, before the test runs, because evidence that has been looked at is no
longer sealed.

Records are keyed by `record_key`: a fingerprint of the exact prices plus the
sealed fraction, so different data, or a different seal, starts a new record.

A court that silently forgets is worse than none, so a damaged record on disk
raises LedgerError instead of quietly starting afresh.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


class LedgerError(RuntimeError):
    """The court record on disk is damaged or incomplete."""


def fingerprint(prices: pd.DataFrame) -> str:
    """A short hash of the asset names and every price and date."""
    digest = hashlib.sha256()
    digest.update(",".join(map(str, prices.columns)).encode())
    digest.update(pd.util.hash_pandas_object(prices, index=True).values.tobytes())
    return digest.hexdigest()[:16]


def record_key(prices: pd.DataFrame, sealed_fraction: float) -> str:
    return f"{fingerprint(prices)}-s{round(sealed_fraction * 100)}"


def record_seed(key: str) -> int:
    """A reproducible random seed for a record, so the same data gets the same bootstrap."""
    return int(key[:8], 16)


class Ledger:
    """The court record. In memory by default; pass a directory to keep it across sessions.

    On disk: index.json (what was tried, verdicts, seal openings) and one .npy
    file of daily returns per trial. Everything read from a Ledger is a copy,
    so recording a new trial never changes what an earlier caller was given.
    """

    def __init__(self, directory: str | Path | None = None):
        self.dir = Path(directory).expanduser() if directory else None
        self.index: dict[str, dict] = {}
        self._returns: dict[str, list[np.ndarray]] = {}  # loaded lazily per key
        if self.dir and (self.dir / "index.json").exists():
            try:
                self.index = json.loads((self.dir / "index.json").read_text())
            except (OSError, ValueError) as e:  # ValueError covers bad JSON and bad UTF-8
                raise LedgerError(
                    f"The court record {self.dir / 'index.json'} can't be read ({e}). Move it "
                    "aside to start a new record: the court won't silently forget what was tried."
                ) from e

    # --- reading ---
    def history(self, key: str) -> list[dict]:
        """Every trial on this data, oldest first."""
        return list(self.index.get(key, {}).get("trials", []))

    def seals(self, key: str) -> list[dict]:
        """Every opening of the sealed evidence for this data, including any still running."""
        return list(self.index.get(key, {}).get("seals", []))

    def prior_returns(self, key: str) -> list[np.ndarray]:
        """The daily returns of every configuration tried before, one T x N array per trial."""
        if key not in self._returns:
            self._returns[key] = [self._load(trial) for trial in self.history(key)]
        return list(self._returns[key])

    def _load(self, trial: dict) -> np.ndarray:
        path = self.dir / trial["file"] if self.dir and trial.get("file") else None
        if path is None or not path.exists():
            raise LedgerError(
                f"The court record is missing the returns of the trial '{trial.get('idea')}' "
                f"({path}). Restore the file, or clear this dataset's record to start again."
            )
        return np.load(path).astype(float)

    # --- writing ---
    def record(
        self, key: str, idea: str, returns: np.ndarray, verdict: str, family: str | None = None
    ) -> None:
        """Add a trial's configurations to the record."""
        returns = np.asarray(returns, dtype=float)
        self.prior_returns(key)  # load what is on disk before appending to it
        entry = self._entry(key)
        item = {
            "ts": round(time.time()),
            "idea": idea[:160],
            "family": family,
            "n": int(returns.shape[1]),
            "verdict": verdict,
        }
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            item["file"] = f"{key}_{len(entry['trials']):04d}.npy"
            np.save(self.dir / item["file"], returns.astype(np.float32))
        entry["trials"].append(item)
        self._returns[key].append(returns)
        self._save()

    def record_seal_opening(self, key: str, commitment: str, params: dict) -> None:
        """Note that the sealed evidence is being opened for this commitment.

        Called before the test runs, so the opening counts whatever happens next.
        """
        self._entry(key)["seals"].append(
            {"ts": round(time.time()), "commitment": commitment, "params": params, "result": None}
        )
        self._save()

    def record_seal_result(self, key: str, commitment: str, result: dict) -> None:
        """Attach the outcome to the most recent opening for this commitment."""
        for seal in reversed(self._entry(key)["seals"]):
            if seal["commitment"] == commitment and seal["result"] is None:
                seal["result"] = result
                break
        self._save()

    def clear(self, key: str | None = None) -> None:
        """Forget one dataset's record, or everything."""
        for k in list(self.index) if key is None else [key]:
            for trial in self.index.get(k, {}).get("trials", []):
                if self.dir and trial.get("file"):
                    (self.dir / trial["file"]).unlink(missing_ok=True)
            self.index.pop(k, None)
            self._returns.pop(k, None)
        self._save()

    def _entry(self, key: str) -> dict:
        return self.index.setdefault(key, {"trials": [], "seals": []})

    def _save(self) -> None:
        """Write the index atomically: a crash mid-write never leaves half a record."""
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            partial = self.dir / "index.json.partial"
            partial.write_text(json.dumps(self.index))
            partial.replace(self.dir / "index.json")
