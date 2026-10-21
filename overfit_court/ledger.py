"""The court remembers.

Every configuration ever tried on a dataset is kept: its daily returns on the
open evidence. The next trial on the same data is judged against all of them,
so the Reality Check's bar for luck and the distinct-bet count both grow as
genuinely new ideas are tried. Each opening of the sealed evidence is recorded
too, because evidence that has been looked at is no longer sealed.

Records are keyed by `record_key`: a fingerprint of the exact prices plus the
sealed fraction, so different data, or a different seal, starts a new record.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


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
    file of daily returns per trial.
    """

    def __init__(self, directory: str | Path | None = None):
        self.dir = Path(directory).expanduser() if directory else None
        self.index: dict[str, dict] = {}
        self._returns: dict[str, list[np.ndarray]] = {}  # loaded lazily per key
        if self.dir and (self.dir / "index.json").exists():
            try:
                self.index = json.loads((self.dir / "index.json").read_text())
            except json.JSONDecodeError:
                self.index = {}  # a corrupt index starts a fresh record

    # --- reading ---
    def history(self, key: str) -> list[dict]:
        """Every trial on this data, oldest first."""
        return self.index.get(key, {}).get("trials", [])

    def seals(self, key: str) -> list[dict]:
        """Every time the sealed evidence for this data was opened."""
        return self.index.get(key, {}).get("seals", [])

    def prior_returns(self, key: str) -> list[np.ndarray]:
        """The daily returns of every configuration tried before, one T x N array per trial."""
        if key not in self._returns:
            arrays = []
            for trial in self.history(key):
                path = self.dir / trial["file"] if self.dir and trial.get("file") else None
                if path and path.exists():
                    arrays.append(np.load(path).astype(float))
            self._returns[key] = arrays
        return self._returns[key]

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
        self._returns.setdefault(key, []).append(returns)
        self._save()

    def record_seal(self, key: str, commitment: str, summary: dict) -> None:
        """Note that the sealed evidence was opened, and for which commitment."""
        self._entry(key)["seals"].append(
            {"ts": round(time.time()), "commitment": commitment, **summary}
        )
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
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "index.json").write_text(json.dumps(self.index))
