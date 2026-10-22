"""The court remembers.

Every configuration ever tried on a dataset is kept: its daily returns on the
open evidence. The next trial on the same data is judged against all of them,
so the Reality Check's bar for luck and the distinct-bet count both grow as
genuinely new ideas are tried. Each opening of the sealed evidence is recorded
too, before the test runs, because evidence that has been looked at is no
longer sealed.

Records are keyed by `record_key`: a fingerprint of the exact prices plus the
sealed fraction, so different data, or a different seal, starts a new record.

A court that silently forgets is worse than none. A damaged record on disk
raises LedgerError instead of quietly starting afresh. Every change is made
under a lock, starting from what is on disk at that moment, and every read
re-reads the index, so two runs on the same record (a trial and a fishing
expedition, say) always see, and never overwrite, each other's work.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout

LOCK_TIMEOUT_S = 30.0


class LedgerError(RuntimeError):
    """The court record on disk is damaged, incomplete or locked."""


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
    file of daily returns per trial. Returns are kept at float32 precision in
    memory too, so a record reopened from disk gives exactly the same numbers.
    Lists read from a Ledger are new lists, so recording a new trial never
    changes one that an earlier caller holds.
    """

    def __init__(self, directory: str | Path | None = None):
        self.dir = Path(directory).expanduser() if directory else None
        self._index: dict[str, dict] = self._read_index() if self.dir else {}
        self._in_memory: dict[str, list[np.ndarray]] = {}  # returns per key, without a directory
        self._loaded: dict[str, np.ndarray] = {}  # returns files already read, by file name

    # --- reading ---
    @property
    def index(self) -> dict[str, dict]:
        """The whole record. On disk it is re-read each time: another run may have added to it."""
        if self.dir:
            self._index = self._read_index()
        return self._index

    def history(self, key: str) -> list[dict]:
        """Every trial on this data, oldest first."""
        return list(self.index.get(key, {}).get("trials", []))

    def seals(self, key: str) -> list[dict]:
        """Every opening of the sealed evidence for this data, including any still running."""
        return list(self.index.get(key, {}).get("seals", []))

    def prior_returns(self, key: str) -> list[np.ndarray]:
        """The daily returns of every configuration tried before, one T x N array per trial."""
        if not self.dir:
            return list(self._in_memory.get(key, []))
        return [self._load(trial) for trial in self.history(key)]

    def _load(self, trial: dict) -> np.ndarray:
        """A trial's returns from disk. Each file is written once, so it is read once."""
        name = trial.get("file")
        if name not in self._loaded:
            path = self.dir / name if name else None
            if path is None or not path.exists():
                raise LedgerError(
                    f"The court record is missing the returns of the trial '{trial.get('idea')}' "
                    f"({path}). Restore the file, or clear this dataset's record to start again."
                )
            self._loaded[name] = np.load(path).astype(float)
        return self._loaded[name]

    # --- writing ---
    def record(
        self, key: str, idea: str, returns: np.ndarray, verdict: str, family: str | None = None
    ) -> None:
        """Add a trial's configurations to the record."""
        returns = np.asarray(returns, dtype=np.float32)  # the precision the record keeps
        with self._changing() as index:
            entry = index.setdefault(key, {"trials": [], "seals": []})
            item = {
                "ts": round(time.time()),
                "idea": idea[:160],
                "family": family,
                "n": int(returns.shape[1]),
                "verdict": verdict,
            }
            if self.dir:
                # A name never used before, even after the record is cleared.
                item["file"] = f"{key}_{len(entry['trials']):04d}_{secrets.token_hex(4)}.npy"
                np.save(self.dir / item["file"], returns)
            entry["trials"].append(item)
        if not self.dir:
            self._in_memory.setdefault(key, []).append(returns.astype(float))

    def record_seal_opening(self, key: str, commitment: str, params: dict) -> int:
        """Note that the sealed evidence is being opened for this commitment.

        Called before the test runs, so the opening counts whatever happens
        next. Returns how many openings were on record before this one, counted
        under the lock so that a concurrent opening can't be missed.
        """
        opening = {"ts": round(time.time()), "commitment": commitment, "params": params}
        with self._changing() as index:
            entry = index.setdefault(key, {"trials": [], "seals": []})
            earlier = len(entry["seals"])
            entry["seals"].append({**opening, "result": None})
        return earlier

    def record_seal_result(self, key: str, commitment: str, result: dict) -> None:
        """Attach the outcome to the most recent opening for this commitment."""
        with self._changing() as index:
            for seal in reversed(index.get(key, {}).get("seals", [])):
                if seal["commitment"] == commitment and seal["result"] is None:
                    seal["result"] = result
                    break

    def clear(self, key: str | None = None) -> None:
        """Forget one dataset's record, or everything."""
        with self._changing() as index:
            for k in list(index) if key is None else [key]:
                for trial in index.get(k, {}).get("trials", []):
                    if self.dir and trial.get("file"):
                        (self.dir / trial["file"]).unlink(missing_ok=True)
                        self._loaded.pop(trial["file"], None)
                index.pop(k, None)
                self._in_memory.pop(k, None)

    # --- on disk ---
    def _read_index(self) -> dict:
        path = self.dir / "index.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as e:  # ValueError covers bad JSON and bad UTF-8
            raise LedgerError(
                f"The court record {path} can't be read ({e}). Move it aside to start a new "
                "record: the court won't silently forget what was tried."
            ) from e

    @contextmanager
    def _changing(self):
        """Change the record. On disk: under the lock, starting from the index as it is
        on disk now, written back atomically (a crash mid-write never leaves half a record)."""
        if not self.dir:
            yield self._index
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._locked():
            index = self._read_index()
            yield index
            partial = self.dir / "index.json.partial"
            partial.write_text(json.dumps(index))
            partial.replace(self.dir / "index.json")
            self._index = index

    @contextmanager
    def _locked(self):
        """Hold the record's lock: an operating-system lock on index.lock, so a run that
        dies holding it (killed, out of memory) releases it, and the file left behind is
        harmless."""
        lock = FileLock(self.dir / "index.lock")
        try:
            lock.acquire(timeout=LOCK_TIMEOUT_S)
        except Timeout:
            raise LedgerError(
                f"The court record {self.dir} has been locked by another run for over "
                f"{LOCK_TIMEOUT_S:.0f} seconds. Try again when it has finished."
            ) from None
        try:
            yield
        finally:
            lock.release()
