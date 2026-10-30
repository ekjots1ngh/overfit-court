"""Prior-art search with Tavily: has this "edge" already been published?

Two kinds of source answer that, and each query is searched in both at once:

- papers: the academic record (SSRN, NBER, arXiv and journal publishers), where
  an anomaly is first documented, with the parameters it was found with;
- practice: practitioner research (Quantpedia, Alpha Architect, AQR and others),
  where you learn whether it still works after costs and after publication.

Every search runs at advanced depth and returns up to three passages per
source, kept whole: the passage that states a paper's parameters is rarely the
first one. One document is one hit whatever its URL (an SSRN abstract page and
its sol3 copy, an NBER landing page and its PDF, two arXiv versions, a DOI under
several paths), and the lists are merged by reciprocal rank fusion, since
relevance scores from different queries are not on one scale. Practice keeps at
least two of the slots the clerk reads: that is where "does it still work after
costs" lives. The clerk (Nemotron Nano) then reads the hits by number and can
only cite what is in front of it.

Prior art is context, not evidence: it never changes a verdict. So a failed
search is reported and the trial carries on. Every search is logged with its
latency and the credits it used (`searches`); totals are in `usage()`.

A public app serves many visitors from one process, so results are cached by
(query, source kind) for a day, and a daily credit cap (OC_DAILY_TAVILY_CREDITS)
skips the live search once reached. In demo mode, or past the cap, a search
recorded from a live run is replayed if docs/live/tavily has one for the query,
labelled with the date it was recorded:

    python -m overfit_court.research --record "query" ["query" ...]
    python -m overfit_court.research --family momentum
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from tavily import TavilyClient

from . import library
from .config import Settings, get_settings
from .llm import Spend

SOURCES = {
    "papers": (
        "ssrn.com",
        "nber.org",
        "arxiv.org",
        "sciencedirect.com",
        "jstor.org",
        "onlinelibrary.wiley.com",
        "cambridge.org",
        "tandfonline.com",
        "academic.oup.com",
        "pm-research.com",
        "ideas.repec.org",
    ),
    "practice": (
        "quantpedia.com",
        "alphaarchitect.com",
        "aqr.com",
        "researchaffiliates.com",
        "man.com",
        "cfainstitute.org",
        "robotwealth.com",
        "quantstart.com",
        "robeco.com",
        "quantocracy.com",
        "thinknewfound.com",
        "dimensional.com",
    ),
}
MAX_QUERIES = 3
MAX_RESULTS = 5  # per search
CHUNKS_PER_SOURCE = 3  # passages per result at advanced depth (Tavily's maximum)
KEEP = 8  # results the clerk reads, best first
PRACTICE_SLOTS = 2  # of those, at least this many from practitioner research
RRF_K = 60  # reciprocal rank fusion's usual constant: a low rank still counts a little
ADVANCED_CREDITS = 2  # what Tavily charges an advanced search, when a reply doesn't say
CACHE_SECONDS = 24 * 3600
RECORDINGS = Path(__file__).resolve().parent.parent / "docs" / "live" / "tavily"

DEMO_NOTE = "Prior-art search runs in live mode; no search was run for this demo trial."
CAP_NOTE = "The daily Tavily credit cap was reached, so the prior-art search was skipped."

CREDITS = Spend()  # Tavily credits used by live searches today, process-wide (like llm.SPEND)


class SearchCache:
    """Results by (query, source kind), kept for a day, shared by every trial in the process."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], tuple[float, dict, list[dict]]] = {}

    def get(self, query: str, kind: str) -> tuple[dict, list[dict]] | None:
        with self._lock:
            entry = self._entries.get((query, kind))
            if entry is None or time.time() - entry[0] > CACHE_SECONDS:
                return None
            _, record, found = entry
            return dict(record), [dict(hit) for hit in found]

    def put(self, query: str, kind: str, record: dict, found: list[dict]) -> None:
        with self._lock:
            self._entries[(query, kind)] = (time.time(), dict(record), [dict(h) for h in found])


CACHE = SearchCache()


@dataclass
class PriorArtSearch:
    hits: list[dict]  # merged, deduplicated, best first
    note: str | None = None  # why there are no hits, or that the hits are a replay
    searches: list[dict] = field(default_factory=list)  # one per search: what, how long, credits

    def usage(self) -> dict:
        """Totals for the cost table: searches run, failed, from cache, credits, slowest."""
        credits = [s["credits"] for s in self.searches if s.get("credits") is not None]
        recorded = sorted({s["recorded"] for s in self.searches if s.get("recorded")})
        return {
            "searches": len(self.searches),
            "failed": sum(1 for s in self.searches if s.get("error")),
            "cached": sum(1 for s in self.searches if s.get("cached")),
            "credits": sum(credits) if credits else None,
            "slowest_s": max((s["latency_s"] for s in self.searches), default=0.0),
            "recorded": ", ".join(recorded) or None,  # the date(s) of a replayed live search
        }


def prior_art(queries: Iterable[str], settings: Settings, max_results: int = MAX_RESULTS):
    """Search papers and practice for each query (at most three), all in parallel.

    Demo mode, and a day whose credit cap is spent, replay recorded live searches
    instead (see `record`), or report that no search ran.
    """
    queries = [q for q in dict.fromkeys(queries) if isinstance(q, str) and q.strip()]
    queries = queries[:MAX_QUERIES]
    if not queries:
        return PriorArtSearch([], "The plan had no prior-art search queries.")
    if settings.mock_search:
        return _replay(queries, DEMO_NOTE)
    cap = settings.daily_tavily_credits
    if cap and CREDITS.today() >= cap:
        return _replay(queries, CAP_NOTE)
    jobs = [(query, kind) for query in queries for kind in SOURCES]

    def search(job: tuple[str, str]) -> tuple[dict, list[dict]]:
        query, kind = job
        cached = CACHE.get(query, kind)
        if cached is not None:
            record, found = cached
            record.update({"cached": True, "credits": 0, "latency_s": 0.0})
            return record, found
        record = {"query": query, "kind": kind, "results": 0, "credits": None, "error": None}
        start = time.perf_counter()
        try:
            response = _search(query, kind, settings, max_results)
        except Exception as e:  # a search outage must not stop the trial
            record["error"] = f"{type(e).__name__}: {e}"[:200]
            response = {}
        record["latency_s"] = round(time.perf_counter() - start, 2)
        found = _hits(query, kind, response)
        record["results"] = len(found)
        record["credits"] = (response.get("usage") or {}).get("credits")
        if record["error"] is None:
            CREDITS.add(record["credits"] if record["credits"] is not None else ADVANCED_CREDITS)
            CACHE.put(query, kind, record, found)
        return record, found

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        done = list(pool.map(search, jobs))
    return _assemble(done)


def _search(query: str, kind: str, settings: Settings, max_results: int) -> dict:
    """One live Tavily search, the way the court asks it (its own client per thread)."""
    client = TavilyClient(api_key=settings.tavily_api_key)
    try:
        # chunks_per_source is a documented search option; tavily-python passes any
        # extra keyword argument through to the request.
        return client.search(
            query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=list(SOURCES[kind]),
            include_usage=True,
            chunks_per_source=CHUNKS_PER_SOURCE,
        )
    finally:
        close = getattr(client, "close", None)
        if close:
            close()


def _hits(query: str, kind: str, response: dict) -> list[dict]:
    """A search reply as hits, in the order Tavily ranked them, content kept whole."""
    return [
        {
            "query": query,
            "kind": kind,
            "title": result.get("title") or result.get("url", ""),
            "url": result.get("url", ""),
            "score": float(result.get("score") or 0.0),
            "content": result.get("content") or "",
        }
        for result in response.get("results") or []
        if isinstance(result, dict) and result.get("url")
    ]


def _assemble(done: list[tuple[dict, list[dict]]], note: str | None = None) -> PriorArtSearch:
    searches = [record for record, _ in done]
    hits = merge([found for _, found in done])
    if not hits:
        failures = [s["error"] for s in searches if s.get("error")]
        note = f"The prior-art search failed: {failures[0]}" if failures else "Nothing was found."
    return PriorArtSearch(hits, note, searches)


# --- merging ---
def canonical(url: str) -> str:
    """One key per document, whatever the URL.

    An SSRN abstract id, an NBER paper number, an arXiv id without its version,
    a DOI under any path; otherwise the host without "www." and the path, with
    no query string or fragment.
    """
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.lower().rstrip("/")
    if host.endswith("ssrn.com"):
        found = re.search(r"abstract(?:_id)?=(\d+)", f"{path}?{parts.query}")
        if found:
            return f"ssrn:{found.group(1)}"
    if host.endswith("nber.org"):
        found = re.search(r"/(w\d+)(?:/|\.|$)", path)
        if found:
            return f"nber:{found.group(1)}"
    if host.endswith("arxiv.org"):
        found = re.match(r"/(?:abs|pdf|html)/(.+?)(?:v\d+)?(?:\.pdf)?$", path)
        if found:
            return f"arxiv:{found.group(1)}"
    doi = re.search(r"(10\.\d{4,9}/[^\s?#]+?)(?:\.pdf)?$", path)
    if doi:
        return f"doi:{doi.group(1)}"
    return f"{host}{path}"


def merge(lists: list[list[dict]], keep: int = KEEP, practice_slots: int = PRACTICE_SLOTS):
    """The hits of several searches as one list, best first.

    Reciprocal rank fusion: a hit scores 1 / (RRF_K + rank) in every list it
    appears in, so a document several searches found rises, and the raw Tavily
    scores of different queries are never compared. Of one document's copies,
    the one Tavily scored highest is kept. Practitioner research keeps at least
    `practice_slots` of the `keep` slots when it has that many hits.
    """
    fused: dict[str, dict] = {}
    for found in lists:
        for rank, hit in enumerate(found, start=1):
            key = canonical(hit["url"])
            entry = fused.get(key)
            if entry is None:
                entry = fused[key] = {**hit, "rrf": 0.0, "found_in": 0}
            elif hit["score"] > entry["score"]:
                entry.update({k: hit[k] for k in ("query", "kind", "title", "url", "score")})
                entry["content"] = hit["content"]
            entry["rrf"] += 1.0 / (RRF_K + rank)
            entry["found_in"] += 1
    ranked = sorted(fused.values(), key=lambda hit: -hit["rrf"])
    chosen = ranked[:keep]
    practice = [hit for hit in ranked if hit["kind"] == "practice"]
    wanted = min(practice_slots, len(practice))
    while sum(1 for hit in chosen if hit["kind"] == "practice") < wanted:
        lowest_paper = max(
            (i for i, hit in enumerate(chosen) if hit["kind"] != "practice"), default=None
        )
        if lowest_paper is None:
            break
        chosen.pop(lowest_paper)
        chosen.append(next(hit for hit in practice if hit not in chosen))
    chosen.sort(key=lambda hit: -hit["rrf"])
    for hit in chosen:
        hit["rrf"] = round(hit["rrf"], 5)
    return chosen


# --- recorded live searches, replayed in demo mode ---
def slug(query: str) -> str:
    """The file name a query's recording is kept under."""
    return re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:80] or "query"


def recording_path(query: str, folder: Path = RECORDINGS) -> Path:
    return folder / f"{slug(query)}.json"


def load_recording(query: str, folder: Path | None = None) -> dict | None:
    """A query's recorded live search, if there is one and it reads as one."""
    path = recording_path(query, folder or RECORDINGS)
    try:
        recording = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(recording, dict) or not isinstance(recording.get("responses"), dict):
        return None
    return recording


def _replay(queries: list[str], note_when_absent: str) -> PriorArtSearch:
    """Recorded live searches for these queries, if any; otherwise no hits and the note."""
    done = []
    for query in queries:
        recording = load_recording(query)
        if recording is None:
            continue
        recorded = str(recording.get("recorded") or "an unknown date")
        for kind in SOURCES:
            response = recording["responses"].get(kind) or {}
            found = _hits(query, kind, response)
            record = {
                "query": query,
                "kind": kind,
                "results": len(found),
                "credits": (response.get("usage") or {}).get("credits"),
                "error": None,
                "latency_s": float((recording.get("latency_s") or {}).get(kind) or 0.0),
                "recorded": recorded,
            }
            done.append((record, found))
    if not done:
        return PriorArtSearch([], note_when_absent)
    dates = sorted({record["recorded"] for record, _ in done})
    return _assemble(done, f"Replayed from a recording of the live search on {', '.join(dates)}.")


def record(queries: list[str], settings: Settings, folder: Path = RECORDINGS) -> list[Path]:
    """Run the live searches for each query and save the raw replies for replay.

    One JSON file per query: both source kinds' replies as Tavily sent them,
    their latency, and the date. Needs TAVILY_API_KEY; a failed search raises,
    so a recording never holds a half result.
    """
    if settings.mock_search:
        raise RuntimeError("TAVILY_API_KEY is not set, so there is no live search to record.")
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for query in queries:
        responses, latency = {}, {}
        for kind in SOURCES:
            start = time.perf_counter()
            responses[kind] = _search(query, kind, settings, MAX_RESULTS)
            latency[kind] = round(time.perf_counter() - start, 2)
        path = recording_path(query, folder)
        path.write_text(
            json.dumps(
                {
                    "query": query,
                    "recorded": datetime.now(timezone.utc).date().isoformat(),
                    "max_results": MAX_RESULTS,
                    "responses": responses,
                    "latency_s": latency,
                },
                indent=2,
            )
        )
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m overfit_court.research",
        description="Record live Tavily searches for replay in demo mode.",
    )
    parser.add_argument("--record", nargs="*", default=[], metavar="QUERY", help="queries")
    parser.add_argument(
        "--family", action="append", default=[], help="record a library family's own queries"
    )
    parser.add_argument("--out", default=str(RECORDINGS), help="folder for the recordings")
    args = parser.parse_args(argv)
    queries = list(args.record)
    for name in args.family:
        if name not in library.FAMILIES:
            print(f"No family called {name!r}; the families are {', '.join(library.FAMILIES)}.")
            return 1
        queries += list(library.FAMILIES[name].search_queries)
    if not queries:
        parser.print_help()
        return 1
    try:
        paths = record(queries, get_settings(), Path(args.out))
    except Exception as e:
        print(f"Recording failed: {type(e).__name__}: {e}")
        return 1
    for path in paths:
        recording = json.loads(path.read_text())
        counts = ", ".join(
            f"{len(reply.get('results') or [])} {kind}"
            for kind, reply in recording["responses"].items()
        )
        print(f"{path}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
