"""Prior-art search with Tavily: has this "edge" already been published?

Two kinds of source answer that, and each query is searched in both at once:

- papers: the academic record (SSRN, NBER, arXiv and journal publishers), where
  an anomaly is first documented, with the parameters it was found with;
- practice: practitioner research (Quantpedia, Alpha Architect, AQR and others),
  where you learn whether it still works after costs and after publication.

Results are merged, duplicates dropped (the best-scoring copy kept), and ranked
by Tavily's relevance score. The clerk (Nemotron Nano) then reads them by
number and can only cite what is in front of it.

Prior art is context, not evidence: it never changes a verdict. So a failed
search is reported and the trial carries on. Every search is logged with its
latency and the credits it used.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from tavily import TavilyClient

from .config import Settings

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
    ),
}
MAX_QUERIES = 3
KEEP = 8  # results the clerk reads, best first


@dataclass
class PriorArtSearch:
    hits: list[dict]  # merged, deduplicated, best first
    note: str | None = None  # why there are no hits, when there aren't any
    searches: list[dict] = field(default_factory=list)  # one per search: what, how long, credits

    def usage(self) -> dict:
        """Totals for the cost table: searches run, credits used, slowest search."""
        credits = [s["credits"] for s in self.searches if s.get("credits") is not None]
        return {
            "searches": len(self.searches),
            "failed": sum(1 for s in self.searches if s.get("error")),
            "credits": sum(credits) if credits else None,
            "slowest_s": max((s["latency_s"] for s in self.searches), default=0.0),
        }


def prior_art(queries: list[str], settings: Settings, max_results: int = 5) -> PriorArtSearch:
    """Search papers and practice for each query (at most three), all in parallel."""
    if settings.mock_search:
        return PriorArtSearch(
            [], "No live prior-art search was run (set TAVILY_API_KEY to enable it)."
        )
    if not queries:
        return PriorArtSearch([], "The plan had no prior-art search queries.")
    client = TavilyClient(api_key=settings.tavily_api_key)
    jobs = [(query, kind) for query in queries[:MAX_QUERIES] for kind in SOURCES]

    def search(job: tuple[str, str]) -> tuple[dict, list[dict]]:
        query, kind = job
        record = {"query": query, "kind": kind, "results": 0, "credits": None, "error": None}
        start = time.perf_counter()
        try:
            response = client.search(
                query,
                search_depth="advanced",
                max_results=max_results,
                include_domains=list(SOURCES[kind]),
                include_usage=True,
            )
        except Exception as e:  # a search outage must not stop the trial
            record["error"] = f"{type(e).__name__}: {e}"[:200]
            response = {}
        record["latency_s"] = round(time.perf_counter() - start, 2)
        results = response.get("results") or []
        record["results"] = len(results)
        record["credits"] = (response.get("usage") or {}).get("credits")
        found = [
            {
                "query": query,
                "kind": kind,
                "title": result.get("title") or result.get("url", ""),
                "url": result.get("url", ""),
                "score": float(result.get("score") or 0.0),
                "content": (result.get("content") or "")[:600],
            }
            for result in results
            if result.get("url")
        ]
        return record, found

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        done = list(pool.map(search, jobs))
    searches = [record for record, _ in done]
    best: dict[str, dict] = {}
    for _, found in done:
        for hit in found:
            if hit["url"] not in best or hit["score"] > best[hit["url"]]["score"]:
                best[hit["url"]] = hit
    hits = sorted(best.values(), key=lambda hit: -hit["score"])[:KEEP]

    note = None
    if not hits:
        failures = [s["error"] for s in searches if s["error"]]
        note = f"The prior-art search failed: {failures[0]}" if failures else "Nothing was found."
    return PriorArtSearch(hits, note, searches)
