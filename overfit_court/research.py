"""Prior-art search with Tavily: has this "edge" already been published, and so arbitraged?

Prior art is context, not evidence: it never changes a verdict. So a failed
search is reported and the trial carries on.
"""

from __future__ import annotations

from tavily import TavilyClient

from .config import Settings


def prior_art(
    queries: list[str], settings: Settings, max_results: int = 4
) -> tuple[list[dict], str | None]:
    """Search for each query (at most three).

    Returns (hits, note). `note` explains why there are no live results when
    there aren't any: demo mode, no queries, or the search failing.
    """
    if settings.mock_search:
        return [], "No live prior-art search was run (set TAVILY_API_KEY to enable it)."
    if not queries:
        return [], "The plan had no prior-art search queries."
    client = TavilyClient(api_key=settings.tavily_api_key)
    hits, failures = [], []
    for query in queries[:3]:
        try:
            response = client.search(query, max_results=max_results, search_depth="basic")
        except Exception as e:  # a search outage must not stop the trial
            failures.append(f"{type(e).__name__}: {e}"[:200])
            continue
        for result in response.get("results", []):
            hits.append(
                {
                    "query": query,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": (result.get("content") or "")[:600],
                }
            )
    note = None
    if not hits:
        note = f"The prior-art search failed: {failures[0]}" if failures else "Nothing was found."
    return hits, note
