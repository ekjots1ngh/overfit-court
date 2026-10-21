"""Prior-art search with Tavily: has this "edge" already been published, and so arbitraged?"""

from __future__ import annotations

from tavily import TavilyClient

from .config import Settings


def prior_art(queries: list[str], settings: Settings, max_results: int = 4) -> list[dict]:
    """Search for each query (at most three) and return the hits.

    Without a Tavily key this returns a single placeholder with no URL, which
    the clerk recognises as "no live search was run".
    """
    if settings.mock_search or not queries:
        return [
            {
                "query": q,
                "title": "(demo mode) set TAVILY_API_KEY for live prior-art search",
                "url": "",
                "content": "",
            }
            for q in queries[:1]
        ]
    client = TavilyClient(api_key=settings.tavily_api_key)
    hits = []
    for query in queries[:3]:
        response = client.search(query, max_results=max_results, search_depth="basic")
        for result in response.get("results", []):
            hits.append(
                {
                    "query": query,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": (result.get("content") or "")[:600],
                }
            )
    return hits
