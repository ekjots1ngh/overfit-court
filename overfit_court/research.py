"""Prior-art search with Tavily: has this 'edge' already been published or arbitraged?"""
from __future__ import annotations

from .config import Settings


def prior_art(queries: list[str], settings: Settings, max_results: int = 4) -> list[dict]:
    if settings.mock_search or not queries:
        return [{"query": q, "title": "(mock) set TAVILY_API_KEY for live prior-art search",
                 "url": "", "content": ""} for q in queries[:1]]
    from tavily import TavilyClient
    client = TavilyClient(api_key=settings.tavily_api_key)
    hits = []
    for q in queries[:3]:
        res = client.search(q, max_results=max_results, search_depth="basic")
        for r in res.get("results", []):
            hits.append({"query": q, "title": r.get("title", ""), "url": r.get("url", ""),
                         "content": (r.get("content") or "")[:600]})
    return hits
