"""A stand-in for Tavily's search API, for tests.

`fake_tavily(...)` returns a class to monkeypatch in as `research.TavilyClient`:
its searches answer from scripted results per source kind (a list, or a
function of the query), can fail on demand, and record every call. The default
results have the URL variants of one paper that a live search returns.
"""

TEXT = (
    "Stocks are ranked on past 12-month returns, skipping the most recent month; "
    "the top and bottom 30% form the winner and loser portfolios. " * 6
)
PAPERS = [
    {
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=299107",
        "title": "Returns to Buying Winners and Selling Losers",
        "score": 0.86,
        "content": TEXT,
    },
    {
        "url": "https://ssrn.com/abstract=299107",
        "title": "Returns to Buying Winners and Selling Losers",
        "score": 0.84,
        "content": TEXT,
    },
    {"url": "https://www.nber.org/papers/w20439", "title": "Momentum Crashes", "score": 0.81},
    {
        "url": "https://www.nber.org/system/files/working_papers/w20439/w20439.pdf",
        "title": "Momentum Crashes",
        "score": 0.80,
    },
    {"url": "https://arxiv.org/abs/1234.5678", "title": "Momentum on arXiv", "score": 0.77},
    {"url": "https://arxiv.org/pdf/1234.5678v2", "title": "Momentum on arXiv", "score": 0.76},
    {
        "url": "https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.12021",
        "title": "Value and Momentum Everywhere",
        "score": 0.74,
    },
]
PRACTICE = [
    {
        "url": "https://quantpedia.com/strategies/momentum-factor-effect-in-stocks",
        "title": "Momentum Factor Effect in Stocks",
        "score": 0.66,
        "content": TEXT,
    },
    {
        "url": "https://www.aqr.com/Insights/Fact-Fiction-and-Momentum-Investing",
        "title": "Fact, Fiction and Momentum Investing",
        "score": 0.61,
    },
    {
        "url": "https://alphaarchitect.com/quantitative-momentum/",
        "title": "Quantitative Momentum Investing",
        "score": 0.55,
    },
]


def kind_of(include_domains) -> str:
    return "papers" if "ssrn.com" in (include_domains or ()) else "practice"


def fake_tavily(papers=PAPERS, practice=PRACTICE, *, credits=2, fail=()):
    """A TavilyClient stand-in. `papers` and `practice` are result lists or functions of the
    query; a kind or query named in `fail` raises, as an outage would."""

    class FakeTavily:
        calls: list[dict] = []

        def __init__(self, api_key=None):
            self.api_key = api_key

        def search(self, query, **options):
            kind = kind_of(options.get("include_domains"))
            FakeTavily.calls.append({"query": query, "kind": kind, **options})
            if kind in fail or query in fail:
                raise ConnectionError("tavily is down")
            rows = papers if kind == "papers" else practice
            rows = rows(query) if callable(rows) else rows
            results = [{"content": "", **row} for row in rows][: options.get("max_results") or 5]
            return {"results": results, "usage": {"credits": credits}, "response_time": 0.1}

    return FakeTavily
