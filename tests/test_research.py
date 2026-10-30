"""The Tavily prior-art search: merging, caching, the credit cap, recordings and pre-flight."""

import json

import pytest
from fake_tavily import PAPERS, PRACTICE, fake_tavily

from overfit_court import preflight, research
from overfit_court.config import get_settings


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """Each test starts with an empty cache, no credits spent today and no recordings."""
    monkeypatch.setattr(research, "CACHE", research.SearchCache())
    monkeypatch.setattr(research, "CREDITS", research.Spend())
    monkeypatch.setattr(research, "RECORDINGS", research.RECORDINGS / "none")


def test_one_document_is_one_hit_whatever_its_url():
    same = [
        "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=299107",
        "https://ssrn.com/abstract=299107",
        "http://www.ssrn.com/abstract=299107#fragment",
    ]
    assert len({research.canonical(u) for u in same}) == 1
    assert research.canonical("https://www.nber.org/papers/w20439") == research.canonical(
        "https://www.nber.org/system/files/working_papers/w20439/w20439.pdf"
    )
    assert research.canonical("https://arxiv.org/abs/1234.5678") == research.canonical(
        "https://arxiv.org/pdf/1234.5678v2"
    )
    assert research.canonical("https://onlinelibrary.wiley.com/doi/10.1111/jofi.12021") == (
        research.canonical("https://onlinelibrary.wiley.com/doi/pdf/10.1111/JOFI.12021")
    )
    assert research.canonical("https://www.aqr.com/Insights/x?utm=1") == "aqr.com/insights/x"
    assert research.canonical("https://www.nber.org/papers/w1") != research.canonical(
        "https://www.nber.org/papers/w10"
    )


def test_the_search_asks_the_way_the_court_needs_and_keeps_content_whole(settings, monkeypatch):
    fake = fake_tavily()
    monkeypatch.setattr(research, "TavilyClient", fake)
    settings.tavily_api_key = "test"
    search = research.prior_art(["cross-sectional momentum"], settings)
    [papers, practice] = fake.calls
    assert papers["search_depth"] == "advanced" and papers["chunks_per_source"] == 3
    assert papers["include_usage"] is True and "academic.oup.com" in papers["include_domains"]
    assert "quantocracy.com" in practice["include_domains"]
    longest = max(len(hit["content"]) for hit in search.hits)
    assert longest == len(PAPERS[0]["content"]) > 600  # no 600-character cut
    # Five papers came back (max_results): one under two URLs, and NBER's page and PDF.
    titles = [hit["title"] for hit in search.hits]
    assert len(titles) == len(set(titles)) == 6
    assert search.hits[0]["found_in"] == 2  # the SSRN paper, found twice, ranks first
    assert all(hit["rrf"] > 0 for hit in search.hits)
    assert search.searches[0]["results"] == 5 and search.searches[1]["results"] == 3


def test_rank_fusion_keeps_practice_slots_and_merges_across_queries(settings, monkeypatch):
    many_papers = [
        {"url": f"https://ssrn.com/abstract={i}", "title": f"Paper {i}", "score": 0.99 - i / 100}
        for i in range(10)
    ]
    fake = fake_tavily(papers=many_papers, practice=PRACTICE)
    monkeypatch.setattr(research, "TavilyClient", fake)
    settings.tavily_api_key = "test"
    search = research.prior_art(["q1", "q2"], settings, max_results=10)
    kinds = [hit["kind"] for hit in search.hits]
    assert len(kinds) == research.KEEP and kinds.count("practice") >= research.PRACTICE_SLOTS
    # Found by both queries at rank 1: the top practice hit outranks a paper found once.
    assert [hit["rrf"] for hit in search.hits] == sorted(
        (hit["rrf"] for hit in search.hits), reverse=True
    )
    assert search.usage() == {
        "searches": 4,
        "failed": 0,
        "cached": 0,
        "credits": 8,
        "slowest_s": pytest.approx(search.usage()["slowest_s"]),
        "recorded": None,
    }


def test_results_are_cached_for_a_day_and_credits_are_counted(settings, monkeypatch):
    fake = fake_tavily()
    monkeypatch.setattr(research, "TavilyClient", fake)
    settings.tavily_api_key = "test"
    first = research.prior_art(["momentum"], settings)
    again = research.prior_art(["momentum"], settings)
    assert len(fake.calls) == 2  # the second trial paid nothing
    assert [hit["url"] for hit in again.hits] == [hit["url"] for hit in first.hits]
    assert again.usage()["cached"] == 2 and again.usage()["credits"] == 0
    assert research.CREDITS.today() == 4
    stale = monkeypatch  # a day later the cache is empty again
    stale.setattr(research, "CACHE_SECONDS", -1)
    research.prior_art(["momentum"], settings)
    assert len(fake.calls) == 4


def test_failed_searches_are_not_cached_and_are_counted(settings, monkeypatch):
    fake = fake_tavily(fail=("practice",))
    monkeypatch.setattr(research, "TavilyClient", fake)
    settings.tavily_api_key = "test"
    search = research.prior_art(["momentum"], settings)
    assert search.usage()["failed"] == 1 and search.hits  # papers still came back
    assert research.CREDITS.today() == 2  # a failed request costs nothing
    research.prior_art(["momentum"], settings)
    assert len(fake.calls) == 3  # papers from the cache, practice tried again


def test_the_daily_credit_cap_skips_the_live_search(settings, monkeypatch):
    fake = fake_tavily()
    monkeypatch.setattr(research, "TavilyClient", fake)
    settings.tavily_api_key = "test"
    settings.daily_tavily_credits = 3
    assert research.prior_art(["momentum"], settings).hits  # 4 credits: over the cap now
    skipped = research.prior_art(["reversal"], settings)
    assert skipped.hits == [] and skipped.note == research.CAP_NOTE
    assert len(fake.calls) == 2 and skipped.usage()["searches"] == 0


def test_demo_mode_runs_no_search_and_never_mentions_the_key(settings):
    search = research.prior_art(["momentum"], settings)  # no Tavily key
    assert search.hits == [] and search.usage()["searches"] == 0
    assert search.note == research.DEMO_NOTE
    assert "TAVILY_API_KEY" not in search.note and "live mode" in search.note


def test_a_recorded_live_search_is_replayed_in_demo_mode_with_its_date(
    settings, monkeypatch, tmp_path
):
    fake = fake_tavily()
    monkeypatch.setattr(research, "TavilyClient", fake)
    live = get_settings()
    live.tavily_api_key = "test"
    [path] = research.record(["cross-sectional momentum"], live, tmp_path)
    recording = json.loads(path.read_text())
    assert path.name == "cross-sectional-momentum.json"
    assert set(recording["responses"]) == {"papers", "practice"} and recording["recorded"]

    monkeypatch.setattr(research, "RECORDINGS", tmp_path)
    replay = research.prior_art(["cross-sectional momentum", "no recording"], settings)
    assert len(fake.calls) == 2  # the replay made no call
    assert replay.hits[0]["title"] == "Returns to Buying Winners and Selling Losers"
    assert replay.usage()["searches"] == 2 and replay.usage()["recorded"] == recording["recorded"]
    assert recording["recorded"] in replay.note and "recording" in replay.note
    # Past the daily cap the recording stands in too.
    live.daily_tavily_credits = 1
    research.CREDITS.add(5)
    assert research.prior_art(["cross-sectional momentum"], live).usage()["recorded"]
    # A recording that doesn't read as one is ignored.
    path.write_text("not json")
    assert research.prior_art(["cross-sectional momentum"], settings).note == research.DEMO_NOTE


def test_the_record_command(settings, monkeypatch, tmp_path, capsys):
    assert research.main(["--record", "q", "--out", str(tmp_path)]) == 1  # no key: nothing recorded
    assert "TAVILY_API_KEY" in capsys.readouterr().out
    monkeypatch.setattr(research, "TavilyClient", fake_tavily())
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    assert research.main(["--family", "momentum", "--out", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("*.json"))) == 2  # the family's two queries
    assert "5 papers, 3 practice" in capsys.readouterr().out
    assert research.main(["--family", "nope", "--out", str(tmp_path)]) == 1


def test_preflight_runs_the_courts_own_search(settings, monkeypatch, capsys):
    settings.tavily_api_key = "test"
    monkeypatch.setattr(research, "TavilyClient", fake_tavily())
    preflight.check_tavily(settings)
    out = capsys.readouterr().out
    assert "[PASS] Tavily papers search" in out and "[PASS] Tavily practice search" in out
    assert "2 credits" in out
    monkeypatch.setattr(research, "CACHE", research.SearchCache())
    monkeypatch.setattr(research, "TavilyClient", fake_tavily(practice=[]))
    preflight.check_tavily(settings)
    out = capsys.readouterr().out
    assert "[FAIL] Tavily practice search" in out and "[PASS] Tavily papers search" in out


def test_settings_can_drop_the_models_and_keep_tavily(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "n")
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.setenv("OC_DAILY_TAVILY_CREDITS", "900")
    settings = get_settings()
    assert settings.daily_tavily_credits == 900
    kept = settings.without_models()
    assert kept.mock_llm and not kept.mock_search
    assert settings.offline().mock_search
    monkeypatch.setenv("OC_DAILY_TAVILY_CREDITS", "-1")
    with pytest.raises(ValueError, match="at least 0"):
        get_settings()
