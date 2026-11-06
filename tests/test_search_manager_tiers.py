"""
Unit tests for search/search_manager.py: tiering, canonical URLs, and junk filtering.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.search_manager import (
    source_tier,
    canonical_url,
    is_junk_result,
    remove_duplicates,
    _domain_matches,
)


def test_canonical_url_normalizes_case_and_trailing_slash():
    url1 = "HTTPS://WWW.Example.COM/Path/To/Page/"
    url2 = "https://www.example.com/Path/To/Page"
    assert canonical_url(url1) == canonical_url(url2)


def test_canonical_url_strips_utm_and_tracking_params():
    url_clean = "https://example.com/article"
    url_tracked = "https://example.com/article?utm_source=twitter&utm_medium=social&ref=feed"
    assert canonical_url(url_tracked) == canonical_url(url_clean)


def test_source_tier_identifies_institutional_tier1():
    assert source_tier("https://cs.stanford.edu/research/paper") == 1
    assert source_tier("https://www.ox.ac.uk/news") == 1
    assert source_tier("https://www.nih.gov/health-information") == 1
    assert source_tier("https://arxiv.org/abs/2301.00001") == 1
    assert source_tier("https://ieeexplore.ieee.org/document/12345") == 1
    assert source_tier("https://www.nature.com/articles/s41586-024") == 1


def test_source_tier_identifies_general_web_tier2():
    assert source_tier("https://techblog.com/ai-post") == 2
    assert source_tier("https://huggingface.co/blog/model-release") == 2
    assert source_tier("https://stackoverflow.com/questions/12345") == 2


def test_source_tier_identifies_junk_tier3():
    assert source_tier("https://www.merriam-webster.com/dictionary/intelligence") == 3
    assert source_tier("https://www.amazon.com/dp/B000123") == 3
    assert source_tier("https://www.fandom.com/wiki/AI") == 3


def test_adult_check_on_hostname_only():
    # URL path mentioning a word should not trigger adult filter if host is clean
    assert is_junk_result("https://pornhub.com/video") is True
    # Host that discusses internet censorship shouldn't falsely trip if host is academic
    assert is_junk_result("https://law.harvard.edu/study-of-adult-filtering") is False


def test_shopping_content_not_applied_to_tier1():
    # Academic paper discussing e-commerce customer behavior should not be dropped
    paper_url = "https://link.springer.com/article/10.1007/s11747-020-00720-7"
    content = "The authors examine customer ratings and buy now decisions on conversion."
    assert is_junk_result(paper_url, title="Consumer Behavior Study", content=content) is False


def test_remove_duplicates_drops_tier3_and_dedups_canonical():
    results = [
        {
            "url": "https://cs.mit.edu/paper/",
            "title": "Deep Learning at MIT",
            "content": "Research findings.",
        },
        {
            "url": "https://cs.mit.edu/paper?utm_source=rss",
            "title": "Deep Learning at MIT (Tracked)",
            "content": "Research findings.",
        },
        {
            "url": "https://www.merriam-webster.com/dictionary/learning",
            "title": "Learning Definition",
            "content": "Definition.",
        },
        {
            "url": "https://techblog.com/overview",
            "title": "A Tech Overview",
            "content": "Blog article.",
        },
    ]

    filtered = remove_duplicates(results)
    # Merriam-Webster should be dropped (Tier 3)
    # The duplicate mit.edu URL should be deduped
    assert len(filtered) == 2
    titles = [r["title"] for r in filtered]
    assert "Deep Learning at MIT" in titles
    assert "A Tech Overview" in titles
    assert "Learning Definition" not in titles
    # Verify tiers
    mit_result = next(r for r in filtered if "mit.edu" in r["url"])
    tech_result = next(r for r in filtered if "techblog.com" in r["url"])
    assert mit_result["tier"] == 1
    assert tech_result["tier"] == 2

