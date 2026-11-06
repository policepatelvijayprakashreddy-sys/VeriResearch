import logging
import re
import time
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import requests
from ddgs import DDGS

from search.cache import get_cached, set_cached

from config import (
    DEFAULT_MAX_RESULTS,
    MAX_SEARCH_RETRIES,
    SEARCH_RETRY_DELAY_SECONDS,
    SEARXNG_INSTANCES,
    SEARXNG_TIMEOUT_SECONDS,
    ACADEMIC_DOMAINS,
    STRICT_SOURCES,
)

logger = logging.getLogger(__name__)

# Domains that provide lexical definitions, course platforms, shopping, or generic non-research noise
JUNK_DOMAINS = [
    # Dictionary / thesaurus
    "merriam-webster.com",
    "dictionary.cambridge.org",
    "vocabulary.com",
    "wiktionary.org",
    "thesaurus.com",
    "collinsdictionary.com",
    "dictionary.com",
    "definitions.net",
    "urbandictionary.com",
    # E-commerce & shopping
    "amazon.com",
    "amazon.co.uk",
    "amazon.in",
    "amazon.de",
    "ebay.com",
    "ebay.co.uk",
    "walmart.com",
    "homedepot.com",
    "lowes.com",
    "etsy.com",
    "alibaba.com",
    "aliexpress.com",
    "target.com",
    "bestbuy.com",
    "wayfair.com",
    # Wikis / fandom / general encyclopaedia noise
    "fandom.com",
    "wikia.com",
    # Social media & video
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "linkedin.com",
    # Q&A / forum
    "quora.com",
    "reddit.com",
    "stackexchange.com",
    # Course platforms (ads)
    "udemy.com",
    "coursera.org",
    "edx.org",
    "skillshare.com",
    "pluralsight.com",
    # Model / dataset hubs (not peer-reviewed)
    "ollama.com",
    # Content farms & marketing blogs (not research)
    "hostinger.com",
    "simplilearn.com",
    "vocal.media",
    "tutorialspoint.com",
    "javatpoint.com",
    "w3schools.com",
    "geeksforgeeks.org",
    # Generic news / tabloids
    "buzzfeed.com",
    "huffpost.com",
]

# Words that indicate a shopping page, generic info page, or non-research noise
JUNK_TITLE_PATTERNS = [
    # Dictionary / language
    "definition & meaning",
    "definition and meaning",
    "english meaning",
    "meaning & pronunciation",
    "how to use in a sentence",
    "synonyms & antonyms",
    "translation in",
    # Shopping / commercial
    "shop through",
    "free shipping",
    "buy online",
    "for sale",
    "best price",
    "add to cart",
    "create a course",
    # Fandom / game wikis
    "| fandom",
    "wiki | fandom",
    # E-commerce terms in title
    "customer reviews",
    "deals & discounts",
    "lowest price",
    "free delivery",
    "in stock",
    "order now",
]

# Single-word generic Wikipedia pages that are often mistakenly returned when DDG truncates
JUNK_WIKI_SLUGS = {
    "comparison", "error", "bias", "test", "definition",
    "overview", "survey", "chain", "network", "model", "system",
    "tool", "tools", "product", "products", "item", "items"
}


ADULT_DOMAINS = [
    "xhamster", "pornhub", "xvideos", "xnxx", "redtube",
    "youporn", "chaturbate", "spankbang", "beeg", "tube8"
]

TRUSTED_SUFFIXES = (
    ".edu", ".gov", ".ac.uk", ".ac.in", ".gov.in", ".edu.au", ".ac.jp",
    ".mil", ".int", ".europa.eu"
)

TRUSTED_DOMAINS = {
    "arxiv.org", "doi.org", "nature.com", "science.org", "sciencedirect.com",
    "springer.com", "ieee.org", "acm.org", "nih.gov", "who.int", "nist.gov",
    "biorxiv.org", "medrxiv.org", "wiley.com", "tandfonline.com", "plos.org",
    "frontiersin.org", "mdpi.com", "openreview.net", "aclanthology.org",
    "jmlr.org", "owasp.org", "semanticscholar.org", "openalex.org",
}

ALL_TRUSTED_DOMAINS = set(TRUSTED_DOMAINS) | set(ACADEMIC_DOMAINS)


def _domain_matches(host: str, target: str) -> bool:
    host = host.lower()
    target = target.lower()
    return host == target or host.endswith("." + target)


def canonical_url(url: str) -> str:
    """
    Normalize a URL for deduplication:
    - Lowercase scheme and hostname (strip ports & credentials).
    - Strip trailing slash from path.
    - Remove tracking / marketing parameters (utm_*, ref, gclid, etc.).
    - Remove URL fragment (#...).
    """
    url = url.strip()
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower() or "https"
        host = (parsed.hostname or "").lower()
        if not host:
            return url.lower().rstrip("/")
        path = parsed.path.rstrip("/") or "/"
        qs = {
            k: v
            for k, v in parse_qs(parsed.query).items()
            if not k.lower().startswith("utm_") and k.lower() not in ("ref", "source", "fbclid", "gclid")
        }
        query = urlencode(qs, doseq=True)
        return urlunparse((scheme, host, path, parsed.params, query, ""))
    except Exception:
        return url.lower().rstrip("/")


def is_junk_result(url: str, title: str = "", content: str = "") -> bool:
    """Return True if URL, title, or content indicates an e-commerce page, dictionary, course ad, or adult/generic noise."""
    url_lower = url.lower()
    title_lower = title.lower()
    content_lower = content.lower() if content else ""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return True

    # Check adult sites strictly on hostname to prevent false positives on paths/queries
    if any(ad in host for ad in ADULT_DOMAINS):
        return True

    # Match junk domains by exact host or domain suffix
    if any(_domain_matches(host, d) for d in JUNK_DOMAINS):
        return True

    if any(pattern in title_lower for pattern in JUNK_TITLE_PATTERNS):
        return True

    # E-commerce shopping patterns: check only if not already an institutional/academic source
    is_trusted = host.endswith(TRUSTED_SUFFIXES) or any(_domain_matches(host, d) for d in ALL_TRUSTED_DOMAINS)
    if not is_trusted:
        if any(pattern in content_lower for pattern in ["add to cart", "buy now", "free shipping on orders"]):
            return True

    if "wikipedia.org/wiki/" in url_lower:
        slug = url_lower.split("wikipedia.org/wiki/")[-1].strip("/").lower()
        if slug in JUNK_WIKI_SLUGS:
            return True

    return False


def source_tier(url: str, title: str = "", content: str = "") -> int:
    """
    Assign a trust tier to a search result:
    Tier 1 = trusted / peer-reviewed / institutional (.edu, .gov, academic registries & publishers).
    Tier 2 = unknown / general web (usable as unverified context; dropped if STRICT_SOURCES=True).
    Tier 3 = junk / dropped (e-commerce, dictionary, adult, spam).
    """
    if not url:
        return 3

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return 3

    # Check junk / adult
    if is_junk_result(url, title, content):
        return 3

    # Tier 1 check: institutional suffixes or trusted/academic domains
    if host.endswith(TRUSTED_SUFFIXES) or any(_domain_matches(host, d) for d in ALL_TRUSTED_DOMAINS):
        return 1

    return 2


# ============================================
# NORMALIZE RESULT
# ============================================

def normalize_result(result):
    """
    Convert a DDGS result into the format
    expected by the research agents.

    A ``tier`` field is added so callers can sort or filter by trust level
    without re-parsing the URL later.
    """

    url = result.get(
        "href",
        result.get(
            "url",
            ""
        )
    )
    title = result.get("title", "")
    content = result.get("body", result.get("content", ""))

    return {
        "title": title,
        "url": url,
        "content": content,
        "tier": source_tier(url, title, content),
    }


# ============================================
# REMOVE DUPLICATES AND JUNK
# ============================================

def remove_duplicates(results):
    """
    Remove duplicate results and filter out dictionary/junk sites.

    Drops Tier 3 (junk / non-research).
    When STRICT_SOURCES=True, only Tier 1 (academic/institutional) results
    are kept.  Otherwise Tier 1 and Tier 2 results are retained, tagged with their
    trust tier so the caller can sort or filter as needed.
    Deduplication uses canonical URL normalization (strips utm parameters,
    trailing slashes, and case).
    """

    unique = []
    seen_canonical = set()

    for result in results:
        url = (result.get("url") or result.get("href") or "").strip()
        if not url:
            continue

        title = result.get("title", "")
        content = result.get("content", "")

        tier = result.get("tier")
        if tier is None or tier not in (1, 2, 3):
            tier = source_tier(url, title, content)

        # Drop Tier 3 (junk, spam, adult, course ads)
        if tier == 3:
            logger.debug("[Search Manager] Dropping tier-3 junk result: %s (%s)", title, url)
            continue

        # Optionally enforce strict-sources mode (keep only Tier 1)
        if STRICT_SOURCES and tier > 1:
            logger.debug("[Search Manager] Dropping tier-%d result (STRICT_SOURCES): %s", tier, url)
            continue

        canon = canonical_url(url)
        if not canon or canon in seen_canonical:
            continue

        seen_canonical.add(canon)
        unique.append({**result, "url": url, "tier": tier})

    return unique


# ============================================
# CLEAN QUERY
# ============================================

def clean_query(query: str, max_terms: int = 7) -> str:
    """
    Clean and optimize an LLM-generated search query before
    sending it to DuckDuckGo or SearXNG.

    1. Removes prefixes like 'QUERY:', 'Search query:'.
    2. Strips boolean operators (AND, OR, NOT) and complex punctuation.
    3. Strips parentheses, brackets, and quotes that break search parsers.
    4. Limits query terms to max_terms (e.g. 7) to avoid keyword truncation.
    """
    if not query:
        return ""

    query = query.strip()

    # Strip conversational prefixes
    for prefix in ["QUERY:", "Query:", "SEARCH QUERY:", "Search query:", "Search:", "GAP:"]:
        if query.startswith(prefix):
            query = query[len(prefix):].strip()

    # Remove quotes
    query = query.strip('"\'`')

    # Remove boolean operators that confuse simple search engines
    query = re.sub(r"\b(AND|OR|NOT)\b", " ", query)

    # Remove parentheses, brackets, braces, colons, semicolons, question marks, commas
    query = re.sub(r"[()\[\]{}:;?,\n\r]", " ", query)

    # Collapse whitespace
    query = re.sub(r"\s+", " ", query).strip()

    # If query is too long, strip stop words and keep the top technical keywords
    words = query.split()
    if len(words) > max_terms:
        STOP_WORDS = {
            "a", "an", "the", "and", "or", "of", "for", "in", "on", "at", "to",
            "from", "by", "with", "about", "into", "through", "during", "is", "are",
            "was", "were", "what", "which", "how", "why", "investigate", "investigating",
            "explore", "exploring", "study", "studying", "understanding", "comparison",
            "comparing", "different", "various"
        }
        filtered = [w for w in words if w.lower() not in STOP_WORDS]
        if len(filtered) >= 3:
            query = " ".join(filtered[:max_terms])
        else:
            query = " ".join(words[:max_terms])

    return query.strip()


# ============================================
# SINGLE SEARCH ATTEMPT
# ============================================

def _search_once(
    query,
    max_results,
):
    """
    Perform one DDGS search attempt.
    """
    query = clean_query(query)

    results = []

    logger.info("[Search Manager]")
    logger.info('Searching for: "%s"', query)

    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                query,
                max_results=max_results,
                safesearch="on",
            )

            for result in search_results:

                normalized = normalize_result(
                    result
                )

                if normalized["url"]:

                    results.append(
                        normalized
                    )

    except Exception as error:

        logger.warning("DDGS search failed: %s", error)

        return []

    return remove_duplicates(
        results
    )


# ============================================
# SEARXNG FALLBACK (no API key)
# ============================================

def _search_searxng(
    query,
    max_results,
):
    """
    Query public SearXNG instances after DDGS fails.
    """

    query = clean_query(query)

    for instance in SEARXNG_INSTANCES:

        logger.info("[SearXNG Fallback] Trying instance: %s", instance)

        try:

            response = requests.get(
                f"{instance}/search",
                params={
                    "q": query,
                    "format": "json",
                    "language": "en",
                },
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (research-agent; "
                        "+https://github.com/)"
                    )
                },
                timeout=SEARXNG_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            data = response.json()
            raw_results = data.get(
                "results",
                []
            )[:max_results]

            results = []

            for item in raw_results:

                url = item.get(
                    "url",
                    ""
                )

                if not url:
                    continue

                results.append(
                    {
                        "title": item.get(
                            "title",
                            ""
                        ),
                        "url": url,
                        "content": item.get(
                            "content",
                            ""
                        ),
                    }
                )

            results = remove_duplicates(results)

            if results:

                logger.info(
                    "[SearXNG Fallback] %s returned %d results.",
                    instance, len(results)
                )

                return results

        except Exception as error:

            logger.warning(
                "[SearXNG Fallback] %s failed: %s", instance, error
            )

            continue

    logger.warning(
        "[SearXNG Fallback] All instances failed or returned no results."
    )

    return []


# ============================================
# MAIN SEARCH FUNCTION
# ============================================

def search(
    query,
    max_results=DEFAULT_MAX_RESULTS
):
    """
    Reliable web search wrapper.

    The rest of the project should call this
    function rather than DDGS directly.
    """

    if not query or not query.strip():

        print(
            "\n[Search Manager] "
            "Empty query received. Skipping search."
        )

        return []

    query = clean_query(query)
    if not query:
        return []

    from config import SEARCH_CACHE_ENABLED
    if SEARCH_CACHE_ENABLED:
        cached = get_cached(query, max_results)
        if cached is not None:
            clean_cached = remove_duplicates(cached)
            if clean_cached:
                return clean_cached

    for attempt in range(
        1,
        MAX_SEARCH_RETRIES + 1
    ):

        logger.info("[Search Attempt %d/%d]", attempt, MAX_SEARCH_RETRIES)

        results = _search_once(
            query,
            max_results
        )

        if results:

            logger.info(
                "Search Manager found %d unique sources.", len(results)
            )

            # Sort tier-1 (academic/institutional) sources first so the
            # LLM sees the highest-trust snippets at the top of the context.
            results.sort(key=lambda r: r.get("tier", 2))
            
            if SEARCH_CACHE_ENABLED:
                set_cached(query, max_results, results)

            return results

        # ------------------------------------
        # Search failed
        # ------------------------------------

        if attempt < MAX_SEARCH_RETRIES:

            logger.info("No results returned. Retrying...")

            time.sleep(
                SEARCH_RETRY_DELAY_SECONDS
            )

    # ========================================
    # FALLBACK: SEARXNG
    # ========================================

    logger.warning(
        "[Search Manager] DDGS exhausted all retries. Falling back to SearXNG..."
    )

    results = _search_searxng(
        query,
        max_results
    )

    if results:

        logger.info(
            "Search Manager found %d unique sources via SearXNG.", len(results)
        )

        results.sort(key=lambda r: r.get("tier", 2))

        if SEARCH_CACHE_ENABLED:
            set_cached(query, max_results, results)

        return results

    # ========================================
    # ALL ATTEMPTS FAILED
    # ========================================

    logger.error(
        '[Search Manager] All search providers failed. Query: "%s"', query
    )

    return []

# ============================================
# OPTIONAL TEST
# ============================================

if __name__ == "__main__":

    test_query = (
        "artificial intelligence "
        "healthcare applications"
    )

    results = search(
        test_query,
        max_results=5
    )

    print(
        "\n========== RESULTS =========="
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n{i}. "
            f"{result['title']}"
        )

        print(
            result["url"]
        )