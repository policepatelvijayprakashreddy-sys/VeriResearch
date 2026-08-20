import time

import requests
from ddgs import DDGS

from config import (
    DEFAULT_MAX_RESULTS,
    MAX_SEARCH_RETRIES,
    SEARCH_RETRY_DELAY_SECONDS,
    SEARXNG_INSTANCES,
    SEARXNG_TIMEOUT_SECONDS,
)


# ============================================
# NORMALIZE RESULT
# ============================================

def normalize_result(result):
    """
    Convert a DDGS result into the format
    expected by the research agents.
    """

    return {
        "title": result.get(
            "title",
            ""
        ),

        "url": result.get(
            "href",
            result.get(
                "url",
                ""
            )
        ),

        "content": result.get(
            "body",
            result.get(
                "content",
                ""
            )
        )
    }


# ============================================
# REMOVE DUPLICATES
# ============================================

def remove_duplicates(results):
    """
    Remove duplicate results using URLs.
    """

    unique = []

    seen_urls = set()

    for result in results:

        url = result.get(
            "url",
            ""
        ).strip().lower()

        if not url:
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        unique.append(result)

    return unique


# ============================================
# CLEAN QUERY
# ============================================

def clean_query(query):
    """
    Clean an LLM-generated search query before
    sending it to the search provider.
    """

    query = query.strip()

    # Remove wrapping quotation marks
    if (
        len(query) >= 2
        and query.startswith('"')
        and query.endswith('"')
    ):
        query = query[1:-1].strip()

    return query


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

    print(
        f'\n[Search Manager]'
    )

    print(
        f'Searching for: "{query}"'
    )

    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                query,
                max_results=max_results
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

        print(
            f"DDGS search failed: {error}"
        )

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

        print(
            f"\n[SearXNG Fallback] "
            f"Trying instance: {instance}"
        )

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

                print(
                    f"[SearXNG Fallback] "
                    f"{instance} returned "
                    f"{len(results)} results."
                )

                return results

        except Exception as error:

            print(
                f"[SearXNG Fallback] "
                f"{instance} failed: {error}"
            )

            continue

    print(
        "[SearXNG Fallback] "
        "All SearXNG instances failed or "
        "returned no results."
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

    for attempt in range(
        1,
        MAX_SEARCH_RETRIES + 1
    ):

        print(
            f"\n[Search Attempt "
            f"{attempt}/{MAX_SEARCH_RETRIES}"
        )

        results = _search_once(
            query,
            max_results
        )

        if results:

            print(
                f"Search Manager found "
                f"{len(results)} unique sources."
            )

            return results

        # ------------------------------------
        # Search failed
        # ------------------------------------

        if attempt < MAX_SEARCH_RETRIES:

            print(
                "No results returned. "
                "Retrying..."
            )

            time.sleep(
                SEARCH_RETRY_DELAY_SECONDS
            )

    # ========================================
    # FALLBACK: SEARXNG
    # ========================================

    print(
        "\n[Search Manager] "
        "DDGS exhausted all retries. "
        "Falling back to SearXNG..."
    )

    results = _search_searxng(
        query,
        max_results
    )

    if results:

        print(
            f"Search Manager found "
            f"{len(results)} unique sources via SearXNG."
        )

        return results

    # ========================================
    # ALL ATTEMPTS FAILED
    # ========================================

    print(
        "\n[Search Manager]"
    )

    print(
        "All search providers failed (DDGS and SearXNG)."
    )

    print(
        f'Query: "{query}"'
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