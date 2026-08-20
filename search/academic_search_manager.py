import re
import time

import requests

from config import (
    SEMANTIC_SCHOLAR_BASE_URL,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_MAX_RESULTS,
    SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
    CROSSREF_BASE_URL,
    CROSSREF_MAILTO,
    CROSSREF_MAX_RESULTS,
    CROSSREF_TIMEOUT_SECONDS,
)


# ============================================================
# ACADEMIC SEARCH CONFIGURATION
# ============================================================

SEMANTIC_SCHOLAR_MAX_RETRIES = 2
SEMANTIC_SCHOLAR_RETRY_DELAY_SECONDS = 3


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):
    """
    Normalize text so titles can be compared reliably.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TITLE SIMILARITY
# ============================================================

def title_similarity(
    title_a,
    title_b
):
    """
    Calculate a simple token-based similarity
    between two academic titles.

    This avoids treating nearly identical titles
    as completely different papers.
    """

    a = set(
        normalize_text(title_a).split()
    )

    b = set(
        normalize_text(title_b).split()
    )

    if not a or not b:
        return 0.0

    intersection = len(
        a.intersection(b)
    )

    union = len(
        a.union(b)
    )

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# NORMALIZE ACADEMIC RESULT
# ============================================================

def normalize_academic_result(
    title="",
    url="",
    abstract="",
    authors=None,
    year=None,
    doi="",
    source="",
):
    """
    Convert academic search results into
    one common format.
    """

    if authors is None:
        authors = []

    return {
        "title": title.strip() if title else "",

        "url": url.strip() if url else "",

        "content": (
            abstract.strip()
            if abstract
            else ""
        ),

        "authors": authors,

        "year": year,

        "doi": (
            doi.strip()
            if doi
            else ""
        ),

        "source": source,
    }


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(
    results,
    title_similarity_threshold=0.85
):
    """
    Remove duplicate academic papers.

    Duplicate detection uses:

    1. DOI
    2. URL
    3. Highly similar titles
    """

    unique = []

    seen_dois = set()

    seen_urls = set()

    for result in results:

        doi = (
            result.get(
                "doi",
                ""
            )
            .strip()
            .lower()
        )

        url = (
            result.get(
                "url",
                ""
            )
            .strip()
            .lower()
        )

        title = result.get(
            "title",
            ""
        )

        # ----------------------------------------------------
        # DOI DUPLICATE
        # ----------------------------------------------------

        if doi:

            if doi in seen_dois:

                continue

            seen_dois.add(
                doi
            )

        # ----------------------------------------------------
        # URL DUPLICATE
        # ----------------------------------------------------

        if url:

            if url in seen_urls:

                continue

            seen_urls.add(
                url
            )

        # ----------------------------------------------------
        # TITLE DUPLICATE
        # ----------------------------------------------------

        duplicate_title = False

        for existing in unique:

            similarity = title_similarity(
                title,
                existing.get(
                    "title",
                    ""
                )
            )

            if (
                similarity
                >= title_similarity_threshold
            ):

                duplicate_title = True

                break

        if duplicate_title:

            continue

        unique.append(
            result
        )

    return unique
def metadata_quality_score(result):
    """
    Score the completeness of an academic source.

    Higher score = more useful metadata.
    """

    score = 0

    if result.get("title"):
        score += 1

    if result.get("authors"):
        score += 2

    if result.get("year"):
        score += 1

    if result.get("doi"):
        score += 2

    if result.get("content"):
        score += 2

    if result.get("url"):
        score += 1

    return score


# ============================================================
# SEMANTIC SCHOLAR
# ============================================================

def search_semantic_scholar(
    query,
    max_results=SEMANTIC_SCHOLAR_MAX_RESULTS,
):
    """
    Search Semantic Scholar.

    Handles rate limiting with a small,
    controlled retry strategy.
    """

    print(
        "\n[Academic Search]"
    )

    print(
        f'Semantic Scholar query: "{query}"'
    )

    url = (
        f"{SEMANTIC_SCHOLAR_BASE_URL}"
        "/paper/search"
    )

    params = {
        "query": query,

        "limit": max_results,

        "fields": (
            "title,"
            "authors,"
            "year,"
            "abstract,"
            "url,"
            "externalIds,"
            "venue"
        ),
    }

    headers = {
        "User-Agent": (
            "AI-Research-Agent/1.0"
        )
    }

    if SEMANTIC_SCHOLAR_API_KEY:

        headers[
            "x-api-key"
        ] = SEMANTIC_SCHOLAR_API_KEY

    for attempt in range(
        1,
        SEMANTIC_SCHOLAR_MAX_RETRIES + 1
    ):

        print(
            f"Semantic Scholar attempt "
            f"{attempt}/"
            f"{SEMANTIC_SCHOLAR_MAX_RETRIES}"
        )

        try:

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=(
                    SEMANTIC_SCHOLAR_TIMEOUT_SECONDS
                ),
            )

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if response.status_code == 429:

                print(
                    "Semantic Scholar rate "
                    "limit reached (429)."
                )

                if attempt < (
                    SEMANTIC_SCHOLAR_MAX_RETRIES
                ):

                    print(
                        "Waiting before retry..."
                    )

                    time.sleep(
                        SEMANTIC_SCHOLAR_RETRY_DELAY_SECONDS
                    )

                    continue

                print(
                    "Semantic Scholar retries "
                    "exhausted."
                )

                return []

            response.raise_for_status()

            data = response.json()

            break

        except requests.RequestException as error:

            print(
                "Semantic Scholar failed:"
            )

            print(
                error
            )

            if attempt < (
                SEMANTIC_SCHOLAR_MAX_RETRIES
            ):

                print(
                    "Retrying..."
                )

                time.sleep(
                    SEMANTIC_SCHOLAR_RETRY_DELAY_SECONDS
                )

                continue

            return []

        except ValueError as error:

            print(
                "Semantic Scholar returned "
                "invalid JSON:"
            )

            print(
                error
            )

            return []

    else:

        return []

    # ========================================================
    # PROCESS PAPERS
    # ========================================================

    papers = data.get(
        "data",
        []
    )

    results = []

    for paper in papers:

        authors = [
            author.get(
                "name",
                ""
            )

            for author in paper.get(
                "authors",
                []
            )

            if author.get(
                "name"
            )
        ]

        external_ids = (
            paper.get(
                "externalIds",
                {}
            )
            or {}
        )

        doi = external_ids.get(
            "DOI",
            ""
        )

        results.append(
            normalize_academic_result(

                title=paper.get(
                    "title",
                    ""
                ),

                url=paper.get(
                    "url",
                    ""
                ),

                abstract=paper.get(
                    "abstract",
                    ""
                ),

                authors=authors,

                year=paper.get(
                    "year"
                ),

                doi=doi,

                source="Semantic Scholar",
            )
        )

    results = remove_duplicates(
        results
    )

    results = rank_academic_results(
        results
    )

    print(
        f"Semantic Scholar returned "
        f"{len(results)} unique papers."
    )

    return results


# ============================================================
# CROSSREF
# ============================================================

def search_crossref(
    query,
    max_results=CROSSREF_MAX_RESULTS,
):
    """
    Search Crossref as the academic fallback.
    """

    print(
        "\n[Academic Fallback]"
    )

    print(
        f'Crossref query: "{query}"'
    )

    params = {
        "query.bibliographic": query,

        "rows": max_results,

        "select": (
            "title,"
            "author,"
            "published,"
            "DOI,"
            "URL,"
            "abstract,"
            "container-title"
        ),
    }

    headers = {
        "User-Agent": (
            "AI-Research-Agent/1.0"
        )
    }

    if CROSSREF_MAILTO:

        headers["User-Agent"] = (
            "AI-Research-Agent/1.0 "
            f"(mailto:{CROSSREF_MAILTO})"
        )

    try:

        response = requests.get(
            CROSSREF_BASE_URL,
            params=params,
            headers=headers,
            timeout=CROSSREF_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as error:

        print(
            "Crossref failed:"
        )

        print(
            error
        )

        return []

    except ValueError as error:

        print(
            "Crossref returned invalid JSON:"
        )

        print(
            error
        )

        return []

    items = (
        data.get(
            "message",
            {}
        )
        .get(
            "items",
            []
        )
    )

    results = []

    for item in items:

        # ----------------------------------------------------
        # AUTHORS
        # ----------------------------------------------------

        authors = []

        for author in item.get(
            "author",
            []
        ):

            given = author.get(
                "given",
                ""
            )

            family = author.get(
                "family",
                ""
            )

            name = (
                f"{given} {family}"
            ).strip()

            if name:

                authors.append(
                    name
                )

        # ----------------------------------------------------
        # YEAR
        # ----------------------------------------------------

        published = (
            item.get(
                "published",
                {}
            )
            .get(
                "date-parts",
                []
            )
        )

        year = None

        if published:

            if published[0]:

                year = published[0][0]

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        titles = item.get(
            "title",
            []
        )

        title = ""

        if titles:

            title = titles[0]

        # ----------------------------------------------------
        # ABSTRACT
        # ----------------------------------------------------

        abstract = item.get(
            "abstract",
            ""
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        results.append(
            normalize_academic_result(

                title=title,

                url=item.get(
                    "URL",
                    ""
                ),

                abstract=abstract,

                authors=authors,

                year=year,

                doi=item.get(
                    "DOI",
                    ""
                ),

                source="Crossref",
            )
        )

    # --------------------------------------------------------
    # DEDUPLICATE
    # --------------------------------------------------------

    results = remove_duplicates(
        results
    )

    results = rank_academic_results(
        results
    )

    print(
        f"Crossref returned "
        f"{len(results)} unique papers."
    )

    return results


# ============================================================
# MAIN ACADEMIC SEARCH
# ============================================================

def academic_search(
    query,
    max_results=SEMANTIC_SCHOLAR_MAX_RESULTS,
):
    """
    Academic search pipeline:

        Semantic Scholar
              ↓
        controlled retry
              ↓
        if failure
              ↓
           Crossref
              ↓
        normalization
              ↓
        deduplication
              ↓
        clean academic sources
    """

    if not query or not query.strip():

        print(
            "\n[Academic Search]"
        )

        print(
            "Empty academic query."
        )

        return []

    query = query.strip()

    # ========================================================
    # PRIMARY
    # ========================================================

    results = search_semantic_scholar(
        query,
        max_results
    )

    if results:

        return results

    # ========================================================
    # FALLBACK
    # ========================================================

    print(
        "\n[Academic Search]"
    )

    print(
        "Semantic Scholar returned "
        "no usable results."
    )

    print(
        "Falling back to Crossref..."
    )

    results = search_crossref(
        query,
        max_results
    )

    if results:

        return results

    # ========================================================
    # COMPLETE FAILURE
    # ========================================================

    print(
        "\n[Academic Search]"
    )

    print(
        "All academic search providers failed."
    )

    return []


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_query = (
        "artificial intelligence "
        "healthcare"
    )

    results = academic_search(
        test_query,
        max_results=5
    )

    print(
        "\n========== ACADEMIC RESULTS =========="
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
            f"Authors: "
            f"{', '.join(result['authors'])}"
        )

        print(
            f"Year: "
            f"{result['year']}"
        )

        print(
            f"DOI: "
            f"{result['doi']}"
        )

        print(
            f"URL: "
            f"{result['url']}"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )
def rank_academic_results(results):
    """
    Rank academic sources by metadata completeness.
    """

    return sorted(
        results,
        key=metadata_quality_score,
        reverse=True
    )