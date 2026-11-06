import logging
import re
import time

import requests

from search.cache import get_cached, set_cached

from config import (
    SEMANTIC_SCHOLAR_BASE_URL,
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_MAX_RESULTS,
    SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
    CROSSREF_BASE_URL,
    CROSSREF_MAILTO,
    CROSSREF_MAX_RESULTS,
    CROSSREF_TIMEOUT_SECONDS,
    OPENALEX_BASE_URL,
    OPENALEX_MAILTO,
    OPENALEX_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# Bump this version string whenever the set of providers or their field lists
# change, so that cached results keyed on the old format are automatically
# skipped rather than returned incomplete.
_ACAD_CACHE_VERSION = "v3"  # v1=SS+CR, v2=+OpenAlex, v3=+extra SS fields


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
    venue="",
    publication_types=None,
    citation_count=None,
    is_open_access=None,
):
    """
    Convert academic search results into
    one common format.
    """

    if authors is None:
        authors = []

    if publication_types is None:
        publication_types = []

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

        "venue": (
            venue.strip()
            if venue
            else ""
        ),

        # Extra fields from Semantic Scholar / OpenAlex
        "publication_types": publication_types,
        "citation_count": citation_count,
        "is_open_access": is_open_access,
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

        raw_doi = result.get("doi", "") or ""
        doi = re.sub(r"^(https?://doi\.org/|doi:)", "", raw_doi.strip(), flags=re.IGNORECASE).lower()

        url = (
            result.get(
                "url",
                ""
            )
            .strip()
            .lower()
        )

        title = (result.get("title") or "").strip()
        if not title or title.lower() in ("untitled", "unknown title", "none", "no title"):
            continue

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


def rank_academic_results(results):
    """
    Preserve the search engine's relevance order (engines return by relevance).
    Drop records that have no abstract/content — the LLM cannot analyse them —
    but only if at least one record does have content; otherwise keep all so
    the caller always gets something back.
    """
    usable = [r for r in results if r.get("content")]
    return usable if usable else results


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

    logger.info("\n[Academic Search]")
    logger.info('Semantic Scholar query: "%s"', query)

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
            "venue,"
            "publicationTypes,"
            "citationCount,"
            "openAccessPdf,"
            "isOpenAccess"
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

        logger.info(
            "Semantic Scholar attempt %d/%d",
            attempt, SEMANTIC_SCHOLAR_MAX_RETRIES
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

                logger.warning("Semantic Scholar rate limit reached (429).")

                if attempt < (
                    SEMANTIC_SCHOLAR_MAX_RETRIES
                ):

                    logger.info("Waiting before retry...")

                    time.sleep(
                        SEMANTIC_SCHOLAR_RETRY_DELAY_SECONDS
                    )

                    continue

                logger.warning("Semantic Scholar retries exhausted.")

                return []

            response.raise_for_status()

            data = response.json()

            break

        except requests.RequestException as error:

            logger.error("Semantic Scholar request failed: %s", error)

            if attempt < (
                SEMANTIC_SCHOLAR_MAX_RETRIES
            ):

                logger.info("Retrying...")

                time.sleep(
                    SEMANTIC_SCHOLAR_RETRY_DELAY_SECONDS
                )

                continue

            return []

        except ValueError as error:

            logger.error("Semantic Scholar returned invalid JSON: %s", error)

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

                venue=paper.get("venue", "") or "",

                publication_types=paper.get("publicationTypes") or [],

                citation_count=paper.get("citationCount"),

                is_open_access=paper.get("isOpenAccess"),
            )
        )

    results = remove_duplicates(
        results
    )

    results = rank_academic_results(
        results
    )

    logger.info("Semantic Scholar returned %d unique papers.", len(results))

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

    logger.info("\n[Academic Fallback]")
    logger.info('Crossref query: "%s"', query)

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

        logger.error("Crossref request failed: %s", error)

        return []

    except ValueError as error:

        logger.error("Crossref returned invalid JSON: %s", error)

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
        # ABSTRACT & XML CLEANING
        # ----------------------------------------------------

        raw_abstract = item.get(
            "abstract",
            ""
        )
        abstract = re.sub(r"<[^>]+>", "", raw_abstract).strip() if raw_abstract else ""

        # ----------------------------------------------------
        # VENUE (container-title)
        # ----------------------------------------------------

        container_titles = item.get("container-title", [])
        venue = container_titles[0].strip() if container_titles and isinstance(container_titles, list) else ""

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

                venue=venue,
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

    logger.info("Crossref returned %d unique papers.", len(results))

    return results


# ============================================================
# OPENALEX
# ============================================================

def search_openalex(
    query,
    max_results=SEMANTIC_SCHOLAR_MAX_RESULTS,
):
    """
    Search OpenAlex for academic works.

    Advantages over Crossref:
    - Better topical search (not citation-string matching).
    - Includes is_retracted flag — retracted papers are skipped.
    - Abstract is reconstructed from the inverted index.
    - Open-access metadata and venue data are reliable.
    """

    logger.info("\n[OpenAlex Search]")
    logger.info('OpenAlex query: "%s"', query)

    mailto = OPENALEX_MAILTO or None

    params = {
        "search": query,
        "per-page": max_results,
        "select": (
            "id,doi,title,authorships,publication_year,"
            "abstract_inverted_index,primary_location,"
            "is_retracted,is_oa,type"
        ),
    }
    if mailto:
        params["mailto"] = mailto

    try:
        response = requests.get(
            f"{OPENALEX_BASE_URL}/works",
            params=params,
            timeout=OPENALEX_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        logger.error("OpenAlex request failed: %s", error)
        return []

    except ValueError as error:
        logger.error("OpenAlex returned invalid JSON: %s", error)
        return []

    raw_works = data.get("results", [])

    if not raw_works:
        logger.info("OpenAlex returned zero results.")
        return []

    results = []

    for work in raw_works:

        # Skip retracted papers
        if work.get("is_retracted"):
            logger.debug("Skipping retracted work: %s", work.get("id"))
            continue

        # Reconstruct abstract from inverted index
        inv = work.get("abstract_inverted_index") or {}
        if inv:
            pos_to_word = {pos: word for word, positions in inv.items() for pos in positions}
            abstract = " ".join(pos_to_word[i] for i in sorted(pos_to_word))
        else:
            abstract = ""

        # Authors
        authors = [
            a["author"]["display_name"]
            for a in work.get("authorships", [])
            if (a.get("author") or {}).get("display_name")
        ]

        # DOI — strip the full URL prefix
        raw_doi = work.get("doi") or ""
        doi = re.sub(r"^https?://doi\.org/", "", raw_doi, flags=re.IGNORECASE)

        # URL — prefer DOI landing page, fall back to OpenAlex ID page
        url = raw_doi or work.get("id") or ""

        # Venue
        primary_loc = work.get("primary_location") or {}
        src = primary_loc.get("source") or {}
        venue = src.get("display_name") or ""

        # Publication type
        pub_type = work.get("type") or ""

        results.append(
            normalize_academic_result(
                title=work.get("title") or "",
                url=url,
                abstract=abstract,
                authors=authors,
                year=work.get("publication_year"),
                doi=doi,
                source="OpenAlex",
                venue=venue,
                publication_types=[pub_type] if pub_type else [],
                is_open_access=work.get("is_oa"),
            )
        )

    results = remove_duplicates(results)
    results = rank_academic_results(results)

    logger.info("OpenAlex returned %d unique papers.", len(results))

    return results


# ============================================================
# MAIN ACADEMIC SEARCH
# ============================================================

def academic_search(
    query,
    max_results=SEMANTIC_SCHOLAR_MAX_RESULTS,
):
    """
    Academic search pipeline.

    Runs all three providers concurrently, then merges and deduplicates:

        Semantic Scholar  +  OpenAlex  +  Crossref (parallel)
                              ↓
                     remove_duplicates (DOI + title)
                              ↓
                     rank_academic_results (filter no-abstract, keep order)
                              ↓
                     clean academic sources

    Why parallel instead of fallback?
    - Semantic Scholar has the best abstracts but can rate-limit.
    - OpenAlex has better topical relevance and skips retractions.
    - Crossref fills DOI gaps even when abstracts are sparse.
    Running all three and deduplicating by DOI gives better coverage
    with no extra wall-clock cost (network I/O is concurrent).
    """

    if not query or not query.strip():
        logger.warning("[Academic Search] Empty academic query.")
        return []

    query = query.strip()

    from config import SEARCH_CACHE_ENABLED
    # Cache key includes a version stamp so stale cached results
    # (missing OpenAlex data) are not returned after the upgrade.
    cache_key = f"acad:{_ACAD_CACHE_VERSION}:{query}"

    if SEARCH_CACHE_ENABLED:
        cached = get_cached(cache_key, max_results)
        if cached is not None:
            logger.info("[Academic Search] Cache hit for query: %s", query)
            return cached

    # ========================================================
    # RUN ALL PROVIDERS CONCURRENTLY
    # ========================================================

    import concurrent.futures

    provider_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(search_semantic_scholar, query, max_results): "Semantic Scholar",
            executor.submit(search_openalex, query, max_results): "OpenAlex",
            executor.submit(search_crossref, query, max_results): "Crossref",
        }

        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                provider_results[name] = future.result()
            except Exception as exc:
                logger.error("[Academic Search] %s raised an exception: %s", name, exc)
                provider_results[name] = []

    ss_results = provider_results.get("Semantic Scholar", [])
    oa_results = provider_results.get("OpenAlex", [])
    cr_results = provider_results.get("Crossref", [])

    if not ss_results:
        logger.warning(
            "[Academic Search] Semantic Scholar returned no results (network error or zero results)."
        )
    if not oa_results:
        logger.warning(
            "[Academic Search] OpenAlex returned no results (network error or zero results)."
        )
    if not cr_results:
        logger.info(
            "[Academic Search] Crossref returned no results."
        )

    # ========================================================
    # MERGE: Semantic Scholar first (best abstracts), then
    # OpenAlex (topical relevance), then Crossref (DOI coverage)
    # ========================================================

    combined = ss_results + oa_results + cr_results

    if not combined:
        logger.error("[Academic Search] All academic search providers returned nothing.")
        return []

    results = remove_duplicates(combined)
    results = rank_academic_results(results)

    logger.info(
        "[Academic Search] Final: %d unique papers from %d+%d+%d raw.",
        len(results), len(ss_results), len(oa_results), len(cr_results),
    )

    if SEARCH_CACHE_ENABLED:
        set_cached(cache_key, max_results, results)

    return results


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