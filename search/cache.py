"""
Search Cache
============

Simple JSON-file cache for web search results.

Keyed by (query, max_results).
Entries expire after SEARCH_CACHE_TTL_HOURS (default 24h).

Usage
-----
    from search.cache import get_cached, set_cached

    results = get_cached(query, max_results)
    if results is None:
        results = _fetch_from_network(query, max_results)
        set_cached(query, max_results, results)
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

from config import SEARCH_CACHE_DIR, SEARCH_CACHE_TTL_HOURS

logger = logging.getLogger(__name__)


def _cache_key(query: str, max_results: int) -> str:
    """Stable, filesystem-safe cache key for (query, max_results)."""
    raw = f"{query.lower().strip()}|{max_results}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(key: str) -> str:
    return os.path.join(SEARCH_CACHE_DIR, f"{key}.json")


def get_cached(query: str, max_results: int):
    """
    Return cached search results if fresh, else None.

    Parameters
    ----------
    query : str
    max_results : int

    Returns
    -------
    list | None
        Cached results list, or None if cache miss / expired.
    """
    try:
        key  = _cache_key(query, max_results)
        path = _cache_path(key)

        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)

        cached_at = datetime.fromisoformat(entry["cached_at"])
        age_hours  = (
            datetime.now(timezone.utc) - cached_at
        ).total_seconds() / 3600

        if age_hours > SEARCH_CACHE_TTL_HOURS:
            logger.debug("[Cache] Expired entry for: %s", query[:60])
            os.remove(path)
            return None

        logger.info(
            "[Cache] HIT for: \"%s\" (age: %.1fh)", query[:60], age_hours
        )
        return entry["results"]

    except Exception as error:
        logger.debug("[Cache] Read error (treating as miss): %s", error)
        return None


def set_cached(query: str, max_results: int, results: list) -> None:
    """
    Persist search results to cache.

    Parameters
    ----------
    query : str
    max_results : int
    results : list
        The search result list to cache.
    """
    try:
        os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)

        key  = _cache_key(query, max_results)
        path = _cache_path(key)

        entry = {
            "query":      query,
            "max_results": max_results,
            "cached_at":  datetime.now(timezone.utc).isoformat(),
            "results":    results,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        logger.debug("[Cache] Stored %d results for: %s", len(results), query[:60])

    except Exception as error:
        logger.warning("[Cache] Write failed (non-fatal): %s", error)


def clear_cache() -> int:
    """
    Delete all cache entries. Returns number of files deleted.
    """
    if not os.path.exists(SEARCH_CACHE_DIR):
        return 0

    deleted = 0
    for filename in os.listdir(SEARCH_CACHE_DIR):
        if filename.endswith(".json"):
            os.remove(os.path.join(SEARCH_CACHE_DIR, filename))
            deleted += 1

    logger.info("[Cache] Cleared %d entries.", deleted)
    return deleted
