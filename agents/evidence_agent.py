import hashlib
import logging
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field

from llm_client import create_llm

logger = logging.getLogger(__name__)


def _stable_source_key(source: dict) -> str:
    """Create a deterministic key from a source DOI or URL."""
    identifier = (
        source.get("doi")
        or source.get("url")
        or source.get("title")
        or ""
    )
    identifier = str(identifier).strip().lower().rstrip("/")
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return f"src_{digest[:16]}"


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace only, preserving the original wording."""
    return " ".join((text or "").split())


def _normalize_for_matching(text: str) -> str:
    """Normalize formatting for comparison, not for stored quotations."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def _split_into_sentences(content: str) -> list[str]:
    """Split content into meaningful sentence candidates."""
    if not content:
        return []
    raw_sents = re.split(r"(?<=[.!?])\s+", content.strip())
    sentences = []
    for s in raw_sents:
        s_clean = re.sub(r"\s+", " ", s).strip()
        if s_clean:
            sentences.append(s_clean)
    return sentences


def _recover_verbatim_passage(
    candidate_quote: str,
    source_content: str,
    min_similarity: float = 0.85,
) -> Optional[str]:
    """
    Locate a likely passage, but return only text copied from the source.
    A fuzzy match locates a candidate; it does not prove claim support.
    """
    if not candidate_quote or not source_content:
        return None

    candidate_norm = _normalize_for_matching(candidate_quote)
    if len(candidate_norm) < 25:
        return None

    sentences = _split_into_sentences(source_content)
    if not sentences:
        sentences = [source_content.strip()]

    # Build candidate passages: individual sentences and adjacent pairs/triplets
    candidates = list(sentences)
    for i in range(len(sentences) - 1):
        candidates.append(f"{sentences[i]} {sentences[i+1]}")
    if len(candidate_norm.split()) >= 25:
        for i in range(len(sentences) - 2):
            candidates.append(f"{sentences[i]} {sentences[i+1]} {sentences[i+2]}")

    candidate_clean = candidate_norm.strip(" .!?,;:\"'“”‘’")
    candidate_lower = candidate_clean.lower()
    candidate_words = set(re.findall(r"\b\w+\b", candidate_lower))

    best_match = None
    best_score = 0.0

    for cand in candidates:
        cand_norm = _normalize_for_matching(cand)
        cand_clean = cand_norm.strip(" .!?,;:\"'“”‘’")
        cand_lower = cand_clean.lower()

        # Check exact containment or normalized substring
        if candidate_clean == cand_clean or candidate_lower == cand_lower:
            return cand

        if candidate_clean in cand_norm or candidate_lower in cand_lower:
            return cand

        # SequenceMatcher similarity
        ratio = SequenceMatcher(None, candidate_lower, cand_lower).ratio()
        if ratio >= best_score:
            best_score = ratio
            best_match = cand

        # Check word containment for cases where candidate is slightly paraphrased or missing intro words
        cand_words = set(re.findall(r"\b\w+\b", cand_lower))
        if candidate_words and cand_words:
            overlap = len(candidate_words & cand_words)
            token_containment = overlap / len(candidate_words)
            if token_containment >= min_similarity and ratio >= 0.70:
                score = token_containment * 0.5 + ratio * 0.5
                if score > best_score:
                    best_score = score
                    best_match = cand

    if best_match and best_score >= min_similarity:
        return best_match

    return None


# Backward-compatible alias
_recover_verbatim_evidence = _recover_verbatim_passage


def _stable_evidence_id(
    source_key: str,
    evidence: str,
    claim: str,
) -> str:
    """Create a deterministic ID for a claim and its supporting passage."""
    identity = "\n".join([
        source_key,
        _normalize_whitespace(evidence),
        _normalize_whitespace(claim),
    ])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"ev_{digest[:16]}"


class EvidenceItem(BaseModel):
    source_id: str = Field(
        description="The source URL or a stable source identifier."
    )

    source_title: str = Field(
        description="The title of the source."
    )

    source_url: str = Field(
        description="The exact URL of the source."
    )

    evidence: str = Field(
        description="Directly supported evidence from the source content."
    )

    claim: str = Field(
        description="A concise research claim supported by the evidence."
    )

    source_key: str = Field(
        default="",
        description="Stable identifier derived from the source DOI or URL.",
    )

    evidence_id: str = Field(
        default="",
        description="Stable identifier for this source-backed claim and passage.",
    )


class EvidenceOutput(BaseModel):
    items: List[EvidenceItem] = Field(
        description="Evidence items extracted from the supplied sources."
    )


def _extract_batch(topic, batch_with_indices, structured_llm):
    """Extract evidence for a single batch of sources."""
    valid_sources = [
        (idx, s) for idx, s in batch_with_indices
        if (s.get("content") or "").strip()
    ]
    if not valid_sources:
        return []

    def _is_paper(s):
        return bool(s.get("doi")) or "doi.org" in (s.get("url") or "")

    source_information = "\n\n".join(
        f"SOURCE_ID: source_{idx}\n"
        f"TYPE: {'ACADEMIC PAPER' if _is_paper(source) else 'WEB PAGE'}\n"
        f"TITLE: {source.get('title', '')}\n"
        f"URL: {source.get('url', '')}\n"
        f"CONTENT: {(source.get('content') or '')[:1500]}"
        for idx, source in valid_sources
    )

    prompt = f"""
You are an evidence extraction agent.

Research topic:
{topic}

Sources:
{source_information}

Extract only evidence directly supported by the source content.
For each useful source, provide:

- source_id: MUST strictly be the exact SOURCE_ID (e.g. source_1, source_2) corresponding to the CONTENT where the quote was taken.
- source_title: the source title
- source_url: the exact source URL
- evidence: MUST be an exact quote or verbatim passage copied directly from CONTENT. Do not paraphrase or use titles as evidence.
- claim: a concise research claim (at least 6 words) supported by that evidence. Never leave blank.
- For TYPE: ACADEMIC PAPER, return up to 3 separate items (repeat the same source_id) covering that paper's most important findings, results or conclusions. Each item needs its own verbatim quote and claim.
- For TYPE: WEB PAGE, return at most 1 item.

Do not create or modify URLs.
Do not invent facts, sources, authors, dates, statistics, or conclusions.
Do not include evidence that is not present in the supplied source content.
"""

    source_lookup = {
        f"source_{idx}": source
        for idx, source in valid_sources
    }
    try:
        response = structured_llm.invoke(prompt)
        if not response or not hasattr(response, "items"):
            return []

        validated_items = []
        for item in response.items:
            # Reject references to sources outside this batch.
            source = source_lookup.get(item.source_id)
            if source is None:
                logger.warning(
                    "[Evidence Agent] Rejected unknown source_id: %s",
                    item.source_id,
                )
                continue

            content = source.get("content") or ""
            evidence = item.evidence or ""
            claim = item.claim or ""

            # Evidence must be present and match the supplied source text.
            normalized_evidence = _normalize_whitespace(evidence)
            normalized_content = _normalize_whitespace(content)
            if not normalized_evidence or normalized_evidence not in normalized_content:
                recovered = _recover_verbatim_passage(
                    item.evidence,
                    content,
                )
                if recovered is None:
                    logger.warning(
                        "[Evidence Agent] Rejected unsupported quote for %s",
                        item.source_id,
                    )
                    continue

                # Store the original source passage, never the generated paraphrase.
                item.evidence = recovered
                evidence = recovered
                normalized_evidence = _normalize_whitespace(evidence)

            if len(claim.split()) < 6:
                logger.warning(
                    "[Evidence Agent] Rejected short claim for %s",
                    item.source_id,
                )
                continue

            source_key = _stable_source_key(source)
            evidence_id = _stable_evidence_id(
                source_key,
                evidence,
                claim,
            )

            # Never trust the LLM to supply authoritative source metadata.
            validated_items.append(
                EvidenceItem(
                    source_id=item.source_id,
                    source_title=source.get("title") or "",
                    source_url=source.get("url") or "",
                    evidence=evidence,
                    claim=claim,
                    source_key=source_key,
                    evidence_id=evidence_id,
                )
            )
        return validated_items
    except Exception as error:
        logger.error(
            "[Evidence Agent] Batch structured LLM call failed: %s",
            error,
        )
        return []


def extract_evidence(topic, sources, batch_size=5):
    """Extract source-grounded evidence for a research topic across batches of sources."""
    if not sources:
        return EvidenceOutput(items=[])

    llm = create_llm()
    structured_llm = llm.with_structured_output(EvidenceOutput)

    all_items = []
    # Process sources in chunks so later sources (e.g. academic literature) don't get starved
    for start in range(0, len(sources), batch_size):
        chunk = [
            (start + i + 1, s)
            for i, s in enumerate(sources[start : start + batch_size])
        ]
        items = _extract_batch(topic, chunk, structured_llm)
        all_items.extend(items)

    return EvidenceOutput(items=all_items)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_sources = [
        {
            "title": (
                "Artificial intelligence "
                "in healthcare"
            ),
            "url": "https://example.com/paper",
            "content": (
                "Artificial intelligence "
                "can assist healthcare "
                "professionals with diagnostic "
                "tasks."
            ),
        }
    ]

    result = extract_evidence(
        "Artificial Intelligence in Healthcare",
        test_sources,
    )

    print(
        "\n=============================="
    )

    print(
        "EXTRACTED EVIDENCE"
    )

    print(
        "=============================="
    )

    for item in result.items:

        print(
            f"\nSource: "
            f"{item.source_id}"
        )

        print(
            f"Title: "
            f"{item.source_title}"
        )

        print(
            f"Evidence: "
            f"{item.evidence}"
        )

        print(
            f"Claim: "
            f"{item.claim}"
        )
