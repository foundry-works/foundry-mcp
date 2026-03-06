"""Source-deepening verification strategy for deep research.

When claim verification produces UNSUPPORTED verdicts, distinguishes between
claims that need *new* sources (widening) and claims that need *better reading*
of existing sources (deepening).

Three deepening strategies:
  1. **Inferential** — comparative/recommendation claims that are unsupported
     by design (synthesis conclusions). No action needed.
  2. **Deepen-window** — factual claims where the source has rich raw_content
     (>16K chars) but verification only checked an 8K keyword window. Re-verify
     with a 3x window.
  3. **Deepen-extract** — factual claims where the source is thin (<4K chars).
     Re-extract the URL or resolve a DOI to get richer content.
  4. **Widen** — factual claims that genuinely need new sources via web search.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from foundry_mcp.core.research.models.deep_research import (
        ClaimVerdict,
        ClaimVerificationResult,
        DeepResearchState,
    )
    from foundry_mcp.core.research.models.sources import ResearchSource
    from foundry_mcp.core.research.providers.tavily_extract import (
        TavilyExtractProvider,
    )

from foundry_mcp.core.research.workflows.deep_research._constants import (
    VERIFICATION_SOURCE_DEEPEN_MAX_CHARS,
)

logger = logging.getLogger(__name__)

# Claim types considered "inferential" (synthesis conclusions, not factual).
_INFERENTIAL_CLAIM_TYPES = frozenset({"comparative", "positive"})

# Recommendation language patterns that reinforce inferential classification.
_RECOMMENDATION_PATTERNS = re.compile(
    r"(?:best|better|worse|ideal|recommend|should|prefer|top pick|overall)",
    re.IGNORECASE,
)

# Minimum raw_content length for "rich" classification (deepen_window).
_RICH_CONTENT_THRESHOLD = 16_000

# Maximum raw_content length for "thin" classification (deepen_extract).
_THIN_CONTENT_THRESHOLD = 4_000


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------


@dataclass
class DeepClassification:
    """Result of classifying UNSUPPORTED claims into deepening strategies."""

    inferential: list[ClaimVerdict] = field(default_factory=list)
    deepen_window: list[ClaimVerdict] = field(default_factory=list)
    deepen_extract: list[ClaimVerdict] = field(default_factory=list)
    widen: list[ClaimVerdict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1: Classify UNSUPPORTED claims
# ---------------------------------------------------------------------------


def classify_unsupported_claims(
    verification_result: "ClaimVerificationResult",
    citation_map: dict[int, "ResearchSource"],
) -> DeepClassification:
    """Classify UNSUPPORTED claims into deepening vs. widening vs. inferential.

    Args:
        verification_result: Completed claim verification result.
        citation_map: Mapping from citation number to ResearchSource.

    Returns:
        DeepClassification with four lists: inferential, deepen_window,
        deepen_extract, and widen.
    """
    result = DeepClassification()

    for claim in verification_result.details:
        if claim.verdict != "UNSUPPORTED":
            continue

        # Check for inferential claims (synthesis/comparative).
        if claim.claim_type in _INFERENTIAL_CLAIM_TYPES and _RECOMMENDATION_PATTERNS.search(
            claim.claim
        ):
            result.inferential.append(claim)
            continue

        # For factual claims, check the cited source content.
        if claim.cited_sources:
            best_source = _get_richest_cited_source(claim, citation_map)
            if best_source is not None:
                content_len = len(best_source.raw_content or "")
                if content_len >= _RICH_CONTENT_THRESHOLD:
                    result.deepen_window.append(claim)
                    continue
                elif content_len < _THIN_CONTENT_THRESHOLD:
                    result.deepen_extract.append(claim)
                    continue

        # No cited source or medium-length content — needs new sources.
        result.widen.append(claim)

    return result


def _get_richest_cited_source(
    claim: "ClaimVerdict",
    citation_map: dict[int, "ResearchSource"],
) -> Optional["ResearchSource"]:
    """Return the cited source with the most raw_content, or None."""
    best: Optional["ResearchSource"] = None
    best_len = -1
    for src_num in claim.cited_sources:
        source = citation_map.get(src_num)
        if source is None:
            continue
        content_len = len(source.raw_content or "")
        if content_len > best_len:
            best = source
            best_len = content_len
    return best


# ---------------------------------------------------------------------------
# Step 2: Re-verify with expanded windows
# ---------------------------------------------------------------------------


async def reverify_with_expanded_window(
    claims: list["ClaimVerdict"],
    citation_map: dict[int, "ResearchSource"],
    llm_call_fn: Callable,
    *,
    max_chars: int = VERIFICATION_SOURCE_DEEPEN_MAX_CHARS,
) -> list["ClaimVerdict"]:
    """Re-verify claims using a larger content window from the same source.

    Uses ``_multi_window_truncate`` with a 3x budget, or passes full
    raw_content if it fits. Does NOT fetch anything new — purely re-reads
    existing content.

    Args:
        claims: Claims classified as ``deepen_window``.
        citation_map: Mapping from citation number to ResearchSource.
        llm_call_fn: Async callable with signature
            ``(system_prompt, user_prompt) -> str``.
        max_chars: Maximum characters per source for expanded window.

    Returns:
        The same claim objects with potentially upgraded verdicts.
    """
    if not claims:
        return claims

    from foundry_mcp.core.research.workflows.deep_research.phases.claim_verification import (
        _VERIFICATION_SYSTEM_PROMPT,
        _parse_verification_response,
    )

    upgraded_count = 0

    for claim in claims:
        user_prompt = _build_expanded_verification_prompt(
            claim, citation_map, max_chars
        )
        if user_prompt is None:
            continue

        try:
            response = await llm_call_fn(_VERIFICATION_SYSTEM_PROMPT, user_prompt)
            if not isinstance(response, str) or not response.strip():
                continue

            parsed = _parse_verification_response(response)
            new_verdict = parsed.get("verdict", "UNSUPPORTED")

            if new_verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                old_verdict = claim.verdict
                claim.verdict = new_verdict
                claim.evidence_quote = parsed.get("evidence_quote")
                claim.explanation = (
                    f"[Upgraded from {old_verdict} via expanded-window re-verification] "
                    f"{parsed.get('explanation', '')}"
                )
                upgraded_count += 1
        except Exception as exc:
            logger.warning(
                "Expanded-window re-verification failed for claim %r: %s",
                claim.claim[:80],
                exc,
            )

    logger.info(
        "Expanded-window re-verification: %d/%d claims upgraded",
        upgraded_count,
        len(claims),
    )
    return claims


def _build_expanded_verification_prompt(
    claim: "ClaimVerdict",
    citation_map: dict[int, "ResearchSource"],
    max_chars: int,
) -> Optional[str]:
    """Build a verification prompt with an expanded content window."""
    from foundry_mcp.core.research.workflows.deep_research.phases.claim_verification import (
        _multi_window_truncate,
        _resolve_source_text,
    )

    source_sections: list[str] = []

    for src_num in claim.cited_sources:
        source = citation_map.get(src_num)
        if source is None:
            continue
        text = _resolve_source_text(source)
        if text is None:
            continue

        truncated = _multi_window_truncate(text, claim.claim, max_chars)
        header = f"### Source [{src_num}]: {source.title}"
        if source.url:
            header += f"\nURL: {source.url}"
        source_sections.append(f"{header}\n\n{truncated}")

    if not source_sections:
        return None

    sources_text = "\n\n---\n\n".join(source_sections)
    return (
        f"## Source Content (expanded window)\n\n{sources_text}\n\n"
        f"## Claim to Verify\n\n"
        f'"{claim.claim}"\n'
        f"Claim type: {claim.claim_type}\n"
        f"Cited sources: {', '.join(f'[{n}]' for n in claim.cited_sources)}\n\n"
        f"## Task\n\n"
        f"Does the source content SUPPORT, CONTRADICT, or provide NO EVIDENCE for this claim?\n"
        f"Return a JSON object with: verdict, evidence_quote, explanation"
    )


# ---------------------------------------------------------------------------
# Step 3: Re-extract thin sources
# ---------------------------------------------------------------------------


async def deepen_thin_sources(
    claims: list["ClaimVerdict"],
    citation_map: dict[int, "ResearchSource"],
    _state: "DeepResearchState",
    extract_provider: Optional["TavilyExtractProvider"],
) -> int:
    """Re-extract content for sources that were too thin for verification.

    For each thin source:
    1. If source has URL → re-extract via TavilyExtractProvider
    2. Update source.raw_content with the richer content

    Args:
        claims: Claims classified as ``deepen_extract``.
        citation_map: Mapping from citation number to ResearchSource.
        state: Deep research state (for source mutation).
        extract_provider: TavilyExtractProvider instance, or None to skip.

    Returns:
        Number of sources successfully deepened.
    """
    if not claims or extract_provider is None:
        return 0

    # Collect unique source URLs to re-extract.
    sources_to_deepen: dict[int, "ResearchSource"] = {}
    for claim in claims:
        for src_num in claim.cited_sources:
            if src_num in sources_to_deepen:
                continue
            source = citation_map.get(src_num)
            if source is None or not source.url:
                continue
            sources_to_deepen[src_num] = source

    if not sources_to_deepen:
        return 0

    urls = [s.url for s in sources_to_deepen.values() if s.url]
    if not urls:
        return 0

    deepened_count = 0

    try:
        extracted = await extract_provider.extract(
            urls=urls[:10],  # Tavily max 10 URLs per call
            extract_depth="advanced",
            format="markdown",
        )

        # Build URL → extracted content map
        url_to_content: dict[str, str] = {}
        for ex_source in extracted:
            if ex_source.raw_content and ex_source.url:
                url_to_content[ex_source.url] = ex_source.raw_content
            elif ex_source.content and ex_source.url:
                url_to_content[ex_source.url] = ex_source.content

        # Update original sources with richer content
        for src_num, source in sources_to_deepen.items():
            new_content = url_to_content.get(source.url or "")
            if new_content and len(new_content) > len(source.raw_content or ""):
                # Preserve original content for audit
                source.metadata["_pre_deepen_content"] = source.raw_content or source.content or source.snippet or ""
                source.raw_content = new_content
                deepened_count += 1

    except Exception as exc:
        logger.warning("Source deepening extraction failed: %s", exc)

    logger.info(
        "Source deepening: %d/%d sources re-extracted with richer content",
        deepened_count,
        len(sources_to_deepen),
    )
    return deepened_count
