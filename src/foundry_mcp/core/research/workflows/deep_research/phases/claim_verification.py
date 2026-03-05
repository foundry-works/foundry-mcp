"""Post-synthesis claim verification pipeline.

Standalone module with free async functions (not a mixin) that verifies
factual claims in the generated report against cited source material.

Two-pass approach:
  Pass 1 — Claim extraction: LLM extracts verifiable factual claims as JSON.
  Pass 2 — Claim-source alignment: Each claim is verified against its cited
           sources, producing SUPPORTED / CONTRADICTED / UNSUPPORTED /
           PARTIALLY_SUPPORTED verdicts.

Contradicted claims trigger single-pass targeted re-synthesis corrections.

Called from ``workflow_execution.py`` between synthesis completion and
``mark_completed()``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig
    from foundry_mcp.core.research.models.sources import ResearchSource

from foundry_mcp.core.research.models.deep_research import (
    ClaimVerdict,
    ClaimVerificationResult,
    DeepResearchState,
)
from foundry_mcp.core.research.workflows.deep_research._concurrency import (
    check_gather_cancellation,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (
    VERIFICATION_SOURCE_MAX_CHARS,
)
from foundry_mcp.core.research.workflows.deep_research._token_budget import (
    CHARS_PER_TOKEN,
)

logger = logging.getLogger(__name__)

from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    LLMCallResult,
    execute_llm_call,
)

# Stopwords for keyword-proximity truncation (common function words).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "have",
        "been",
        "will",
        "would",
        "could",
        "should",
        "their",
        "there",
        "which",
        "about",
        "where",
        "these",
        "those",
        "does",
        "into",
        "also",
        "more",
        "than",
        "only",
        "most",
        "each",
        "some",
        "when",
        "they",
        "were",
        "other",
    }
)

# Claim types ordered by verification priority (highest first).
_CLAIM_TYPE_PRIORITY: dict[str, int] = {
    "negative": 0,
    "quantitative": 1,
    "comparative": 2,
    "positive": 3,
}

# Valid verdicts.
_VALID_VERDICTS: frozenset[str] = frozenset(
    {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED"}
)

# Valid claim types.
_VALID_CLAIM_TYPES: frozenset[str] = frozenset(
    {"negative", "quantitative", "comparative", "positive"}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_source_text(source: "ResearchSource") -> Optional[str]:
    """Return the best available text content for a source.

    Falls back through raw_content → content → snippet.
    Prefers raw_content (full page text) over content (compressed summary)
    because summaries strip specific numbers and factual details that
    claim verification needs.
    Returns None if no text is available.
    """
    return source.raw_content or source.content or source.snippet


def _multi_window_truncate(
    text: str,
    claim_text: str,
    max_chars: int,
    max_windows: int = 3,
) -> str:
    """Truncate source text to multiple windows centered on claim-relevant keywords.

    Instead of a single window around the first keyword match, finds all keyword
    positions, scores candidate windows by keyword density, and returns the top
    N non-overlapping windows concatenated with ``[...]`` separators.

    Args:
        text: Full source text.
        claim_text: The claim being verified (used to extract keywords).
        max_chars: Total character budget across all windows.
        max_windows: Maximum number of non-overlapping windows.

    Returns:
        Concatenated windows with ``[...]`` separators, within *max_chars* budget.
    """
    if len(text) <= max_chars:
        return text

    # Extract keywords from claim (words >= 4 chars, not stopwords).
    keywords = [
        w.lower()
        for w in re.split(r"\s+", claim_text)
        if len(w) >= 4 and w.lower() not in _STOPWORDS
    ]

    if not keywords:
        return text[:max_chars]

    text_lower = text.lower()

    # Find ALL positions of each keyword (case-insensitive).
    keyword_positions: list[tuple[int, str]] = []
    for kw in keywords:
        start = 0
        while True:
            pos = text_lower.find(kw, start)
            if pos == -1:
                break
            keyword_positions.append((pos, kw))
            start = pos + 1

    if not keyword_positions:
        # Fallback: prefix truncation.
        return text[:max_chars]

    # Sort positions by document order.
    keyword_positions.sort(key=lambda x: x[0])

    # Cluster keyword positions by proximity.
    cluster_radius = max_chars // max_windows
    clusters: list[list[tuple[int, str]]] = []
    current_cluster: list[tuple[int, str]] = [keyword_positions[0]]

    for pos, kw in keyword_positions[1:]:
        if pos - current_cluster[-1][0] <= cluster_radius:
            current_cluster.append((pos, kw))
        else:
            clusters.append(current_cluster)
            current_cluster = [(pos, kw)]
    clusters.append(current_cluster)

    # Score clusters by distinct keyword count (not total occurrences).
    def _cluster_score(cluster: list[tuple[int, str]]) -> int:
        return len({kw for _, kw in cluster})

    # Select top N clusters by score.
    scored = sorted(clusters, key=_cluster_score, reverse=True)
    selected = scored[:max_windows]

    # Sort selected clusters by document position (median position).
    selected.sort(key=lambda c: c[len(c) // 2][0])

    # Adaptive window sizing: distribute full budget across selected clusters.
    _SEPARATOR = "\n[...]\n"
    num_separators = len(selected) - 1
    total_separator_chars = len(_SEPARATOR) * num_separators
    available_for_windows = max(1, max_chars - total_separator_chars)
    window_size = max(1, available_for_windows // len(selected))

    # Extract windows, ensuring non-overlap.
    windows: list[str] = []
    prev_end = -1

    for cluster in selected:
        positions = [p for p, _ in cluster]
        median_pos = positions[len(positions) // 2]

        # Center window on median position.
        half = window_size // 2
        w_start = max(0, median_pos - half)
        w_end = min(len(text), w_start + window_size)
        w_start = max(0, w_end - window_size)

        # Ensure non-overlapping with previous window.
        if w_start <= prev_end:
            w_start = prev_end + 1
            w_end = min(len(text), w_start + window_size)

        if w_start >= len(text) or w_start >= w_end:
            continue

        windows.append(text[w_start:w_end])
        prev_end = w_end

    if not windows:
        return text[:max_chars]

    return _SEPARATOR.join(windows)


def _sort_claims_by_priority(claims: list[ClaimVerdict]) -> list[ClaimVerdict]:
    """Sort claims by verification priority: negative > quantitative > comparative > positive.

    Within each type, claims with more cited sources come first.
    """
    return sorted(
        claims,
        key=lambda c: (
            _CLAIM_TYPE_PRIORITY.get(c.claim_type, 99),
            -len(c.cited_sources),
        ),
    )


# ---------------------------------------------------------------------------
# Pass 1: Claim Extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a factual claim extraction assistant. Your task is to identify verifiable \
factual claims that are backed by inline citations [N] in the report section below.

For each inline citation [N] you find, extract the specific factual claim it supports.

Rules:
- ONLY extract claims that have an explicit [N] citation adjacent to them
- Skip opinions, recommendations, subjective assessments, and uncited statements
- Classify each claim:
  - "negative" — "X does NOT do Y", "X is not available"
  - "quantitative" — specific numbers, dates, prices, ratios, percentages
  - "comparative" — "X is better/worse than Y", "X has Y but Z does not"
  - "positive" — "X does Y", "X supports Z"
- If a single sentence cites multiple sources [1][2], extract ONE claim for that sentence \
with all citation numbers in cited_sources

For each claim, return:
- "claim": the exact factual assertion (do not paraphrase)
- "claim_type": one of "negative", "quantitative", "comparative", "positive"
- "cited_sources": list of citation numbers (integers) from the inline [N] references
- "report_section": the section heading this claim appears under
- "quote_context": the exact sentence containing this claim and its citation(s)

Return a JSON array of claim objects. Return ONLY the JSON array, no other text.
If no cited claims are found in this section, return an empty array: []
"""


# Regex to detect bibliography/references headings (anchored to avoid false positives).
_BIBLIOGRAPHY_HEADING_RE = re.compile(
    r"(?i)^(bibliography|references|sources|works cited)$"
)

# Minimum section size before merging with the next section.
_MIN_SECTION_CHARS = 500


def _split_report_into_sections(report: str) -> list[dict[str, str]]:
    """Split report into section-level chunks for parallel extraction.

    Splits on ``## `` and ``### `` headings.  Each chunk includes its heading
    and body text.  Sections smaller than :data:`_MIN_SECTION_CHARS` are merged
    with the following section to avoid trivially small extraction calls.

    Returns:
        List of dicts with ``"section"`` (heading text) and ``"content"``
        (full text including heading line).
    """
    # Split on level-2 and level-3 headings.  The regex matches headings at
    # the start of the string or after a newline.
    parts = re.split(r"(?:^|\n)(#{2,3} .+)", report)

    # ``re.split`` with a capturing group interleaves non-match / match:
    #   [preamble, heading1, body1, heading2, body2, ...]
    # Build raw sections from these pairs.
    raw_sections: list[dict[str, str]] = []

    # If the first element is non-empty and not a heading, it's preamble text
    # (content before any heading).
    idx = 0
    if parts and parts[0].strip():
        raw_sections.append({"section": "", "content": parts[0]})
        idx = 1
    elif parts:
        # Empty preamble — skip it.
        idx = 1

    # Walk heading/body pairs.
    while idx < len(parts):
        heading = parts[idx].strip() if idx < len(parts) else ""
        body = parts[idx + 1] if idx + 1 < len(parts) else ""
        # Heading text without the leading ## / ###.
        heading_text = re.sub(r"^#{2,3}\s*", "", heading)

        # Exclude bibliography/references sections.
        if _BIBLIOGRAPHY_HEADING_RE.match(heading_text.strip()):
            idx += 2
            continue

        raw_sections.append({
            "section": heading_text,
            "content": heading + body,
        })
        idx += 2

    # Fallback: no headings found → single chunk with full report.
    if not raw_sections:
        return [{"section": "", "content": report}]

    # Discard final chunk if it lacks a heading (truncation boundary fragment),
    # unless it's the only chunk.
    if len(raw_sections) > 1 and not raw_sections[-1]["section"]:
        raw_sections.pop()

    # Merge consecutive small sections (< _MIN_SECTION_CHARS) with the next.
    merged: list[dict[str, str]] = []
    for sec in raw_sections:
        if merged and len(merged[-1]["content"]) < _MIN_SECTION_CHARS:
            # Merge into previous (small) section.
            merged[-1]["content"] += "\n" + sec["content"]
            # Keep the first section's heading (it's the one that was too small).
            if not merged[-1]["section"] and sec["section"]:
                merged[-1]["section"] = sec["section"]
        else:
            merged.append(dict(sec))

    # If the last section is still small after the forward pass, merge it
    # backwards into the previous section (if one exists).
    if len(merged) > 1 and len(merged[-1]["content"]) < _MIN_SECTION_CHARS:
        merged[-2]["content"] += "\n" + merged[-1]["content"]
        merged.pop()

    return merged if merged else [{"section": "", "content": report}]


async def _extract_claims_from_chunk(
    chunk: dict[str, str],
    workflow: Any,
    state: DeepResearchState,
    system_prompt: str,
    provider_id: str,
    timeout: float,
    max_claims_per_chunk: int,
) -> list[ClaimVerdict]:
    """Extract claims from a single report section chunk.

    Args:
        chunk: Dict with ``"section"`` and ``"content"`` keys.
        workflow: DeepResearchWorkflow instance (for lifecycle instrumentation).
        state: Current research state.
        system_prompt: Extraction system prompt.
        provider_id: LLM provider to use.
        timeout: Per-call timeout in seconds.
        max_claims_per_chunk: Max claims to parse from this chunk.

    Returns:
        List of :class:`ClaimVerdict` objects extracted from this chunk.
    """
    chunk_prompt = (
        f"## Section\n\n{chunk['content']}\n\n"
        f"## Task\n\n"
        f"Extract cited factual claims from the section above as a JSON array."
    )
    try:
        ret = await execute_llm_call(
            workflow=workflow,
            state=state,
            phase_name="claim_extraction",
            system_prompt=system_prompt,
            user_prompt=chunk_prompt,
            provider_id=provider_id,
            model=None,
            temperature=0.0,
            timeout=timeout,
            role="claim_verification",
        )
        if isinstance(ret, LLMCallResult):
            extraction_result = ret.result
        else:
            # WorkflowResult returned on error — treat as failure.
            logger.warning(
                "Chunk extraction failed for section %r: lifecycle returned error",
                chunk.get("section", "")[:60],
            )
            return []
        if not extraction_result.success or not extraction_result.content:
            logger.warning(
                "Chunk extraction failed for section %r: LLM call unsuccessful",
                chunk.get("section", "")[:60],
            )
            return []

        claims = _parse_extracted_claims(
            extraction_result.content,
            max_claims=max_claims_per_chunk,
        )
        # Tag each claim with the chunk's section heading.
        section_heading = chunk.get("section", "")
        for claim in claims:
            if not claim.report_section and section_heading:
                claim.report_section = section_heading
        return claims

    except Exception as exc:
        logger.warning(
            "Chunk extraction failed for section %r: %s",
            chunk.get("section", "")[:60],
            exc,
        )
        return []


def _filter_uncited_claims(claims: list[ClaimVerdict]) -> list[ClaimVerdict]:
    """Drop claims that have no explicit citation references.

    The citation-anchored extraction prompt should only produce claims with
    cited_sources, but this filter acts as a safety net — if the LLM
    hallucinated a claim with no [N] reference, it gets dropped here.

    Args:
        claims: Raw extracted claims.

    Returns:
        Claims with at least one entry in cited_sources.
    """
    filtered = [c for c in claims if c.cited_sources]
    dropped = len(claims) - len(filtered)
    if dropped:
        logger.info("Dropped %d claims with no citation references", dropped)
    return filtered


# Regex for removing citation brackets during deduplication.
_CITATION_BRACKET_RE = re.compile(r"\[\d+\]")


async def _extract_claims_chunked(
    report: str,
    workflow: Any,
    state: DeepResearchState,
    provider_id: str,
    timeout: float,
    max_claims: int,
    max_concurrent: int,
    metadata: Optional[dict[str, Any]] = None,
) -> list[ClaimVerdict]:
    """Extract claims from report using parallel section-level chunking.

    Splits the report into section chunks, extracts claims from each in
    parallel with bounded concurrency, then merges and deduplicates results.

    Args:
        report: The (possibly truncated) report text.
        workflow: DeepResearchWorkflow instance (for lifecycle instrumentation).
        state: Current research state.
        provider_id: LLM provider to use.
        timeout: Per-call timeout in seconds.
        max_claims: Max total claims to return.
        max_concurrent: Max parallel extraction calls.
        metadata: Optional state metadata dict to populate with extraction stats.

    Returns:
        Merged, deduplicated list of ClaimVerdict objects.
    """
    chunks = _split_report_into_sections(report)
    max_claims_per_chunk = max(10, max_claims // len(chunks)) if chunks else max_claims

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_chunk(chunk: dict[str, str]) -> list[ClaimVerdict]:
        async with semaphore:
            return await _extract_claims_from_chunk(
                chunk=chunk,
                workflow=workflow,
                state=state,
                system_prompt=_EXTRACTION_SYSTEM_PROMPT,
                provider_id=provider_id,
                timeout=timeout,
                max_claims_per_chunk=max_claims_per_chunk,
            )

    tasks = [_run_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    check_gather_cancellation(results)

    # Merge claims and log per-chunk results.
    all_claims: list[ClaimVerdict] = []
    claims_per_chunk: list[int] = []
    chunks_succeeded = 0
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            logger.warning(
                "Chunk %d/%d extraction raised: %s", idx + 1, len(chunks), res
            )
            claims_per_chunk.append(0)
        elif isinstance(res, list):
            count = len(res)
            all_claims.extend(res)
            claims_per_chunk.append(count)
            chunks_succeeded += 1
            logger.info(
                "Chunk %d/%d extracted %d claims (section: %r)",
                idx + 1,
                len(chunks),
                count,
                chunks[idx].get("section", "")[:60],
            )
        else:
            claims_per_chunk.append(0)

    # Deduplicate by normalized claim text.
    seen: set[str] = set()
    deduped: list[ClaimVerdict] = []
    for claim in all_claims:
        normalized = _CITATION_BRACKET_RE.sub("", claim.claim.lower()).strip()
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(claim)

    # Filter uncited claims.
    deduped = _filter_uncited_claims(deduped)

    # Cap at max_claims.
    if max_claims and len(deduped) > max_claims:
        deduped = deduped[:max_claims]

    # Populate metadata.
    if metadata is not None:
        metadata["extraction_strategy"] = "chunked"
        metadata["extraction_chunks_attempted"] = len(chunks)
        metadata["extraction_chunks_succeeded"] = chunks_succeeded
        metadata["extraction_claims_per_chunk"] = claims_per_chunk

    logger.info(
        "Chunked extraction complete: %d chunks (%d succeeded), %d claims after dedup+filter",
        len(chunks),
        chunks_succeeded,
        len(deduped),
    )

    return deduped


def _parse_extracted_claims(response: str, max_claims: int = 0) -> list[ClaimVerdict]:
    """Parse LLM response into ClaimVerdict objects.

    Handles JSON wrapped in markdown code fences, truncated output,
    and malformed entries gracefully.

    Args:
        response: Raw LLM response text.
        max_claims: When > 0, stop parsing after ``max_claims * 2`` entries
            to bound memory usage from a hallucinating model.
    """
    text = response.strip()

    # Strip markdown code fences if present.
    if text.startswith("```"):
        # Remove opening fence (possibly with language tag).
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        # Remove closing fence.
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to recover: find the outermost array.
        start = text.find("[")
        if start == -1:
            logger.warning("Claim extraction: no JSON array found in response")
            return []
        # Find the matching close bracket, or truncate.
        end = text.rfind("]")
        if end == -1 or end <= start:
            # Truncated output — try adding a closing bracket.
            # This may produce semantically incorrect JSON if the truncation
            # split a nested object, but it recovers the parseable prefix.
            logger.warning(
                "Claim extraction: JSON array appears truncated; "
                "appending ']' to attempt recovery — parsed claims may be incomplete"
            )
            text = text[start:] + "]"
        else:
            text = text[start : end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Claim extraction: failed to parse JSON after recovery attempt")
            return []

    if not isinstance(data, list):
        logger.warning("Claim extraction: expected JSON array, got %s", type(data).__name__)
        return []

    # Early cap to prevent memory pressure from hallucinating models.
    parse_cap = max_claims * 2 if max_claims > 0 else 0

    claims: list[ClaimVerdict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        claim_text = item.get("claim", "")
        if not claim_text:
            continue
        claim_type = item.get("claim_type", "positive")
        if claim_type not in _VALID_CLAIM_TYPES:
            claim_type = "positive"
        cited = item.get("cited_sources", [])
        if not isinstance(cited, list):
            cited = []
        cited = [int(c) for c in cited if isinstance(c, (int, float))]
        claims.append(
            ClaimVerdict(
                claim=claim_text,
                claim_type=claim_type,
                cited_sources=cited,
                report_section=item.get("report_section"),
                quote_context=item.get("quote_context"),
            )
        )
        if parse_cap and len(claims) >= parse_cap:
            logger.warning(
                "Claim extraction: hit parse cap (%d); truncating to prevent memory pressure",
                parse_cap,
            )
            break
    return claims


def _filter_claims_for_verification(
    claims: list[ClaimVerdict],
    sample_rate: float,
    max_claims: int,
) -> list[ClaimVerdict]:
    """Filter and prioritize claims for verification.

    - All negative and quantitative claims: always verify.
    - Comparative claims: always verify.
    - Positive claims: deterministic sampling via SHA-256 hash.
    - When filtered claims exceed max_claims, truncate by priority.
    """
    to_verify: list[ClaimVerdict] = []
    for c in claims:
        if c.claim_type in ("negative", "quantitative", "comparative"):
            to_verify.append(c)
        elif c.claim_type == "positive":
            # Deterministic sampling.
            digest = hashlib.sha256(c.claim.encode()).hexdigest()
            hash_val = int(digest, 16) % 100
            if hash_val < int(sample_rate * 100):
                to_verify.append(c)

    # Sort by priority and truncate.
    to_verify = _sort_claims_by_priority(to_verify)
    if len(to_verify) > max_claims:
        logger.info(
            "Claim verification: %d claims exceed max_claims=%d, truncating",
            len(to_verify),
            max_claims,
        )
        to_verify = to_verify[:max_claims]
    return to_verify


def _apply_token_budget(
    claims: list[ClaimVerdict],
    citation_map: dict[int, "ResearchSource"],
    max_input_tokens: int,
) -> list[ClaimVerdict]:
    """Drop claims from the tail of the priority list when estimated tokens exceed budget.

    Estimation uses CHARS_PER_TOKEN from _token_budget.py (canonical constant).
    Includes a fixed overhead estimate for system prompt and JSON structure.
    """
    # Fixed overhead for system prompt + JSON structure per verification call.
    _SYSTEM_PROMPT_OVERHEAD_TOKENS = 500

    estimated_total = 0.0
    kept: list[ClaimVerdict] = []
    for claim in claims:
        claim_tokens = float(_SYSTEM_PROMPT_OVERHEAD_TOKENS)
        for src_num in claim.cited_sources:
            source = citation_map.get(src_num)
            if source is None:
                continue
            text = _resolve_source_text(source)
            if text is None:
                continue
            # Truncated text length for estimation.
            text_len = min(len(text), VERIFICATION_SOURCE_MAX_CHARS)
            claim_tokens += text_len / CHARS_PER_TOKEN
        if estimated_total + claim_tokens > max_input_tokens and kept:
            logger.info(
                "Token budget: dropping %d claims (estimated %.0f tokens exceeds %d limit)",
                len(claims) - len(kept),
                estimated_total + claim_tokens,
                max_input_tokens,
            )
            break
        estimated_total += claim_tokens
        kept.append(claim)
    return kept


# ---------------------------------------------------------------------------
# Pass 2: Claim-Source Alignment
# ---------------------------------------------------------------------------

_VERIFICATION_SYSTEM_PROMPT = """\
You are a claim verification assistant. You will be given a claim from a research \
report and excerpts from the source material it cites. Your task is to determine \
whether the source excerpts SUPPORT, CONTRADICT, or provide NO EVIDENCE for the claim.

Verdict definitions:
- SUPPORTED: The source excerpts explicitly confirm the claim or contain information \
fully consistent with it.
- CONTRADICTED: The source excerpts explicitly state something that DIRECTLY CONFLICTS \
with the claim. The source must contain a clear counter-statement — not merely the \
absence of confirming information.
- PARTIALLY_SUPPORTED: The source excerpts confirm part of the claim but not all of it, \
or confirm it with different specifics (e.g., different numbers, dates, or scope).
- UNSUPPORTED: The source excerpts do not contain enough information to confirm or deny \
the claim. This includes cases where the topic is not mentioned at all. When in doubt \
between CONTRADICTED and UNSUPPORTED, choose UNSUPPORTED.

IMPORTANT: You are seeing excerpts, not the full source. Absence of information in these \
excerpts does NOT mean the source contradicts the claim.

Return a JSON object with exactly these fields:
- "verdict": one of "SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED"
- "evidence_quote": the exact quote from the source that supports your verdict (REQUIRED \
for CONTRADICTED — if you cannot quote a directly conflicting statement, use UNSUPPORTED)
- "explanation": a brief explanation of your verdict

Return ONLY the JSON object, no other text.
"""


def _build_verification_user_prompt(
    claim: ClaimVerdict,
    citation_map: dict[int, "ResearchSource"],
) -> tuple[Optional[str], str]:
    """Build the verification prompt for a single claim.

    Returns a tuple of (prompt, source_resolution) where source_resolution
    indicates the best content tier used across all cited sources:
    - ``"full_content"``: raw_content was available for at least one source
    - ``"compressed_only"``: only compressed content was available
    - ``"snippet_only"``: only snippet text was available
    - ``"no_content"``: no source text could be resolved
    - ``"citation_not_found"``: all citation numbers were missing from the map

    The prompt is None when no usable source text was found.
    """
    source_sections: list[str] = []
    # Track best resolution tier across all cited sources.
    # Priority: full_content > compressed_only > snippet_only
    _TIER_RANK = {"full_content": 3, "compressed_only": 2, "snippet_only": 1}
    best_tier: Optional[str] = None
    all_citations_missing = True

    for src_num in claim.cited_sources:
        source = citation_map.get(src_num)
        if source is None:
            logger.warning("Claim verification: citation [%d] not in citation map, skipping", src_num)
            continue
        all_citations_missing = False

        # Determine which content tier this source resolved to.
        if source.raw_content:
            tier = "full_content"
        elif source.content:
            tier = "compressed_only"
        elif source.snippet:
            tier = "snippet_only"
        else:
            tier = None

        text = _resolve_source_text(source)
        if text is None:
            logger.warning(
                "Claim verification: source [%d] (%s) has no verifiable content, skipping",
                src_num,
                source.title,
            )
            continue

        # Update best tier.
        if tier and (best_tier is None or _TIER_RANK.get(tier, 0) > _TIER_RANK.get(best_tier, 0)):
            best_tier = tier

        truncated = _multi_window_truncate(text, claim.claim, VERIFICATION_SOURCE_MAX_CHARS)
        header = f"### Source [{src_num}]: {source.title}"
        if source.url:
            header += f"\nURL: {source.url}"
        source_sections.append(f"{header}\n\n{truncated}")

    if not source_sections:
        if all_citations_missing and claim.cited_sources:
            return None, "citation_not_found"
        return None, "no_content"

    # Determine final resolution (best_tier should be set if we have sections).
    resolution = best_tier or "no_content"

    sources_text = "\n\n---\n\n".join(source_sections)
    prompt = (
        f"## Source Content\n\n{sources_text}\n\n"
        f"## Claim to Verify\n\n"
        f'"{claim.claim}"\n'
        f"Claim type: {claim.claim_type}\n"
        f"Cited sources: {', '.join(f'[{n}]' for n in claim.cited_sources)}\n\n"
        f"## Task\n\n"
        f"Does the source content SUPPORT, CONTRADICT, or provide NO EVIDENCE for this claim?\n"
        f"Return a JSON object with: verdict, evidence_quote, explanation"
    )
    return prompt, resolution


def _parse_verification_response(response: str) -> dict[str, Any]:
    """Parse a verification LLM response into verdict fields."""
    text = response.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {"verdict": "UNSUPPORTED", "evidence_quote": None, "explanation": "Failed to parse verification response"}
        else:
            return {"verdict": "UNSUPPORTED", "evidence_quote": None, "explanation": "Failed to parse verification response"}

    if not isinstance(data, dict):
        return {"verdict": "UNSUPPORTED", "evidence_quote": None, "explanation": "Invalid response format"}

    verdict = data.get("verdict", "UNSUPPORTED")
    if verdict not in _VALID_VERDICTS:
        verdict = "UNSUPPORTED"
    return {
        "verdict": verdict,
        "evidence_quote": data.get("evidence_quote"),
        "explanation": data.get("explanation"),
    }


async def _verify_single_claim(
    claim: ClaimVerdict,
    citation_map: dict[int, "ResearchSource"],
    workflow: Any,
    state: DeepResearchState,
    provider_id: str,
    timeout: float,
    semaphore: asyncio.Semaphore,
) -> ClaimVerdict:
    """Verify a single claim against its cited sources."""
    user_prompt, source_resolution = _build_verification_user_prompt(claim, citation_map)
    claim.source_resolution = source_resolution
    if user_prompt is None:
        claim.verdict = "UNSUPPORTED"
        claim.explanation = "No source content available for verification"
        return claim

    async with semaphore:
        try:
            ret = await execute_llm_call(
                workflow=workflow,
                state=state,
                phase_name="claim_verification",
                system_prompt=_VERIFICATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                provider_id=provider_id,
                model=None,
                temperature=0.0,
                timeout=timeout,
                role="claim_verification",
            )
            if isinstance(ret, LLMCallResult):
                result = ret.result
            else:
                result = ret
            if result.success and result.content:
                parsed = _parse_verification_response(result.content)
                claim.verdict = parsed["verdict"]
                claim.evidence_quote = parsed.get("evidence_quote")
                claim.explanation = parsed.get("explanation")
                # Structural gate: CONTRADICTED without evidence quote
                # is downgraded to UNSUPPORTED (defense in depth).
                if claim.verdict == "CONTRADICTED" and not claim.evidence_quote:
                    logger.info(
                        "Downgrading CONTRADICTED to UNSUPPORTED for claim %r: "
                        "no evidence quote provided",
                        claim.claim[:80],
                    )
                    claim.verdict = "UNSUPPORTED"
                    claim.explanation = (
                        f"Originally CONTRADICTED but no contradicting quote provided. "
                        f"Original explanation: {claim.explanation}"
                    )
            else:
                claim.verdict = "UNSUPPORTED"
                claim.explanation = "Verification LLM call failed"
        except Exception as exc:
            logger.warning("Verification failed for claim %r: %s", claim.claim[:80], exc)
            claim.verdict = "UNSUPPORTED"
            claim.explanation = f"Verification error: {exc}"
    return claim


async def _verify_claims_batch(
    claims: list[ClaimVerdict],
    citation_map: dict[int, "ResearchSource"],
    workflow: Any,
    state: DeepResearchState,
    provider_id: str,
    timeout: float,
    max_concurrent: int,
) -> list[ClaimVerdict]:
    """Verify a batch of claims in parallel (bounded concurrency)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _verify_single_claim(claim, citation_map, workflow, state, provider_id, timeout, semaphore)
        for claim in claims
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    check_gather_cancellation(results)
    # Filter out any unexpected exceptions that bypassed _verify_single_claim's
    # internal try/except (e.g. BaseException subclasses).
    return [r for r in results if isinstance(r, ClaimVerdict)]


# ---------------------------------------------------------------------------
# Correction Application
# ---------------------------------------------------------------------------

_CORRECTION_SYSTEM_PROMPT = """\
You are a factual correction assistant. You will be given a section of a research \
report that contains a factually incorrect claim, along with the source evidence \
that contradicts it.

Your task is to rewrite ONLY the portion containing the false claim to align with \
the source evidence. Preserve the surrounding text, formatting, and tone exactly. \
Do not add hedging language, meta-commentary, or annotations — just fix the factual error.

Return ONLY the corrected text passage (the replacement for the provided context window). \
No explanations, no markdown code fences.
"""


def _extract_context_window(report: str, quote_context: str) -> Optional[tuple[str, int, int]]:
    """Find the context window around a claim's quote_context in the report.

    Returns (window_text, start_idx, end_idx) or None if not found.
    The window is ~500 chars before and after, expanded outward to paragraph
    boundaries (\\n\\n).
    """
    idx = report.find(quote_context)
    if idx == -1:
        return None

    match_start = idx
    match_end = idx + len(quote_context)

    # Expand ~500 chars in each direction, then outward to paragraph boundaries.
    raw_start = max(0, match_start - 500)
    raw_end = min(len(report), match_end + 500)

    # Expand backward to nearest \n\n (or start of string).
    para_start = report.rfind("\n\n", 0, raw_start)
    window_start = para_start if para_start != -1 else 0

    # Expand forward to nearest \n\n (or end of string).
    para_end = report.find("\n\n", raw_end)
    window_end = para_end + 2 if para_end != -1 else len(report)

    # Clamp at heading boundaries so corrections never span sections.
    # Backward: keep claim's own section heading, drop earlier sections.
    backward_region = report[window_start:match_start]
    heading_hits = list(_HEADING_BOUNDARY_RE.finditer(backward_region))
    if heading_hits:
        last_hit = heading_hits[-1]
        clamped_start = window_start + last_hit.start()
        if clamped_start < match_start:
            window_start = clamped_start

    # Forward: stop at next section heading.
    forward_region = report[match_end:window_end]
    fwd_hit = _HEADING_BOUNDARY_RE.search(forward_region)
    if fwd_hit:
        clamped_end = match_end + fwd_hit.start()
        if clamped_end > match_end:
            window_end = clamped_end

    return report[window_start:window_end], window_start, window_end


# ---------------------------------------------------------------------------
# Heading-boundary repair
# ---------------------------------------------------------------------------

_HEADING_TRUNCATED_RE = re.compile(r"^(#{1,6}\s+\S.*\w)-\s*$", re.MULTILINE)

_HEADING_RE = re.compile(
    r"^(#{1,6}\s+[^\n]*?[a-z0-9)\]\"'\u2019\u201d!?.;:*\u2014\u2013\-])([A-Z][a-z])",
    re.MULTILINE,
)
_SAMELINE_FUSION_RE = re.compile(
    r"^(#{1,6}\s+.+?[.!?)\]\"'\u2019\u201d*\u2014\u2013:;\-])([A-Z][a-z])",
    re.MULTILINE,
)
_HEADING_TABLE_FUSION_RE = re.compile(
    r"^(#{1,6}\s+[^|\n]+?)\s*(\|(?:[^|\n]*\|){2,}.*)$", re.MULTILINE
)
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_HEADING_BOUNDARY_RE = re.compile(r"\n(?=#{1,6}\s)")


def _repair_truncated_headings(text: str) -> str:
    """Rejoin headings that were split mid-word across lines.

    Detects patterns like ``## Sign-\\n\\nUp Bonuses`` where the synthesis LLM
    broke a heading at a hyphenation point, and merges the continuation back
    onto the heading line.
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        m = _HEADING_TRUNCATED_RE.match(lines[i])
        if not m:
            result.append(lines[i])
            i += 1
            continue

        # Found a heading ending with `word-` — scan forward for continuation.
        heading_line = lines[i]
        j = i + 1
        # Skip blank lines.
        while j < len(lines) and lines[j].strip() == "":
            j += 1

        if j >= len(lines):
            # Truncated heading at end of document — leave as-is.
            result.append(heading_line)
            i += 1
            continue

        continuation = lines[j]
        if re.match(r"^#{1,6}\s+", continuation):
            # Continuation is another heading — don't merge.
            result.append(heading_line)
            i += 1
            continue

        # Merge: heading keeps the hyphen, continuation is appended.
        merged = heading_line.rstrip() + continuation
        result.append(merged)
        # Skip the blank lines and the continuation line we consumed.
        i = j + 1
        continue

    return "\n".join(result)


def _repair_heading_boundaries(original_window: str, corrected_text: str) -> str:
    """Ensure markdown headings are followed by a blank line in corrected text.

    When the correction LLM rewrites a context window that contains markdown
    headings, it sometimes concatenates the heading line with the following
    body paragraph (e.g. ``### TitleBody text...``).  This helper detects
    such fusions and inserts the missing ``\\n\\n`` separator.

    Only structural whitespace after headings is enforced — heading *content*
    may be legitimately modified by the correction.
    """
    if not _HEADING_LINE_RE.search(original_window):
        # Original had no headings — nothing to repair.
        return corrected_text

    # Pattern: heading text immediately followed by an uppercase letter on the
    # same line (no newline between heading and body).
    repaired = _HEADING_RE.sub(r"\1\n\n\2", corrected_text)
    repaired = _SAMELINE_FUSION_RE.sub(r"\1\n\n\2", repaired)  # fallback pass
    repaired = _HEADING_TABLE_FUSION_RE.sub(r"\1\n\n\2", repaired)  # table-on-heading

    # Also handle a heading line followed by a single \n (not \n\n) then body.
    # Split into lines and check each heading.
    lines = repaired.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        result.append(lines[i])
        if re.match(r"^#{1,6}\s+", lines[i]):
            # This is a heading line.  If the next line is non-empty and not
            # another heading, and there's no blank line between, insert one.
            if (
                i + 1 < len(lines)
                and lines[i + 1].strip() != ""
                and not re.match(r"^#{1,6}\s+", lines[i + 1])
            ):
                # Check if the next line is already blank (empty string).
                result.append("")
        i += 1
    repaired = "\n".join(result)

    # Collapse triple+ blank lines back to double.
    repaired = re.sub(r"\n{3,}", "\n\n", repaired)

    return repaired


def repair_heading_boundaries_global(report: str) -> str:
    """Run heading-boundary repair on the entire report."""
    if not report:
        return report
    # Pass a dummy original with a heading to force repair logic to activate.
    report = _repair_heading_boundaries("# dummy heading\n\ntext", report)
    # Final pass: rejoin headings truncated mid-word across lines.
    # Runs after boundary repair because _HEADING_RE can false-positive on
    # hyphenated words (e.g. "Sign-Up") — truncation repair cleans those up.
    return _repair_truncated_headings(report)


async def _correct_single_claim(
    claim: ClaimVerdict,
    state: DeepResearchState,
    workflow: Any,
    provider_id: str,
    timeout: float,
) -> bool:
    """Apply a single correction for a CONTRADICTED claim.

    Returns True if the correction was successfully applied.
    Mutates state.report in place.
    """
    report = state.report
    if not report or not claim.quote_context:
        return False

    window_result = _extract_context_window(report, claim.quote_context)

    if window_result is not None:
        original_window, _, _ = window_result
        user_prompt = (
            f"## Original Report Section\n\n{original_window}\n\n"
            f"## Incorrect Claim\n\n\"{claim.claim}\"\n\n"
            f"## Source Evidence Contradicting This Claim\n\n"
            f"Evidence: {claim.evidence_quote or 'N/A'}\n"
            f"Explanation: {claim.explanation or 'N/A'}\n\n"
            f"## Task\n\n"
            f"Rewrite the section above, correcting ONLY the incorrect claim. "
            f"Preserve all other content, formatting, and tone exactly."
        )
    else:
        # Fallback: full-report correction with section hint.
        logger.info(
            "Claim verification: quote_context not found for claim %r, "
            "using full-report correction with section hint %r",
            claim.claim[:60],
            claim.report_section,
        )
        section_hint = f" in the section '{claim.report_section}'" if claim.report_section else ""
        user_prompt = (
            f"## Full Research Report\n\n{report}\n\n"
            f"## Incorrect Claim{section_hint}\n\n\"{claim.claim}\"\n\n"
            f"## Source Evidence Contradicting This Claim\n\n"
            f"Evidence: {claim.evidence_quote or 'N/A'}\n"
            f"Explanation: {claim.explanation or 'N/A'}\n\n"
            f"## Task\n\n"
            f"Rewrite the report, correcting ONLY this specific incorrect claim. "
            f"Change as little text as possible — fix only the factual error."
        )

    try:
        ret = await execute_llm_call(
            workflow=workflow,
            state=state,
            phase_name="claim_verification_correction",
            system_prompt=_CORRECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            provider_id=provider_id,
            model=None,
            temperature=0.0,
            timeout=timeout,
            role="claim_verification",
        )
        if isinstance(ret, LLMCallResult):
            result = ret.result
        else:
            result = ret
        if not result.success or not result.content:
            logger.warning("Correction LLM call failed for claim %r", claim.claim[:60])
            return False

        corrected_text = result.content.strip()

        # Repair heading/body concatenation caused by the LLM.
        if window_result is not None:
            original_window_for_repair, _, _ = window_result
            corrected_text = _repair_heading_boundaries(
                original_window_for_repair, corrected_text
            )

        if window_result is not None:
            original_window, _, _ = window_result

            # Replace the first occurrence of the original window.
            # state.report is guaranteed non-None here (checked at function entry).
            assert state.report is not None  # for type checker
            new_report = state.report.replace(original_window, corrected_text, 1)

            # Sanity check: paragraph boundaries should survive.
            # Verify that the \n\n delimiters bracketing the original window are still present.
            if new_report == state.report:
                # No replacement happened (window already changed by prior correction).
                logger.warning(
                    "Correction: original window no longer found in report for claim %r",
                    claim.claim[:60],
                )
                return False

            state.report = new_report
            claim.correction_applied = True
            claim.corrected_text = corrected_text
            return True
        else:
            # Full-report correction: replace entire report.
            if len(corrected_text) < len(report) * 0.5:
                # Sanity: correction is suspiciously short — reject it.
                logger.warning(
                    "Correction: full-report replacement is < 50%% of original length, rejecting"
                )
                return False
            state.report = corrected_text
            claim.correction_applied = True
            claim.corrected_text = corrected_text
            return True

    except Exception as exc:
        logger.warning("Correction failed for claim %r: %s", claim.claim[:60], exc)
        return False


async def apply_corrections(
    state: DeepResearchState,
    config: "ResearchConfig",
    verification_result: ClaimVerificationResult,
    workflow: Any,
    provider_id: Optional[str] = None,
) -> None:
    """Apply corrections for CONTRADICTED claims.

    Corrections are applied sequentially (not in parallel) to avoid racing
    on state.report when context windows overlap.

    Also handles UNSUPPORTED annotations when configured.
    """
    from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
        resolve_phase_provider,
    )

    resolved_provider = provider_id or resolve_phase_provider(config, "claim_verification", "synthesis")
    max_corrections = config.deep_research_claim_verification_max_corrections
    timeout = config.deep_research_claim_verification_timeout

    # Collect CONTRADICTED claims, sorted by priority.
    contradicted = _sort_claims_by_priority(
        [d for d in verification_result.details if d.verdict == "CONTRADICTED"]
    )

    corrections_applied = 0
    for claim in contradicted:
        if corrections_applied >= max_corrections:
            logger.info(
                "Correction budget exhausted (%d/%d), skipping remaining contradicted claims",
                corrections_applied,
                max_corrections,
            )
            break
        success = await _correct_single_claim(
            claim, state, workflow, resolved_provider, timeout
        )
        if success:
            corrections_applied += 1

    verification_result.corrections_applied = corrections_applied

    # Handle UNSUPPORTED annotations (opt-in).
    if config.deep_research_claim_verification_annotate_unsupported and state.report:
        for claim in verification_result.details:
            if claim.verdict != "UNSUPPORTED" or not claim.quote_context:
                continue
            if claim.quote_context in state.report:
                # Insert annotation after the sentence containing the claim.
                state.report = state.report.replace(
                    claim.quote_context,
                    claim.quote_context + " (unverified)",
                    1,
                )
            else:
                logger.info(
                    "Annotation skip: quote_context not found for UNSUPPORTED claim %r",
                    claim.claim[:60],
                )

    # Global heading-boundary repair after all corrections.
    if corrections_applied > 0 and state.report:
        state.report = repair_heading_boundaries_global(state.report)


# ---------------------------------------------------------------------------
# Citation remapping for UNSUPPORTED claims
# ---------------------------------------------------------------------------

# Max sources to send per LLM remapping call to control token costs.
_REMAP_MAX_CANDIDATE_SOURCES = 10

# Max total remapping LLM calls per report.
_REMAP_MAX_LLM_CALLS = 15

_REMAP_SYSTEM_PROMPT = """\
You are a citation-matching assistant. You will be given a factual claim from a \
research report and excerpts from several candidate sources. Your task is to \
determine which source, if any, contains evidence that supports the claim.

Return a JSON object with exactly these fields:
- "best_source": the citation number (integer) of the source that best supports \
the claim, or null if none of the sources contain supporting evidence
- "confidence": one of "high", "medium", or "low"
- "evidence_quote": a brief quote from the matching source that supports the claim \
(empty string if best_source is null)

Return ONLY the JSON object, no other text.
"""


async def remap_unsupported_citations(
    state: DeepResearchState,
    verification_result: ClaimVerificationResult,
    workflow: Any,
    provider_id: str,
    timeout: float = 60.0,
    max_concurrent: int = 5,
) -> int:
    """Remap citations for UNSUPPORTED claims to better-matching sources.

    For each UNSUPPORTED claim that has ``cited_sources``, searches all available
    sources for one whose content actually supports the claim. Uses LLM-based
    matching for accuracy.

    Mutates ``state.report`` in place (replacing ``[old]`` with ``[new]`` within
    the claim's ``quote_context`` region). Also updates ``claim.cited_sources``
    on the verdict objects.

    Args:
        state: Research state with populated report and sources.
        verification_result: Verification result with per-claim details.
        workflow: DeepResearchWorkflow instance (for lifecycle instrumentation).
        provider_id: LLM provider for remapping calls.
        timeout: Per-call timeout in seconds.
        max_concurrent: Maximum parallel remapping LLM calls.

    Returns:
        Number of citations successfully remapped.
    """
    if not state.report:
        return 0

    citation_map = state.get_citation_map()
    if not citation_map:
        return 0

    # Collect UNSUPPORTED claims that have citations to remap.
    candidates = [
        claim
        for claim in verification_result.details
        if claim.verdict in ("UNSUPPORTED", "PARTIALLY_SUPPORTED")
        and claim.cited_sources
        and claim.quote_context
    ]

    if not candidates:
        return 0

    # Build source content snippets for candidate matching.
    # Exclude sources with no usable content.
    available_sources: dict[int, tuple[str, str]] = {}  # citation_num -> (title, text_snippet)
    for cit_num, source in citation_map.items():
        text = _resolve_source_text(source)
        if not text:
            continue
        title = source.title or source.url or f"Source {cit_num}"
        # Truncate to keep prompt size manageable.
        snippet = text[:VERIFICATION_SOURCE_MAX_CHARS]
        available_sources[cit_num] = (title, snippet)

    if not available_sources:
        return 0

    semaphore = asyncio.Semaphore(max_concurrent)
    remapped_count = 0
    llm_calls_made = 0

    async def _remap_single_claim(claim: ClaimVerdict) -> bool:
        """Attempt to remap a single claim's citation. Returns True if remapped."""
        nonlocal llm_calls_made

        if llm_calls_made >= _REMAP_MAX_LLM_CALLS:
            return False

        # Build candidate source list, excluding the already-cited sources
        # (which were already checked during verification and found lacking).
        already_cited = set(claim.cited_sources)
        candidate_nums = [
            n for n in available_sources if n not in already_cited
        ]

        if not candidate_nums:
            return False

        # Limit candidates to control token costs.
        candidate_nums = candidate_nums[:_REMAP_MAX_CANDIDATE_SOURCES]

        # Build the prompt with source excerpts.
        source_sections = []
        for cit_num in candidate_nums:
            title, snippet = available_sources[cit_num]
            # Use multi-window truncation to focus on claim-relevant content.
            focused = _multi_window_truncate(
                snippet, claim.claim, max_chars=3000, max_windows=2
            )
            source_sections.append(
                f"### Source [{cit_num}]: {title}\n\n{focused}"
            )

        user_prompt = (
            f"## Claim\n\n\"{claim.claim}\"\n\n"
            f"## Candidate Sources\n\n"
            + "\n\n".join(source_sections)
        )

        async with semaphore:
            llm_calls_made += 1
            try:
                ret = await execute_llm_call(
                    workflow=workflow,
                    state=state,
                    phase_name="claim_verification_remap",
                    system_prompt=_REMAP_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    provider_id=provider_id,
                    model=None,
                    temperature=0.0,
                    timeout=timeout,
                    role="claim_verification",
                )
                if isinstance(ret, LLMCallResult):
                    result = ret.result
                else:
                    result = ret
                if not result.success or not result.content:
                    return False

                parsed = _parse_remap_response(result.content)
                if parsed is None:
                    return False

                best_source, confidence, evidence_quote = parsed

                if best_source is None or best_source not in available_sources:
                    # No matching source found — remove the citation from the report.
                    _remove_citations_from_report(state, claim)
                    return False

                if confidence == "low":
                    # Low confidence — don't remap, just remove.
                    _remove_citations_from_report(state, claim)
                    return False

                # Remap: replace old citation numbers with new one in the report.
                success = _apply_citation_remap(
                    state, claim, claim.cited_sources, best_source
                )
                if success:
                    claim.cited_sources = [best_source]
                    claim.evidence_quote = evidence_quote or claim.evidence_quote
                    return True
                return False

            except Exception as exc:
                logger.warning(
                    "Citation remap failed for claim %r: %s",
                    claim.claim[:60],
                    exc,
                )
                return False

    # Run remapping with bounded concurrency.
    tasks = [_remap_single_claim(claim) for claim in candidates]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if r is True:
            remapped_count += 1

    verification_result.citations_remapped = remapped_count

    if remapped_count > 0:
        logger.info(
            "Citation remapping: %d/%d UNSUPPORTED citations remapped",
            remapped_count,
            len(candidates),
        )

    return remapped_count


def _parse_remap_response(
    content: str,
) -> Optional[tuple[Optional[int], str, str]]:
    """Parse the LLM remapping response.

    Returns (best_source, confidence, evidence_quote) or None on parse failure.
    """
    content = content.strip()
    # Strip markdown code fences if present.
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Citation remap: failed to parse JSON response")
        return None

    best_source = data.get("best_source")
    if best_source is not None:
        try:
            best_source = int(best_source)
        except (TypeError, ValueError):
            best_source = None

    confidence = data.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    evidence_quote = data.get("evidence_quote", "")

    return (best_source, confidence, evidence_quote)


def _apply_citation_remap(
    state: DeepResearchState,
    claim: ClaimVerdict,
    old_citations: list[int],
    new_citation: int,
) -> bool:
    """Replace citation numbers in the report within the claim's quote_context region.

    Only modifies the first occurrence of the quote_context to avoid unintended
    changes elsewhere in the report.

    Returns True if the replacement was made.
    """
    if not state.report or not claim.quote_context:
        return False

    if claim.quote_context not in state.report:
        logger.info(
            "Citation remap: quote_context not found in report for claim %r",
            claim.claim[:60],
        )
        return False

    # Build the modified quote_context with remapped citations.
    modified_context = claim.quote_context
    for old_cit in old_citations:
        modified_context = modified_context.replace(
            f"[{old_cit}]", f"[{new_citation}]"
        )

    if modified_context == claim.quote_context:
        # No citation references found in the quote_context text.
        return False

    state.report = state.report.replace(
        claim.quote_context, modified_context, 1
    )
    # Update quote_context to reflect the change (for subsequent operations).
    claim.quote_context = modified_context
    return True


def _remove_citations_from_report(
    state: DeepResearchState,
    claim: ClaimVerdict,
) -> None:
    """Remove citation brackets from the claim's quote_context in the report.

    Leaves the factual text intact but removes the ``[N]`` markers.
    """
    if not state.report or not claim.quote_context:
        return

    if claim.quote_context not in state.report:
        return

    import re as _re

    modified_context = claim.quote_context
    for cit in claim.cited_sources:
        # Remove [N] (and optional trailing space).
        modified_context = modified_context.replace(f"[{cit}]", "")

    # Clean up double spaces left by removed citations.
    modified_context = _re.sub(r"  +", " ", modified_context)

    if modified_context != claim.quote_context:
        state.report = state.report.replace(
            claim.quote_context, modified_context, 1
        )
        claim.quote_context = modified_context
        claim.cited_sources = []


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def extract_and_verify_claims(
    state: DeepResearchState,
    config: "ResearchConfig",
    provider_id: str,
    workflow: Any,
    timeout: float = 120.0,
) -> ClaimVerificationResult:
    """Run the full claim extraction and verification pipeline.

    Args:
        state: Research state with populated report and sources.
        config: Research configuration.
        provider_id: LLM provider for verification calls.
        workflow: DeepResearchWorkflow instance (for lifecycle instrumentation).
        timeout: Per-call timeout in seconds.

    Returns:
        ClaimVerificationResult with extraction and verification details.
    """
    result = ClaimVerificationResult()

    if not state.report:
        logger.warning("Claim verification: no report to verify")
        state.metadata["claim_verification_skipped"] = "no_report"
        return result

    # --- Pass 1: Claim Extraction ---
    # Cap report input to avoid timeout on very large reports.
    _MAX_EXTRACTION_CHARS = 30_000
    if len(state.report) > _MAX_EXTRACTION_CHARS:
        # Prefer the first portion (exec summary + body) over the bibliography.
        extraction_report = state.report[:_MAX_EXTRACTION_CHARS]
        logger.info(
            "Claim extraction: truncated report from %d to %d chars",
            len(state.report),
            _MAX_EXTRACTION_CHARS,
        )
    else:
        extraction_report = state.report
    try:
        all_claims = await _extract_claims_chunked(
            report=extraction_report,
            workflow=workflow,
            state=state,
            provider_id=provider_id,
            timeout=timeout,
            max_claims=config.deep_research_claim_verification_max_claims,
            max_concurrent=config.deep_research_claim_verification_max_concurrent,
            metadata=state.metadata,
        )
    except Exception as exc:
        logger.warning("Claim extraction failed: %s", exc)
        state.metadata["claim_verification_skipped"] = "extraction_failed"
        return result

    if not all_claims:
        logger.info("Claim extraction returned no claims")
        state.metadata["claim_verification_skipped"] = "extraction_failed"
        return result

    result.claims_extracted = len(all_claims)

    # --- Filter and prioritize ---
    to_verify = _filter_claims_for_verification(
        all_claims,
        sample_rate=config.deep_research_claim_verification_sample_rate,
        max_claims=config.deep_research_claim_verification_max_claims,
    )

    if not to_verify:
        logger.info("No claims passed filtering for verification")
        return result

    # --- Token budget check ---
    citation_map = state.get_citation_map()
    to_verify = _apply_token_budget(
        to_verify,
        citation_map,
        max_input_tokens=config.deep_research_claim_verification_max_input_tokens,
    )

    result.claims_filtered = len(to_verify)

    if not to_verify:
        logger.info("All claims dropped by token budget")
        return result

    # --- Pass 2: Claim-Source Alignment ---
    verified_claims = await _verify_claims_batch(
        to_verify,
        citation_map,
        workflow,
        state,
        provider_id,
        timeout=timeout,
        max_concurrent=config.deep_research_claim_verification_max_concurrent,
    )

    # Aggregate results.
    result.claims_verified = len(verified_claims)
    result.details = verified_claims
    for claim in verified_claims:
        if claim.verdict == "SUPPORTED":
            result.claims_supported += 1
        elif claim.verdict == "CONTRADICTED":
            result.claims_contradicted += 1
        elif claim.verdict == "UNSUPPORTED":
            result.claims_unsupported += 1
        elif claim.verdict == "PARTIALLY_SUPPORTED":
            result.claims_partially_supported += 1

    return result
