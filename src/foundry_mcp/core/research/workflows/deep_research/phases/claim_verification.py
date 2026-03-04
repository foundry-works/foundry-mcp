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
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig
    from foundry_mcp.core.research.models.sources import ResearchSource
    from foundry_mcp.core.research.workflows.base import WorkflowResult

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

# Type alias for the LLM execution callable passed as a dependency.
# Matches ResearchWorkflowBase._execute_provider_async signature.
ExecuteFn = Callable[..., Any]

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
    execute_fn: ExecuteFn,
    system_prompt: str,
    provider_id: str,
    timeout: float,
    max_claims_per_chunk: int,
) -> list[ClaimVerdict]:
    """Extract claims from a single report section chunk.

    Args:
        chunk: Dict with ``"section"`` and ``"content"`` keys.
        execute_fn: LLM execution callable.
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
        extraction_result: "WorkflowResult" = await execute_fn(
            prompt=chunk_prompt,
            system_prompt=system_prompt,
            provider_id=provider_id,
            timeout=timeout,
            phase="claim_extraction",
            max_tokens=4096,
            max_retries=1,
            retry_delay=2.0,
        )
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
    execute_fn: ExecuteFn,
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
        execute_fn: LLM execution callable.
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
                execute_fn=execute_fn,
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
) -> Optional[str]:
    """Build the verification prompt for a single claim.

    Returns None if no source text could be resolved for any cited source.
    """
    source_sections: list[str] = []
    for src_num in claim.cited_sources:
        source = citation_map.get(src_num)
        if source is None:
            logger.warning("Claim verification: citation [%d] not in citation map, skipping", src_num)
            continue
        text = _resolve_source_text(source)
        if text is None:
            logger.warning(
                "Claim verification: source [%d] (%s) has no verifiable content, skipping",
                src_num,
                source.title,
            )
            continue
        truncated = _multi_window_truncate(text, claim.claim, VERIFICATION_SOURCE_MAX_CHARS)
        header = f"### Source [{src_num}]: {source.title}"
        if source.url:
            header += f"\nURL: {source.url}"
        source_sections.append(f"{header}\n\n{truncated}")

    if not source_sections:
        return None

    sources_text = "\n\n---\n\n".join(source_sections)
    return (
        f"## Source Content\n\n{sources_text}\n\n"
        f"## Claim to Verify\n\n"
        f'"{claim.claim}"\n'
        f"Claim type: {claim.claim_type}\n"
        f"Cited sources: {', '.join(f'[{n}]' for n in claim.cited_sources)}\n\n"
        f"## Task\n\n"
        f"Does the source content SUPPORT, CONTRADICT, or provide NO EVIDENCE for this claim?\n"
        f"Return a JSON object with: verdict, evidence_quote, explanation"
    )


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
    execute_fn: ExecuteFn,
    provider_id: str,
    timeout: float,
    semaphore: asyncio.Semaphore,
) -> ClaimVerdict:
    """Verify a single claim against its cited sources."""
    user_prompt = _build_verification_user_prompt(claim, citation_map)
    if user_prompt is None:
        claim.verdict = "UNSUPPORTED"
        claim.explanation = "No source content available for verification"
        return claim

    async with semaphore:
        try:
            result: "WorkflowResult" = await execute_fn(
                prompt=user_prompt,
                system_prompt=_VERIFICATION_SYSTEM_PROMPT,
                provider_id=provider_id,
                timeout=timeout,
                max_tokens=2000,
                phase="claim_verification",
            )
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
    execute_fn: ExecuteFn,
    provider_id: str,
    timeout: float,
    max_concurrent: int,
) -> list[ClaimVerdict]:
    """Verify a batch of claims in parallel (bounded concurrency)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        _verify_single_claim(claim, citation_map, execute_fn, provider_id, timeout, semaphore)
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

    return report[window_start:window_end], window_start, window_end


async def _correct_single_claim(
    claim: ClaimVerdict,
    state: DeepResearchState,
    execute_fn: ExecuteFn,
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
        result: "WorkflowResult" = await execute_fn(
            prompt=user_prompt,
            system_prompt=_CORRECTION_SYSTEM_PROMPT,
            provider_id=provider_id,
            timeout=timeout,
            phase="claim_verification_correction",
        )
        if not result.success or not result.content:
            logger.warning("Correction LLM call failed for claim %r", claim.claim[:60])
            return False

        corrected_text = result.content.strip()

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
    execute_fn: ExecuteFn,
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
            claim, state, execute_fn, resolved_provider, timeout
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


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def extract_and_verify_claims(
    state: DeepResearchState,
    config: "ResearchConfig",
    provider_id: str,
    execute_fn: ExecuteFn,
    timeout: float = 120.0,
) -> ClaimVerificationResult:
    """Run the full claim extraction and verification pipeline.

    Args:
        state: Research state with populated report and sources.
        config: Research configuration.
        provider_id: LLM provider for verification calls.
        execute_fn: Async callable matching _execute_provider_async signature.
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
            execute_fn=execute_fn,
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
        execute_fn,
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
