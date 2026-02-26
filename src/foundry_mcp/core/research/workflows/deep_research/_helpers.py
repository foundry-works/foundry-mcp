"""Shared pure utility functions used by multiple phases.

These are stateless functions with no instance access. Called as
module-level functions (not via ``self``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    _extract_domain,
)

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig

logger = logging.getLogger(__name__)


def extract_json(content: str) -> Optional[str]:
    """Extract JSON object from content that may contain other text.

    Handles cases where JSON is wrapped in markdown code blocks
    or mixed with explanatory text.

    Args:
        content: Raw content that may contain JSON

    Returns:
        Extracted JSON string or None if not found
    """
    # First, try to find JSON in code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(code_block_pattern, content)
    for match in matches:
        match = match.strip()
        if match.startswith("{"):
            return match

    # Try to find raw JSON object
    # Look for the outermost { ... } pair
    brace_start = content.find("{")
    if brace_start == -1:
        return None

    # Find matching closing brace, skipping braces inside JSON strings.
    depth = 0
    in_string = False
    escape = False
    for i, char in enumerate(content[brace_start:], brace_start):
        if escape:
            escape = False
            continue
        if char == "\\":
            if in_string:
                escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[brace_start : i + 1]

    return None


def fidelity_level_from_score(fidelity_score: float) -> str:
    """Convert fidelity score (0-1) to fidelity level string.

    Args:
        fidelity_score: Numeric fidelity from 0.0 to 1.0

    Returns:
        Fidelity level: 'full', 'condensed', 'compressed', or 'minimal'
    """
    if fidelity_score >= 0.9:
        return "full"
    elif fidelity_score >= 0.6:
        return "condensed"
    elif fidelity_score >= 0.3:
        return "compressed"
    else:
        return "minimal"


def truncate_at_boundary(content: str, target_length: int) -> str:
    """Truncate content at a natural boundary (paragraph, sentence).

    Args:
        content: Content to truncate
        target_length: Target length in characters

    Returns:
        Truncated content with ellipsis marker
    """
    if len(content) <= target_length:
        return content

    truncated = content[:target_length]

    # Try to find paragraph boundary in last 20%
    search_start = int(target_length * 0.8)
    para_break = truncated.rfind("\n\n", search_start)
    if para_break > search_start // 2:
        truncated = truncated[:para_break]
    else:
        # Try sentence boundary
        sentence_break = truncated.rfind(". ", search_start)
        if sentence_break > search_start // 2:
            truncated = truncated[: sentence_break + 1]

    return truncated.strip() + "\n\n[... content truncated for context limits]"


def truncate_to_token_estimate(text: str, max_tokens: int) -> str:
    """Truncate text to fit within an estimated token budget.

    Uses the 4 chars/token heuristic (same as open_deep_research) to
    estimate the character budget, then truncates at a natural boundary
    via ``truncate_at_boundary()``.

    Args:
        text: Text to truncate
        max_tokens: Maximum token budget for the text

    Returns:
        Truncated text if it exceeds the budget, otherwise the original text
    """
    # 4 chars per token heuristic
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return truncate_at_boundary(text, max_chars)


# ---------------------------------------------------------------------------
# Structured truncation helpers for message-aware token recovery (Phase 4)
# ---------------------------------------------------------------------------

# Section keywords that should NOT be truncated — they contain structural
# information essential for the LLM to produce a good response.
_PROTECTED_SECTION_KEYWORDS: frozenset[str] = frozenset(
    {
        "instructions",
        "research query",
        "research brief",
    }
)


def _split_prompt_sections(prompt: str) -> list[tuple[str, str]]:
    """Split a structured prompt into ``(header, content)`` pairs.

    Splits at markdown header boundaries (lines starting with ``#``,
    ``##``, or ``###``).  The first entry may have an empty header if
    the prompt starts with non-header content.

    Args:
        prompt: The prompt string to split

    Returns:
        List of ``(header_line, content_after_header)`` tuples
    """
    lines = prompt.split("\n")
    sections: list[tuple[str, str]] = []
    current_header = ""
    current_content: list[str] = []

    for line in lines:
        if re.match(r"^#{1,3}\s", line):
            # Save previous section
            if current_header or current_content:
                sections.append((current_header, "\n".join(current_content)))
            current_header = line
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_header or current_content:
        sections.append((current_header, "\n".join(current_content)))

    return sections


def structured_truncate_blocks(prompt: str, max_tokens: int) -> str:
    """Truncate longest content blocks in a structured prompt.

    Splits the prompt into sections at markdown header boundaries,
    identifies truncatable content sections (excluding protected sections
    like Instructions and Research Query), and truncates the longest ones
    first to bring the total size within the token budget.

    Falls back to simple character-based truncation via
    ``truncate_at_boundary()`` if no section structure is found.

    Args:
        prompt: The structured prompt to truncate
        max_tokens: Target token budget

    Returns:
        Truncated prompt string
    """
    max_chars = max_tokens * 4
    if len(prompt) <= max_chars:
        return prompt

    sections = _split_prompt_sections(prompt)
    if len(sections) <= 1:
        return truncate_at_boundary(prompt, max_chars)

    # Identify truncatable sections (by index and original content length)
    truncatable: list[tuple[int, int]] = []
    for i, (header, content) in enumerate(sections):
        header_lower = header.lower()
        is_protected = any(kw in header_lower for kw in _PROTECTED_SECTION_KEYWORDS)
        if not is_protected and len(content) > 100:
            truncatable.append((i, len(content)))

    if not truncatable:
        return truncate_at_boundary(prompt, max_chars)

    # Sort by content length descending — truncate largest sections first
    truncatable.sort(key=lambda x: x[1], reverse=True)

    # Iteratively truncate until within budget
    excess = len(prompt) - max_chars
    for idx, orig_len in truncatable:
        if excess <= 0:
            break
        header, content = sections[idx]
        # Truncate this section's content by up to 50%
        cut = min(excess, orig_len // 2)
        target_len = max(100, orig_len - cut)
        truncated_content = truncate_at_boundary(content, target_len)
        saved = orig_len - len(truncated_content)
        sections[idx] = (header, truncated_content)
        excess -= saved

    # Reassemble
    parts: list[str] = []
    for header, content in sections:
        if header:
            parts.append(header)
        if content:
            parts.append(content)

    return "\n".join(parts)


# Quality tiers for source scoring in quality-aware dropping.
_QUALITY_SCORES: dict[str, int] = {"high": 3, "medium": 2, "low": 1}

# Compiled patterns that mark the start of a source entry in various phases.
_SOURCE_ENTRY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^- \*\*\[\d+\]\*\*:"),  # Synthesis: - **[N]**: title [quality]
    re.compile(r"^- \[\d+\]:"),  # Synthesis fallback: - [N]: title
    re.compile(r"^Source \d+ \(ID:"),  # Analysis: Source N (ID: src-id):
    re.compile(r"^\[\d+\] Title:"),  # Compression: [N] Title: ...
]

# Patterns that end a source entry (start of next entry or new section).
_SOURCE_BOUNDARY_PATTERNS: list[re.Pattern[str]] = [
    *_SOURCE_ENTRY_PATTERNS,
    re.compile(r"^#{1,3}\s"),  # Markdown section header
    re.compile(r"^---\s*Topic\s+\d+"),  # Topic divider in analysis
]


def structured_drop_sources(prompt: str, max_tokens: int) -> str:
    """Drop source entries from a structured prompt to fit within token budget.

    Identifies individual source entries by pattern matching.  Each source
    is scored by quality markers (``[high]``, ``[medium]``, ``[low]``)
    when present, falling back to inverse length as a heuristic (longer
    entries are dropped first within the same quality tier).  Sources are
    dropped in ascending quality order (lowest quality first).

    Falls back to character-based truncation if no source entries are found.

    Args:
        prompt: The structured prompt containing source entries
        max_tokens: Target token budget

    Returns:
        Prompt with lowest-quality sources removed
    """
    max_chars = max_tokens * 4
    if len(prompt) <= max_chars:
        return prompt

    lines = prompt.split("\n")

    # Identify source entries: (start_line, end_line, quality_score, char_length)
    entries: list[dict[str, int]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_source_start = any(p.match(line) for p in _SOURCE_ENTRY_PATTERNS)
        if not is_source_start:
            i += 1
            continue

        start = i
        # Extract quality score from the entry's first line
        quality = 2  # default: medium
        line_lower = line.lower()
        for qname, qscore in _QUALITY_SCORES.items():
            if f"[{qname}]" in line_lower:
                quality = qscore
                break

        # Advance past this entry's content
        i += 1
        while i < len(lines):
            if any(p.match(lines[i]) for p in _SOURCE_BOUNDARY_PATTERNS):
                break
            i += 1

        entry_text = "\n".join(lines[start:i])
        entries.append(
            {
                "start": start,
                "end": i,
                "quality": quality,
                "length": len(entry_text),
            }
        )

    if not entries:
        return truncate_at_boundary(prompt, max_chars)

    # Sort: lowest quality first, then largest first within same quality
    entries.sort(key=lambda e: (e["quality"], -e["length"]))

    # Mark lines to drop
    current_len = len(prompt)
    drop_lines: set[int] = set()
    for entry in entries:
        if current_len <= max_chars:
            break
        for line_idx in range(entry["start"], entry["end"]):
            drop_lines.add(line_idx)
        current_len -= entry["length"]

    if not drop_lines:
        return truncate_at_boundary(prompt, max_chars)

    # Rebuild prompt without dropped lines
    kept = [line for idx, line in enumerate(lines) if idx not in drop_lines]
    result = "\n".join(kept)

    # Final fallback if still over budget
    if len(result) > max_chars:
        result = truncate_at_boundary(result, max_chars)

    return result


def estimate_token_limit_for_model(model: Optional[str], token_limits: dict[str, int]) -> Optional[int]:
    """Look up context window size for a model using substring matching.

    Checks the *token_limits* registry for the first key that appears
    as a substring in *model* (case-insensitive).  Returns ``None`` if
    no match is found.

    Args:
        model: Model identifier string (e.g. ``"claude-3.5-sonnet-20240620"``)
        token_limits: Mapping of model name substrings to context window sizes

    Returns:
        Context window size in tokens, or None if the model is unknown
    """
    if not model:
        return None
    model_lower = model.lower()
    for pattern, limit in token_limits.items():
        if pattern.lower() in model_lower:
            return limit
    return None


@dataclass
class TopicReflectionDecision:
    """Structured decision from a topic research reflection step.

    Captures whether the topic researcher should continue searching,
    has completed research, needs to refine its query, or wants to
    extract full content from promising URLs.
    """

    continue_searching: bool = False
    refined_query: Optional[str] = None
    research_complete: bool = False
    rationale: str = ""
    urls_to_extract: Optional[list[str]] = None

    def to_dict(self) -> dict:
        """Serialize to dict for audit/logging."""
        return {
            "continue_searching": self.continue_searching,
            "refined_query": self.refined_query,
            "research_complete": self.research_complete,
            "rationale": self.rationale,
            "urls_to_extract": self.urls_to_extract,
        }


def parse_reflection_decision(text: str) -> TopicReflectionDecision:
    """Parse a topic reflection LLM response into a structured decision.

    Attempts JSON extraction first, then falls back to regex-based
    parsing for key fields if JSON extraction fails.

    Args:
        text: Raw LLM response text (may contain JSON or prose)

    Returns:
        TopicReflectionDecision with extracted fields. On total
        parse failure, returns a conservative default (stop searching,
        not complete — lets the outer loop decide).
    """
    # Try JSON extraction first
    json_str = extract_json(text)
    if json_str:
        try:
            data = json.loads(json_str)
            # Parse urls_to_extract: accept list of strings, cap at reasonable limit
            raw_urls = data.get("urls_to_extract")
            urls_to_extract: Optional[list[str]] = None
            if isinstance(raw_urls, list):
                urls_to_extract = [
                    str(u).strip() for u in raw_urls
                    if isinstance(u, str) and u.strip().startswith("http")
                ][:5]  # hard cap for safety
                if not urls_to_extract:
                    urls_to_extract = None
            return TopicReflectionDecision(
                continue_searching=bool(data.get("continue_searching", False)),
                refined_query=data.get("refined_query"),
                research_complete=bool(data.get("research_complete", False)),
                rationale=str(data.get("rationale", "")),
                urls_to_extract=urls_to_extract,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.debug("Topic reflection JSON parse failed: %s", exc)

    # Fallback: regex extraction for key fields
    decision = TopicReflectionDecision()

    # Look for research_complete signal
    if re.search(r'"?research_complete"?\s*:\s*true', text, re.IGNORECASE):
        decision.research_complete = True
        decision.rationale = "Extracted research_complete=true via fallback parsing"
        return decision

    # Look for continue_searching signal
    continue_match = re.search(r'"?continue_searching"?\s*:\s*(true|false)', text, re.IGNORECASE)
    if continue_match:
        decision.continue_searching = continue_match.group(1).lower() == "true"

    # Look for refined_query
    query_match = re.search(r'"?refined_query"?\s*:\s*"([^"]+)"', text)
    if query_match:
        decision.refined_query = query_match.group(1)
        if decision.refined_query and not decision.research_complete:
            decision.continue_searching = True

    # Look for rationale
    rationale_match = re.search(r'"?rationale"?\s*:\s*"([^"]*)"', text)
    if rationale_match:
        decision.rationale = rationale_match.group(1)
    elif not decision.rationale:
        decision.rationale = "Parsed via fallback regex extraction"

    return decision


@dataclass
class ClarificationDecision:
    """Structured decision from the clarification phase.

    Captures whether the query needs clarification (with a question)
    or is understood (with a verification statement restating the LLM's
    understanding of the query).
    """

    need_clarification: bool = False
    question: str = ""
    verification: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict for audit/logging."""
        return {
            "need_clarification": self.need_clarification,
            "question": self.question,
            "verification": self.verification,
        }


def parse_clarification_decision(text: str) -> ClarificationDecision:
    """Parse a clarification LLM response into a structured decision.

    Attempts JSON extraction first, then falls back to regex-based
    parsing for key fields if JSON extraction fails.

    Args:
        text: Raw LLM response text (may contain JSON or prose)

    Returns:
        ClarificationDecision with extracted fields.  On total parse
        failure, returns a safe default (no clarification needed,
        empty verification).
    """
    if not text:
        return ClarificationDecision()

    json_str = extract_json(text)
    if json_str:
        try:
            data = json.loads(json_str)
            return ClarificationDecision(
                need_clarification=bool(data.get("need_clarification", False)),
                question=str(data.get("question", "")),
                verification=str(data.get("verification", "")),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.debug("Clarification decision JSON parse failed: %s", exc)

    # Fallback: regex extraction
    decision = ClarificationDecision()

    # Look for need_clarification signal
    nc_match = re.search(r'"?need_clarification"?\s*:\s*(true|false)', text, re.IGNORECASE)
    if nc_match:
        decision.need_clarification = nc_match.group(1).lower() == "true"

    # Look for question
    q_match = re.search(r'"?question"?\s*:\s*"([^"]*)"', text)
    if q_match:
        decision.question = q_match.group(1)

    # Look for verification
    v_match = re.search(r'"?verification"?\s*:\s*"([^"]*)"', text)
    if v_match:
        decision.verification = v_match.group(1)

    if not decision.question and not decision.verification:
        decision.verification = "Parsed via fallback regex extraction"

    return decision


def safe_resolve_model_for_role(
    config: "ResearchConfig",
    role: str,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve ``(provider_id, model)`` for a role, returning ``(None, None)`` on failure.

    Wraps ``config.resolve_model_for_role(role)`` with defensive error
    handling so callers don't need repeated try/except blocks.

    Args:
        config: ResearchConfig instance
        role: Model role (e.g. ``"summarization"``, ``"compression"``)

    Returns:
        ``(provider_id, model)`` on success, ``(None, None)`` if the config
        object doesn't support role resolution or the role is invalid.
    """
    try:
        provider, model = config.resolve_model_for_role(role)
        return provider, model
    except (AttributeError, TypeError, ValueError):
        logger.debug("Role resolution unavailable for %s, using defaults", role)
        return None, None


def resolve_phase_provider(config: "ResearchConfig", *phase_names: str) -> str:
    """Resolve LLM provider ID by trying phase-specific config attrs in order.

    Walks *phase_names* and checks
    ``config.deep_research_{name}_provider`` for each.  Returns the
    first non-None value found, falling back to ``config.default_provider``.

    Args:
        config: ResearchConfig instance
        *phase_names: Config attribute suffixes to check in order
            (e.g. ``"topic_reflection"``, ``"reflection"``).

    Returns:
        Provider ID string (never None).
    """
    for name in phase_names:
        value = getattr(config, f"deep_research_{name}_provider", None)
        if value is not None:
            return value
    return config.default_provider


# ---------------------------------------------------------------------------
# Content similarity deduplication (Phase 5.3)
# ---------------------------------------------------------------------------

# N-gram size for shingling
_SHINGLE_SIZE: int = 5


def _char_ngrams(text: str, n: int = _SHINGLE_SIZE) -> set[str]:
    """Generate character n-grams (shingles) from text.

    Args:
        text: Input text (should be pre-normalized)
        n: N-gram size

    Returns:
        Set of character n-grams
    """
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _normalize_content_for_dedup(text: str) -> str:
    """Normalize text for content similarity comparison.

    Lowercases, removes extra whitespace, strips punctuation-heavy
    boilerplate (e.g. navigation, footer text) by collapsing whitespace.

    Args:
        text: Raw source content

    Returns:
        Normalized text string
    """
    if not text:
        return ""
    normalized = text.lower()
    # Remove common boilerplate markers before whitespace collapse
    normalized = re.sub(r"copyright \d{4}.*?(?:all rights reserved\.?)?", "", normalized)
    # Collapse all whitespace (including newlines) into single spaces
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def content_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts using character n-grams.

    Uses shingling (character n-grams) and Jaccard coefficient to estimate
    content overlap.  This is fast, requires no external dependencies, and
    works well for detecting mirror/syndicated content.

    Args:
        text_a: First text
        text_b: Second text

    Returns:
        Similarity score between 0.0 (no overlap) and 1.0 (identical)
    """
    norm_a = _normalize_content_for_dedup(text_a)
    norm_b = _normalize_content_for_dedup(text_b)

    if not norm_a or not norm_b:
        return 0.0

    # Quick length-ratio check: if lengths differ by more than 3x,
    # similarity will be low regardless — skip expensive shingling.
    len_ratio = min(len(norm_a), len(norm_b)) / max(len(norm_a), len(norm_b))
    if len_ratio < 0.3:
        return 0.0

    shingles_a = _char_ngrams(norm_a)
    shingles_b = _char_ngrams(norm_b)

    if not shingles_a or not shingles_b:
        return 0.0

    intersection = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b)

    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Novelty tagging for researcher stop decisions (Phase 3)
# ---------------------------------------------------------------------------

# Thresholds for novelty classification
_NOVELTY_DUPLICATE_THRESHOLD: float = 0.7
_NOVELTY_RELATED_THRESHOLD: float = 0.3


@dataclass
class NoveltyTag:
    """Novelty classification for a single search result.

    Attributes:
        tag: Display tag — ``"[NEW]"``, ``"[RELATED: <title>]"``, or ``"[DUPLICATE]"``
        category: One of ``"new"``, ``"related"``, ``"duplicate"``
        similarity: Highest similarity score against existing sources
        matched_title: Title of the most similar existing source (if any)
    """

    tag: str
    category: str
    similarity: float
    matched_title: Optional[str] = None


def compute_novelty_tag(
    new_content: str,
    new_url: Optional[str],
    existing_sources: Sequence[tuple[str, str, Optional[str]]],
    *,
    duplicate_threshold: float = _NOVELTY_DUPLICATE_THRESHOLD,
    related_threshold: float = _NOVELTY_RELATED_THRESHOLD,
) -> NoveltyTag:
    """Classify a new search result's novelty against existing sources.

    Uses a two-pass approach: first checks URL domain overlap for a cheap
    signal, then falls back to content similarity via ``content_similarity()``.

    Args:
        new_content: Content (or summary) of the new source.
        new_url: URL of the new source (may be None).
        existing_sources: List of ``(content, title, url)`` tuples for
            sources already found for this sub-query.
        duplicate_threshold: Similarity >= this is ``[DUPLICATE]``.
        related_threshold: Similarity >= this (and < duplicate) is ``[RELATED]``.

    Returns:
        NoveltyTag with classification and metadata.
    """
    if not existing_sources:
        return NoveltyTag(tag="[NEW]", category="new", similarity=0.0)

    best_sim = 0.0
    best_title: Optional[str] = None

    for ex_content, ex_title, ex_url in existing_sources:
        # Quick URL-domain check: same domain boosts similarity estimate
        domain_boost = 0.0
        if new_url and ex_url:
            new_domain = _extract_domain(new_url)
            ex_domain = _extract_domain(ex_url)
            if new_domain and ex_domain and new_domain == ex_domain:
                domain_boost = 0.1

        sim = content_similarity(new_content, ex_content) + domain_boost
        sim = min(sim, 1.0)  # cap at 1.0

        if sim > best_sim:
            best_sim = sim
            best_title = ex_title

    if best_sim >= duplicate_threshold:
        return NoveltyTag(
            tag="[DUPLICATE]",
            category="duplicate",
            similarity=best_sim,
            matched_title=best_title,
        )
    elif best_sim >= related_threshold:
        # Truncate title for readability
        display_title = best_title[:60] + "..." if best_title and len(best_title) > 60 else best_title
        return NoveltyTag(
            tag=f"[RELATED: {display_title}]",
            category="related",
            similarity=best_sim,
            matched_title=best_title,
        )
    else:
        return NoveltyTag(tag="[NEW]", category="new", similarity=best_sim)


def build_novelty_summary(
    tags: list[NoveltyTag],
) -> str:
    """Build a one-line novelty summary for the search results header.

    Args:
        tags: List of NoveltyTag results for all sources in a search batch.

    Returns:
        Summary string like ``"Novelty: 3 new, 1 related, 1 duplicate out of 5 results"``
    """
    new_count = sum(1 for t in tags if t.category == "new")
    related_count = sum(1 for t in tags if t.category == "related")
    dup_count = sum(1 for t in tags if t.category == "duplicate")
    total = len(tags)
    return f"Novelty: {new_count} new, {related_count} related, {dup_count} duplicate out of {total} results"
