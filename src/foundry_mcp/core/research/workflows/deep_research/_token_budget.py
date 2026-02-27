"""Token budget, fidelity, and structured truncation helpers."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Approximate characters-per-token heuristic used across budget calculations.
# Defined here as the single source of truth; other modules import this constant.
CHARS_PER_TOKEN: int = 4


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
    if max_tokens <= 0:
        logger.warning(
            "truncate_to_token_estimate called with max_tokens=%d; returning full text (budget exhausted upstream)",
            max_tokens,
        )
        return text
    max_chars = max_tokens * CHARS_PER_TOKEN
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
    max_chars = max_tokens * CHARS_PER_TOKEN
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

    # Multi-pass truncation: each pass cuts up to 50% of each section,
    # so a single pass is insufficient when a section is >2x over budget.
    # Up to 3 passes brings even very large prompts within budget.
    for _pass in range(3):
        total = sum(len(h) + len(c) for h, c in sections)
        excess = total - max_chars
        if excess <= 0:
            break

        # Re-identify truncatable sections with current lengths
        truncatable = []
        for i, (header, content) in enumerate(sections):
            header_lower = header.lower()
            is_protected = any(kw in header_lower for kw in _PROTECTED_SECTION_KEYWORDS)
            if not is_protected and len(content) > 100:
                truncatable.append((i, len(content)))

        if not truncatable:
            break

        # Sort by content length descending — truncate largest first
        truncatable.sort(key=lambda x: x[1], reverse=True)

        for idx, cur_len in truncatable:
            if excess <= 0:
                break
            header, content = sections[idx]
            # Truncate this section's content by up to 50%
            cut = min(excess, cur_len // 2)
            target_len = max(100, cur_len - cut)
            truncated_content = truncate_at_boundary(content, target_len)
            saved = cur_len - len(truncated_content)
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
    max_chars = max_tokens * CHARS_PER_TOKEN
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
