"""Shared pure utility functions used by multiple phases.

These are stateless functions with no instance access. Called as
module-level functions (not via ``self``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig


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

    # Find matching closing brace
    depth = 0
    for i, char in enumerate(content[brace_start:], brace_start):
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
