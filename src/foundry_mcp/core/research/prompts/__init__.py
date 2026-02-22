"""Prompt templates for research workflows.

Provides versioned, secure prompt templates for LLM-based operations
like summarization. All prompts are designed to handle untrusted content
safely by explicitly ignoring embedded instructions.

Usage:
    from foundry_mcp.core.research.prompts import (
        get_summarization_prompt,
        SummarizationPromptVersion,
    )

    prompt = get_summarization_prompt(
        level="key_points",
        content="Content to summarize...",
        source_id="source-123",
    )
"""

from enum import Enum
from pathlib import Path
from typing import Optional

# Directory containing prompt templates
_PROMPTS_DIR = Path(__file__).parent


class SummarizationPromptVersion(str, Enum):
    """Available versions of the summarization prompt."""

    V1 = "v1"


# Level-specific instructions for summarization
LEVEL_INSTRUCTIONS = {
    "raw": "Return the content unchanged. No summarization needed.",
    "condensed": (
        "Condense the content while preserving key details and nuance.\n"
        "- Retain important context and supporting details\n"
        "- Preserve the original structure where helpful\n"
        "- Target approximately 50-70% of the original length\n"
        "- Use complete sentences"
    ),
    "key_points": (
        "Extract the key points as a concise bullet list.\n"
        "- Focus on main ideas, findings, and conclusions\n"
        "- Omit redundant or tangential information\n"
        "- Target approximately 20-40% of the original length\n"
        "- Use bullet points (- or *) for clarity"
    ),
    "headline": (
        "Summarize in a single sentence or brief headline.\n"
        "- Capture the essential message\n"
        "- Maximum 1-2 lines\n"
        "- Be specific rather than vague\n"
        "- Avoid filler words"
    ),
}


def _load_template(version: SummarizationPromptVersion) -> str:
    """Load a prompt template from disk.

    Args:
        version: Template version to load

    Returns:
        Template string with placeholders

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    template_path = _PROMPTS_DIR / f"summarization_{version.value}.txt"
    return template_path.read_text(encoding="utf-8")


def get_summarization_prompt(
    level: str,
    content: str,
    *,
    source_id: Optional[str] = None,
    max_tokens: int = 500,
    version: SummarizationPromptVersion = SummarizationPromptVersion.V1,
) -> str:
    """Generate a summarization prompt with the given parameters.

    Creates a prompt that treats the content as untrusted, explicitly
    instructs the model to ignore embedded instructions, and preserves
    source provenance when provided.

    Args:
        level: Summarization level (raw, condensed, key_points, headline)
        content: The content to summarize (treated as UNTRUSTED)
        source_id: Optional source identifier for provenance tracking
        max_tokens: Maximum output tokens for the summary
        version: Prompt template version to use

    Returns:
        Rendered prompt string ready for LLM consumption

    Example:
        prompt = get_summarization_prompt(
            level="key_points",
            content="Long article about AI...",
            source_id="article-123",
            max_tokens=500,
        )
    """
    # Load template
    template = _load_template(version)

    # Get level instruction
    level_lower = level.lower()
    level_instruction = LEVEL_INSTRUCTIONS.get(level_lower, LEVEL_INSTRUCTIONS["key_points"])

    # Build source provenance section
    if source_id:
        source_provenance = (
            f"SOURCE PROVENANCE:\n"
            f"The content being summarized is from source: {source_id}\n"
            f"Include this source reference in your summary if relevant."
        )
    else:
        source_provenance = ""

    # Render template
    return template.format(
        level=level_lower,
        level_instruction=level_instruction,
        content=content,
        source_id=source_id or "unknown",
        max_tokens=max_tokens,
        source_provenance=source_provenance,
    )


def get_level_instruction(level: str) -> str:
    """Get the instruction text for a summarization level.

    Args:
        level: Summarization level name

    Returns:
        Level-specific instruction text
    """
    return LEVEL_INSTRUCTIONS.get(level.lower(), LEVEL_INSTRUCTIONS["key_points"])


# Template cache for performance
_TEMPLATE_CACHE: dict[SummarizationPromptVersion, str] = {}


def get_cached_template(version: SummarizationPromptVersion) -> str:
    """Get a cached prompt template.

    Args:
        version: Template version

    Returns:
        Cached template string
    """
    if version not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[version] = _load_template(version)
    return _TEMPLATE_CACHE[version]


def clear_template_cache() -> None:
    """Clear the template cache."""
    _TEMPLATE_CACHE.clear()
