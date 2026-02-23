"""Shared pure utility functions used by multiple phases.

These are stateless functions with no instance access. Called as
module-level functions (not via ``self``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

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
    has completed research, or needs to refine its query.
    """

    continue_searching: bool = False
    refined_query: Optional[str] = None
    research_complete: bool = False
    rationale: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict for audit/logging."""
        return {
            "continue_searching": self.continue_searching,
            "refined_query": self.refined_query,
            "research_complete": self.research_complete,
            "rationale": self.rationale,
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
            return TopicReflectionDecision(
                continue_searching=bool(data.get("continue_searching", False)),
                refined_query=data.get("refined_query"),
                research_complete=bool(data.get("research_complete", False)),
                rationale=str(data.get("rationale", "")),
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
