"""Model/provider resolution and LLM response parsing helpers."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from foundry_mcp.core.research.workflows.deep_research._json_parsing import (
    extract_json,
)

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig

logger = logging.getLogger(__name__)


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
