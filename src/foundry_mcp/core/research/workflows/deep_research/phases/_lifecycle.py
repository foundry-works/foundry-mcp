"""Shared LLM call lifecycle helpers for deep research phase mixins.

Extracts the common boilerplate around LLM provider calls: heartbeat updates,
audit events, ContextWindowError handling, metrics emission, token tracking,
and PhaseMetrics recording. Each phase mixin calls these helpers instead of
duplicating ~88 lines of lifecycle code.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import foundry_mcp.config as _config_pkg
from foundry_mcp.core.errors.provider import ContextWindowError
from foundry_mcp.core.observability import get_metrics
from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.models.fidelity import PhaseMetrics
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
    estimate_token_limit_for_model,
    safe_resolve_model_for_role,
)
from foundry_mcp.core.research.workflows.deep_research._token_budget import (
    CHARS_PER_TOKEN,
    structured_drop_sources,
    structured_truncate_blocks,
    truncate_to_token_estimate,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM model context-window sizes (tokens)
# ---------------------------------------------------------------------------

# Hardcoded fallback used when the JSON config file is missing or unreadable.
_FALLBACK_MODEL_TOKEN_LIMITS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-opus": 200_000,
    "claude-sonnet": 200_000,
    "claude-haiku": 200_000,
    "gpt-5.3-codex-spark": 128_000,
    "gpt-5.3-codex": 400_000,
    "gpt-5.3": 400_000,
    "gpt-5.2-codex": 400_000,
    "gpt-5.2": 400_000,
    "gpt-5.1-codex-mini": 400_000,
    "gpt-5.1-codex-max": 400_000,
    "gpt-5.1-codex": 400_000,
    "gpt-5.1": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5": 400_000,
    "gpt-4.1-mini": 1_048_576,
    "gpt-4.1-nano": 1_048_576,
    "gpt-4.1": 1_048_576,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gemini-3.1-pro": 1_000_000,
    "gemini-3.1-flash": 1_000_000,
    "gemini-3.1": 1_000_000,
    "gemini-3-pro": 1_000_000,
    "gemini-3-flash": 1_000_000,
    "gemini-3": 1_000_000,
    "gemini-2.5-flash-lite": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash-lite": 1_048_576,
    "gemini-2.0-pro": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "glm-5-code": 204_800,
    "glm-5": 204_800,
    "glm-4.7-flash": 202_752,
    "glm-4.7": 202_752,
    "glm-4.6": 202_752,
    "glm-4.5-flash": 131_072,
    "glm-4.5": 131_072,
}


def _sort_limits_longest_first(limits: dict[str, int]) -> dict[str, int]:
    """Sort token limit entries by key length descending.

    Ensures the most specific (longest) substring patterns are matched first
    by ``estimate_token_limit_for_model``, regardless of JSON key ordering.
    """
    return dict(sorted(limits.items(), key=lambda item: len(item[0]), reverse=True))


def _load_model_token_limits() -> dict[str, int]:
    """Load model token limits from the external JSON config file.

    Falls back to the hardcoded ``_FALLBACK_MODEL_TOKEN_LIMITS`` if the
    config file is missing, unreadable, or malformed.

    The config file is located at ``foundry_mcp/config/model_token_limits.json``.
    Entries are sorted by key length descending so that more-specific
    substrings (e.g. ``"gpt-4.1-mini"``) match before less-specific ones
    (e.g. ``"gpt-4.1"``), making match order deterministic regardless of
    JSON key ordering.
    """
    config_path = Path(_config_pkg.__file__).resolve().parent / "model_token_limits.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        limits = data.get("limits", {})
        if not isinstance(limits, dict) or not limits:
            logger.warning("model_token_limits.json has empty/invalid 'limits', using fallback")
            return _sort_limits_longest_first(dict(_FALLBACK_MODEL_TOKEN_LIMITS))
        parsed = {str(k): int(v) for k, v in limits.items()}
        validated: dict[str, int] = {}
        for name, value in parsed.items():
            if value < 1000:
                logger.warning(
                    "model_token_limits.json: skipping %r = %d (below minimum 1000)",
                    name,
                    value,
                )
                continue
            validated[name] = value
        if not validated:
            logger.warning("model_token_limits.json: no valid entries after validation, using fallback")
            return _sort_limits_longest_first(dict(_FALLBACK_MODEL_TOKEN_LIMITS))
        return _sort_limits_longest_first(validated)
    except FileNotFoundError:
        logger.debug("model_token_limits.json not found at %s, using fallback", config_path)
        return _sort_limits_longest_first(dict(_FALLBACK_MODEL_TOKEN_LIMITS))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Failed to load model_token_limits.json: %s, using fallback", exc)
        return _sort_limits_longest_first(dict(_FALLBACK_MODEL_TOKEN_LIMITS))


@functools.lru_cache(maxsize=1)
def get_model_token_limits() -> dict[str, int]:
    """Return the model-name → context-window-size mapping (lazy-loaded).

    Loaded from ``foundry_mcp/config/model_token_limits.json`` on first call,
    with a hardcoded fallback for resilience.  Cached after the first call.

    Used by ``execute_llm_call()`` for progressive token-limit recovery when
    ``ContextWindowError.max_tokens`` is not provided by the provider.
    """
    return _load_model_token_limits()


# ---------------------------------------------------------------------------
# Supervision message history truncation
# ---------------------------------------------------------------------------

# Reserve this fraction of the model's context window for the supervision
# message history.  The rest is needed for the system prompt, coverage data,
# and the LLM's response tokens.
_SUPERVISION_HISTORY_BUDGET_FRACTION: float = 0.4

# Re-export under the old private name for backward compatibility with tests.
_CHARS_PER_TOKEN: int = CHARS_PER_TOKEN

# Type-aware budget allocation: reasoning messages (think + delegation) get
# 60% of the budget; findings messages (tool_result / research_findings)
# get 40%.  This preserves the supervisor's gap analyses and strategic
# reasoning — the highest-value content for subsequent rounds.
_REASONING_BUDGET_FRACTION: float = 0.6
_FINDINGS_BUDGET_FRACTION: float = 0.4

# Default number of most-recent think messages to unconditionally preserve
# regardless of budget constraints.
_DEFAULT_PRESERVE_LAST_N_THINKS: int = 2

# When truncating findings message bodies, keep this many leading characters
# as a header/summary before discarding the rest.  Enough to preserve the
# first few lines of compressed findings.
_FINDINGS_BODY_TRUNCATION_HEADER_CHARS: int = 300


def _msg_content(msg: dict[str, Any]) -> str:
    """Extract message content as a string (safe for ``len()``)."""
    c = msg.get("content")
    return c if isinstance(c, str) else ""


def _is_reasoning_message(msg: dict[str, Any]) -> bool:
    """Return True if the message is a reasoning message (think or delegation)."""
    msg_type = msg.get("type", "")
    return msg_type in ("think", "delegation")


def _is_findings_message(msg: dict[str, Any]) -> bool:
    """Return True if the message is a findings message (tool_result/research_findings)."""
    return msg.get("role") == "tool_result" or msg.get("type") == "research_findings"


def _is_evidence_inventory(msg: dict[str, Any]) -> bool:
    """Return True if the message is an evidence inventory (lower-priority findings).

    Evidence inventories are compact source listings appended alongside
    compressed research findings.  During truncation they are dropped before
    regular research_findings messages from the same round.
    """
    return msg.get("type") == "evidence_inventory"


def _truncate_findings_body(content: str) -> str:
    """Truncate a findings message body, preserving the header/summary portion.

    Keeps the first ``_FINDINGS_BODY_TRUNCATION_HEADER_CHARS`` characters and
    appends an ellipsis marker.
    """
    if len(content) <= _FINDINGS_BODY_TRUNCATION_HEADER_CHARS:
        return content
    # Try to break at a newline boundary within the header region
    cut_point = content.rfind("\n", 0, _FINDINGS_BODY_TRUNCATION_HEADER_CHARS)
    if cut_point < _FINDINGS_BODY_TRUNCATION_HEADER_CHARS // 2:
        cut_point = _FINDINGS_BODY_TRUNCATION_HEADER_CHARS
    return content[:cut_point] + "\n[... truncated for context budget ...]"


def truncate_supervision_messages(
    messages: list[dict[str, Any]],
    model: Optional[str],
    token_limits: Optional[dict[str, int]] = None,
    preserve_last_n_thinks: int = _DEFAULT_PRESERVE_LAST_N_THINKS,
) -> list[dict[str, Any]]:
    """Truncate supervision message history using type-aware budgeting.

    Implements a two-bucket truncation strategy that prioritises reasoning
    context (think + delegation messages) over findings (tool_result messages):

    - **Reasoning bucket (60%):** think and delegation messages containing gap
      analyses and strategic reasoning — the highest-value content for
      multi-round supervision quality.
    - **Findings bucket (40%):** tool_result / research_findings messages
      containing compressed research findings — re-derivable from state.

    Within each bucket, oldest messages are removed first.  For findings
    messages, message bodies are truncated (keeping headers/summaries) before
    dropping entire messages.

    The ``preserve_last_n_thinks`` parameter unconditionally preserves the N
    most recent think messages regardless of budget, since the supervisor's
    most recent gap analyses are critical for reasoning continuity.

    Args:
        messages: The supervision message history to potentially truncate.
        model: Model identifier for context-window lookup.
        token_limits: Token limit registry (defaults to ``get_model_token_limits()``).
        preserve_last_n_thinks: Number of most recent think messages to
            unconditionally preserve (default: 2).

    Returns:
        The (possibly truncated) message list.  Returns the original list
        unchanged if it fits within budget.
    """
    if not messages:
        return messages

    limits = token_limits if token_limits is not None else get_model_token_limits()
    max_tokens = estimate_token_limit_for_model(model, limits)
    if max_tokens is None:
        max_tokens = FALLBACK_CONTEXT_WINDOW

    budget_chars = int(max_tokens * _SUPERVISION_HISTORY_BUDGET_FRACTION * _CHARS_PER_TOKEN)

    # Compute total character count of all message contents
    total_chars = sum(len(_msg_content(msg)) for msg in messages)
    if total_chars <= budget_chars:
        return messages

    # --- Identify protected think messages (last N thinks) ---
    think_indices: list[int] = [
        idx for idx, msg in enumerate(messages)
        if msg.get("type") == "think"
    ]
    protected_think_indices: set[int] = set(
        think_indices[-preserve_last_n_thinks:]
    ) if think_indices else set()

    # --- Partition messages into reasoning and findings buckets ---
    reasoning_indices: list[int] = []
    findings_indices: list[int] = []
    for idx, msg in enumerate(messages):
        if _is_reasoning_message(msg):
            reasoning_indices.append(idx)
        elif _is_findings_message(msg):
            findings_indices.append(idx)
        else:
            # Unknown type — treat as findings (lower priority)
            findings_indices.append(idx)

    reasoning_budget = int(budget_chars * _REASONING_BUDGET_FRACTION)
    findings_budget = int(budget_chars * _FINDINGS_BUDGET_FRACTION)

    reasoning_chars = sum(len(_msg_content(messages[i])) for i in reasoning_indices)
    findings_chars = sum(len(_msg_content(messages[i])) for i in findings_indices)

    # --- Phase 1: Truncate findings bodies (keep headers) ---
    # Before dropping whole messages, truncate long findings bodies.
    body_truncated: dict[int, str] = {}  # idx -> truncated content

    if findings_chars > findings_budget:
        # Sort findings by round (oldest first) for truncation
        findings_by_round = sorted(findings_indices, key=lambda i: messages[i].get("round", 0))
        for idx in findings_by_round:
            if findings_chars <= findings_budget:
                break
            content = _msg_content(messages[idx])
            if len(content) > _FINDINGS_BODY_TRUNCATION_HEADER_CHARS:
                truncated_content = _truncate_findings_body(content)
                saved = len(content) - len(truncated_content)
                findings_chars -= saved
                body_truncated[idx] = truncated_content

    # --- Phase 2: Drop evidence inventories from oldest rounds first ---
    # Evidence inventories are compact and lower-priority than research
    # findings.  Dropping them first preserves the detailed compressed
    # findings that the supervisor relies on for gap analysis.
    findings_to_remove: set[int] = set()
    if findings_chars > findings_budget:
        inventory_indices = sorted(
            (i for i in findings_indices if _is_evidence_inventory(messages[i])),
            key=lambda i: messages[i].get("round", 0),
        )
        for idx in inventory_indices:
            if findings_chars <= findings_budget:
                break
            content_len = len(body_truncated.get(idx, _msg_content(messages[idx])))
            findings_chars -= content_len
            findings_to_remove.add(idx)

    # --- Phase 3: Drop oldest remaining findings messages if still over budget ---
    if findings_chars > findings_budget:
        findings_by_round = sorted(findings_indices, key=lambda i: messages[i].get("round", 0))
        for idx in findings_by_round:
            if findings_chars <= findings_budget:
                break
            if idx in findings_to_remove:
                continue  # Already removed in phase 2
            content_len = len(body_truncated.get(idx, _msg_content(messages[idx])))
            findings_chars -= content_len
            findings_to_remove.add(idx)

    # --- Phase 4: Drop oldest reasoning messages if over budget ---
    # Protected think messages are never removed.
    reasoning_to_remove: set[int] = set()
    if reasoning_chars > reasoning_budget:
        reasoning_by_round = sorted(reasoning_indices, key=lambda i: messages[i].get("round", 0))
        for idx in reasoning_by_round:
            if reasoning_chars <= reasoning_budget:
                break
            if idx in protected_think_indices:
                continue  # Never remove protected thinks
            reasoning_chars -= len(_msg_content(messages[idx]))
            reasoning_to_remove.add(idx)

    # --- Phase 5: Rebalance — if one bucket is under budget, donate surplus ---
    # If reasoning is still over after phase 3 (due to protected thinks),
    # try to steal from unused findings budget, and vice versa.
    reasoning_remaining = sum(
        len(_msg_content(messages[i]))
        for i in reasoning_indices if i not in reasoning_to_remove
    )
    findings_remaining = sum(
        len(body_truncated.get(i, _msg_content(messages[i])))
        for i in findings_indices if i not in findings_to_remove
    )

    if reasoning_remaining > reasoning_budget and findings_remaining < findings_budget:
        # Donate unused findings budget to reasoning
        surplus = findings_budget - findings_remaining
        reasoning_budget += surplus
        # Re-check: can we keep some reasoning messages we were going to remove?
        restored = set()
        for idx in sorted(reasoning_to_remove, key=lambda i: messages[i].get("round", 0), reverse=True):
            msg_chars = len(_msg_content(messages[idx]))
            # Guard: never let combined total exceed the global budget
            if findings_remaining + reasoning_remaining + msg_chars > budget_chars:
                break
            if reasoning_remaining + msg_chars <= reasoning_budget:
                reasoning_remaining += msg_chars
                restored.add(idx)
        reasoning_to_remove -= restored

    elif findings_remaining > findings_budget and reasoning_remaining < reasoning_budget:
        # Donate unused reasoning budget to findings
        surplus = reasoning_budget - reasoning_remaining
        findings_budget += surplus
        # Re-check: can we keep some findings messages we were going to remove?
        restored = set()
        for idx in sorted(findings_to_remove, key=lambda i: messages[i].get("round", 0), reverse=True):
            msg_chars = len(body_truncated.get(idx, _msg_content(messages[idx])))
            # Guard: never let combined total exceed the global budget
            if findings_remaining + reasoning_remaining + msg_chars > budget_chars:
                break
            if findings_remaining + msg_chars <= findings_budget:
                findings_remaining += msg_chars
                restored.add(idx)
        findings_to_remove -= restored

    # --- Build result ---
    all_removed = findings_to_remove | reasoning_to_remove
    if not all_removed and not body_truncated:
        return messages

    result: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if idx in all_removed:
            continue
        if idx in body_truncated:
            result.append({**msg, "content": body_truncated[idx]})
        else:
            result.append(msg)

    removed_count = len(all_removed)
    truncated_count = len(body_truncated) - len(body_truncated.keys() & all_removed)
    result_chars = sum(len(_msg_content(m)) for m in result)

    logger.info(
        "Type-aware supervision truncation: removed %d messages, "
        "truncated %d message bodies, protected %d think messages "
        "(budget=%d chars, reasoning=%d/%d, findings=%d/%d, result=%d chars)",
        removed_count,
        truncated_count,
        len(protected_think_indices),
        budget_chars,
        sum(len(_msg_content(messages[i])) for i in reasoning_indices if i not in reasoning_to_remove),
        int(budget_chars * _REASONING_BUDGET_FRACTION),
        sum(
            len(body_truncated.get(i, _msg_content(messages[i])))
            for i in findings_indices if i not in findings_to_remove
        ),
        int(budget_chars * _FINDINGS_BUDGET_FRACTION),
        result_chars,
    )
    return result


# ---------------------------------------------------------------------------
# Provider-specific context-window error detection
# ---------------------------------------------------------------------------

#: Regex patterns matched against exception messages to detect context-window
#: errors that providers raise as generic ``BadRequestError`` or similar.
#: When the exception message matches any pattern, the error is re-classified
#: as a ContextWindowError.
_CONTEXT_WINDOW_ERROR_PATTERNS: list[str] = [
    # Token/context limit exceeded (tight patterns to avoid false positives
    # on "invalid authentication token", "context parameter is required", etc.)
    r"(?i)\b(?:token|context)\s*(?:limit|window|exceeded|too\s+long|overflow)\b",
    r"(?i)\btoken\s+\w+\s+exceeded\b",
    r"(?i)maximum\s+(?:context|token)",
    r"(?i)(?:exceeds?|over)\s+(?:the\s+)?(?:token|context|length)\s+limit",
    # Anthropic: BadRequestError with "prompt is too long"
    r"(?i)prompt\s+is\s+too\s+long",
    # Google: ResourceExhausted / InvalidArgument with token keywords
    r"(?i)\b(?:resource\s*exhausted|context\s*length|token\s*limit)\b",
    # Cross-provider: explicit "too many tokens" phrasing
    r"(?i)too\s+many\s+tokens",
]

#: Exception class names that are known context-window indicators for
#: specific providers, regardless of message content.
#: ``InvalidArgument`` is NOT included here — it's too generic (covers all
#: gRPC bad-request errors).  Instead, ``InvalidArgument`` is checked with
#: a message pattern below.
_CONTEXT_WINDOW_ERROR_CLASSES: set[str] = {
    "ResourceExhausted",  # Google / gRPC
}


def _is_context_window_error(exc: Exception) -> bool:
    """Detect if a generic exception is actually a context-window overflow.

    Checks the exception's class name and message against known provider
    patterns.  Returns ``True`` if the exception should be treated as a
    ``ContextWindowError`` for progressive-truncation recovery.

    This catches errors that the provider layer raises as generic
    ``BadRequestError`` (OpenAI, Anthropic) or ``ResourceExhausted``
    (Google) instead of the canonical ``ContextWindowError``.
    """
    cls_name = type(exc).__name__

    # Fast path: known class names (unconditional match)
    if cls_name in _CONTEXT_WINDOW_ERROR_CLASSES:
        return True

    msg = str(exc)

    # InvalidArgument is a generic gRPC class — only classify as
    # context-window error when the message also mentions tokens/context.
    if cls_name == "InvalidArgument":
        return bool(re.search(r"(?i)\b(?:token|context)\s*(?:limit|window|exceeded|too\s+long|overflow)\b", msg))

    for pattern in _CONTEXT_WINDOW_ERROR_PATTERNS:
        if re.search(pattern, msg):
            return True

    return False


@dataclass
class LLMCallResult:
    """Result of a successful LLM call with provider metadata."""

    result: WorkflowResult
    llm_call_duration_ms: float


# Maximum number of progressive truncation retries on context-window errors.
# 3 retries at 10% reduction each leaves ~72.9% of the original prompt —
# enough to preserve useful context while meaningfully reducing size.
# Going beyond 3 risks degrading prompt quality without fixing the overflow.
_MAX_TOKEN_LIMIT_RETRIES: int = 3

# Each retry truncates the user prompt to this fraction of the previous size.
# 10% per retry is conservative: aggressive truncation (e.g., 50%) loses too
# much context, while smaller steps (e.g., 5%) waste retries on negligible cuts.
_TRUNCATION_FACTOR: float = 0.9  # keep 90%, remove 10%

# Fallback context-window size when neither the error nor the model registry
# provides a concrete limit.  128K tokens matches the smallest common context
# window among popular models (e.g., GPT-4o-class).  Conservative enough to
# avoid overflows on smaller models, large enough for real research prompts.
FALLBACK_CONTEXT_WINDOW: int = 128_000


def _truncate_for_retry(
    user_prompt: str,
    error_max_tokens: Optional[int],
    model: Optional[str],
    retry_count: int,
    truncate_fn: Any,
    estimate_limit_fn: Any,
    token_limits: dict[str, int],
) -> str:
    """Compute a truncated user prompt for a token-limit retry.

    Determines the token budget from (in order): the error's max_tokens,
    the model registry, or the fallback default. Then applies progressive
    reduction (90% per retry) and truncates at a natural boundary.
    """
    max_tokens = error_max_tokens
    if max_tokens is None:
        max_tokens = estimate_limit_fn(model, token_limits)
    if max_tokens is None:
        max_tokens = FALLBACK_CONTEXT_WINDOW

    reduced_budget = int(max_tokens * (_TRUNCATION_FACTOR ** retry_count))
    return truncate_fn(user_prompt, reduced_budget)


# Human-readable names for each truncation strategy tier (for logging).
_TRUNCATION_STRATEGY_NAMES: dict[int, str] = {
    1: "structured_block_truncation",
    2: "quality_aware_source_dropping",
    3: "char_truncation",
}


def _apply_truncation_strategy(
    user_prompt: str,
    error_max_tokens: Optional[int],
    model: Optional[str],
    retry_count: int,
) -> str:
    """Select and apply the appropriate truncation strategy for a retry.

    Implements tiered truncation that preserves the most relevant content:

    - **Retry 1 — structured block truncation:** splits the prompt at
      markdown header boundaries and truncates the longest content
      sections first, preserving structure and protected sections.
    - **Retry 2 — quality-aware source dropping:** identifies individual
      source entries, scores them by quality markers or length, and
      removes the lowest-value sources entirely.
    - **Retry 3 — character-based truncation:** simple tail-chopping at
      a natural boundary (existing fallback behaviour).

    Each strategy falls back to character-based truncation when the
    prompt lacks recognisable structure (e.g. no markdown headers or
    no source-entry patterns).

    Args:
        user_prompt: The current user prompt to truncate
        error_max_tokens: Token limit reported by the provider error
            (may be ``None``)
        model: Model identifier for registry-based limit lookup
        retry_count: Current retry number (1, 2, or 3)

    Returns:
        Truncated user prompt string
    """
    max_tokens = error_max_tokens
    if max_tokens is None:
        max_tokens = estimate_token_limit_for_model(model, get_model_token_limits())
    if max_tokens is None:
        max_tokens = FALLBACK_CONTEXT_WINDOW

    budget = int(max_tokens * (_TRUNCATION_FACTOR ** retry_count))

    if retry_count == 1:
        result = structured_truncate_blocks(user_prompt, budget)
        if result != user_prompt:
            return result
    elif retry_count == 2:
        result = structured_drop_sources(user_prompt, budget)
        if result != user_prompt:
            return result

    # Retry 3 or fallback when structured strategies don't reduce enough
    return truncate_to_token_estimate(user_prompt, budget)


def truncate_prompt_for_retry(
    prompt: str,
    attempt: int,
    max_attempts: int = 3,
) -> str:
    """Truncate a prompt by removing the oldest content for a retry attempt.

    Removes the first *X%* of ``prompt``, preserving the tail (most recent
    context).  The removal percentage grows with each attempt:

    - Attempt 1: remove first 20%
    - Attempt 2: remove first 30%
    - Attempt 3: remove first 40%

    This complements the structural truncation strategies in
    :func:`_apply_truncation_strategy` by providing a simple content-level
    oldest-first removal — useful when the phase-specific outer retry loop
    pre-truncates the prompt before handing it back to
    :func:`execute_llm_call`.

    Args:
        prompt: The prompt string to truncate.
        attempt: The current retry attempt (1-based).  Values outside
            ``[1, max_attempts]`` return the original prompt unchanged.
        max_attempts: Maximum number of retry attempts (default 3).

    Returns:
        The truncated prompt string, or the original if truncation is
        not applicable or the prompt is already at or below the minimum
        threshold.
    """
    _MIN_PROMPT_CHARS = 1000

    if attempt < 1 or attempt > max_attempts:
        return prompt
    if len(prompt) <= _MIN_PROMPT_CHARS:
        return prompt

    # Progressive removal: 20%, 30%, 40% for attempts 1, 2, 3
    removal_pct = 0.1 + (attempt * 0.1)

    chars_to_remove = int(len(prompt) * removal_pct)
    remaining = prompt[chars_to_remove:]

    # Never truncate below minimum threshold
    if len(remaining) < _MIN_PROMPT_CHARS:
        remaining = prompt[-_MIN_PROMPT_CHARS:]

    return remaining


# Maximum outer retry attempts for phase-specific token-limit recovery.
# Each outer retry pre-truncates the prompt before handing it back to
# execute_llm_call (which has its own 3 internal structural retries).
MAX_PHASE_TOKEN_RETRIES: int = 3


def _is_context_window_exceeded(result: Any) -> bool:
    """Check whether a WorkflowResult represents a context-window error.

    Used by phase-specific outer retry loops to detect when
    ``execute_llm_call`` exhausted its internal retries due to
    context-window overflow.

    Args:
        result: A ``WorkflowResult`` returned by ``execute_llm_call``.

    Returns:
        ``True`` if the result's metadata indicates a context-window error.
    """
    if not hasattr(result, "metadata") or not result.metadata:
        return False
    return result.metadata.get("error_type") == "context_window_exceeded"


async def execute_llm_call(
    workflow: Any,
    state: DeepResearchState,
    phase_name: str,
    system_prompt: str,
    user_prompt: str,
    provider_id: Optional[str],
    model: Optional[str],
    temperature: float,
    timeout: float,
    error_metadata: Optional[dict[str, Any]] = None,
    role: Optional[str] = None,
) -> LLMCallResult | WorkflowResult:
    """Execute an LLM call with full lifecycle instrumentation.

    Handles: heartbeat update, state persistence, audit events (started/completed),
    provider call with ContextWindowError handling (including progressive
    truncation recovery), metrics emission, timeout/failure check, token
    tracking, and PhaseMetrics recording.

    **Progressive token-limit recovery:** When a ``ContextWindowError`` (or a
    provider-specific equivalent detected by ``_is_context_window_error``) is
    raised, the user prompt is truncated by 10% and the call is retried, up to
    3 times.  Only the user prompt is truncated — the system prompt is never
    modified.  If all retries are exhausted the original hard-error path is
    taken.

    **Role-based model resolution (Phase 6):** When *role* is provided and
    *provider_id* / *model* are ``None``, the provider and model are resolved
    from ``workflow.config.resolve_model_for_role(role)``.  Explicit
    *provider_id* / *model* values always take precedence over role-based
    resolution.

    Args:
        workflow: The DeepResearchWorkflow instance (provides config, memory, etc.)
        state: Current research state
        phase_name: Phase identifier (e.g. "planning", "analysis")
        system_prompt: System prompt for the LLM call
        user_prompt: User prompt for the LLM call
        provider_id: Explicit provider ID (may be None for phase default)
        model: Model override for the provider
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        error_metadata: Extra fields to include in ContextWindowError response metadata
        role: Model role for cost-optimized routing (e.g. "research", "report",
            "reflection", "summarization", "compression", "clarification").
            When set, resolves provider/model from config if not explicitly
            provided.

    Returns:
        LLMCallResult on success (caller uses .result for the WorkflowResult),
        or WorkflowResult directly on error (ContextWindowError, timeout, failure).
        Callers use ``isinstance(ret, WorkflowResult)`` to branch on error.
    """
    # Role-based model resolution (Phase 6): when role is provided and
    # provider_id / model are not explicitly set, resolve from config.
    if role:
        role_provider, role_model = safe_resolve_model_for_role(workflow.config, role)
        if provider_id is None:
            provider_id = role_provider
        if model is None:
            model = role_model

    effective_provider = provider_id

    # Heartbeat + persist
    llm_call_start_time = time.perf_counter()
    state.last_heartbeat_at = datetime.now(timezone.utc)
    workflow.memory.save_deep_research(state)

    # Audit: llm.call.started
    workflow._write_audit_event(
        state,
        "llm.call.started",
        data={
            "provider": effective_provider,
            "task_id": state.id,
            "phase": phase_name,
        },
    )

    # ------------------------------------------------------------------
    # Provider call with progressive token-limit recovery
    # ------------------------------------------------------------------
    current_user_prompt = user_prompt
    token_limit_retries = 0
    result: Optional[WorkflowResult] = None
    last_context_error: ContextWindowError | Exception | None = None

    for attempt in range(_MAX_TOKEN_LIMIT_RETRIES + 1):  # 0 = initial, 1-3 = retries
        try:
            result = await workflow._execute_provider_async(
                prompt=current_user_prompt,
                provider_id=effective_provider,
                model=model,
                system_prompt=system_prompt,
                timeout=timeout,
                temperature=temperature,
                phase=phase_name,
                fallback_providers=workflow.config.get_phase_fallback_providers(phase_name),
                max_retries=workflow.config.deep_research_max_retries,
                retry_delay=workflow.config.deep_research_retry_delay,
            )
            # Success — clear any prior context error and break
            last_context_error = None
            break

        except ContextWindowError as e:
            last_context_error = e
            if attempt >= _MAX_TOKEN_LIMIT_RETRIES:
                break  # All retries exhausted

            token_limit_retries += 1
            current_user_prompt = _apply_truncation_strategy(
                current_user_prompt, e.max_tokens, model, token_limit_retries,
            )

            strategy = _TRUNCATION_STRATEGY_NAMES.get(token_limit_retries, "unknown")
            logger.warning(
                "%s phase context window exceeded (attempt %d/%d, strategy=%s), "
                "truncating user prompt and retrying. "
                "prompt_tokens=%s, max_tokens=%s, provider=%s",
                phase_name.capitalize(),
                token_limit_retries,
                _MAX_TOKEN_LIMIT_RETRIES,
                strategy,
                e.prompt_tokens,
                e.max_tokens,
                e.provider,
            )

        except Exception as e:
            if _is_context_window_error(e) and attempt < _MAX_TOKEN_LIMIT_RETRIES:
                last_context_error = e
                token_limit_retries += 1
                current_user_prompt = _apply_truncation_strategy(
                    current_user_prompt, None, model, token_limit_retries,
                )

                strategy = _TRUNCATION_STRATEGY_NAMES.get(token_limit_retries, "unknown")
                logger.warning(
                    "%s phase detected provider-specific context window error "
                    "(attempt %d/%d, strategy=%s), truncating user prompt. "
                    "error_class=%s, message=%s",
                    phase_name.capitalize(),
                    token_limit_retries,
                    _MAX_TOKEN_LIMIT_RETRIES,
                    strategy,
                    type(e).__name__,
                    str(e)[:200],
                )
            else:
                raise

    # ------------------------------------------------------------------
    # All retries exhausted — emit hard error
    # ------------------------------------------------------------------
    if last_context_error is not None:
        llm_call_duration_ms = (time.perf_counter() - llm_call_start_time) * 1000

        prompt_tokens = getattr(last_context_error, "prompt_tokens", None)
        max_tokens_val = getattr(last_context_error, "max_tokens", None)
        truncation_needed = getattr(last_context_error, "truncation_needed", None)
        error_provider = getattr(last_context_error, "provider", None)

        workflow._write_audit_event(
            state,
            "llm.call.completed",
            data={
                "provider": effective_provider,
                "task_id": state.id,
                "duration_ms": llm_call_duration_ms,
                "status": "error",
                "error_type": "context_window_exceeded",
                "token_limit_retries": token_limit_retries,
            },
        )
        get_metrics().histogram(
            "foundry_mcp_research_llm_call_duration_seconds",
            llm_call_duration_ms / 1000.0,
            labels={"provider": effective_provider or "unknown", "status": "error"},
        )

        logger.error(
            "%s phase context window exceeded after %d retries: "
            "prompt_tokens=%s, max_tokens=%s, truncation_needed=%s, provider=%s",
            phase_name.capitalize(),
            token_limit_retries,
            prompt_tokens,
            max_tokens_val,
            truncation_needed,
            error_provider,
        )

        metadata: dict[str, Any] = {
            "research_id": state.id,
            "phase": phase_name,
            "error_type": "context_window_exceeded",
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens_val,
            "truncation_needed": truncation_needed,
            "token_limit_retries": token_limit_retries,
        }
        if error_metadata:
            metadata.update(error_metadata)

        return WorkflowResult(
            success=False,
            content="",
            error=str(last_context_error),
            metadata=metadata,
        )

    # Safety: result must be set at this point (break from loop with no error)
    if result is None:
        return WorkflowResult(
            success=False,
            content="",
            error="LLM call completed without producing a result",
            metadata={"task_id": state.id, "provider": provider_id or "unknown"},
        )

    # Audit + metrics for completion
    llm_call_duration_ms = (time.perf_counter() - llm_call_start_time) * 1000
    llm_call_status = "success" if result.success else "error"
    llm_call_provider: str = result.provider_id or effective_provider or "unknown"

    workflow._write_audit_event(
        state,
        "llm.call.completed",
        data={
            "provider": llm_call_provider,
            "task_id": state.id,
            "duration_ms": llm_call_duration_ms,
            "status": llm_call_status,
        },
    )
    get_metrics().histogram(
        "foundry_mcp_research_llm_call_duration_seconds",
        llm_call_duration_ms / 1000.0,
        labels={"provider": llm_call_provider, "status": llm_call_status},
    )

    # Failure early return
    if not result.success:
        if result.metadata and result.metadata.get("timeout"):
            logger.error(
                "%s phase timed out after exhausting all providers: %s",
                phase_name.capitalize(),
                result.metadata.get("providers_tried", []),
            )
        else:
            logger.error("%s phase LLM call failed: %s", phase_name.capitalize(), result.error)
        return result

    # Token tracking
    if result.tokens_used:
        state.total_tokens_used += result.tokens_used

    # Phase metrics (include token_limit_retries if any occurred)
    phase_metrics_metadata: dict[str, Any] = {}
    if token_limit_retries > 0:
        phase_metrics_metadata["token_limit_retries"] = token_limit_retries
    if role:
        phase_metrics_metadata["role"] = role

    state.phase_metrics.append(
        PhaseMetrics(
            phase=phase_name,
            duration_ms=result.duration_ms or 0.0,
            input_tokens=result.input_tokens or 0,
            output_tokens=result.output_tokens or 0,
            cached_tokens=result.cached_tokens or 0,
            provider_id=result.provider_id,
            model_used=result.model_used,
            metadata=phase_metrics_metadata,
        )
    )

    return LLMCallResult(result=result, llm_call_duration_ms=llm_call_duration_ms)


# Maximum parse-validation retries for structured LLM calls.
_MAX_STRUCTURED_PARSE_RETRIES: int = 3


@dataclass
class StructuredLLMCallResult:
    """Result of a structured LLM call with parsed data.

    Attributes:
        result: The underlying WorkflowResult from the LLM call.
        llm_call_duration_ms: Total time spent across all LLM call attempts.
        parsed: The parsed structured data (output of ``parse_fn``), or
            ``None`` if parsing failed on all attempts.
        parse_retries: Number of parse-validation retries that were needed.
    """

    result: WorkflowResult
    llm_call_duration_ms: float
    parsed: Any
    parse_retries: int = 0


async def execute_structured_llm_call(
    workflow: Any,
    state: DeepResearchState,
    phase_name: str,
    system_prompt: str,
    user_prompt: str,
    provider_id: Optional[str],
    model: Optional[str],
    temperature: float,
    timeout: float,
    parse_fn: Any,
    error_metadata: Optional[dict[str, Any]] = None,
    role: Optional[str] = None,
) -> StructuredLLMCallResult | WorkflowResult:
    """Execute an LLM call expecting structured JSON output.

    Wraps :func:`execute_llm_call` with parse-validation and retry logic.
    On each attempt the LLM response content is passed through *parse_fn*.
    If *parse_fn* raises (``ValueError``, ``json.JSONDecodeError``, etc.),
    the call is retried with a reinforced JSON instruction appended to the
    user prompt, up to ``_MAX_STRUCTURED_PARSE_RETRIES`` times.

    If all parse attempts fail, returns a :class:`StructuredLLMCallResult`
    with ``parsed=None`` and the last successful LLM result — letting the
    caller fall back to unstructured handling.

    Args:
        workflow: The DeepResearchWorkflow instance
        state: Current research state
        phase_name: Phase identifier
        system_prompt: System prompt (should already request JSON output)
        user_prompt: User prompt for the LLM call
        provider_id: Explicit provider ID
        model: Model override
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        parse_fn: Callable ``(content: str) -> T`` that parses the LLM
            response content.  Should raise on validation failure.
        error_metadata: Extra fields for error response metadata
        role: Model role for cost-optimized routing (passed through to
            :func:`execute_llm_call`)

    Returns:
        StructuredLLMCallResult on success or parse-exhaustion (check
        ``.parsed`` for ``None``), or WorkflowResult on LLM-level error.
    """
    total_duration_ms = 0.0
    last_llm_result: Optional[LLMCallResult] = None
    parse_retries = 0

    current_user_prompt = user_prompt

    for attempt in range(_MAX_STRUCTURED_PARSE_RETRIES + 1):  # 0 = initial, 1-3 = retries
        call_result = await execute_llm_call(
            workflow=workflow,
            state=state,
            phase_name=phase_name,
            system_prompt=system_prompt,
            user_prompt=current_user_prompt,
            provider_id=provider_id,
            model=model,
            temperature=temperature,
            timeout=timeout,
            error_metadata=error_metadata,
            role=role,
        )

        # LLM-level error — propagate immediately
        if isinstance(call_result, WorkflowResult):
            return call_result

        last_llm_result = call_result
        total_duration_ms += call_result.llm_call_duration_ms

        # Try parsing the structured output
        content = call_result.result.content or ""
        try:
            parsed = parse_fn(content)
            return StructuredLLMCallResult(
                result=call_result.result,
                llm_call_duration_ms=total_duration_ms,
                parsed=parsed,
                parse_retries=parse_retries,
            )
        except (ValueError, json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(
                "%s phase structured parse failed (attempt %d/%d): %s",
                phase_name.capitalize(),
                attempt + 1,
                _MAX_STRUCTURED_PARSE_RETRIES + 1,
                exc,
            )

            if attempt >= _MAX_STRUCTURED_PARSE_RETRIES:
                break  # All retries exhausted

            parse_retries += 1

            # Reinforce JSON instruction for next attempt
            current_user_prompt = (
                user_prompt
                + "\n\nIMPORTANT: Your previous response could not be parsed as valid JSON. "
                "You MUST respond with ONLY a valid JSON object, no markdown formatting, "
                "no extra text before or after the JSON."
            )

    # Parse exhausted — return with parsed=None so caller can fall back
    assert last_llm_result is not None
    logger.warning(
        "%s phase structured output parsing failed after %d retries, "
        "falling back to unstructured handling",
        phase_name.capitalize(),
        parse_retries,
    )
    return StructuredLLMCallResult(
        result=last_llm_result.result,
        llm_call_duration_ms=total_duration_ms,
        parsed=None,
        parse_retries=parse_retries,
    )


def finalize_phase(
    workflow: Any,
    state: DeepResearchState,
    phase_name: str,
    phase_start_time: float,
) -> None:
    """Emit phase.completed audit event and duration metric.

    Args:
        workflow: The DeepResearchWorkflow instance
        state: Current research state
        phase_name: Phase identifier (e.g. "planning", "analysis")
        phase_start_time: Value from ``time.perf_counter()`` at phase start
    """
    phase_duration_ms = (time.perf_counter() - phase_start_time) * 1000

    workflow._write_audit_event(
        state,
        "phase.completed",
        data={
            "phase_name": phase_name,
            "iteration": state.iteration,
            "task_id": state.id,
            "duration_ms": phase_duration_ms,
        },
    )

    get_metrics().histogram(
        "foundry_mcp_research_phase_duration_seconds",
        phase_duration_ms / 1000.0,
        labels={"phase_name": phase_name, "status": "success"},
    )
