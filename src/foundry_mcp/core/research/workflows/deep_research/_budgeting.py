"""Budget allocation, validation, and digest archive management.

All functions are standalone (no instance state). Called from phase mixins
and the core workflow class via thin delegation methods.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from foundry_mcp.core.research.context_budget import (
    AllocationResult,
    AllocationStrategy,
    ContentItem,
    ContextBudgetManager,
    compute_priority,
    compute_recency_score,
)
from foundry_mcp.core.research.document_digest import (
    DocumentDigestor,
    deserialize_payload,
)
from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import ResearchSource, SourceQuality
from foundry_mcp.core.research.token_management import (
    PreflightResult,
    TokenBudget,
    estimate_tokens,
    get_effective_context,
    get_model_limits,
    get_provider_model_from_spec,
    preflight_count,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (
    ANALYSIS_OUTPUT_RESERVED,
    ANALYSIS_PHASE_BUDGET_FRACTION,
    FINAL_FIT_COMPRESSION_FACTOR,
    FINAL_FIT_MAX_ITERATIONS,
    FINAL_FIT_SAFETY_MARGIN,
    REFINEMENT_OUTPUT_RESERVED,
    REFINEMENT_PHASE_BUDGET_FRACTION,
    REFINEMENT_REPORT_BUDGET_FRACTION,
    SYNTHESIS_OUTPUT_RESERVED,
    SYNTHESIS_PHASE_BUDGET_FRACTION,
)
from foundry_mcp.core.research.workflows.deep_research._token_budget import (
    truncate_at_boundary,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Digest Archive Management
# =============================================================================


def validate_archive_source_id(source_id: str) -> None:
    """Validate source_id is safe to use as an archive path component."""
    if not source_id or not source_id.strip():
        raise ValueError("Invalid source_id for digest archive (empty)")
    source_path = Path(source_id)
    if source_path.is_absolute() or source_path.drive:
        raise ValueError("Invalid source_id for digest archive (absolute path)")
    if ".." in source_path.parts or len(source_path.parts) != 1:
        raise ValueError("Invalid source_id for digest archive (path traversal)")


def ensure_private_dir(path: Path) -> None:
    """Ensure directory exists with owner-only permissions."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def cleanup_digest_archives(source_dir: Path, retention_days: int) -> None:
    """Remove archived digest files older than retention_days."""
    cutoff = time.time() - (retention_days * 86400)
    for path in source_dir.glob("*.txt"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def write_digest_archive(
    *,
    source_id: str,
    source_text_hash: str,
    canonical_text: str,
    retention_days: int,
) -> Path:
    """Write canonical text to the digest archive directory."""
    archive_root = Path.home() / ".foundry-mcp" / "research_archives"
    ensure_private_dir(archive_root)

    validate_archive_source_id(source_id)
    source_dir = archive_root / source_id
    ensure_private_dir(source_dir)

    target_path = source_dir / f"{source_text_hash}.txt"
    if not target_path.exists():
        fd, tmp_path = tempfile.mkstemp(dir=source_dir, prefix="tmp-", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(canonical_text)
            os.replace(tmp_path, target_path)
            try:
                os.chmod(target_path, 0o600)
            except OSError:
                pass
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        try:
            os.utime(target_path, None)
        except OSError:
            pass

    if retention_days > 0:
        cleanup_digest_archives(source_dir, retention_days)

    return target_path


def archive_digest_source(
    *,
    source: ResearchSource,
    digestor: DocumentDigestor,
    raw_content: str,
    page_boundaries: Optional[list[tuple[int, int, int]]],
    source_text_hash: str,
    retention_days: int,
) -> None:
    """Archive canonical text for a digested source.

    Raises ValueError if canonical text is empty or hashes do not match.
    """
    if not raw_content:
        raise ValueError("No raw content available for digest archival")

    if page_boundaries:
        canonical_text, _ = digestor._canonicalize_pages(raw_content, page_boundaries)
    else:
        canonical_text = digestor._normalize_text(raw_content)

    if not canonical_text.strip():
        raise ValueError("Canonical text is empty after normalization")

    computed_hash = digestor._compute_source_hash(canonical_text)
    if computed_hash != source_text_hash:
        raise ValueError(f"Canonical text hash mismatch: computed={computed_hash}, payload={source_text_hash}")

    archive_path = write_digest_archive(
        source_id=source.id,
        source_text_hash=source_text_hash,
        canonical_text=canonical_text,
        retention_days=retention_days,
    )
    source.metadata["_digest_archive_hash"] = source_text_hash
    logger.debug("Archived digest source %s to %s", source.id, archive_path)


# =============================================================================
# Budget Allocation
# =============================================================================


def allocate_source_budget(
    state: DeepResearchState,
    provider_id: Optional[str],
) -> AllocationResult:
    """Allocate token budget across sources for analysis phase.

    Computes phase budget (80% of effective context), converts sources to
    prioritized ContentItems, and allocates budget with PRIORITY_FIRST strategy.

    Args:
        state: Current research state with sources
        provider_id: LLM provider to use for model limits

    Returns:
        AllocationResult with allocated items and fidelity metadata
    """
    # Get model limits for the analysis provider
    provider_spec = provider_id or state.analysis_provider or "claude"
    provider, model = get_provider_model_from_spec(provider_spec)
    limits = get_model_limits(provider, model)

    # Calculate effective context and phase budget
    effective_context = get_effective_context(limits, output_budget=ANALYSIS_OUTPUT_RESERVED)
    phase_budget = int(effective_context * ANALYSIS_PHASE_BUDGET_FRACTION)

    logger.debug(
        "Analysis budget: effective_context=%d, phase_budget=%d (%.0f%%)",
        effective_context,
        phase_budget,
        ANALYSIS_PHASE_BUDGET_FRACTION * 100,
    )

    # Convert sources to ContentItems with priority scores
    content_items: list[ContentItem] = []
    for source in state.sources:
        # Compute recency score from discovered_at
        recency = 0.5  # Default if no timestamp
        if source.discovered_at:
            now = datetime.now(timezone.utc)
            discovered = source.discovered_at
            # Handle timezone-naive datetimes (legacy data)
            if discovered.tzinfo is None:
                discovered = discovered.replace(tzinfo=timezone.utc)
            age_hours = (now - discovered).total_seconds() / 3600
            recency = compute_recency_score(age_hours, max_age_hours=720.0)

        # Compute overall priority (0-1 scale, higher = higher priority)
        priority_score = compute_priority(
            source_quality=source.quality,
            confidence=ConfidenceLevel.MEDIUM,  # Default for sources
            recency_score=recency,
            relevance_score=0.7,  # Assume sources are generally relevant
        )

        # Convert 0-1 score to integer priority (1=highest)
        # 0.9+ -> priority 1, 0.7-0.9 -> priority 2, etc.
        int_priority = max(1, min(5, int((1.0 - priority_score) * 5) + 1))

        # Build content for token estimation
        content = source.content or source.snippet or ""
        if source.is_digest and source.content:
            try:
                payload = deserialize_payload(source.content)
                digest_parts = [
                    payload.summary,
                    *payload.key_points,
                    *[ev.text for ev in payload.evidence_snippets],
                ]
                content = "\n".join(part for part in digest_parts if part)
            except Exception:
                # Fallback to raw digest JSON if parsing fails
                content = source.content or source.snippet or ""

        content_items.append(
            ContentItem(
                id=source.id,
                content=content,
                priority=int_priority,
                source_id=source.id,
                source_ref=source,
                protected=source.quality == SourceQuality.HIGH,  # Protect high-quality sources
            )
        )

    # Allocate budget using ContextBudgetManager
    manager = ContextBudgetManager(provider=provider, model=model)
    result = manager.allocate_budget(
        items=content_items,
        budget=phase_budget,
        strategy=AllocationStrategy.PRIORITY_FIRST,
    )

    return result


def allocate_synthesis_budget(
    state: DeepResearchState,
    provider_id: Optional[str],
) -> AllocationResult:
    """Allocate token budget for synthesis phase.

    Prioritizes findings (full fidelity) over source references (compressed).
    Uses 85% of effective context as phase budget.

    Args:
        state: Current research state with findings and sources
        provider_id: LLM provider to use for model limits

    Returns:
        AllocationResult with allocated items and fidelity metadata
    """
    # Get model limits for the synthesis provider
    provider_spec = provider_id or state.synthesis_provider or "claude"
    provider, model = get_provider_model_from_spec(provider_spec)
    limits = get_model_limits(provider, model)

    # Calculate effective context and phase budget
    effective_context = get_effective_context(limits, output_budget=SYNTHESIS_OUTPUT_RESERVED)
    phase_budget = int(effective_context * SYNTHESIS_PHASE_BUDGET_FRACTION)

    logger.debug(
        "Synthesis budget: effective_context=%d, phase_budget=%d (%.0f%%)",
        effective_context,
        phase_budget,
        SYNTHESIS_PHASE_BUDGET_FRACTION * 100,
    )

    # Build content items: findings first (protected, priority 1),
    # then sources (not protected, lower priority)
    content_items: list[ContentItem] = []

    # Add findings - they get priority and are protected
    for finding in state.findings:
        # Compute confidence-based priority
        confidence_scores = {
            ConfidenceLevel.CONFIRMED: 1,
            ConfidenceLevel.HIGH: 1,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.LOW: 3,
            ConfidenceLevel.SPECULATION: 4,
        }
        int_priority = confidence_scores.get(finding.confidence, 2)

        # Build finding content for token estimation
        confidence_label = finding.confidence.value if hasattr(finding.confidence, "value") else str(finding.confidence)
        source_refs = ", ".join(finding.source_ids) if finding.source_ids else "no sources"
        content = f"[{confidence_label.upper()}] {finding.content}\nSources: {source_refs}"

        content_items.append(
            ContentItem(
                id=finding.id,
                content=content,
                priority=int_priority,
                source_id=None,
                protected=True,  # Findings get full fidelity
            )
        )

    # Add sources - they get compressed more aggressively
    for source in state.sources:
        # Compute recency score from discovered_at
        recency = 0.5  # Default if no timestamp
        if source.discovered_at:
            now = datetime.now(timezone.utc)
            discovered = source.discovered_at
            # Handle timezone-naive datetimes (legacy data)
            if discovered.tzinfo is None:
                discovered = discovered.replace(tzinfo=timezone.utc)
            age_hours = (now - discovered).total_seconds() / 3600
            recency = compute_recency_score(age_hours, max_age_hours=720.0)

        # Compute overall priority (0-1 scale, higher = higher priority)
        priority_score = compute_priority(
            source_quality=source.quality,
            confidence=ConfidenceLevel.MEDIUM,  # Default for sources
            recency_score=recency,
            relevance_score=0.5,  # Lower relevance for synthesis (sources are secondary)
        )

        # Convert 0-1 score to integer priority (1=highest)
        # Start at priority 5 (after findings) and add based on score
        # 0.9+ -> priority 5, 0.7-0.9 -> priority 6, etc.
        int_priority = 5 + max(0, min(4, int((1.0 - priority_score) * 5)))

        # Build source reference content (more compressed than analysis)
        content_parts = [f"{source.id}: {source.title}"]
        if source.url:
            content_parts.append(f"URL: {source.url}")
        # Include only snippet for sources in synthesis (not full content)
        if source.snippet:
            content_parts.append(f"Snippet: {source.snippet[:200]}...")
        content = "\n".join(content_parts)

        content_items.append(
            ContentItem(
                id=source.id,
                content=content,
                priority=int_priority,
                source_id=source.id,
                source_ref=source,
                protected=False,  # Sources can be dropped if needed
            )
        )

    # Allocate budget using ContextBudgetManager
    manager = ContextBudgetManager(provider=provider, model=model)
    result = manager.allocate_budget(
        items=content_items,
        budget=phase_budget,
        strategy=AllocationStrategy.PRIORITY_FIRST,
    )

    return result


def compute_refinement_budget(
    provider_id: Optional[str],
    state: DeepResearchState,
) -> tuple[int, int, int]:
    """Compute token budgets for refinement phase.

    Calculates phase budget and allocates portions for report summary,
    gaps, and findings context.

    Args:
        provider_id: LLM provider to use for model limits
        state: Current research state

    Returns:
        Tuple of (phase_budget, report_budget, remaining_budget)
    """
    # Get model limits for the refinement provider
    provider_spec = provider_id or state.refinement_provider or "claude"
    provider, model = get_provider_model_from_spec(provider_spec)
    limits = get_model_limits(provider, model)

    # Calculate effective context and phase budget
    effective_context = get_effective_context(limits, output_budget=REFINEMENT_OUTPUT_RESERVED)
    phase_budget = int(effective_context * REFINEMENT_PHASE_BUDGET_FRACTION)

    # Allocate budget: 50% for report, 50% for gaps/findings
    report_budget = int(phase_budget * REFINEMENT_REPORT_BUDGET_FRACTION)
    remaining_budget = phase_budget - report_budget

    logger.debug(
        "Refinement budget: phase=%d, report=%d, remaining=%d",
        phase_budget,
        report_budget,
        remaining_budget,
    )

    return phase_budget, report_budget, remaining_budget


# =============================================================================
# Report Summarization
# =============================================================================


def extract_report_summary(report: str, char_limit: int) -> str:
    """Extract summary from report preserving structure.

    Prioritizes:
    1. Executive Summary section (if present)
    2. Conclusions section (if present)
    3. Key Findings headings
    4. First portion of content

    Args:
        report: Full report content
        char_limit: Maximum characters allowed

    Returns:
        Truncated/summarized report
    """
    if len(report) <= char_limit:
        return report

    summary_parts = []
    remaining = char_limit

    # Try to extract Executive Summary
    exec_start = report.find("## Executive Summary")
    if exec_start == -1:
        exec_start = report.find("# Executive Summary")

    if exec_start >= 0:
        # Find next section
        next_section = report.find("\n## ", exec_start + 5)
        if next_section == -1:
            next_section = report.find("\n# ", exec_start + 5)
        if next_section == -1:
            next_section = min(exec_start + 1500, len(report))

        exec_content = report[exec_start:next_section].strip()
        if len(exec_content) < remaining:
            summary_parts.append(exec_content)
            remaining -= len(exec_content) + 20  # Account for separators

    # Try to extract Conclusions
    concl_start = report.find("## Conclusions")
    if concl_start == -1:
        concl_start = report.find("# Conclusions")

    if concl_start >= 0 and remaining > 200:
        # Find next section or end
        next_section = report.find("\n## ", concl_start + 5)
        if next_section == -1:
            next_section = report.find("\n# ", concl_start + 5)
        if next_section == -1:
            next_section = len(report)

        concl_content = report[concl_start:next_section].strip()
        if len(concl_content) < remaining:
            summary_parts.append(concl_content)
            remaining -= len(concl_content) + 20

    # If we have space, add beginning of report
    if remaining > 300 and not summary_parts:
        # Take first portion
        summary_parts.append(report[:remaining])
    elif remaining > 300:
        # Add note about truncation
        summary_parts.append(f"\n\n[Report truncated - {len(report)} chars total]")

    return "\n\n---\n\n".join(summary_parts)


def summarize_report_for_refinement(
    report: str,
    target_tokens: int,
) -> tuple[str, str]:
    """Summarize report content to fit within token budget.

    Uses heuristic truncation with key section preservation.
    Full LLM-based summarization would be async, so this function
    uses intelligent truncation instead.

    Args:
        report: Full report content
        target_tokens: Target token budget for report

    Returns:
        Tuple of (summarized_report, fidelity_level)
    """
    # Estimate current token count
    current_tokens = estimate_tokens(report)

    if current_tokens <= target_tokens:
        return report, "full"

    # Calculate compression ratio needed
    ratio = target_tokens / current_tokens

    if ratio >= 0.7:
        fidelity = "condensed"
    elif ratio >= 0.4:
        fidelity = "compressed"
    else:
        fidelity = "minimal"

    # Use character limit based on token budget (~4 chars/token)
    char_limit = target_tokens * 4

    # Extract key sections with smart truncation
    summarized = extract_report_summary(report, char_limit)

    logger.info(
        "Report summarized for refinement: %d -> %d tokens (fidelity=%s)",
        current_tokens,
        estimate_tokens(summarized),
        fidelity,
    )

    return summarized, fidelity


# =============================================================================
# Final-Fit Validation
# =============================================================================


def final_fit_validate(
    system_prompt: str,
    user_prompt: str,
    provider_id: Optional[str],
    model: Optional[str],
    output_reserved: int,
    phase: str,
) -> tuple[bool, PreflightResult, str, str]:
    """Validate assembled payload fits within context budget.

    Performs preflight token counting on the full payload (system + user prompts).
    If over budget, attempts to compress prompts with capped retry loop.

    Args:
        system_prompt: System prompt content
        user_prompt: User prompt content
        provider_id: LLM provider to use
        model: Model override
        output_reserved: Tokens reserved for output
        phase: Phase name for logging

    Returns:
        Tuple of (valid, preflight_result, final_system_prompt, final_user_prompt)
    """
    # Get model limits
    provider_spec = provider_id or "claude"
    provider, model_name = get_provider_model_from_spec(provider_spec)
    limits = get_model_limits(provider, model_name if model is None else model)

    # Create token budget
    budget = TokenBudget(
        total_budget=limits.context_window,
        reserved_output=output_reserved,
        safety_margin=FINAL_FIT_SAFETY_MARGIN,
    )

    # Combine prompts for total token count
    full_payload = f"{system_prompt}\n\n{user_prompt}"

    current_system = system_prompt
    current_user = user_prompt

    for iteration in range(FINAL_FIT_MAX_ITERATIONS):
        # Recompute payload
        if iteration > 0:
            full_payload = f"{current_system}\n\n{current_user}"

        # Run preflight check
        result = preflight_count(
            full_payload,
            budget,
            provider=provider,
            model=model_name,
            is_final_fit=(iteration > 0),
            warn_on_heuristic=False,  # Suppress warnings during loop
        )

        if result.valid:
            logger.info(
                "Final-fit validation passed for %s: %d tokens (%.1f%% of budget, iteration %d)",
                phase,
                result.estimated_tokens,
                result.usage_fraction * 100,
                iteration + 1,
            )
            return True, result, current_system, current_user

        # Over budget - try to compress
        if iteration + 1 >= FINAL_FIT_MAX_ITERATIONS:
            logger.warning(
                "Final-fit validation failed for %s after %d iterations: %d tokens exceeds budget by %d",
                phase,
                iteration + 1,
                result.estimated_tokens,
                result.overflow_tokens,
            )
            break

        # Calculate compression target
        target_tokens = int(result.effective_budget * FINAL_FIT_COMPRESSION_FACTOR)
        excess_tokens = result.estimated_tokens - target_tokens

        logger.info(
            "Final-fit compression needed for %s: reducing by ~%d tokens (iteration %d)",
            phase,
            excess_tokens,
            iteration + 1,
        )

        # Apply compression to user prompt (preserve system prompt)
        # Estimate character reduction needed (~4 chars/token)
        char_reduction = excess_tokens * 4
        current_length = len(current_user)
        target_length = max(100, current_length - char_reduction)

        if target_length >= current_length:
            # Can't compress further
            logger.warning("Cannot compress user prompt further for %s", phase)
            break

        # Truncate user prompt at a reasonable boundary
        current_user = truncate_at_boundary(current_user, target_length)

    # Return failed result with last attempt's prompts
    return False, result, current_system, current_user
