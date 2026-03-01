"""Synthesis phase mixin for DeepResearchWorkflow.

Generates a comprehensive markdown report from analyzed findings and sources.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig
    from foundry_mcp.core.research.memory import ResearchMemory

from foundry_mcp.core.research.context_budget import AllocationResult
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    ResearchLandscape,
)
from foundry_mcp.core.research.models.sources import ResearchMode, SourceType
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._budgeting import (
    allocate_synthesis_budget,
    final_fit_validate,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (
    SYNTHESIS_OUTPUT_RESERVED,
)
from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
    sanitize_external_content,
)
from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
    estimate_token_limit_for_model,
)
from foundry_mcp.core.research.workflows.deep_research._token_budget import (
    fidelity_level_from_score,
    truncate_at_boundary,
)
from foundry_mcp.core.research.workflows.deep_research.phases._citation_postprocess import (
    postprocess_citations,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    FALLBACK_CONTEXT_WINDOW,
    execute_llm_call,
    finalize_phase,
    get_model_token_limits,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auto-save report as markdown file
# ---------------------------------------------------------------------------

from pathlib import Path


def _slugify_query(query: str, max_len: int = 80) -> str:
    """Convert a research query into a filesystem-safe slug."""
    slug = query.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")
    return slug[:max_len].rstrip("-")


def _save_report_markdown(state: DeepResearchState) -> Optional[str]:
    """Save the report as a markdown file in the current working directory.

    Returns the output path on success, or None if saving failed.
    Failure is non-fatal — logs a warning but does not break the workflow.
    """
    if not state.report:
        return None

    try:
        slug = _slugify_query(state.original_query)
        if not slug:
            slug = "deep-research-report"

        output_dir = Path.cwd()
        output_path = output_dir / f"{slug}.md"

        # Collision handling: append research ID suffix if file exists
        if output_path.exists():
            id_suffix = state.id.replace("deepres-", "")[:8]
            output_path = output_dir / f"{slug}-{id_suffix}.md"

        output_path.write_text(state.report, encoding="utf-8")
        logger.info("Auto-saved research report to %s", output_path)
        return str(output_path)
    except Exception:
        logger.warning("Failed to auto-save research report", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Query-type classification and structure guidance
# ---------------------------------------------------------------------------

# Keywords/patterns for classifying the intent of a research query.
_COMPARISON_PATTERNS = re.compile(
    r"\b(compar\w*|vs\.?|versus|differ\w*|contrast|between .+ and )\b",
    re.IGNORECASE,
)
_ENUMERATION_PATTERNS = re.compile(
    r"\b(list\b|top \d|best \d|options|alternatives|examples of|types of)",
    re.IGNORECASE,
)
_HOWTO_PATTERNS = re.compile(
    r"\b(how to|how do|steps to|guide to|tutorial|setup|install\w*|implement\w*|build\w*)\b",
    re.IGNORECASE,
)


def _classify_query_type(query: str) -> str:
    """Classify a research query into a structural type.

    Returns one of: ``"comparison"``, ``"enumeration"``, ``"howto"``,
    or ``"explanation"`` (the default).
    """
    if _COMPARISON_PATTERNS.search(query):
        return "comparison"
    if _ENUMERATION_PATTERNS.search(query):
        return "enumeration"
    if _HOWTO_PATTERNS.search(query):
        return "howto"
    return "explanation"


_STRUCTURE_GUIDANCE: dict[str, str] = {
    "comparison": """\
For **comparison** queries, use this structure:
# Research Report: [Topic]
## Executive Summary
## Overview of [Subject A]
## Overview of [Subject B]
## Comparative Analysis
## Conclusions""",
    "enumeration": """\
For **list/enumeration** queries, use this structure:
# Research Report: [Topic]
## Executive Summary
## [Item 1]
## [Item 2]
## [Item N]
Each item should be its own section when depth is needed. For short lists a single section with a table or bullet list is acceptable.""",
    "howto": """\
For **how-to** queries, use this structure:
# Research Report: [Topic]
## Executive Summary
## Prerequisites
## Step 1: [Action]
## Step 2: [Action]
## Step N: [Action]
## Conclusions""",
    "explanation": """\
For **explanation/overview** queries, use this structure:
# Research Report: [Topic]
## Executive Summary
## Key Findings
### [Theme/Category 1]
### [Theme/Category 2]
## Conclusions""",
}


# ---------------------------------------------------------------------------
# Findings-specific truncation for token-limit recovery (PLAN Phase 4)
# ---------------------------------------------------------------------------

# Maximum retries with findings-specific truncation before giving up.
_MAX_FINDINGS_TRUNCATION_RETRIES: int = 3

# Each retry reduces the findings char budget by this factor (10% per retry).
_FINDINGS_TRUNCATION_FACTOR: float = 0.9

# FALLBACK_CONTEXT_WINDOW imported from _lifecycle.py (canonical source)

# Markers that delineate the start of the findings section in the user prompt.
_FINDINGS_START_MARKERS: list[str] = [
    "## Unified Research Digest",
    "## Research Findings by Topic",
    "## Findings to Synthesize",
]

# Marker that delineates the end of the findings section.
_SOURCE_REF_MARKER: str = "## Source Reference"

# ---------------------------------------------------------------------------
# Supplementary raw notes injection (Phase 3 — ODR alignment)
# ---------------------------------------------------------------------------

# Minimum headroom (fraction of context window) required before injecting
# supplementary raw notes into the synthesis prompt.
_SUPPLEMENTARY_HEADROOM_THRESHOLD: float = 0.10  # 10% of context window

# Maximum fraction of the context window that supplementary notes may occupy.
# Prevents raw notes from dominating the prompt even when headroom is large.
_SUPPLEMENTARY_MAX_FRACTION: float = 0.25  # 25% of context window


def _truncate_findings_section(
    user_prompt: str,
    max_findings_chars: int,
) -> tuple[str, int]:
    """Truncate only the findings section of the synthesis user prompt.

    Identifies the findings portion (between the research brief and the
    source reference section) and truncates it to ``max_findings_chars``,
    preserving the header (query + brief), source reference, and
    instructions intact.

    This mirrors open_deep_research's strategy of truncating findings
    content specifically rather than applying generic prompt-level
    truncation that might remove the system prompt or source context.

    Args:
        user_prompt: The full synthesis user prompt.
        max_findings_chars: Maximum characters allowed for the findings
            section.

    Returns:
        A tuple of (possibly truncated prompt, characters dropped).
        If the findings section is already within budget or cannot be
        identified, returns the original prompt with 0 chars dropped.
    """
    # Find the end of findings (start of source reference)
    source_ref_idx = user_prompt.find(_SOURCE_REF_MARKER)
    if source_ref_idx == -1:
        return user_prompt, 0

    # Find the start of the findings section
    findings_start = -1
    for marker in _FINDINGS_START_MARKERS:
        idx = user_prompt.find(marker)
        if idx != -1:
            findings_start = idx
            break

    if findings_start == -1:
        return user_prompt, 0

    # Split: header + findings + tail
    header = user_prompt[:findings_start]
    findings = user_prompt[findings_start:source_ref_idx]
    tail = user_prompt[source_ref_idx:]

    original_len = len(findings)
    if original_len <= max_findings_chars:
        return user_prompt, 0

    truncated_findings = truncate_at_boundary(findings, max_findings_chars)
    chars_dropped = original_len - len(truncated_findings)

    return header + truncated_findings + "\n\n" + tail, chars_dropped


def _estimate_findings_section_length(user_prompt: str) -> int:
    """Return the character length of the findings section in the synthesis prompt.

    Falls back to the full prompt length when the findings section cannot be
    identified, so callers always get a usable value.
    """
    source_ref_idx = user_prompt.find(_SOURCE_REF_MARKER)
    if source_ref_idx == -1:
        return len(user_prompt)

    findings_start = -1
    for marker in _FINDINGS_START_MARKERS:
        idx = user_prompt.find(marker)
        if idx != -1:
            findings_start = idx
            break

    if findings_start == -1:
        return len(user_prompt)

    return source_ref_idx - findings_start


class SynthesisPhaseMixin:
    """Synthesis phase methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)

    See ``DeepResearchWorkflowProtocol`` in ``_protocols.py`` for the
    full structural contract.
    """

    config: ResearchConfig
    memory: ResearchMemory

    # Stubs for Pyright — canonical signatures live in _protocols.py
    if TYPE_CHECKING:
        from foundry_mcp.core.research.models.deep_research import DeepResearchState as _S

        def _write_audit_event(
            self, state: _S | None, event_name: str, *, data: dict[str, Any] | None = ..., level: str = ...
        ) -> None: ...
        def _check_cancellation(self, state: _S) -> None: ...

    async def _execute_synthesis_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute synthesis phase: generate comprehensive report from findings.

        This phase:
        1. Builds a synthesis prompt with all findings grouped by theme
        2. Includes source references for citation
        3. Generates a structured markdown report with:
           - Executive summary
           - Key findings organized by theme
           - Source citations
           - Knowledge gaps and limitations
           - Conclusions with actionable insights
        4. Stores the report in state.report

        Args:
            state: Current research state with findings from analysis
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with synthesis outcome
        """
        # Check for early termination when no findings exist.
        early_result, degraded_mode = self._handle_empty_findings(state)
        if early_result is not None:
            return early_result

        logger.info(
            "Starting synthesis phase: %d findings, %d sources, %d topic results with compressed findings",
            len(state.findings),
            len(state.sources),
            sum(1 for tr in state.topic_research_results if tr.compressed_findings),
        )

        # Emit phase.started audit event
        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "synthesis",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        # Allocate budget, build prompts, run final-fit validation.
        system_prompt, user_prompt = self._prepare_synthesis_budget_and_prompts(
            state,
            provider_id,
            degraded_mode,
        )

        # Check for cancellation before making provider call
        self._check_cancellation(state)

        # LLM call with findings-specific token-limit recovery.
        llm_result = await self._execute_synthesis_llm_with_retry(
            state,
            provider_id,
            system_prompt,
            user_prompt,
            timeout,
        )
        if isinstance(llm_result, WorkflowResult):
            return llm_result

        # Unpack successful LLM result.
        result_content, findings_retries, total_chars_dropped, used_fallback = llm_result

        # Extract report, post-process citations, audit, and finalize.
        return self._finalize_synthesis_report(
            state=state,
            result=result_content,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            phase_start_time=phase_start_time,
            findings_retries=findings_retries,
            total_chars_dropped=total_chars_dropped,
            used_fallback=used_fallback,
            degraded_mode=degraded_mode,
        )

    # ------------------------------------------------------------------
    # Extracted helpers for _execute_synthesis_async
    # ------------------------------------------------------------------

    def _handle_empty_findings(
        self,
        state: DeepResearchState,
    ) -> tuple[Optional[WorkflowResult], bool]:
        """Handle the case when no findings exist for synthesis.

        Checks for raw notes, compressed findings, and analysis findings.
        Returns a WorkflowResult for early exit when there is truly nothing
        to synthesize, or ``(None, degraded_mode)`` to continue.

        Args:
            state: Current research state.

        Returns:
            A tuple of (early_result_or_none, degraded_mode_flag).
        """
        has_compressed = any(tr.compressed_findings for tr in state.topic_research_results)
        has_raw_notes = bool(state.raw_notes)
        degraded_mode = False

        if not state.findings and not state.compressed_digest and not has_compressed:
            if has_raw_notes:
                # Phase 3b (ODR alignment): Degraded mode — synthesize
                # directly from raw notes when compression failed or
                # produced no output.  This prevents the "empty report"
                # path when raw researcher data exists.
                degraded_mode = True
                logger.warning(
                    "No compressed findings available; falling back to raw "
                    "notes for synthesis (degraded mode, %d note entries)",
                    len(state.raw_notes),
                )
            else:
                logger.warning("No findings to synthesize")
                # Generate a minimal report even without findings
                state.report = self._generate_empty_report(state)
                self._write_audit_event(
                    state,
                    "synthesis_result",
                    data={
                        "provider_id": None,
                        "model_used": None,
                        "tokens_used": None,
                        "duration_ms": None,
                        "system_prompt": None,
                        "user_prompt": None,
                        "raw_response": None,
                        "report": state.report,
                        "empty_report": True,
                    },
                    level="warning",
                )
                return WorkflowResult(
                    success=True,
                    content=state.report,
                    metadata={
                        "research_id": state.id,
                        "finding_count": 0,
                        "empty_report": True,
                    },
                ), degraded_mode

        return None, degraded_mode

    def _prepare_synthesis_budget_and_prompts(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        degraded_mode: bool,
    ) -> tuple[str, str]:
        """Allocate token budget, build prompts, and run final-fit validation.

        Args:
            state: Current research state.
            provider_id: LLM provider to use.
            degraded_mode: Whether synthesis is running from raw notes.

        Returns:
            A tuple of (system_prompt, user_prompt) ready for the LLM call.
        """
        # Allocate token budget for findings and sources
        allocation_result = allocate_synthesis_budget(
            state=state,
            provider_id=provider_id,
        )

        # Update state with allocation metadata
        # Store overall fidelity in metadata (content_fidelity is now per-item dict)
        state.dropped_content_ids = allocation_result.dropped_ids
        allocation_dict = allocation_result.to_dict()
        allocation_dict["overall_fidelity_level"] = fidelity_level_from_score(allocation_result.fidelity)
        state.content_allocation_metadata = allocation_dict

        logger.info(
            "Synthesis budget allocation: %d items allocated, %d dropped, fidelity=%.1f%%",
            len(allocation_result.items),
            len(allocation_result.dropped_ids),
            allocation_result.fidelity * 100,
        )

        # Build the synthesis prompt with allocated content
        system_prompt = self._build_synthesis_system_prompt(state)
        user_prompt = self._build_synthesis_user_prompt(
            state,
            allocation_result,
            degraded_mode=degraded_mode,
        )

        # Final-fit validation before provider dispatch
        valid, _preflight, system_prompt, user_prompt = final_fit_validate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.synthesis_provider,
            model=state.synthesis_model,
            output_reserved=SYNTHESIS_OUTPUT_RESERVED,
            phase="synthesis",
        )

        if not valid:
            logger.warning("Synthesis phase final-fit validation failed, proceeding with truncated prompts")

        return system_prompt, user_prompt

    async def _execute_synthesis_llm_with_retry(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        system_prompt: str,
        user_prompt: str,
        timeout: float,
    ) -> "WorkflowResult | tuple[Any, int, int, bool]":
        """Execute the synthesis LLM call with findings-specific retry logic.

        When synthesis hits a token limit, truncates the findings section
        specifically (preserving system prompt, source reference, and
        instructions).  Falls back to generic prompt truncation when
        findings-specific truncation is insufficient.

        Args:
            state: Current research state.
            provider_id: LLM provider to use.
            system_prompt: The synthesis system prompt.
            user_prompt: The synthesis user prompt.
            timeout: Request timeout in seconds.

        Returns:
            On success: ``(llm_result, findings_retries,
            total_chars_dropped, used_generic_fallback)``.
            On failure: a ``WorkflowResult`` with the error.
        """
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            _is_context_window_exceeded,
        )

        current_user_prompt = user_prompt
        findings_retries = 0
        total_findings_chars_dropped = 0
        max_findings_chars: Optional[int] = None
        used_generic_fallback = False
        result = None

        for outer_attempt in range(_MAX_FINDINGS_TRUNCATION_RETRIES + 1):  # 0 = initial
            call_result = await execute_llm_call(
                workflow=self,
                state=state,
                phase_name="synthesis",
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                provider_id=provider_id or state.synthesis_provider,
                model=state.synthesis_model,
                temperature=0.5,
                timeout=timeout,
                error_metadata={
                    "finding_count": len(state.findings),
                    "guidance": "Try reducing the number of findings or source content included",
                    "findings_retries": findings_retries,
                },
                role="report",
            )

            if isinstance(call_result, WorkflowResult):
                if _is_context_window_exceeded(call_result) and outer_attempt < _MAX_FINDINGS_TRUNCATION_RETRIES:
                    # Apply findings-specific truncation (or generic fallback).
                    (
                        current_user_prompt,
                        findings_retries,
                        total_findings_chars_dropped,
                        max_findings_chars,
                        used_generic_fallback,
                    ) = self._apply_findings_truncation(
                        state=state,
                        user_prompt=user_prompt,
                        findings_retries=findings_retries,
                        max_findings_chars=max_findings_chars,
                        used_generic_fallback=used_generic_fallback,
                    )
                    continue

                # Non-retryable error or retries exhausted
                if findings_retries > 0:
                    self._write_audit_event(
                        state,
                        "synthesis_retry_exhausted",
                        data={
                            "findings_retries": findings_retries,
                            "total_chars_dropped": total_findings_chars_dropped,
                            "used_generic_fallback": used_generic_fallback,
                            "error": call_result.error,
                        },
                        level="warning",
                    )
                return call_result

            # Success — record retry metadata if retries were needed
            if findings_retries > 0:
                self._write_audit_event(
                    state,
                    "synthesis_retry_succeeded",
                    data={
                        "findings_retries": findings_retries,
                        "total_chars_dropped": total_findings_chars_dropped,
                        "used_generic_fallback": used_generic_fallback,
                    },
                )
            result = call_result.result
            break

        if result is None:
            # All outer retries exhausted — should be caught above, safety net
            return WorkflowResult(
                success=False,
                content="",
                error="Synthesis failed: all token-limit retries exhausted",
                metadata={
                    "research_id": state.id,
                    "findings_retries": findings_retries,
                    "total_chars_dropped": total_findings_chars_dropped,
                    "used_generic_fallback": used_generic_fallback,
                },
            )

        return (result, findings_retries, total_findings_chars_dropped, used_generic_fallback)

    def _apply_findings_truncation(
        self,
        state: DeepResearchState,
        user_prompt: str,
        findings_retries: int,
        max_findings_chars: Optional[int],
        used_generic_fallback: bool,
    ) -> tuple[str, int, int, Optional[int], bool]:
        """Truncate findings for a single retry attempt and emit audit event.

        Determines the new findings character budget, applies findings-
        specific truncation, and falls back to generic prompt truncation
        when the findings section is already small enough.

        Args:
            state: Current research state (for audit events).
            user_prompt: The *original* (un-truncated) user prompt.
            findings_retries: Current retry counter (pre-increment).
            max_findings_chars: Previous findings budget (None on first retry).
            used_generic_fallback: Whether generic fallback was already used.

        Returns:
            ``(current_user_prompt, findings_retries, total_chars_dropped,
            max_findings_chars, used_generic_fallback)`` — updated state
            for the caller to thread back into the loop.
        """
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            truncate_prompt_for_retry,
        )

        findings_retries += 1

        # Determine findings char budget based on actual content
        if max_findings_chars is None:
            # First retry: 30% cut from actual findings length
            findings_section_len = _estimate_findings_section_length(user_prompt)
            max_findings_chars = int(findings_section_len * 0.7)
        else:
            # Subsequent retries: reduce by 10% from previous budget
            max_findings_chars = int(
                max_findings_chars * _FINDINGS_TRUNCATION_FACTOR,
            )

        # Try findings-specific truncation
        truncated_prompt, chars_dropped = _truncate_findings_section(
            user_prompt,  # Always truncate from original
            max_findings_chars,
        )
        total_findings_chars_dropped = (
            len(user_prompt) - len(truncated_prompt) if truncated_prompt != user_prompt else 0
        )

        if truncated_prompt != user_prompt:
            current_user_prompt = truncated_prompt
            logger.warning(
                "Synthesis findings-specific retry %d/%d: "
                "truncating findings to %d chars "
                "(dropped %d chars this pass, %d total)",
                findings_retries,
                _MAX_FINDINGS_TRUNCATION_RETRIES,
                max_findings_chars,
                chars_dropped,
                total_findings_chars_dropped,
            )
        else:
            # Findings truncation didn't help (section already small or
            # not found) — fall back to generic lifecycle-level truncation.
            used_generic_fallback = True
            current_user_prompt = truncate_prompt_for_retry(
                user_prompt,
                findings_retries,
                _MAX_FINDINGS_TRUNCATION_RETRIES,
            )
            logger.warning(
                "Synthesis findings-specific retry %d/%d: "
                "findings section already within budget, "
                "falling back to generic prompt truncation",
                findings_retries,
                _MAX_FINDINGS_TRUNCATION_RETRIES,
            )

        # Audit: truncation metrics
        self._write_audit_event(
            state,
            "synthesis_findings_truncation",
            data={
                "retry": findings_retries,
                "max_retries": _MAX_FINDINGS_TRUNCATION_RETRIES,
                "max_findings_chars": max_findings_chars,
                "chars_dropped": chars_dropped,
                "total_chars_dropped": total_findings_chars_dropped,
                "used_generic_fallback": used_generic_fallback,
                "original_prompt_len": len(user_prompt),
                "truncated_prompt_len": len(current_user_prompt),
            },
        )

        return (
            current_user_prompt,
            findings_retries,
            total_findings_chars_dropped,
            max_findings_chars,
            used_generic_fallback,
        )

    def _finalize_synthesis_report(
        self,
        state: DeepResearchState,
        result: Any,
        system_prompt: str,
        user_prompt: str,
        phase_start_time: float,
        findings_retries: int,
        total_chars_dropped: int,
        used_fallback: bool,
        degraded_mode: bool,
    ) -> WorkflowResult:
        """Extract the report, post-process citations, audit, and build the result.

        Args:
            state: Current research state.
            result: The successful LLM result object (has ``.content``,
                ``.provider_id``, etc.).
            system_prompt: The system prompt used for synthesis.
            user_prompt: The user prompt used for synthesis.
            phase_start_time: ``time.perf_counter()`` value at phase start.
            findings_retries: Number of findings-specific retries performed.
            total_chars_dropped: Total characters dropped during retries.
            used_fallback: Whether generic prompt truncation was used.
            degraded_mode: Whether synthesis ran from raw notes.

        Returns:
            WorkflowResult with the final synthesis report.
        """
        # Extract the markdown report from the response
        report = self._extract_markdown_report(result.content)

        if not report:
            logger.warning("Failed to extract report from synthesis response")
            # Use raw content as fallback
            report = result.content

        # Post-process citations: remove dangling refs, append Sources section
        report, citation_metadata = postprocess_citations(report, state)

        # Store report in state
        state.report = report

        # PLAN-3: Build research landscape metadata (pure data transformation)
        try:
            landscape = self._build_research_landscape(state)
            state.extensions.research_landscape = landscape
            logger.info(
                "Built research landscape: %d timeline entries, %d venues, %d top-cited papers",
                len(landscape.timeline),
                len(landscape.venue_distribution),
                len(landscape.top_cited_papers),
            )
        except Exception:
            logger.warning("Failed to build research landscape", exc_info=True)

        # Auto-save report as markdown file
        output_path = _save_report_markdown(state)
        if output_path:
            state.report_output_path = output_path

        # Save state
        self.memory.save_deep_research(state)
        synthesis_audit_data: dict[str, Any] = {
            "provider_id": result.provider_id,
            "model_used": result.model_used,
            "tokens_used": result.tokens_used,
            "duration_ms": result.duration_ms,
            "report_length": len(state.report),
            "citation_postprocess": citation_metadata,
            "degraded_mode": degraded_mode,
        }
        if self.config.audit_verbosity == "full":
            synthesis_audit_data["system_prompt"] = system_prompt
            synthesis_audit_data["user_prompt"] = user_prompt
            synthesis_audit_data["raw_response"] = result.content
            synthesis_audit_data["report"] = state.report
        else:
            synthesis_audit_data["system_prompt_length"] = len(system_prompt)
            synthesis_audit_data["user_prompt_length"] = len(user_prompt)
            synthesis_audit_data["raw_response_length"] = len(result.content)
        self._write_audit_event(
            state,
            "synthesis_result",
            data=synthesis_audit_data,
        )

        logger.info(
            "Synthesis phase complete: report length %d chars",
            len(state.report),
        )

        finalize_phase(self, state, "synthesis", phase_start_time)

        return WorkflowResult(
            success=True,
            content=state.report,
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "finding_count": len(state.findings),
                "source_count": len(state.sources),
                "report_length": len(state.report),
                "iteration": state.iteration,
                "degraded_mode": degraded_mode,
            },
        )

    # ------------------------------------------------------------------
    # PLAN-3: Research landscape builder
    # ------------------------------------------------------------------

    def _build_research_landscape(self, state: DeepResearchState) -> ResearchLandscape:
        """Extract structured landscape metadata from research sources.

        Pure data transformation — iterates state.sources, aggregates by year,
        venue, field, citation count, and author. No additional API or LLM calls.
        """
        timeline_data: dict[int, dict[str, Any]] = {}
        venue_dist: dict[str, int] = {}
        field_dist: dict[str, int] = {}
        author_freq: dict[str, int] = {}
        source_type_count: dict[str, int] = {}
        cited_papers: list[dict[str, Any]] = []

        for source in state.sources:
            # Source type breakdown
            st = source.source_type.value if hasattr(source.source_type, "value") else str(source.source_type)
            source_type_count[st] = source_type_count.get(st, 0) + 1

            meta = source.metadata or {}

            # Year-based timeline
            year = meta.get("year")
            if year and isinstance(year, int):
                if year not in timeline_data:
                    timeline_data[year] = {"year": year, "count": 0, "key_papers": []}
                timeline_data[year]["count"] += 1
                citation_count = meta.get("citation_count", 0) or 0
                if citation_count >= 10:
                    timeline_data[year]["key_papers"].append(
                        {"title": source.title, "citation_count": citation_count}
                    )

            # Venue distribution
            venue = meta.get("venue") or meta.get("journal")
            if venue and isinstance(venue, str) and venue.strip():
                venue_dist[venue.strip()] = venue_dist.get(venue.strip(), 0) + 1

            # Field distribution
            fields = meta.get("fields_of_study") or meta.get("fields") or []
            if isinstance(fields, list):
                for f in fields:
                    if isinstance(f, str) and f.strip():
                        field_dist[f.strip()] = field_dist.get(f.strip(), 0) + 1

            # Author frequency
            authors = meta.get("authors") or []
            if isinstance(authors, list):
                for author in authors:
                    name = author.get("name", author) if isinstance(author, dict) else str(author)
                    if name and isinstance(name, str) and name.strip():
                        author_freq[name.strip()] = author_freq.get(name.strip(), 0) + 1

            # Collect papers with citation counts for top-cited ranking
            citation_count = meta.get("citation_count")
            if citation_count is not None and isinstance(citation_count, (int, float)):
                author_names = []
                for a in (meta.get("authors") or []):
                    if isinstance(a, dict):
                        author_names.append(a.get("name", ""))
                    elif isinstance(a, str):
                        author_names.append(a)
                cited_papers.append({
                    "title": source.title,
                    "authors": ", ".join(author_names),
                    "year": meta.get("year"),
                    "citation_count": int(citation_count),
                    "doi": meta.get("doi", ""),
                })

        # Sort timeline ascending by year
        timeline = sorted(timeline_data.values(), key=lambda x: x["year"])
        # Sort key_papers descending within each year
        for entry in timeline:
            entry["key_papers"] = sorted(
                entry["key_papers"], key=lambda x: x.get("citation_count", 0), reverse=True
            )[:5]

        # Top-cited papers descending
        top_cited = sorted(cited_papers, key=lambda x: x.get("citation_count", 0), reverse=True)[:20]

        # Sort author frequency descending
        sorted_authors = dict(sorted(author_freq.items(), key=lambda x: x[1], reverse=True)[:30])

        return ResearchLandscape(
            timeline=timeline,
            venue_distribution=venue_dist,
            field_distribution=field_dist,
            top_cited_papers=top_cited,
            author_frequency=sorted_authors,
            source_type_breakdown=source_type_count,
        )

    def _build_synthesis_system_prompt(self, state: DeepResearchState) -> str:
        """Build system prompt for report synthesis.

        The prompt is state-aware: it detects query language and adapts
        structural guidance based on query type.

        Args:
            state: Current research state used for language and structure hints

        Returns:
            System prompt string
        """
        query_type = _classify_query_type(state.original_query)
        structure_guidance = _STRUCTURE_GUIDANCE.get(query_type, _STRUCTURE_GUIDANCE["explanation"])

        base_prompt = f"""You are a research synthesizer. Your task is to create a comprehensive, well-structured research report from analyzed findings.

## Report Structure

Select a report structure suited to the query type. A structural hint is provided in the user prompt.

{structure_guidance}

These are suggestions. Section is a fluid concept — you can structure your report however you think is best, including in ways not listed above. Make sure sections are cohesive and make sense for the reader.

Include analysis of Conflicting Information and Limitations where they exist, but integrate them naturally into the relevant sections rather than forcing separate subsections. A Conclusions section with actionable insights is always valuable.

## Section Writing Rules

- Use ## for each section title (Markdown format). Consistent heading levels enable downstream rendering and table-of-contents generation.
- Write in paragraph form by default; use bullet points only when listing discrete items. Paragraph form supports nuanced argumentation and flowing analysis; bullet points fragment reasoning into disconnected pieces that lose causal connections.
- Each section should be as long as necessary to deeply answer the question with the information gathered. Sections are expected to be thorough and detailed. You are writing a deep research report and users expect comprehensive answers.
- Do not refer to yourself or comment on the report itself — just write the report.

## Writing Quality

- Write directly and authoritatively. Do not hedge with openers like "it appears that", "it seems", or "based on available information". Hedging undermines reader trust in findings that may actually be well-supported — if confidence is genuinely low, express that through explicit caveats tied to evidence, not reflexive hedging.
- Never use meta-commentary about the report itself ("based on the research", "the findings show", "this report examines"). Meta-commentary wastes space and breaks reading flow — the user wants the content, not commentary about the content.
- Never refer to yourself ("as an AI", "I found that", "in my analysis"). Self-reference breaks the illusion of an authoritative research report and distracts from the findings.
- Use clear, professional language.
- Include all relevant findings. Do not omit information for brevity.

## Citations

- For the **first reference** to each source, use an inline markdown link followed by its citation number: `[Title](URL) [N]`. This makes sources immediately navigable in rendered markdown.
- For **subsequent references** to the same source, use just the citation number: `[N]`.
- The citation numbers correspond to the numbered sources provided in the input.
- Do NOT generate a Sources or References section — it will be appended automatically.
- Distinguish between high-confidence findings (well-supported) and lower-confidence insights.
- Citations are extremely important. Pay careful attention to getting these right. Users will often use citations to find more information on specific points.

## Language

CRITICAL: The report MUST be written in the same language as the original research query.
If the query is in English, write in English. If the query is in Chinese, write entirely in Chinese. If in Spanish, write entirely in Spanish.
The research and findings may be in English, but you must translate information to match the query language.

IMPORTANT: Return ONLY the markdown report, no preamble or meta-commentary."""

        # PLAN-3: Add academic-specific synthesis instructions
        if state.research_mode == ResearchMode.ACADEMIC:
            base_prompt += """

## Study Comparison Table (Academic Mode)

If the research involves multiple empirical studies, include a markdown comparison table
in the "Methodological Approaches" section with columns:

| Study | Year | Method | Sample | Key Finding | Effect Size | Limitations |
|-------|------|--------|--------|-------------|-------------|-------------|

Only include this table when there are 3+ empirical studies with sufficient methodological
detail. Populate from the findings provided — do not invent data. Use "Not reported" for
missing values rather than omitting the study.

## Research Gaps & Future Directions (Academic Mode)

For the "Research Gaps & Future Directions" section:
- Base this on the identified research gaps provided in the input
- Distinguish between completely unexplored areas and partially addressed topics
- For each gap, suggest specific research questions or methodological approaches
- Prioritize gaps by their potential impact on the field"""

        return base_prompt

    def _build_synthesis_user_prompt(
        self,
        state: DeepResearchState,
        allocation_result: Optional[AllocationResult] = None,
        *,
        degraded_mode: bool = False,
    ) -> str:
        """Build user prompt with findings and sources for synthesis.

        Args:
            state: Current research state
            allocation_result: Optional budget allocation result for token-aware prompts
            degraded_mode: When True, build prompt from raw_notes because no
                compressed findings are available (Phase 3b ODR alignment).

        Returns:
            User prompt string
        """
        # Build source_id → citation_number mapping for inline references
        id_to_citation = state.source_id_to_citation()

        prompt_parts = [
            f"# Research Query\n{sanitize_external_content(state.original_query)}",
            "",
            f"## Research Brief\n{sanitize_external_content(state.research_brief or 'Direct research on the query')}",
            "",
        ]

        # Phase 3b (ODR alignment): Degraded mode — synthesize directly
        # from raw notes when no compressed findings exist.
        if degraded_mode and state.raw_notes:
            prompt_parts.extend(
                [
                    "## Research Notes (uncompressed)",
                    "",
                    "**Note:** Compressed findings were not available for this session. "
                    "The following are uncompressed research notes from topic researchers. "
                    "Synthesize a comprehensive report from these notes, extracting key "
                    "findings, identifying themes, and noting any gaps.",
                    "",
                ]
            )
            # Join raw notes with separators, truncate to fit context
            context_window = (
                estimate_token_limit_for_model(
                    state.synthesis_model,
                    get_model_token_limits(),
                )
                or FALLBACK_CONTEXT_WINDOW
            )
            # Reserve space for the rest of the prompt and output
            max_notes_tokens = int(context_window * 0.60)
            raw_notes_text = "\n---\n".join(sanitize_external_content(note) for note in state.raw_notes)
            truncated = truncate_at_boundary(raw_notes_text, max_notes_tokens * 4)
            prompt_parts.append(truncated)
            prompt_parts.append("")

            return self._build_synthesis_tail(
                state,
                prompt_parts,
                id_to_citation,
                allocation_result,
            )

        # When a global compressed digest is available, use it as the
        # primary findings source.  The digest already contains deduplicated
        # cross-topic findings with consistent citations, contradictions,
        # and gaps — so we can skip the per-finding enumeration below.
        if state.compressed_digest:
            prompt_parts.extend(
                [
                    "## Unified Research Digest",
                    "",
                    sanitize_external_content(state.compressed_digest),
                    "",
                ]
            )
            # Still include the source reference and instructions below
            # (skip to source reference section)
            return self._build_synthesis_tail(
                state,
                prompt_parts,
                id_to_citation,
                allocation_result,
            )

        # ---------------------------------------------------------------
        # Phase 3 PLAN: Direct-from-compressed-findings path.
        #
        # When the ANALYSIS phase is skipped (collapsed pipeline), there
        # are no analysis findings but per-topic compressed_findings exist.
        # Build the synthesis prompt from those directly.
        # ---------------------------------------------------------------
        compressed_topics = [tr for tr in state.topic_research_results if tr.compressed_findings]
        if not state.findings and compressed_topics:
            prompt_parts.extend(
                [
                    "## Research Findings by Topic",
                    "",
                ]
            )
            for tr in compressed_topics:
                # Resolve sub-query text for section header
                sq = state.get_sub_query(tr.sub_query_id)
                topic_label = sanitize_external_content(sq.query) if sq else tr.sub_query_id
                prompt_parts.append(f"### {topic_label}")
                prompt_parts.append("")
                prompt_parts.append(sanitize_external_content(tr.compressed_findings or ""))
                prompt_parts.append("")

            # Add contradictions and gaps (may still be populated from supervision)
            self._append_contradictions_and_gaps(state, prompt_parts, id_to_citation)

            return self._build_synthesis_tail(
                state,
                prompt_parts,
                id_to_citation,
                allocation_result,
            )

        # ---------------------------------------------------------------
        # Standard path: build from analysis findings.
        # ---------------------------------------------------------------
        prompt_parts.extend(
            [
                "## Findings to Synthesize",
                "",
            ]
        )

        # Group findings by category if available
        categorized: dict[str, list] = {}

        for finding in state.findings:
            category = finding.category or "General"
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(finding)

        # Add findings by category - findings are protected, always included at full fidelity
        for category, findings in categorized.items():
            prompt_parts.append(f"### {category}")
            for f in findings:
                confidence_label = f.confidence.value if hasattr(f.confidence, "value") else str(f.confidence)
                # Map source IDs to citation numbers
                citation_refs = [f"[{id_to_citation[sid]}]" for sid in f.source_ids if sid in id_to_citation]
                source_refs = ", ".join(citation_refs) if citation_refs else "no sources"
                prompt_parts.append(f"- [{confidence_label.upper()}] {f.content}")
                prompt_parts.append(f"  Sources: {source_refs}")
            prompt_parts.append("")

        # Add detected contradictions and knowledge gaps
        self._append_contradictions_and_gaps(state, prompt_parts, id_to_citation)

        return self._build_synthesis_tail(
            state,
            prompt_parts,
            id_to_citation,
            allocation_result,
        )

    def _append_contradictions_and_gaps(
        self,
        state: DeepResearchState,
        prompt_parts: list[str],
        id_to_citation: dict[str, int],
    ) -> None:
        """Append contradiction and gap sections to the synthesis prompt.

        Shared by both the compressed-findings path and the standard
        findings path.

        Args:
            state: Current research state
            prompt_parts: Accumulated prompt sections (mutated in-place)
            id_to_citation: source-id → citation-number mapping
        """
        if state.contradictions:
            prompt_parts.append("## Contradictions Detected")
            prompt_parts.append(
                "The following contradictions were identified between findings. "
                "Address these explicitly in the report's 'Conflicting Information' section."
            )
            for contradiction in state.contradictions:
                severity_label = sanitize_external_content(contradiction.severity).upper()
                prompt_parts.append(f"- [{severity_label}] {sanitize_external_content(contradiction.description)}")
                prompt_parts.append(f"  Conflicting findings: {', '.join(contradiction.finding_ids)}")
                if contradiction.resolution:
                    prompt_parts.append(
                        f"  Suggested resolution: {sanitize_external_content(contradiction.resolution)}"
                    )
                if contradiction.preferred_source_id:
                    cn = id_to_citation.get(contradiction.preferred_source_id)
                    if cn is not None:
                        prompt_parts.append(f"  Preferred source: [{cn}]")
            prompt_parts.append("")

        # PLAN-3: Enhanced gap injection for academic queries
        is_academic = state.research_mode == ResearchMode.ACADEMIC
        unresolved_gaps = [g for g in state.gaps if not g.resolved]
        resolved_gaps = [g for g in state.gaps if g.resolved]

        if unresolved_gaps and is_academic:
            prompt_parts.append("## Identified Research Gaps (from iterative analysis)")
            for i, gap in enumerate(unresolved_gaps, 1):
                prompt_parts.append(f"{i}. {sanitize_external_content(gap.description)} (priority: {gap.priority})")
            prompt_parts.append(
                "\nIncorporate these gaps into a 'Research Gaps & Future Directions' "
                "section. Frame them constructively — what specific studies or "
                "methodologies would address each gap?"
            )
            prompt_parts.append("")

            if resolved_gaps:
                prompt_parts.append("## Partially Addressed Gaps")
                for gap in resolved_gaps:
                    resolution = sanitize_external_content(gap.resolution_notes or "No details")
                    prompt_parts.append(
                        f"- {sanitize_external_content(gap.description)} — Addressed by: {resolution}"
                    )
                prompt_parts.append("")
        elif state.gaps:
            # Non-academic: original compact format
            prompt_parts.append("## Knowledge Gaps Identified")
            for gap in state.gaps:
                status = "addressed" if gap.resolved else "unresolved"
                prompt_parts.append(f"- [{status}] {sanitize_external_content(gap.description)}")
            prompt_parts.append("")

    def _build_synthesis_tail(
        self,
        state: DeepResearchState,
        prompt_parts: list[str],
        id_to_citation: dict[str, int],
        allocation_result: Optional[AllocationResult] = None,
    ) -> str:
        """Append source reference and instructions to the synthesis prompt.

        Shared by both the standard findings path and the compressed-digest
        path in ``_build_synthesis_user_prompt``.

        Args:
            state: Current research state
            prompt_parts: Accumulated prompt sections (mutated in-place)
            id_to_citation: source-id → citation-number mapping
            allocation_result: Optional budget allocation result

        Returns:
            Complete user prompt string
        """
        # Add source reference list with citation numbers - use allocation-aware content
        prompt_parts.append("## Source Reference (use these citation numbers in your report)")

        if allocation_result:
            # Use allocated sources in priority order, applying token limits
            for item in allocation_result.items:
                # Skip findings (they're in the findings section)
                if not item.id.startswith("src-"):
                    continue

                source = next((s for s in state.sources if s.id == item.id), None)
                if not source:
                    continue

                cn = source.citation_number
                label = f"[{cn}]" if cn is not None else f"[{source.id}]"
                quality = source.quality.value if hasattr(source.quality, "value") else str(source.quality)
                safe_title = sanitize_external_content(source.title)
                prompt_parts.append(f"- **{label}**: {safe_title} [{quality}]")
                if source.url:
                    prompt_parts.append(f"  URL: {sanitize_external_content(source.url)}")

                # Apply token-aware content limit for snippets
                if item.needs_summarization:
                    # Compressed: use allocated tokens to estimate character limit (~4 chars/token)
                    char_limit = max(50, item.allocated_tokens * 4)
                    if source.snippet:
                        safe_snippet = sanitize_external_content(source.snippet)
                        snippet = safe_snippet[:char_limit]
                        if len(safe_snippet) > char_limit:
                            snippet += "..."
                        prompt_parts.append(f"  Snippet: {snippet}")
                else:
                    # Full fidelity: include snippet up to 200 chars
                    if source.snippet:
                        safe_snippet = sanitize_external_content(source.snippet)
                        snippet = safe_snippet[:200]
                        if len(safe_snippet) > 200:
                            snippet += "..."
                        prompt_parts.append(f"  Snippet: {snippet}")

            # Note dropped sources if any
            if allocation_result.dropped_ids:
                dropped_sources = [sid for sid in allocation_result.dropped_ids if sid.startswith("src-")]
                if dropped_sources:
                    prompt_parts.append(
                        f"\n*Note: {len(dropped_sources)} additional source(s) omitted for context limits*"
                    )
        else:
            # Fallback: use first 30 sources (legacy behavior)
            for source in state.sources[:30]:
                cn = source.citation_number
                label = f"[{cn}]" if cn is not None else f"[{source.id}]"
                quality = source.quality.value if hasattr(source.quality, "value") else str(source.quality)
                safe_title = sanitize_external_content(source.title)
                prompt_parts.append(f"- {label}: {safe_title} [{quality}]")
                if source.url:
                    prompt_parts.append(f"  URL: {sanitize_external_content(source.url)}")

        prompt_parts.append("")

        # Add synthesis instructions with query-type structural hint
        query_type = _classify_query_type(state.original_query)
        query_type_labels = {
            "comparison": "comparison (side-by-side analysis of alternatives)",
            "enumeration": "list/enumeration (discrete items or options)",
            "howto": "how-to (step-by-step procedural guide)",
            "explanation": "explanation/overview (topical deep-dive)",
        }
        type_label = query_type_labels.get(query_type, query_type)

        prompt_parts.extend(
            [
                "## Instructions",
                f"Generate a comprehensive research report addressing the query: '{sanitize_external_content(state.original_query)}'",
                "",
                f"**Query type hint:** {type_label} — adapt the report structure accordingly.",
                "",
                f"This is iteration {state.iteration} of {state.max_iterations}.",
                f"Total findings: {len(state.findings)}",
                f"Total sources: {len(state.sources)}",
                f"Unresolved gaps: {len(state.unresolved_gaps())}",
                "",
                "Create a well-structured markdown report following the format specified.",
            ]
        )

        # Phase 3 (ODR alignment): Inject supplementary raw notes when
        # token headroom permits.  This gives the synthesizer access to
        # uncompressed researcher evidence that may have been lost during
        # compression, improving report depth without risking token-limit
        # failures.
        prompt = "\n".join(prompt_parts)
        prompt = self._inject_supplementary_raw_notes(state, prompt)
        return prompt

    def _inject_supplementary_raw_notes(
        self,
        state: DeepResearchState,
        prompt: str,
    ) -> str:
        """Append supplementary raw notes when token headroom allows.

        Calculates remaining token budget after the primary synthesis
        prompt and injects a ``## Supplementary Research Notes`` section
        with truncated raw notes if headroom exceeds the configured
        threshold.  This matches the ODR pattern where ``raw_notes``
        provides a safety net of uncompressed evidence for synthesis.

        Args:
            state: Current research state (with ``raw_notes`` populated
                by Phase 1).
            prompt: The already-built synthesis user prompt.

        Returns:
            Prompt with supplementary section appended, or the original
            prompt if no raw notes exist or headroom is insufficient.
        """
        if not state.raw_notes:
            return prompt

        # Determine context window size for the synthesis model, reduced by
        # the configured safety margin so the final prompt stays within the
        # safe token budget (mirrors the margin applied elsewhere).
        raw_context_window = (
            estimate_token_limit_for_model(
                state.synthesis_model,
                get_model_token_limits(),
            )
            or FALLBACK_CONTEXT_WINDOW
        )
        safety_margin = getattr(self.config, "token_safety_margin", 0.15)
        context_window = int(raw_context_window * (1.0 - safety_margin))

        # Estimate current prompt size in tokens (4 chars ≈ 1 token heuristic)
        current_tokens = len(prompt) // 4

        # Estimate the system prompt token count so we don't over-allocate
        # headroom. The system prompt occupies context-window space alongside
        # the user prompt but was previously ignored in this calculation.
        system_prompt = self._build_synthesis_system_prompt(state)
        system_prompt_tokens = len(system_prompt) // 4

        # Available headroom = effective_context - system_prompt - current_tokens - output_reserved
        headroom_tokens = context_window - system_prompt_tokens - current_tokens - SYNTHESIS_OUTPUT_RESERVED
        headroom_fraction = headroom_tokens / context_window

        if headroom_fraction < _SUPPLEMENTARY_HEADROOM_THRESHOLD:
            logger.debug(
                "Skipping supplementary raw notes: headroom %.1f%% < threshold %.1f%%",
                headroom_fraction * 100,
                _SUPPLEMENTARY_HEADROOM_THRESHOLD * 100,
            )
            return prompt

        # Cap supplementary budget at _SUPPLEMENTARY_MAX_FRACTION of context
        max_supplementary_tokens = min(
            headroom_tokens,
            int(context_window * _SUPPLEMENTARY_MAX_FRACTION),
        )

        # Build the raw notes text (sanitize web-sourced content)
        raw_notes_text = "\n---\n".join(sanitize_external_content(note) for note in state.raw_notes)
        supplementary = truncate_at_boundary(
            raw_notes_text,
            max_supplementary_tokens * 4,  # tokens → chars
        )

        logger.info(
            "Injecting supplementary raw notes: %d chars (headroom %.1f%%, budget %d tokens)",
            len(supplementary),
            headroom_fraction * 100,
            max_supplementary_tokens,
        )

        return (
            prompt
            + "\n\n## Supplementary Research Notes\n"
            + "The following are uncompressed research notes from topic researchers. "
            + "Prefer the compressed findings above for accuracy; use these notes "
            + "for additional detail, specific data points, or evidence not captured "
            + "in the compressed summaries.\n\n"
            + supplementary
        )

    def _extract_markdown_report(self, content: str) -> Optional[str]:
        """Extract markdown report from LLM response.

        The response should be pure markdown, but this handles cases where
        the LLM wraps it in code blocks or adds preamble.

        Args:
            content: Raw LLM response content

        Returns:
            Extracted markdown report or None if extraction fails
        """
        if not content:
            return None

        # If content starts with markdown heading, it's likely clean
        if content.strip().startswith("#"):
            return content.strip()

        # Check for markdown code block wrapper
        if "```markdown" in content or "```md" in content:
            # Extract content between code blocks
            pattern = r"```(?:markdown|md)?\s*([\s\S]*?)```"
            matches = re.findall(pattern, content)
            if matches:
                return matches[0].strip()

        # Check for generic code block
        if "```" in content:
            pattern = r"```\s*([\s\S]*?)```"
            matches = re.findall(pattern, content)
            for match in matches:
                # Check if it looks like markdown (has headings)
                if match.strip().startswith("#") or "##" in match:
                    return match.strip()

        # Look for first heading and take everything from there
        heading_match = re.search(r"^(#[^\n]+)", content, re.MULTILINE)
        if heading_match:
            start_pos = heading_match.start()
            return content[start_pos:].strip()

        # If nothing else works, return the trimmed content
        return content.strip() if len(content.strip()) > 50 else None

    def _generate_empty_report(self, state: DeepResearchState) -> str:
        """Generate a minimal report when no findings are available.

        Args:
            state: Current research state

        Returns:
            Minimal markdown report
        """
        safe_query = sanitize_external_content(state.original_query)
        safe_brief = sanitize_external_content(state.research_brief or "No research brief generated.")
        return f"""# Research Report

## Executive Summary

Research was conducted on the query: "{safe_query}"

Unfortunately, the analysis phase did not yield extractable findings from the gathered sources. This may indicate:
- The sources lacked relevant information
- The query may need refinement
- Additional research iterations may be needed

## Research Query

{safe_query}

## Research Brief

{safe_brief}

## Sources Examined

{len(state.sources)} source(s) were examined during this research session.

## Recommendations

1. Consider refining the research query for more specific results
2. Try additional research iterations if available
3. Review the gathered sources manually for relevant information

---

*Report generated with no extractable findings. Iteration {state.iteration}/{state.max_iterations}.*
"""
