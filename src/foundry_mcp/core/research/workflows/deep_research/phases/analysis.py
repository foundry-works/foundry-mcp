"""Analysis phase mixin for DeepResearchWorkflow.

Extracts findings from gathered sources via LLM analysis, with a digest
pipeline that extracts, ranks, selects, and digests source content before
the main analysis call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.observability import get_metrics
from foundry_mcp.core.providers import ContextWindowError
from foundry_mcp.core.research.context_budget import AllocationResult
from foundry_mcp.core.research.document_digest import (
    DigestConfig,
    DigestPolicy,
    DigestResult,
    DocumentDigestor,
    deserialize_payload,
    serialize_payload,
)
from foundry_mcp.core.research.models import (
    ConfidenceLevel,
    DeepResearchState,
    FidelityLevel,
    PhaseMetrics,
    ResearchSource,
    SourceQuality,
)
from foundry_mcp.core.research.pdf_extractor import PDFExtractor
from foundry_mcp.core.research.summarization import ContentSummarizer
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._budgeting import (
    allocate_source_budget,
    archive_digest_source,
    final_fit_validate,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (
    ANALYSIS_OUTPUT_RESERVED,
)
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    extract_json,
    fidelity_level_from_score,
)

if TYPE_CHECKING:
    from foundry_mcp.core.research.workflows.deep_research._monolith import (
        DeepResearchWorkflow,
    )

logger = logging.getLogger(__name__)


class AnalysisPhaseMixin:
    """Analysis phase methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)
    """

    async def _execute_analysis_async(
        self: DeepResearchWorkflow,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute analysis phase: extract findings from sources.

        This phase:
        1. Builds prompt with gathered source summaries
        2. Uses LLM to extract key findings
        3. Assesses confidence levels for each finding
        4. Identifies knowledge gaps requiring follow-up
        5. Updates source quality assessments

        Args:
            state: Current research state with gathered sources
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with analysis outcome
        """
        if not state.sources:
            logger.warning("No sources to analyze")
            return WorkflowResult(
                success=True,
                content="No sources to analyze",
                metadata={"research_id": state.id, "finding_count": 0},
            )

        logger.info(
            "Starting analysis phase: %d sources to analyze",
            len(state.sources),
        )

        # Emit phase.started audit event
        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "analysis",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        # Execute digest step: extract content, rank, select, and digest sources
        # This step runs BEFORE budget allocation to ensure digested content is used
        # for token counting and allocation decisions
        digest_stats = await self._execute_digest_step_async(
            state=state,
            query=state.original_query,
        )

        # Record digest statistics in state metadata
        if digest_stats["sources_digested"] > 0:
            state.metadata = state.metadata or {}
            state.metadata["digest_stats"] = digest_stats
            self._write_audit_event(
                state,
                "digest.completed",
                data={
                    "sources_extracted": digest_stats["sources_extracted"],
                    "sources_ranked": digest_stats["sources_ranked"],
                    "sources_selected": digest_stats["sources_selected"],
                    "sources_digested": digest_stats["sources_digested"],
                    "errors": len(digest_stats["digest_errors"]),
                },
            )

        # Allocate token budget for sources
        allocation_result = allocate_source_budget(
            state=state,
            provider_id=provider_id,
        )

        # Update state with allocation metadata
        # Store overall fidelity in metadata (content_fidelity is now per-item dict)
        state.dropped_content_ids = allocation_result.dropped_ids
        allocation_dict = allocation_result.to_dict()
        allocation_dict["overall_fidelity_level"] = fidelity_level_from_score(
            allocation_result.fidelity
        )
        state.content_allocation_metadata = allocation_dict

        logger.info(
            "Budget allocation: %d sources allocated, %d dropped, fidelity=%.1f%%",
            len(allocation_result.items),
            len(allocation_result.dropped_ids),
            allocation_result.fidelity * 100,
        )

        # Build the analysis prompt with allocated sources
        system_prompt = self._build_analysis_system_prompt(state)
        user_prompt = self._build_analysis_user_prompt(state, allocation_result)

        # Final-fit validation before provider dispatch
        valid, _preflight, system_prompt, user_prompt = final_fit_validate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.analysis_provider,
            model=state.analysis_model,
            output_reserved=ANALYSIS_OUTPUT_RESERVED,
            phase="analysis",
        )

        if not valid:
            logger.warning(
                "Analysis phase final-fit validation failed, proceeding with truncated prompts"
            )

        # Check for cancellation before making provider call
        self._check_cancellation(state)

        # Execute LLM call with context window error handling and timeout protection
        effective_provider = provider_id or state.analysis_provider
        llm_call_start_time = time.perf_counter()
        # Update heartbeat and persist interim state for progress visibility
        state.last_heartbeat_at = datetime.now(timezone.utc)
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "llm.call.started",
            data={
                "provider": effective_provider,
                "task_id": state.id,
                "phase": "analysis",
            },
        )
        try:
            result = await self._execute_provider_async(
                prompt=user_prompt,
                provider_id=effective_provider,
                model=state.analysis_model,
                system_prompt=system_prompt,
                timeout=timeout,
                temperature=0.3,  # Lower temperature for analytical tasks
                phase="analysis",
                fallback_providers=self.config.get_phase_fallback_providers("analysis"),
                max_retries=self.config.deep_research_max_retries,
                retry_delay=self.config.deep_research_retry_delay,
            )
        except ContextWindowError as e:
            llm_call_duration_ms = (time.perf_counter() - llm_call_start_time) * 1000
            self._write_audit_event(
                state,
                "llm.call.completed",
                data={
                    "provider": effective_provider,
                    "task_id": state.id,
                    "duration_ms": llm_call_duration_ms,
                    "status": "error",
                    "error_type": "context_window_exceeded",
                },
            )
            get_metrics().histogram(
                "foundry_mcp_research_llm_call_duration_seconds",
                llm_call_duration_ms / 1000.0,
                labels={"provider": effective_provider or "unknown", "status": "error"},
            )
            logger.error(
                "Analysis phase context window exceeded: prompt_tokens=%s, "
                "max_tokens=%s, truncation_needed=%s, provider=%s, source_count=%d",
                e.prompt_tokens,
                e.max_tokens,
                e.truncation_needed,
                e.provider,
                len(state.sources),
            )
            return WorkflowResult(
                success=False,
                content="",
                error=str(e),
                metadata={
                    "research_id": state.id,
                    "phase": "analysis",
                    "error_type": "context_window_exceeded",
                    "prompt_tokens": e.prompt_tokens,
                    "max_tokens": e.max_tokens,
                    "truncation_needed": e.truncation_needed,
                    "source_count": len(state.sources),
                    "guidance": "Try reducing max_sources_per_query or processing sources in batches",
                },
            )

        # Emit llm.call.completed audit event
        llm_call_duration_ms = (time.perf_counter() - llm_call_start_time) * 1000
        llm_call_status = "success" if result.success else "error"
        llm_call_provider: str = result.provider_id or effective_provider or "unknown"
        self._write_audit_event(
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

        if not result.success:
            # Check if this was a timeout
            if result.metadata and result.metadata.get("timeout"):
                logger.error(
                    "Analysis phase timed out after exhausting all providers: %s",
                    result.metadata.get("providers_tried", []),
                )
            else:
                logger.error("Analysis phase LLM call failed: %s", result.error)
            return result

        # Track token usage
        if result.tokens_used:
            state.total_tokens_used += result.tokens_used

        # Track phase metrics for audit
        state.phase_metrics.append(
            PhaseMetrics(
                phase="analysis",
                duration_ms=result.duration_ms or 0.0,
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
                cached_tokens=result.cached_tokens or 0,
                provider_id=result.provider_id,
                model_used=result.model_used,
            )
        )

        # Parse the response
        parsed = self._parse_analysis_response(result.content, state)

        if not parsed["success"]:
            logger.warning("Failed to parse analysis response")
            self._write_audit_event(
                state,
                "analysis_result",
                data={
                    "provider_id": result.provider_id,
                    "model_used": result.model_used,
                    "tokens_used": result.tokens_used,
                    "duration_ms": result.duration_ms,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": result.content,
                    "parse_success": False,
                    "findings": [],
                    "gaps": [],
                    "quality_updates": [],
                },
                level="warning",
            )
            # Still mark as success but with no findings
            return WorkflowResult(
                success=True,
                content="Analysis completed but no findings extracted",
                metadata={
                    "research_id": state.id,
                    "finding_count": 0,
                    "parse_error": True,
                },
            )

        # Add findings to state
        for finding_data in parsed["findings"]:
            state.add_finding(
                content=finding_data["content"],
                confidence=finding_data["confidence"],
                source_ids=finding_data.get("source_ids", []),
                category=finding_data.get("category"),
            )

        # Add gaps to state
        for gap_data in parsed["gaps"]:
            state.add_gap(
                description=gap_data["description"],
                suggested_queries=gap_data.get("suggested_queries", []),
                priority=gap_data.get("priority", 1),
            )

        # Update source quality assessments
        for quality_update in parsed.get("quality_updates", []):
            source = state.get_source(quality_update["source_id"])
            if source:
                try:
                    source.quality = SourceQuality(quality_update["quality"])
                except ValueError:
                    pass  # Invalid quality value, skip

        # Save state
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "analysis_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": result.content,
                "parse_success": True,
                "findings": parsed["findings"],
                "gaps": parsed["gaps"],
                "quality_updates": parsed.get("quality_updates", []),
            },
        )

        logger.info(
            "Analysis phase complete: %d findings, %d gaps identified",
            len(parsed["findings"]),
            len(parsed["gaps"]),
        )

        # Emit phase.completed audit event
        phase_duration_ms = (time.perf_counter() - phase_start_time) * 1000
        self._write_audit_event(
            state,
            "phase.completed",
            data={
                "phase_name": "analysis",
                "iteration": state.iteration,
                "task_id": state.id,
                "duration_ms": phase_duration_ms,
            },
        )

        # Emit phase duration metric
        get_metrics().histogram(
            "foundry_mcp_research_phase_duration_seconds",
            phase_duration_ms / 1000.0,
            labels={"phase_name": "analysis", "status": "success"},
        )

        return WorkflowResult(
            success=True,
            content=f"Extracted {len(parsed['findings'])} findings and identified {len(parsed['gaps'])} gaps",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "finding_count": len(parsed["findings"]),
                "gap_count": len(parsed["gaps"]),
                "source_count": len(state.sources),
            },
        )

    def _build_analysis_system_prompt(self, state: DeepResearchState) -> str:
        """Build system prompt for source analysis.

        Args:
            state: Current research state (reserved for future state-aware prompts)

        Returns:
            System prompt string
        """
        # state is reserved for future state-aware prompt customization
        _ = state
        return """You are a research analyst. Your task is to analyze research sources and extract key findings, assess their quality, and identify knowledge gaps.

Your response MUST be valid JSON with this exact structure:
{
    "findings": [
        {
            "content": "A clear, specific finding or insight extracted from the sources",
            "confidence": "low|medium|high",
            "source_ids": ["src-xxx", "src-yyy"],
            "category": "optional category/theme"
        }
    ],
    "gaps": [
        {
            "description": "Description of missing information or unanswered question",
            "suggested_queries": ["follow-up query 1", "follow-up query 2"],
            "priority": 1
        }
    ],
    "quality_updates": [
        {
            "source_id": "src-xxx",
            "quality": "low|medium|high"
        }
    ]
}

Guidelines for findings:
- Extract 2-5 key findings from the sources
- Each finding should be a specific, actionable insight
- Confidence levels: "low" (single weak source), "medium" (multiple sources or one authoritative), "high" (multiple authoritative sources agree)
- Include source_ids that support each finding
- Categorize findings by theme when applicable

Guidelines for gaps:
- Identify 1-3 knowledge gaps or unanswered questions
- Provide specific follow-up queries that could fill each gap
- Priority 1 is most important, higher numbers are lower priority

Guidelines for quality_updates:
- Assess source quality based on authority, relevance, and recency
- "low" = questionable reliability, "medium" = generally reliable, "high" = authoritative

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    async def _execute_digest_step_async(
        self: DeepResearchWorkflow,
        state: DeepResearchState,
        query: str,
    ) -> dict[str, Any]:
        """Execute digest step: extract content, rank, select, and digest sources.

        This method implements the digest pipeline for the ANALYSIS phase:
        1. For sources WITHOUT content: extract PDFs (if fetch_pdfs enabled)
        2. Compute ranking on extracted content
        3. Select top N eligible sources
        4. Digest selected sources

        Sources without content (when fetch disabled) are ranked on snippet only
        and marked ineligible for digest.

        Args:
            state: Current research state with sources
            query: Research query for digest conditioning

        Returns:
            Dict with digest statistics:
            - sources_extracted: Number of sources with content extracted
            - sources_ranked: Number of sources ranked
            - sources_selected: Number of sources selected for digest
            - sources_digested: Number of sources successfully digested
            - digest_errors: List of error messages for failed digests
        """
        stats: dict[str, Any] = {
            "sources_extracted": 0,
            "sources_ranked": 0,
            "sources_selected": 0,
            "sources_digested": 0,
            "digest_errors": [],
        }

        # Check if digest is enabled via policy
        policy_str = self.config.deep_research_digest_policy
        if policy_str == "off":
            logger.debug("Digest step skipped: policy is OFF")
            return stats

        policy = DigestPolicy(policy_str)
        fetch_pdfs = self.config.deep_research_digest_fetch_pdfs

        # Step 1: Extract PDF content for sources without content (if fetch enabled)
        if fetch_pdfs:
            pdf_extractor = PDFExtractor()
            for source in state.sources:
                if not source.content and source.url:
                    try:
                        # Check if URL points to a PDF
                        if source.url.lower().endswith(".pdf"):
                            result = await pdf_extractor.extract_from_url(source.url)
                            if result.success and result.text:
                                source.content = result.text
                                source.metadata["_pdf_extracted"] = True
                                source.metadata["_pdf_page_count"] = result.page_count
                                if result.page_offsets:
                                    source.metadata["_pdf_page_offsets"] = result.page_offsets
                                stats["sources_extracted"] += 1
                                logger.debug(
                                    "Extracted PDF content for source %s: %d chars, %d pages",
                                    source.id,
                                    len(result.text),
                                    result.page_count or 0,
                                )
                    except Exception as e:
                        logger.warning(
                            "Failed to extract PDF for source %s: %s",
                            source.id,
                            str(e),
                        )
                        source.metadata["_pdf_extract_error"] = str(e)

                        # Emit audit event for PDF extraction failure
                        # Error handling policy: skip digest, preserve original, emit warning
                        error_msg = str(e)
                        if len(error_msg) > 200:
                            error_msg = error_msg[:200] + "...[truncated]"
                        self._write_audit_event(
                            state,
                            "digest.pdf_extract_error",
                            data={
                                "source_id": source.id,
                                "error_type": type(e).__name__,
                                "message": error_msg,
                                "url": source.url,
                                "correlation_id": state.id,
                            },
                            level="warning",
                        )

        # Step 2: Rank sources based on content/snippet
        # Sources with content are ranked higher than snippet-only sources
        ranked_sources: list[tuple[ResearchSource, float]] = []
        for source in state.sources:
            # Compute ranking score
            score = 0.0

            # Quality contributes to score
            quality_scores = {
                SourceQuality.HIGH: 1.0,
                SourceQuality.MEDIUM: 0.7,
                SourceQuality.LOW: 0.4,
                SourceQuality.UNKNOWN: 0.2,
            }
            score += quality_scores.get(source.quality, 0.2)

            # Content presence boosts score significantly
            if source.content:
                content_len = len(source.content)
                # Normalize content length contribution (max 1.0 at 10k+ chars)
                score += min(1.0, content_len / 10000)
            elif source.snippet:
                # Snippet-only sources get smaller boost
                score += 0.1

            ranked_sources.append((source, score))
            stats["sources_ranked"] += 1

        # Step 3: Sort by score (descending) then by ID (deterministic tiebreaker)
        ranked_sources.sort(key=lambda x: (-x[1], x[0].id))

        # Create digestor with config (used for eligibility + digest)
        max_sources = self.config.deep_research_digest_max_sources
        min_chars = self.config.deep_research_digest_min_chars
        digest_config = DigestConfig(
            policy=policy,
            min_content_length=min_chars,
            max_evidence_snippets=self.config.deep_research_digest_max_evidence_snippets,
            max_snippet_length=self.config.deep_research_digest_evidence_max_chars,
            include_evidence=self.config.deep_research_digest_include_evidence,
        )

        # Create summarizer for digestor (uses digest provider with fallback chain)
        digest_provider = self.config.get_digest_provider(analysis_provider=state.analysis_provider)
        digest_providers = self.config.get_digest_fallback_providers()

        summarizer = ContentSummarizer(
            summarization_provider=digest_provider,
            summarization_providers=digest_providers,
            max_retries=self.config.deep_research_max_retries,
            retry_delay=self.config.deep_research_retry_delay,
            timeout=self.config.deep_research_digest_timeout,
        )
        pdf_extractor = PDFExtractor()

        digestor = DocumentDigestor(
            summarizer=summarizer,
            pdf_extractor=pdf_extractor,
            config=digest_config,
        )

        # Step 4: Select top N eligible for digest
        eligible_sources: list[ResearchSource] = []

        for source, score in ranked_sources:
            if len(eligible_sources) >= max_sources:
                break

            # Skip already-digested sources (prevents double-digest in multi-iteration)
            if source.is_digest:
                source.metadata["_digest_eligible"] = False
                source.metadata["_digest_skip_reason"] = "already_digested"
                continue

            if not source.content:
                source.metadata["_digest_eligible"] = False
                source.metadata["_digest_skip_reason"] = "no_content"
                continue

            # Check eligibility using digestor policy/quality/length rules
            if digestor._is_eligible(source.content, source.quality):
                eligible_sources.append(source)
                source.metadata["_digest_eligible"] = True
                stats["sources_selected"] += 1
            else:
                source.metadata["_digest_eligible"] = False
                source.metadata["_digest_skip_reason"] = digestor._get_skip_reason(
                    source.content,
                    source.quality,
                )

        # Step 5: Digest selected sources
        if not eligible_sources:
            logger.debug("No eligible sources for digest")
            return stats

        # Digest each eligible source with timeout budgets
        # Configured timeout is per-source; batch scales with concurrency
        per_source_timeout = self.config.deep_research_digest_timeout
        max_concurrent = self.config.deep_research_digest_max_concurrent

        # Batch timeout = per_source_timeout * number of concurrent batches
        batch_count = max(1, math.ceil(len(eligible_sources) / max_concurrent))
        batch_timeout = per_source_timeout * batch_count
        logger.debug(
            "Digest timeout budgets: per_source=%.1fs, batch=%.1fs (batches=%d, max_concurrent=%d)",
            per_source_timeout,
            batch_timeout,
            batch_count,
            max_concurrent,
        )

        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
        semaphore = asyncio.Semaphore(max_concurrent)
        stats_lock = asyncio.Lock()

        async def _digest_source(source: ResearchSource) -> None:
            async with semaphore:
                # Store raw content BEFORE digest call for potential archival
                # This is set before and deleted in finally to ensure cleanup
                source.metadata["_raw_content"] = source.content
                content_size = len(source.content) if source.content else 0

                # Emit digest.started audit event (no raw content)
                self._write_audit_event(
                    state,
                    "digest.started",
                    data={
                        "source_id": source.id,
                        "content_size": content_size,
                        "policy": policy.value,
                        "query_hash": query_hash,
                        "correlation_id": state.id,
                    },
                )

                # Page boundaries for PDF locators (if available)
                page_offsets = source.metadata.get("_pdf_page_offsets")
                page_boundaries = None
                if page_offsets:
                    page_boundaries = [
                        (idx + 1, start, end)
                        for idx, (start, end) in enumerate(page_offsets)
                    ]

                try:
                    # Use per-source timeout with cancellation propagation
                    result: DigestResult = await asyncio.wait_for(
                        digestor.digest(
                            source=source.metadata["_raw_content"] or "",
                            query=query,
                            source_id=source.id,
                            quality=source.quality,
                            page_boundaries=page_boundaries,
                        ),
                        timeout=per_source_timeout,
                    )

                    if result.success and result.payload:
                        # Update source with digest payload
                        source.content = serialize_payload(result.payload)
                        source.content_type = "digest/v1"
                        source.metadata["_digest_cache_hit"] = result.cache_hit
                        source.metadata["_digest_duration_ms"] = result.duration_ms
                        async with stats_lock:
                            stats["sources_digested"] += 1
                        if self.config.deep_research_archive_content:
                            try:
                                await asyncio.to_thread(
                                    archive_digest_source,
                                    source=source,
                                    digestor=digestor,
                                    raw_content=source.metadata.get("_raw_content") or "",
                                    page_boundaries=page_boundaries,
                                    source_text_hash=result.payload.source_text_hash,
                                    retention_days=self.config.deep_research_archive_retention_days,
                                )
                            except Exception as archive_error:
                                error_msg = str(archive_error)
                                if len(error_msg) > 200:
                                    error_msg = error_msg[:200] + "...[truncated]"
                                source.metadata["_digest_archive_error"] = error_msg
                                logger.warning(
                                    "Digest archive failed for source %s: %s",
                                    source.id,
                                    error_msg,
                                )

                        # Record fidelity for digested source
                        # Estimate tokens: ~4 chars per token is a reasonable approximation
                        original_tokens = result.payload.original_chars // 4
                        final_tokens = result.payload.digest_chars // 4
                        state.record_item_fidelity(
                            item_id=source.id,
                            phase="digest",
                            level=FidelityLevel.DIGEST,
                            item_type="source",
                            reason="digest_compression",
                            original_tokens=original_tokens,
                            final_tokens=final_tokens,
                        )

                        logger.debug(
                            "Digested source %s: %d -> %d chars (%.1f%% compression)",
                            source.id,
                            result.payload.original_chars,
                            result.payload.digest_chars,
                            result.payload.compression_ratio * 100,
                        )

                        # Emit digest.completed audit event (no raw content)
                        self._write_audit_event(
                            state,
                            "digest.completed",
                            data={
                                "source_id": source.id,
                                "compression_ratio": result.payload.compression_ratio,
                                "cache_hit": result.cache_hit,
                                "duration_ms": result.duration_ms,
                                "correlation_id": state.id,
                            },
                        )
                    elif result.skipped:
                        source.metadata["_digest_skipped"] = True
                        source.metadata["_digest_skip_reason"] = result.skip_reason

                        # Record fidelity as FULL (content unchanged) with warning
                        state.record_item_fidelity(
                            item_id=source.id,
                            phase="digest",
                            level=FidelityLevel.FULL,
                            item_type="source",
                            reason="digest_skipped",
                            warnings=[f"Digest skipped: {result.skip_reason}"],
                        )

                        # Emit digest.skipped audit event
                        self._write_audit_event(
                            state,
                            "digest.skipped",
                            data={
                                "source_id": source.id,
                                "reason": result.skip_reason,
                                "correlation_id": state.id,
                            },
                        )
                    else:
                        async with stats_lock:
                            stats["digest_errors"].append(
                                f"Source {source.id}: digest failed with warnings: {result.warnings}"
                            )

                        # Record fidelity as FULL (content unchanged) with warnings
                        state.record_item_fidelity(
                            item_id=source.id,
                            phase="digest",
                            level=FidelityLevel.FULL,
                            item_type="source",
                            reason="digest_failed",
                            warnings=result.warnings or ["Digest failed without specific error"],
                        )

                        # Emit digest.error audit event for non-exception failures
                        error_msg = (
                            "; ".join(result.warnings)
                            if result.warnings
                            else "Digest failed without specific error"
                        )
                        if len(error_msg) > 200:
                            error_msg = error_msg[:200] + "...[truncated]"
                        self._write_audit_event(
                            state,
                            "digest.error",
                            data={
                                "source_id": source.id,
                                "error_type": "digest_failed",
                                "message": error_msg,
                                "correlation_id": state.id,
                            },
                            level="warning",
                        )

                except asyncio.TimeoutError:
                    logger.warning(
                        "Digest timeout for source %s after %.1fs (budget: per_source=%.1fs)",
                        source.id,
                        per_source_timeout,
                        per_source_timeout,
                    )
                    source.metadata["_digest_timeout"] = True
                    async with stats_lock:
                        stats["digest_errors"].append(
                            f"Source {source.id}: timeout after {per_source_timeout:.1f}s"
                        )

                    # Record fidelity as FULL (content unchanged) with timeout warning
                    state.record_item_fidelity(
                        item_id=source.id,
                        phase="digest",
                        level=FidelityLevel.FULL,
                        item_type="source",
                        reason="digest_timeout",
                        warnings=[f"Digest timeout after {per_source_timeout:.1f}s"],
                    )

                    # Emit digest.error audit event for timeout
                    self._write_audit_event(
                        state,
                        "digest.error",
                        data={
                            "source_id": source.id,
                            "error_type": "timeout",
                            "message": f"Digest timeout after {per_source_timeout:.1f}s (budget: {per_source_timeout:.1f}s)",
                            "correlation_id": state.id,
                        },
                        level="warning",
                    )
                except Exception as e:
                    logger.warning(
                        "Digest error for source %s: %s",
                        source.id,
                        str(e),
                    )
                    source.metadata["_digest_error"] = str(e)
                    async with stats_lock:
                        stats["digest_errors"].append(f"Source {source.id}: {str(e)}")

                    # Record fidelity as FULL (content unchanged) with error warning
                    # Sanitize error message for fidelity record
                    error_msg = str(e)
                    if len(error_msg) > 200:
                        error_msg = error_msg[:200] + "...[truncated]"
                    state.record_item_fidelity(
                        item_id=source.id,
                        phase="digest",
                        level=FidelityLevel.FULL,
                        item_type="source",
                        reason="digest_error",
                        warnings=[f"Digest error ({type(e).__name__}): {error_msg}"],
                    )

                    # Emit digest.error audit event for exception
                    # Sanitize error message: truncate to prevent raw content leakage
                    self._write_audit_event(
                        state,
                        "digest.error",
                        data={
                            "source_id": source.id,
                            "error_type": type(e).__name__,
                            "message": error_msg,
                            "correlation_id": state.id,
                        },
                        level="warning",
                    )
                finally:
                    # Always delete _raw_content to prevent serialization
                    # This ensures raw content is never persisted to disk
                    source.metadata.pop("_raw_content", None)

        # Track which sources have been processed (set in _digest_source on completion)
        processed_source_ids: set[str] = set()

        async def _tracked_digest_source(source: ResearchSource) -> None:
            await _digest_source(source)
            processed_source_ids.add(source.id)

        tasks = [asyncio.create_task(_tracked_digest_source(source)) for source in eligible_sources]
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=batch_timeout,
            )
        except asyncio.TimeoutError:
            remaining_count = sum(1 for t in tasks if not t.done())
            logger.warning(
                "Batch timeout exceeded (%.1fs), cancelling remaining %d sources",
                batch_timeout,
                remaining_count,
            )
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            # Record fidelity and set metadata for sources that weren't processed
            for source in eligible_sources:
                if source.id not in processed_source_ids:
                    # Check if already handled by per-source timeout or error
                    if not source.metadata.get("_digest_timeout") and not source.metadata.get("_digest_error"):
                        source.metadata["_digest_timeout"] = True
                        stats["digest_errors"].append(
                            f"Source {source.id}: batch timeout after {batch_timeout:.1f}s"
                        )
                        state.record_item_fidelity(
                            item_id=source.id,
                            phase="digest",
                            level=FidelityLevel.FULL,
                            item_type="source",
                            reason="digest_timeout",
                            warnings=[f"Batch timeout after {batch_timeout:.1f}s"],
                        )
                        self._write_audit_event(
                            state,
                            "digest.error",
                            data={
                                "source_id": source.id,
                                "error_type": "batch_timeout",
                                "message": f"Batch timeout after {batch_timeout:.1f}s",
                                "correlation_id": state.id,
                            },
                            level="warning",
                        )

        logger.info(
            "Digest step complete: %d extracted, %d ranked, %d selected, %d digested",
            stats["sources_extracted"],
            stats["sources_ranked"],
            stats["sources_selected"],
            stats["sources_digested"],
        )

        return stats

    def _build_analysis_user_prompt(
        self: DeepResearchWorkflow,
        state: DeepResearchState,
        allocation_result: Optional[AllocationResult] = None,
    ) -> str:
        """Build user prompt with source summaries for analysis.

        Args:
            state: Current research state
            allocation_result: Optional budget allocation result for token-aware prompts

        Returns:
            User prompt string
        """

        prompt_parts = [
            f"Original Research Query: {state.original_query}",
            "",
            "Research Brief:",
            state.research_brief or "Direct research on the query",
            "",
            "Sources to Analyze:",
            "",
        ]

        # Build source lookup for allocation info
        allocated_map: dict[str, Any] = {}
        if allocation_result:
            for item in allocation_result.items:
                allocated_map[item.id] = item

        # Add source summaries based on allocation
        sources_to_include = []
        if allocation_result:
            # Use allocated sources in priority order
            for item in allocation_result.items:
                source = next((s for s in state.sources if s.id == item.id), None)
                if source:
                    sources_to_include.append((source, item))
        else:
            # Fallback: use first 20 sources (legacy behavior)
            for source in state.sources[:20]:
                sources_to_include.append((source, None))

        for i, (source, alloc_item) in enumerate(sources_to_include, 1):
            prompt_parts.append(f"Source {i} (ID: {source.id}):")
            prompt_parts.append(f"  Title: {source.title}")
            if source.url:
                prompt_parts.append(f"  URL: {source.url}")

            # Determine content limit based on allocation
            if alloc_item and alloc_item.needs_summarization:
                # Use allocated tokens to estimate character limit (~4 chars/token)
                char_limit = max(100, alloc_item.allocated_tokens * 4)
                snippet_limit = min(500, char_limit // 3)
                content_limit = min(1000, char_limit - snippet_limit)
            else:
                # Full fidelity: use default limits
                snippet_limit = 500
                content_limit = 1000

            if source.snippet:
                snippet = source.snippet[:snippet_limit]
                if len(source.snippet) > snippet_limit:
                    snippet += "..."
                prompt_parts.append(f"  Snippet: {snippet}")

            if source.content:
                # Check if source contains a digest payload
                if source.is_digest:
                    # Parse digest and use evidence snippets for citations
                    try:
                        payload = deserialize_payload(source.content)
                        prompt_parts.append(f"  Summary: {payload.summary[:content_limit]}")
                        if payload.key_points:
                            prompt_parts.append("  Key Points:")
                            for kp in payload.key_points[:5]:
                                prompt_parts.append(f"    - {kp}")
                        if payload.evidence_snippets:
                            prompt_parts.append("  Evidence:")
                            for ev in payload.evidence_snippets[:3]:
                                prompt_parts.append(f"    - \"{ev.text[:200]}\" [{ev.locator}]")
                    except Exception:
                        # Fallback to raw content if parsing fails
                        content = source.content[:content_limit]
                        prompt_parts.append(f"  Content: {content}")
                else:
                    content = source.content[:content_limit]
                    if len(source.content) > content_limit:
                        content += "..."
                    prompt_parts.append(f"  Content: {content}")

            prompt_parts.append("")

        prompt_parts.extend([
            "Please analyze these sources and:",
            "1. Extract 2-5 key findings relevant to the research query",
            "2. Assess confidence levels based on source agreement and authority",
            "3. Identify any knowledge gaps or unanswered questions",
            "4. Assess the quality of each source",
            "",
            "Return your analysis as JSON.",
        ])

        return "\n".join(prompt_parts)

    def _parse_analysis_response(
        self: DeepResearchWorkflow,
        content: str,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Parse LLM response into structured analysis data.

        Args:
            content: Raw LLM response content
            state: Current research state (reserved for context-aware parsing)

        Returns:
            Dict with 'success', 'findings', 'gaps', and 'quality_updates' keys
        """
        # state is reserved for future context-aware parsing
        _ = state
        result = {
            "success": False,
            "findings": [],
            "gaps": [],
            "quality_updates": [],
        }

        if not content:
            return result

        # Try to extract JSON from the response
        json_str = extract_json(content)
        if not json_str:
            logger.warning("No JSON found in analysis response")
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from analysis response: %s", e)
            return result

        # Parse findings
        raw_findings = data.get("findings", [])
        if isinstance(raw_findings, list):
            for f in raw_findings:
                if not isinstance(f, dict):
                    continue
                content_text = f.get("content", "").strip()
                if not content_text:
                    continue

                # Map confidence string to enum
                confidence_str = f.get("confidence", "medium").lower()
                confidence_map = {
                    "low": ConfidenceLevel.LOW,
                    "medium": ConfidenceLevel.MEDIUM,
                    "high": ConfidenceLevel.HIGH,
                    "confirmed": ConfidenceLevel.CONFIRMED,
                    "speculation": ConfidenceLevel.SPECULATION,
                }
                confidence = confidence_map.get(confidence_str, ConfidenceLevel.MEDIUM)

                result["findings"].append({
                    "content": content_text,
                    "confidence": confidence,
                    "source_ids": f.get("source_ids", []),
                    "category": f.get("category"),
                })

        # Parse gaps
        raw_gaps = data.get("gaps", [])
        if isinstance(raw_gaps, list):
            for g in raw_gaps:
                if not isinstance(g, dict):
                    continue
                description = g.get("description", "").strip()
                if not description:
                    continue

                result["gaps"].append({
                    "description": description,
                    "suggested_queries": g.get("suggested_queries", []),
                    "priority": min(max(int(g.get("priority", 1)), 1), 10),
                })

        # Parse quality updates
        raw_quality = data.get("quality_updates", [])
        if isinstance(raw_quality, list):
            for q in raw_quality:
                if not isinstance(q, dict):
                    continue
                source_id = q.get("source_id", "").strip()
                quality = q.get("quality", "").lower()
                if source_id and quality in ("low", "medium", "high", "unknown"):
                    result["quality_updates"].append({
                        "source_id": source_id,
                        "quality": quality,
                    })

        # Mark success if we got at least one finding
        result["success"] = len(result["findings"]) > 0

        return result
