"""Digest step mixin for the analysis phase.

Extracts, ranks, selects, and digests source content before the main
analysis LLM call.  Split from ``analysis.py`` to keep each module focused.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from typing import TYPE_CHECKING, Any

from foundry_mcp.core.research.document_digest import (
    DigestConfig,
    DigestPolicy,
    DigestResult,
    DocumentDigestor,
    serialize_payload,
)
from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.models.fidelity import FidelityLevel
from foundry_mcp.core.research.models.sources import ResearchSource, SourceQuality
from foundry_mcp.core.research.pdf_extractor import PDFExtractor
from foundry_mcp.core.research.summarization import ContentSummarizer
from foundry_mcp.core.research.workflows.deep_research._budgeting import (
    archive_digest_source,
)

logger = logging.getLogger(__name__)


class DigestStepMixin:
    """Digest pipeline methods. Mixed into AnalysisPhaseMixin.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    """

    config: Any
    memory: Any

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _check_cancellation(self, *args: Any, **kwargs: Any) -> None: ...

    async def _execute_digest_step_async(
        self,
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

        for source, _score in ranked_sources:
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
                    page_boundaries = [(idx + 1, start, end) for idx, (start, end) in enumerate(page_offsets)]

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
                            "; ".join(result.warnings) if result.warnings else "Digest failed without specific error"
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
                        stats["digest_errors"].append(f"Source {source.id}: timeout after {per_source_timeout:.1f}s")

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
                        stats["digest_errors"].append(f"Source {source.id}: batch timeout after {batch_timeout:.1f}s")
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
