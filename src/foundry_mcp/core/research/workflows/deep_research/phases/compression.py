"""Per-topic compression mixin for DeepResearchWorkflow.

Compresses each topic's raw sources into citation-rich summaries before
analysis.  Extracted from GatheringPhaseMixin (Phase 3 PA.2) to isolate
the compression logic as an independently testable unit.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from foundry_mcp.core.research.models.deep_research import DeepResearchState, TopicResearchResult
from foundry_mcp.core.research.models.fidelity import PhaseMetrics

logger = logging.getLogger(__name__)


class CompressionMixin:
    """Per-topic compression methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)
    """

    config: Any
    memory: Any

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _check_cancellation(self, *args: Any, **kwargs: Any) -> None: ...
        async def _execute_provider_async(self, *args: Any, **kwargs: Any) -> Any: ...

    # ------------------------------------------------------------------
    # Per-topic compression
    # ------------------------------------------------------------------

    #: Maximum characters per source included in the compression prompt.
    #: ~500 tokens at ~4 chars/token — enough for the LLM to produce a
    #: meaningful citation-rich summary per source while keeping the total
    #: compression prompt within context-window limits when many sources
    #: are processed together.
    _COMPRESSION_SOURCE_CHAR_LIMIT: int = 2000

    async def _compress_topic_findings_async(
        self,
        state: DeepResearchState,
        max_concurrent: int,
        timeout: float,
    ) -> dict[str, Any]:
        """Compress each topic's sources into citation-rich summaries.

        Runs after all topic researchers complete.  For each
        ``TopicResearchResult`` that has sources, builds a compression
        prompt asking the LLM to reformat the raw findings with inline
        citations — preserving all relevant information without
        summarising.  Results are stored in
        ``TopicResearchResult.compressed_findings``.

        Features:
        - Parallel compression across topics, bounded by *max_concurrent*.
        - Progressive token-limit handling (3 retries, 10 % truncation each).
        - Graceful fallback: if compression fails for a topic,
          ``compressed_findings`` stays ``None`` and the analysis phase
          falls through to raw sources.

        Args:
            state: Current research state with topic results and sources.
            max_concurrent: Maximum parallel compression calls.
            timeout: Per-compression LLM call timeout in seconds.

        Returns:
            Dict with compression statistics (topics_compressed,
            topics_failed, total_compression_tokens, input_tokens,
            output_tokens).
        """
        from foundry_mcp.core.errors.provider import ContextWindowError
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            estimate_token_limit_for_model,
            truncate_to_token_estimate,
        )
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            MODEL_TOKEN_LIMITS,
            _TRUNCATION_FACTOR,
            _is_context_window_error,
        )

        results_to_compress = [
            tr for tr in state.topic_research_results
            if tr.source_ids and tr.compressed_findings is None
        ]

        if not results_to_compress:
            return {"topics_compressed": 0, "topics_failed": 0, "total_compression_tokens": 0}

        # Resolve compression provider/model via role-based hierarchy (Phase 6).
        # Uses try/except — hasattr is not safe with mock objects that
        # auto-create attributes on access.
        compression_provider: str = self.config.default_provider
        compression_model: str | None = None
        try:
            compression_provider, compression_model = self.config.resolve_model_for_role("compression")
        except (AttributeError, TypeError, ValueError):
            logger.debug("Role resolution unavailable for compression, using defaults")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def compress_one(
            topic_result: TopicResearchResult,
        ) -> tuple[int, int, bool]:
            """Compress a single topic's findings.

            Returns:
                (input_tokens, output_tokens, success) tuple.
            """
            async with semaphore:
                # Collect sources belonging to this topic
                topic_sources = [
                    s for s in state.sources if s.id in topic_result.source_ids
                ]
                if not topic_sources:
                    return (0, 0, True)

                # Look up the sub-query text for context
                sub_query = state.get_sub_query(topic_result.sub_query_id)
                query_text = sub_query.query if sub_query else "Unknown query"

                # Build source block for the prompt
                source_lines: list[str] = []
                for idx, src in enumerate(topic_sources, 1):
                    source_lines.append(f"[{idx}] Title: {src.title}")
                    if src.url:
                        source_lines.append(f"    URL: {src.url}")
                    content = src.content or src.snippet or ""
                    if content:
                        if len(content) > self._COMPRESSION_SOURCE_CHAR_LIMIT:
                            content = content[:self._COMPRESSION_SOURCE_CHAR_LIMIT] + "..."
                        source_lines.append(f"    Content: {content}")
                    source_lines.append("")

                sources_block = "\n".join(source_lines)

                system_prompt = (
                    "You are a research assistant. Your task is to reformat research "
                    "findings into a structured, citation-rich summary.\n\n"
                    "IMPORTANT RULES:\n"
                    "- DO NOT summarize or remove information. Preserve ALL relevant details.\n"
                    "- Add inline citations using [1], [2], etc. matching the source numbers.\n"
                    "- Organize findings into a coherent narrative grouped by theme.\n"
                    "- Include a source list at the end.\n\n"
                    "OUTPUT FORMAT:\n"
                    "## Queries Made\n"
                    "- List the search queries used\n\n"
                    "## Comprehensive Findings\n"
                    "- All findings with inline citations [1], [2], etc.\n"
                    "- Group by theme when applicable\n\n"
                    "## Source List\n"
                    "- [1] Title — URL\n"
                    "- [2] Title — URL\n"
                )

                user_prompt = (
                    f"Research sub-query: {query_text}\n\n"
                    f"Sources ({len(topic_sources)} total):\n\n"
                    f"{sources_block}\n"
                    "Reformat these findings with inline citations. "
                    "Preserve all relevant information."
                )

                # Progressive token-limit recovery (up to 3 retries)
                max_retries = 3
                current_prompt = user_prompt
                compressed: str | None = None
                input_tokens = 0
                output_tokens = 0

                for attempt in range(max_retries + 1):
                    try:
                        self._check_cancellation(state)

                        result = await self._execute_provider_async(
                            prompt=current_prompt,
                            provider_id=compression_provider,
                            model=compression_model,
                            system_prompt=system_prompt,
                            timeout=timeout,
                            temperature=0.2,
                            phase="compression",
                            fallback_providers=self.config.get_phase_fallback_providers("analysis"),
                            max_retries=1,
                            retry_delay=2.0,
                        )

                        if result.success and result.content:
                            compressed = result.content
                            input_tokens = result.input_tokens or 0
                            output_tokens = result.output_tokens or 0
                        break

                    except ContextWindowError as e:
                        if attempt >= max_retries:
                            break
                        # Truncate user prompt by 10% and retry
                        max_tokens = e.max_tokens or estimate_token_limit_for_model(
                            compression_model, MODEL_TOKEN_LIMITS
                        ) or 128_000
                        reduced = int(max_tokens * (_TRUNCATION_FACTOR ** (attempt + 1)))
                        current_prompt = truncate_to_token_estimate(current_prompt, reduced)
                        logger.warning(
                            "Compression context window exceeded for topic %s "
                            "(attempt %d/%d), truncating and retrying",
                            topic_result.sub_query_id,
                            attempt + 1,
                            max_retries,
                        )
                    except Exception as e:
                        if _is_context_window_error(e) and attempt < max_retries:
                            max_tokens = estimate_token_limit_for_model(
                                compression_model, MODEL_TOKEN_LIMITS
                            ) or 128_000
                            reduced = int(max_tokens * (_TRUNCATION_FACTOR ** (attempt + 1)))
                            current_prompt = truncate_to_token_estimate(current_prompt, reduced)
                            logger.warning(
                                "Compression detected provider-specific context error "
                                "for topic %s (attempt %d/%d)",
                                topic_result.sub_query_id,
                                attempt + 1,
                                max_retries,
                            )
                        else:
                            logger.error(
                                "Compression failed for topic %s: %s",
                                topic_result.sub_query_id,
                                e,
                            )
                            break

                if compressed:
                    topic_result.compressed_findings = compressed
                    return (input_tokens, output_tokens, True)
                return (0, 0, False)

        # Run compression tasks in parallel
        compression_start = time.perf_counter()
        tasks = [compress_one(tr) for tr in results_to_compress]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results after gather completes (no nonlocal mutation)
        total_input_tokens = 0
        total_output_tokens = 0
        topics_compressed = 0
        topics_failed = 0
        for i, gather_result in enumerate(gather_results):
            if isinstance(gather_result, BaseException):
                topics_failed += 1
                logger.error(
                    "Compression task exception for topic %s: %s",
                    results_to_compress[i].sub_query_id,
                    gather_result,
                )
            else:
                inp, out, success = gather_result
                total_input_tokens += inp
                total_output_tokens += out
                if success:
                    topics_compressed += 1
                else:
                    topics_failed += 1
        compression_duration_ms = (time.perf_counter() - compression_start) * 1000

        # Record compression phase metrics
        total_compression_tokens = total_input_tokens + total_output_tokens
        if total_compression_tokens > 0:
            state.total_tokens_used += total_compression_tokens
            state.phase_metrics.append(
                PhaseMetrics(
                    phase="compression",
                    duration_ms=compression_duration_ms,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cached_tokens=0,
                    provider_id=compression_provider,
                    model_used=compression_model,
                    metadata={
                        "topics_compressed": topics_compressed,
                        "topics_failed": topics_failed,
                    },
                )
            )

        self._write_audit_event(
            state,
            "topic_compression_complete",
            data={
                "topics_compressed": topics_compressed,
                "topics_failed": topics_failed,
                "total_compression_tokens": total_compression_tokens,
                "compression_provider": compression_provider,
                "compression_model": compression_model,
            },
        )

        logger.info(
            "Per-topic compression complete: %d compressed, %d failed, %d tokens",
            topics_compressed,
            topics_failed,
            total_compression_tokens,
        )

        return {
            "topics_compressed": topics_compressed,
            "topics_failed": topics_failed,
            "total_compression_tokens": total_compression_tokens,
        }
