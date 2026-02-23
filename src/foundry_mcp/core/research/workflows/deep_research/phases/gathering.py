"""Gathering phase mixin for DeepResearchWorkflow.

Executes sub-queries in parallel across search providers, collects and
deduplicates sources, and optionally follows up with Tavily Extract.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.research.models.deep_research import DeepResearchState, TopicResearchResult
from foundry_mcp.core.research.models.fidelity import PhaseMetrics
from foundry_mcp.core.research.models.sources import SourceQuality
from foundry_mcp.core.research.providers import (
    GoogleSearchProvider,
    PerplexitySearchProvider,
    SearchProvider,
    SearchProviderError,
    SemanticScholarProvider,
    TavilyExtractProvider,
    TavilySearchProvider,
)
from foundry_mcp.core.research.providers.resilience import get_resilience_manager
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    _normalize_title,
    get_domain_quality,
)

logger = logging.getLogger(__name__)


class GatheringPhaseMixin:
    """Gathering phase methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _search_providers (cache dict on instance)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    """

    config: Any
    memory: Any
    _search_providers: dict[str, Any]

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _check_cancellation(self, *args: Any, **kwargs: Any) -> None: ...
        async def _execute_topic_research_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_provider_async(self, *args: Any, **kwargs: Any) -> Any: ...
        def _attach_source_summarizer(self, provider: Any) -> None: ...

    # ------------------------------------------------------------------
    # Per-topic compression
    # ------------------------------------------------------------------

    #: Maximum characters per source included in the compression prompt.
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
            topics_failed, total_compression_tokens).
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
        # Uses try/except for the unpacking — hasattr alone is not safe with
        # mock objects that auto-create attributes.
        compression_provider: str = self.config.default_provider
        compression_model: str | None = None
        try:
            compression_provider, compression_model = self.config.resolve_model_for_role("compression")
        except (AttributeError, TypeError, ValueError):
            compression_provider = getattr(
                self.config, "get_compression_provider", lambda: self.config.default_provider
            )()
            compression_model = getattr(
                self.config, "get_compression_model", lambda: None
            )()

        semaphore = asyncio.Semaphore(max_concurrent)
        total_compression_tokens = 0
        topics_compressed = 0
        topics_failed = 0

        async def compress_one(topic_result: TopicResearchResult) -> None:
            nonlocal total_compression_tokens, topics_compressed, topics_failed

            async with semaphore:
                # Collect sources belonging to this topic
                topic_sources = [
                    s for s in state.sources if s.id in topic_result.source_ids
                ]
                if not topic_sources:
                    return

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
                tokens_used = 0

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
                            tokens_used = result.tokens_used or 0
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
                    total_compression_tokens += tokens_used
                    topics_compressed += 1
                else:
                    topics_failed += 1

        # Run compression tasks in parallel
        compression_start = time.perf_counter()
        tasks = [compress_one(tr) for tr in results_to_compress]
        await asyncio.gather(*tasks, return_exceptions=True)
        compression_duration_ms = (time.perf_counter() - compression_start) * 1000

        # Record compression phase metrics
        if total_compression_tokens > 0:
            state.total_tokens_used += total_compression_tokens
            state.phase_metrics.append(
                PhaseMetrics(
                    phase="compression",
                    duration_ms=compression_duration_ms,
                    input_tokens=total_compression_tokens,
                    output_tokens=0,
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

    # ------------------------------------------------------------------
    # Search provider configuration
    # ------------------------------------------------------------------

    def _get_tavily_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]:
        """Build Tavily search kwargs based on config and research mode.

        Applies parameter precedence:
        1. Config values (highest priority when explicitly set)
        2. Research-mode defaults (academic/technical/general)
        3. Base defaults

        Research mode defaults:
        - general: search_depth=basic, chunks_per_source=3
        - academic: search_depth=advanced, chunks_per_source=5, include_raw_content=markdown
        - technical: search_depth=advanced, chunks_per_source=4, include_raw_content=markdown

        Args:
            state: Current deep research state (for research_mode)

        Returns:
            Dict of kwargs to pass to TavilySearchProvider.search()
        """
        # Start with research-mode defaults
        mode = state.research_mode or self.config.deep_research_mode
        mode_defaults: dict[str, Any] = {
            "general": {
                "search_depth": "basic",
                "chunks_per_source": 3,
                "include_raw_content": False,
            },
            "academic": {
                "search_depth": "advanced",
                "chunks_per_source": 5,
                "include_raw_content": "markdown",
            },
            "technical": {
                "search_depth": "advanced",
                "chunks_per_source": 4,
                "include_raw_content": "markdown",
            },
        }
        kwargs = mode_defaults.get(mode, mode_defaults["general"]).copy()

        # Override with config values (if explicitly set/non-default)
        config = self.config
        default_topic = "general"

        if getattr(config, "tavily_search_depth_configured", False) or config.tavily_search_depth != "basic":
            kwargs["search_depth"] = config.tavily_search_depth
        if config.tavily_topic != default_topic or config.tavily_news_days is not None:
            kwargs["topic"] = config.tavily_topic
        if config.tavily_include_images:
            kwargs["include_images"] = True
        kwargs["include_favicon"] = False  # Not typically needed for research
        if config.tavily_auto_parameters:
            kwargs["auto_parameters"] = True
        if getattr(config, "tavily_chunks_per_source_configured", False) or config.tavily_chunks_per_source != 3:
            kwargs["chunks_per_source"] = config.tavily_chunks_per_source

        # Only include optional parameters when explicitly set
        if config.tavily_news_days is not None:
            kwargs["days"] = config.tavily_news_days
        if config.tavily_country is not None:
            kwargs["country"] = config.tavily_country

        # Handle include_raw_content: config value or mode default, but state.follow_links takes precedence
        if state.follow_links:
            # If follow_links is True, we want raw content
            kwargs["include_raw_content"] = kwargs.get("include_raw_content", "markdown") or "markdown"

        return kwargs

    def _get_perplexity_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]:
        """Build Perplexity search kwargs based on config.

        Applies config values for Perplexity-specific parameters.
        Only includes non-None values to allow provider defaults.

        Args:
            state: Current deep research state (for potential future mode-based defaults)

        Returns:
            Dict of kwargs to pass to PerplexitySearchProvider.search()
        """
        config = self.config
        kwargs: dict[str, Any] = {}

        # Always include non-default values
        default_search_context_size = "medium"
        default_max_tokens = 50000
        default_max_tokens_per_page = 2048

        if config.perplexity_search_context_size != default_search_context_size:
            kwargs["search_context_size"] = config.perplexity_search_context_size
        if config.perplexity_max_tokens != default_max_tokens:
            kwargs["max_tokens"] = config.perplexity_max_tokens
        if config.perplexity_max_tokens_per_page != default_max_tokens_per_page:
            kwargs["max_tokens_per_page"] = config.perplexity_max_tokens_per_page

        # Only include optional parameters when explicitly set (non-None)
        if config.perplexity_recency_filter is not None:
            kwargs["recency_filter"] = config.perplexity_recency_filter
        if config.perplexity_country is not None:
            kwargs["country"] = config.perplexity_country

        return kwargs

    def _get_semantic_scholar_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]:
        """Build Semantic Scholar search kwargs based on config.

        Applies config values for Semantic Scholar-specific parameters.
        Only includes non-default values to allow provider defaults.

        Args:
            state: Current deep research state (for potential future mode-based defaults)

        Returns:
            Dict of kwargs to pass to SemanticScholarProvider.search()
        """
        config = self.config
        kwargs: dict[str, Any] = {}

        # Only include publication_types when explicitly set (non-None)
        if config.semantic_scholar_publication_types is not None:
            kwargs["publication_types"] = config.semantic_scholar_publication_types

        # Only include sort_by when explicitly set (non-None)
        if config.semantic_scholar_sort_by is not None:
            kwargs["sort_by"] = config.semantic_scholar_sort_by

        # Include sort_order only when sort_by is also set (or non-default)
        default_sort_order = "desc"
        if config.semantic_scholar_sort_by is not None or config.semantic_scholar_sort_order != default_sort_order:
            kwargs["sort_order"] = config.semantic_scholar_sort_order

        # Include use_extended_fields only when False (True is the default)
        if not config.semantic_scholar_use_extended_fields:
            kwargs["use_extended_fields"] = False

        return kwargs

    # ------------------------------------------------------------------
    # Search provider factory
    # ------------------------------------------------------------------

    def _get_search_provider(self, provider_name: str) -> Optional[SearchProvider]:
        """Get or create a search provider instance.

        Args:
            provider_name: Name of the provider (e.g., "tavily")

        Returns:
            SearchProvider instance or None if unavailable
        """
        if provider_name in self._search_providers:
            return self._search_providers[provider_name]

        try:
            if provider_name == "tavily":
                provider = TavilySearchProvider()
                # Wire fetch-time summarization if enabled
                if getattr(self.config, "deep_research_fetch_time_summarization", False):
                    self._attach_source_summarizer(provider)
                self._search_providers[provider_name] = provider
                return provider
            if provider_name == "perplexity":
                provider = PerplexitySearchProvider()
                self._search_providers[provider_name] = provider
                return provider
            if provider_name == "google":
                provider = GoogleSearchProvider()
                self._search_providers[provider_name] = provider
                return provider
            if provider_name == "semantic_scholar":
                provider = SemanticScholarProvider()
                self._search_providers[provider_name] = provider
                return provider
            else:
                logger.warning("Unknown search provider: %s", provider_name)
                return None
        except ValueError as e:
            # API key not configured
            logger.error("Failed to initialize %s provider: %s", provider_name, e)
            return None
        except Exception as e:
            logger.error("Error initializing %s provider: %s", provider_name, e)
            return None

    def _attach_source_summarizer(self, provider: SearchProvider) -> None:
        """Attach a SourceSummarizer to a search provider for fetch-time summarization.

        Reads summarization provider/model config and creates a SourceSummarizer
        instance, then sets it on the provider's ``_source_summarizer`` attribute.

        Args:
            provider: SearchProvider instance to configure.
        """
        from foundry_mcp.core.research.providers.shared import SourceSummarizer

        # Resolve summarization provider/model via role-based hierarchy (Phase 6).
        summarization_provider: str = self.config.default_provider
        summarization_model: str | None = None
        try:
            summarization_provider, summarization_model = self.config.resolve_model_for_role("summarization")
        except (AttributeError, TypeError, ValueError):
            summarization_provider = getattr(
                self.config, "get_summarization_provider", lambda: self.config.default_provider
            )()
            summarization_model = getattr(
                self.config, "get_summarization_model", lambda: None
            )()
        max_concurrent = getattr(self.config, "deep_research_max_concurrent", 3)

        provider._source_summarizer = SourceSummarizer(
            provider_id=summarization_provider,
            model=summarization_model,
            timeout=60.0,
            max_concurrent=max_concurrent,
        )

    # ------------------------------------------------------------------
    # Main gathering phase
    # ------------------------------------------------------------------

    async def _execute_gathering_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
        max_concurrent: int,
    ) -> WorkflowResult:
        """Execute gathering phase: parallel sub-query execution.

        This phase:
        1. Gets all pending sub-queries from planning phase
        2. Executes them concurrently with rate limiting
        3. Collects and deduplicates sources
        4. Marks sub-queries as completed/failed

        Args:
            state: Current research state with sub-queries
            provider_id: LLM provider (reserved for future use in gathering)
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent search requests

        Returns:
            WorkflowResult with gathering outcome
        """
        # provider_id is reserved for future use (e.g., LLM-assisted query refinement)
        _ = provider_id
        pending_queries = state.pending_sub_queries()
        if not pending_queries:
            logger.warning("No pending sub-queries for gathering phase")
            return WorkflowResult(
                success=True,
                content="No sub-queries to execute",
                metadata={"research_id": state.id, "source_count": 0},
            )

        logger.info(
            "Starting gathering phase: %d sub-queries, max_concurrent=%d",
            len(pending_queries),
            max_concurrent,
        )

        # Emit phase.started audit event
        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "gathering",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        provider_names = getattr(
            self.config,
            "deep_research_providers",
            ["tavily", "google", "semantic_scholar"],
        )
        available_providers: list[SearchProvider] = []
        unavailable_providers: list[str] = []

        for name in provider_names:
            provider = self._get_search_provider(name)
            if provider is None:
                unavailable_providers.append(name)
                continue
            available_providers.append(provider)

        configured_providers = list(available_providers)
        configured_provider_names = [provider.get_provider_name() for provider in configured_providers]

        # Filter out providers with OPEN circuit breakers
        # HALF_OPEN providers are allowed to enable recovery probes
        resilience_manager = get_resilience_manager()
        circuit_breaker_filtered: list[str] = []
        filtered_providers: list[SearchProvider] = []
        for provider in available_providers:
            provider_name = provider.get_provider_name()
            if resilience_manager.is_provider_available(provider_name):
                filtered_providers.append(provider)
            else:
                circuit_breaker_filtered.append(provider_name)

        if circuit_breaker_filtered:
            logger.warning(
                "Filtered %d provider(s) due to open circuit breaker: %s",
                len(circuit_breaker_filtered),
                circuit_breaker_filtered,
            )

        available_providers = filtered_providers

        if not available_providers:
            # Determine if failure is due to circuit breakers or missing configuration
            if circuit_breaker_filtered:
                # All configured providers have open circuit breakers
                breaker_states = {
                    name: resilience_manager.get_breaker_state(name).value for name in configured_provider_names
                }
                audit_log(
                    "all_providers_circuit_open",
                    provider_names=circuit_breaker_filtered,
                    breaker_states=breaker_states,
                    configured_providers=configured_provider_names,
                    unavailable_providers=unavailable_providers,
                )
                logger.error("All providers have open circuit breakers: %s", breaker_states)
                return WorkflowResult(
                    success=False,
                    content="",
                    error=(
                        f"All search providers temporarily unavailable due to repeated failures. "
                        f"Circuit breakers open for: {', '.join(circuit_breaker_filtered)}. "
                        "Please wait for automatic recovery or check provider health."
                    ),
                )
            else:
                # No providers configured/available
                return WorkflowResult(
                    success=False,
                    content="",
                    error=(
                        "No search providers available. Configure API keys for Tavily, Google, or Semantic Scholar."
                    ),
                )

        # Capture circuit breaker states at start of gathering
        circuit_breaker_states_start = {
            name: resilience_manager.get_breaker_state(name).value for name in configured_provider_names
        }

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        state_lock = asyncio.Lock()

        # Update heartbeat and persist interim state for progress visibility
        state.last_heartbeat_at = datetime.now(timezone.utc)
        self.memory.save_deep_research(state)

        # Track collected sources for deduplication
        seen_urls: set[str] = {s.url for s in state.sources if s.url}
        seen_titles: dict[str, str] = {}
        for source in state.sources:
            normalized_title = _normalize_title(source.title)
            if normalized_title and len(normalized_title) > 20:
                seen_titles.setdefault(normalized_title, source.url or "")
        total_sources_added = 0
        failed_queries = 0

        # --- Topic agent delegation path ---
        # When topic agents are enabled, each sub-query runs its own ReAct
        # loop (search → reflect → refine → search) instead of flat parallel search.
        if getattr(self.config, "deep_research_enable_topic_agents", False):
            topic_max_searches = getattr(self.config, "deep_research_topic_max_searches", 3)

            # Budget splitting: divide max_sources_per_query across topic agents
            # so the aggregate source count stays within a reasonable bound.
            # Each topic gets at least 2 results per provider call.
            num_topics = max(1, len(pending_queries))
            per_topic_max_sources = max(2, state.max_sources_per_query // num_topics)

            logger.info(
                "Topic agent budget: %d topics, %d sources/provider/topic (total budget %d)",
                num_topics,
                per_topic_max_sources,
                state.max_sources_per_query,
            )

            self._check_cancellation(state)

            async def run_topic_agent(sq):
                return await self._execute_topic_research_async(
                    sub_query=sq,
                    state=state,
                    available_providers=available_providers,
                    max_searches=topic_max_searches,
                    max_sources_per_provider=per_topic_max_sources,
                    timeout=timeout,
                    seen_urls=seen_urls,
                    seen_titles=seen_titles,
                    state_lock=state_lock,
                    semaphore=semaphore,
                )

            try:
                tasks = [run_topic_agent(sq) for sq in pending_queries]
                topic_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(topic_results):
                    if isinstance(result, BaseException):
                        failed_queries += 1
                        logger.error("Topic agent exception for sub-query %s: %s", pending_queries[i].id, result)
                    else:
                        total_sources_added += result.sources_found
                        state.topic_research_results.append(result)
                        if result.sources_found == 0:
                            failed_queries += 1

            except asyncio.CancelledError:
                logger.warning("Gathering phase (topic agents) cancelled for research %s", state.id)
                try:
                    state.updated_at = datetime.now(timezone.utc)
                    self.memory.save_deep_research(state)
                except Exception as save_exc:
                    logger.error("Error saving state during topic agent cancellation: %s", save_exc)
                raise
            finally:
                state.updated_at = datetime.now(timezone.utc)

            # --- Per-topic compression step ---
            # Compress each topic's sources into citation-rich summaries
            # before analysis.  Runs only when sources were gathered.
            compression_stats: dict[str, Any] = {}
            if total_sources_added > 0 and state.topic_research_results:
                self._check_cancellation(state)
                compression_stats = await self._compress_topic_findings_async(
                    state=state,
                    max_concurrent=max_concurrent,
                    timeout=timeout,
                )

            # Save state and emit audit events (same as flat path)
            circuit_breaker_states_end = {
                name: resilience_manager.get_breaker_state(name).value for name in configured_provider_names
            }
            self.memory.save_deep_research(state)
            self._write_audit_event(
                state,
                "gathering_result",
                data={
                    "source_count": total_sources_added,
                    "queries_executed": len(pending_queries),
                    "queries_failed": failed_queries,
                    "unique_urls": len(seen_urls),
                    "providers_used": [p.get_provider_name() for p in available_providers],
                    "providers_unavailable": unavailable_providers,
                    "circuit_breaker_states_start": circuit_breaker_states_start,
                    "circuit_breaker_states_end": circuit_breaker_states_end,
                    "topic_agents_enabled": True,
                    "topic_max_searches": topic_max_searches,
                    "per_topic_max_sources": per_topic_max_sources,
                    "compression": compression_stats,
                },
            )

            success = total_sources_added > 0 or failed_queries < len(pending_queries)
            error_msg = None
            if not success and failed_queries == len(pending_queries):
                error_msg = (
                    f"All {failed_queries} topic researchers failed to find sources. "
                    f"Providers used: {[p.get_provider_name() for p in available_providers]}"
                )

            logger.info(
                "Gathering phase (topic agents) complete: %d sources from %d queries (%d failed)",
                total_sources_added,
                len(pending_queries),
                failed_queries,
            )

            phase_duration_ms = (time.perf_counter() - phase_start_time) * 1000
            self._write_audit_event(
                state,
                "phase.completed",
                data={
                    "phase_name": "gathering",
                    "iteration": state.iteration,
                    "task_id": state.id,
                    "duration_ms": phase_duration_ms,
                    "topic_agents_enabled": True,
                },
            )
            get_metrics().histogram(
                "foundry_mcp_research_phase_duration_seconds",
                phase_duration_ms / 1000.0,
                labels={"phase_name": "gathering", "status": "success" if success else "error"},
            )

            return WorkflowResult(
                success=success,
                content=f"Gathered {total_sources_added} sources from {len(pending_queries)} topic researchers",
                error=error_msg,
                metadata={
                    "research_id": state.id,
                    "source_count": total_sources_added,
                    "queries_executed": len(pending_queries),
                    "queries_failed": failed_queries,
                    "unique_urls": len(seen_urls),
                    "providers_used": [p.get_provider_name() for p in available_providers],
                    "topic_agents_enabled": True,
                    "compression": compression_stats,
                },
            )

        # --- Flat parallel search path (original behavior) ---
        try:

            async def execute_sub_query(sub_query) -> tuple[int, Optional[str]]:
                """Execute a single sub-query and return (sources_added, error)."""
                async with semaphore:
                    # Check for cancellation before executing sub-query
                    self._check_cancellation(state)

                    sub_query.status = "executing"

                    provider_errors: list[str] = []
                    added = 0

                    for provider in available_providers:
                        provider_name = provider.get_provider_name()

                        # Check if circuit breaker opened mid-gathering (graceful degradation)
                        if not resilience_manager.is_provider_available(provider_name):
                            logger.warning(
                                "Provider %s circuit breaker opened mid-gathering, skipping for remaining sub-queries",
                                provider_name,
                            )
                            provider_errors.append(f"{provider_name}: circuit breaker open")
                            continue

                        try:
                            # Check for cancellation before making search provider call
                            self._check_cancellation(state)

                            # Build provider-specific kwargs
                            search_kwargs: dict[str, Any] = {
                                "query": sub_query.query,
                                "max_results": state.max_sources_per_query,
                                "sub_query_id": sub_query.id,
                            }

                            # Add provider-specific kwargs
                            if provider_name == "tavily":
                                tavily_kwargs = self._get_tavily_search_kwargs(state)
                                search_kwargs.update(tavily_kwargs)
                            elif provider_name == "perplexity":
                                perplexity_kwargs = self._get_perplexity_search_kwargs(state)
                                search_kwargs.update(perplexity_kwargs)
                                # Perplexity also needs include_raw_content for link following
                                search_kwargs["include_raw_content"] = state.follow_links
                            elif provider_name == "semantic_scholar":
                                semantic_scholar_kwargs = self._get_semantic_scholar_search_kwargs(state)
                                search_kwargs.update(semantic_scholar_kwargs)
                                # Semantic Scholar also gets include_raw_content for consistency
                                search_kwargs["include_raw_content"] = state.follow_links
                            else:
                                # Other providers just get include_raw_content
                                search_kwargs["include_raw_content"] = state.follow_links

                            sources = await asyncio.wait_for(
                                provider.search(**search_kwargs),
                                timeout=timeout,
                            )

                            # Add sources with deduplication
                            for source in sources:
                                async with state_lock:
                                    # URL-based deduplication
                                    if source.url and source.url in seen_urls:
                                        continue  # Skip duplicate URL

                                    # Title-based deduplication (same paper from different domains)
                                    normalized_title = _normalize_title(source.title)
                                    if normalized_title and len(normalized_title) > 20:
                                        if normalized_title in seen_titles:
                                            logger.debug(
                                                "Skipping duplicate by title: %s (already have %s)",
                                                source.url,
                                                seen_titles[normalized_title],
                                            )
                                            continue  # Skip duplicate title
                                        seen_titles[normalized_title] = source.url or ""

                                    if source.url:
                                        seen_urls.add(source.url)
                                        # Apply domain-based quality scoring
                                        if source.quality == SourceQuality.UNKNOWN:
                                            source.quality = get_domain_quality(source.url, state.research_mode)

                                    # Add source to state (centralised citation assignment)
                                    state.append_source(source)
                                    sub_query.source_ids.append(source.id)
                                    added += 1

                            self._write_audit_event(
                                state,
                                "gathering_provider_result",
                                data={
                                    "provider": provider_name,
                                    "sub_query_id": sub_query.id,
                                    "sub_query": sub_query.query,
                                    "sources_added": len(sources),
                                },
                            )
                            # Track search provider query count
                            async with state_lock:
                                state.search_provider_stats[provider_name] = (
                                    state.search_provider_stats.get(provider_name, 0) + 1
                                )
                        except SearchProviderError as e:
                            provider_errors.append(f"{provider_name}: {e}")
                            self._write_audit_event(
                                state,
                                "gathering_provider_result",
                                data={
                                    "provider": provider_name,
                                    "sub_query_id": sub_query.id,
                                    "sub_query": sub_query.query,
                                    "sources_added": 0,
                                    "error": str(e),
                                },
                                level="warning",
                            )
                        except asyncio.TimeoutError:
                            provider_errors.append(f"{provider_name}: timeout after {timeout}s")
                            self._write_audit_event(
                                state,
                                "gathering_provider_result",
                                data={
                                    "provider": provider_name,
                                    "sub_query_id": sub_query.id,
                                    "sub_query": sub_query.query,
                                    "sources_added": 0,
                                    "error": f"timeout after {timeout}s",
                                },
                                level="warning",
                            )
                        except Exception as e:
                            provider_errors.append(f"{provider_name}: {e}")
                            self._write_audit_event(
                                state,
                                "gathering_provider_result",
                                data={
                                    "provider": provider_name,
                                    "sub_query_id": sub_query.id,
                                    "sub_query": sub_query.query,
                                    "sources_added": 0,
                                    "error": str(e),
                                },
                                level="warning",
                            )

                    if added > 0:
                        sub_query.mark_completed(findings=f"Found {added} sources")
                        logger.debug(
                            "Sub-query '%s' completed: %d sources",
                            sub_query.query[:50],
                            added,
                        )
                        return added, None

                    error_summary = "; ".join(provider_errors) or "No sources found"
                    sub_query.mark_failed(error_summary)
                    logger.warning(
                        "Sub-query '%s' failed: %s",
                        sub_query.query[:50],
                        error_summary,
                    )
                    return 0, error_summary

            # Check for cancellation before executing sub-query batch
            self._check_cancellation(state)

            # Execute all sub-queries concurrently
            tasks = [execute_sub_query(sq) for sq in pending_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            for result in results:
                # Check for BaseException (includes Exception, CancelledError, KeyboardInterrupt, etc.)
                # asyncio.gather with return_exceptions=True can return any BaseException
                if isinstance(result, BaseException):
                    failed_queries += 1
                    logger.error("Task exception: %s", result)
                else:
                    added, error = result
                    total_sources_added += added
                    if error:
                        failed_queries += 1

        except asyncio.CancelledError:
            # Handle cancellation: save interim state before re-raising
            logger.warning(
                "Gathering phase cancelled during sub-query execution for research %s",
                state.id,
            )
            try:
                state.updated_at = datetime.now(timezone.utc)
                self.memory.save_deep_research(state)
            except Exception as save_exc:
                logger.error(
                    "Error saving state during gathering cancellation for research %s: %s",
                    state.id,
                    save_exc,
                )
            raise
        finally:
            # Ensure state timestamp is updated on any exit
            state.updated_at = datetime.now(timezone.utc)

        # Capture circuit breaker states at end of gathering
        circuit_breaker_states_end = {
            name: resilience_manager.get_breaker_state(name).value for name in configured_provider_names
        }

        # Save state (normal execution path after finally block)
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "gathering_result",
            data={
                "source_count": total_sources_added,
                "queries_executed": len(pending_queries),
                "queries_failed": failed_queries,
                "unique_urls": len(seen_urls),
                "providers_used": [p.get_provider_name() for p in available_providers],
                "providers_unavailable": unavailable_providers,
                "circuit_breaker_states_start": circuit_breaker_states_start,
                "circuit_breaker_states_end": circuit_breaker_states_end,
            },
        )

        # Determine success
        success = total_sources_added > 0 or failed_queries < len(pending_queries)

        # Build error message if all queries failed
        error_msg = None
        if not success:
            providers_used = [p.get_provider_name() for p in available_providers]
            if failed_queries == len(pending_queries):
                error_msg = (
                    f"All {failed_queries} sub-queries failed to find sources. "
                    f"Providers used: {providers_used}. "
                    f"Unavailable providers: {unavailable_providers}"
                )

        logger.info(
            "Gathering phase complete: %d sources from %d queries (%d failed)",
            total_sources_added,
            len(pending_queries),
            failed_queries,
        )

        # Emit phase.completed audit event
        phase_duration_ms = (time.perf_counter() - phase_start_time) * 1000
        self._write_audit_event(
            state,
            "phase.completed",
            data={
                "phase_name": "gathering",
                "iteration": state.iteration,
                "task_id": state.id,
                "duration_ms": phase_duration_ms,
                "circuit_breaker_states": circuit_breaker_states_end,
            },
        )

        # Emit phase duration metric
        get_metrics().histogram(
            "foundry_mcp_research_phase_duration_seconds",
            phase_duration_ms / 1000.0,
            labels={"phase_name": "gathering", "status": "success" if success else "error"},
        )

        return WorkflowResult(
            success=success,
            content=f"Gathered {total_sources_added} sources from {len(pending_queries)} sub-queries",
            error=error_msg,
            metadata={
                "research_id": state.id,
                "source_count": total_sources_added,
                "queries_executed": len(pending_queries),
                "queries_failed": failed_queries,
                "unique_urls": len(seen_urls),
                "providers_used": [p.get_provider_name() for p in available_providers],
                "providers_unavailable": unavailable_providers,
                "circuit_breaker_states": {
                    "start": circuit_breaker_states_start,
                    "end": circuit_breaker_states_end,
                },
            },
        )

    # ------------------------------------------------------------------
    # Tavily Extract follow-up
    # ------------------------------------------------------------------

    async def _execute_extract_followup_async(
        self,
        state: DeepResearchState,
        max_urls: int = 5,
    ) -> Optional[dict[str, Any]]:
        """Execute Tavily Extract as optional follow-up after gathering phase.

        This step expands URL content for top-ranked sources discovered during search.
        It runs between GATHERING and ANALYSIS phases when enabled via config flag
        ``tavily_extract_in_deep_research``.

        Per acceptance criteria:
        - Extract can expand URLs discovered during search
        - Optional step controlled by config flag: tavily_extract_in_deep_research
        - Max 5 URLs extracted per deep research run (configurable)
        - URL prioritization: top-N by relevance score (quality)
        - Results integrated into source collection with extract_source=true metadata
        - Extraction occurs after search phase, before analysis phase

        Args:
            state: Current research state with sources from gathering
            max_urls: Maximum URLs to extract (default: 5)

        Returns:
            Dict with extraction stats or None on complete failure
        """
        import os

        # Get sources that have URLs but no content yet
        sources_with_urls = [s for s in state.sources if s.url and not s.content]

        if not sources_with_urls:
            logger.debug("No sources need content extraction")
            return {"urls_extracted": 0, "urls_failed": 0, "skipped": "no_eligible_sources"}

        # Prioritize by quality score (HIGH > MEDIUM > LOW > UNKNOWN)
        quality_order = {
            SourceQuality.HIGH: 0,
            SourceQuality.MEDIUM: 1,
            SourceQuality.LOW: 2,
            SourceQuality.UNKNOWN: 3,
        }
        sources_with_urls.sort(key=lambda s: quality_order.get(s.quality, 99))

        # Take top N URLs
        urls_to_extract = [s.url for s in sources_with_urls[:max_urls] if s.url]

        if not urls_to_extract:
            logger.debug("No URLs to extract after filtering")
            return {"urls_extracted": 0, "urls_failed": 0, "skipped": "no_urls_after_filter"}

        logger.info(
            "Executing extract follow-up: %d URLs (max %d)",
            len(urls_to_extract),
            max_urls,
        )

        # Get API key
        api_key = self.config.tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            logger.warning("Tavily API key not available for extract follow-up")
            return {"urls_extracted": 0, "urls_failed": len(urls_to_extract), "error": "no_api_key"}

        try:
            provider = TavilyExtractProvider(api_key=api_key)

            # Execute extraction
            extracted_sources = await provider.extract(
                urls=urls_to_extract,
                extract_depth=self.config.tavily_extract_depth,
                include_images=self.config.tavily_extract_include_images,
            )

            # Map extracted content back to existing sources and add extract_source metadata
            urls_extracted = 0
            for extracted in extracted_sources:
                # Find matching source by URL
                for source in state.sources:
                    if source.url == extracted.url:
                        # Update source with extracted content
                        source.content = extracted.content
                        if extracted.snippet and not source.snippet:
                            source.snippet = extracted.snippet
                        # Add extract_source=true to metadata
                        source.metadata["extract_source"] = True
                        source.metadata["extract_depth"] = extracted.metadata.get("extract_depth")
                        source.metadata["chunk_count"] = extracted.metadata.get("chunk_count")
                        urls_extracted += 1
                        break

            # Save updated state
            self.memory.save_deep_research(state)

            logger.info(
                "Extract follow-up complete: %d/%d URLs extracted",
                urls_extracted,
                len(urls_to_extract),
            )

            return {
                "urls_extracted": urls_extracted,
                "urls_failed": len(urls_to_extract) - urls_extracted,
            }

        except Exception as e:
            logger.error("Extract follow-up failed: %s", e)
            return {
                "urls_extracted": 0,
                "urls_failed": len(urls_to_extract),
                "error": str(e),
            }
