"""Per-topic ReAct research mixin for DeepResearchWorkflow.

Implements parallel sub-topic researcher agents that each run an
independent search → reflect → refine cycle for a single sub-query.
When enabled, the gathering phase delegates to these topic researchers
instead of flat parallel search.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    TopicResearchResult,
)
from foundry_mcp.core.research.models.sources import SourceQuality, SubQuery
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    content_similarity,
    extract_json,
    parse_reflection_decision,
)
from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    _normalize_title,
    get_domain_quality,
)

logger = logging.getLogger(__name__)


class TopicResearchMixin:
    """Per-topic ReAct research methods. Mixed into DeepResearchWorkflow.

    Provides ``_execute_topic_research_async`` which runs a mini ReAct loop
    for a single sub-query: search → reflect → (refine query → search)* →
    compile summary.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _search_providers (cache dict on instance)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _get_search_provider(), _get_tavily_search_kwargs(), etc. (from GatheringPhaseMixin)
    - _execute_provider_async() (from ResearchWorkflowBase)
    """

    config: Any
    memory: Any
    _search_providers: dict[str, Any]

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _check_cancellation(self, *args: Any, **kwargs: Any) -> None: ...
        def _get_search_provider(self, provider_name: str) -> Any: ...
        def _get_tavily_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]: ...
        def _get_perplexity_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]: ...
        def _get_semantic_scholar_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]: ...
        async def _execute_provider_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _compress_single_topic_async(self, *args: Any, **kwargs: Any) -> tuple[int, int, bool]: ...

    # ------------------------------------------------------------------
    # Single-topic ReAct loop
    # ------------------------------------------------------------------

    async def _execute_topic_research_async(
        self,
        sub_query: SubQuery,
        state: DeepResearchState,
        available_providers: list[Any],
        *,
        max_searches: int = 3,
        max_sources_per_provider: int | None = None,
        timeout: float = 120.0,
        seen_urls: set[str],
        seen_titles: dict[str, str],
        state_lock: asyncio.Lock,
        semaphore: asyncio.Semaphore,
    ) -> TopicResearchResult:
        """Execute a single-topic ReAct research loop with mandatory reflection.

        The loop runs: search → mandatory reflect → (refine query → search)*
        → compile summary. Reflection is always performed after each search
        iteration (not conditional on source count). The reflection step
        produces a structured decision that can signal early completion,
        request continued searching with a refined query, or stop.

        Args:
            sub_query: The sub-query to research
            state: Current research state (for config access and source storage)
            available_providers: List of initialized search providers
            max_searches: Maximum search iterations for this topic (hard cap)
            max_sources_per_provider: Max results to request from each provider
                per search call. When None, falls back to state.max_sources_per_query.
                Used for budget splitting across parallel topic researchers.
            timeout: Timeout per search operation
            seen_urls: Shared set of already-seen URLs (for deduplication)
            seen_titles: Shared dict of normalized titles (for deduplication)
            state_lock: Lock for thread-safe state mutations
            semaphore: Semaphore for concurrency control

        Returns:
            TopicResearchResult with per-topic findings
        """
        result = TopicResearchResult(sub_query_id=sub_query.id)
        current_query = sub_query.query
        # Accumulate tokens locally and merge under lock after the loop
        # to avoid a race on state.total_tokens_used from concurrent topics.
        local_tokens_used = 0
        # Track total tool calls (search + extract) toward budget
        tool_calls_used = 0
        async with state_lock:
            sub_query.status = "executing"

        # Resolve extract config once for the loop
        import os as _os

        extract_enabled = (
            getattr(self.config, "deep_research_enable_extract", True)
            and bool(
                getattr(self.config, "tavily_api_key", None)
                or _os.environ.get("TAVILY_API_KEY")
            )
        )
        extract_max_per_iter = getattr(
            self.config, "deep_research_extract_max_per_iteration", 2
        )

        for iteration in range(max_searches):
            self._check_cancellation(state)

            # --- Search step ---
            sources_added = await self._topic_search(
                query=current_query,
                sub_query=sub_query,
                state=state,
                available_providers=available_providers,
                max_sources_per_provider=max_sources_per_provider,
                timeout=timeout,
                seen_urls=seen_urls,
                seen_titles=seen_titles,
                state_lock=state_lock,
                semaphore=semaphore,
            )
            result.searches_performed += 1
            tool_calls_used += 1
            result.sources_found += sources_added

            # --- Cost-aware early-exit heuristic ---
            # If sources are high-quality and diverse, stop regardless of
            # remaining iterations.  This saves reflection + think + search
            # costs when coverage is already sufficient.
            if result.sources_found >= 3 and iteration > 0:
                early_exit = await self._check_early_exit(
                    sub_query=sub_query,
                    state=state,
                    state_lock=state_lock,
                )
                if early_exit:
                    result.early_completion = True
                    result.completion_rationale = (
                        "Early exit: sufficient high-quality diverse sources found"
                    )
                    logger.info(
                        "Topic %r early exit at iteration %d/%d: quality heuristic met",
                        sub_query.id,
                        iteration + 1,
                        max_searches,
                    )
                    break

            # If this is the last allowed iteration, skip reflection
            # (max_searches is the hard cap regardless of reflection)
            if tool_calls_used >= max_searches:
                logger.info(
                    "Topic %r hit max_tool_calls cap (%d/%d), stopping regardless of reflection",
                    sub_query.id,
                    tool_calls_used,
                    max_searches,
                )
                break

            # --- Mandatory reflection step ---
            # Always reflect after every search, not just when sources > 0
            reflection = await self._topic_reflect(
                original_query=sub_query.query,
                current_query=current_query,
                sources_found=result.sources_found,
                iteration=iteration + 1,
                max_iterations=max_searches,
                state=state,
            )
            local_tokens_used += reflection.get("tokens_used", 0)

            # Parse structured reflection decision
            decision = parse_reflection_decision(
                reflection.get("raw_response", "")
            )
            rationale = decision.rationale or reflection.get("assessment", "")
            result.reflection_notes.append(rationale)

            # Log every reflection decision for observability
            logger.info(
                "Topic %r reflection [iter %d/%d]: complete=%s, continue=%s, extract=%d urls, rationale=%s",
                sub_query.id,
                iteration + 1,
                max_searches,
                decision.research_complete,
                decision.continue_searching,
                len(decision.urls_to_extract or []),
                rationale[:120] if rationale else "(empty)",
            )

            # --- Optional extraction step ---
            # If reflection recommends URLs for extraction and we have budget,
            # extract full content before deciding whether to continue.
            if (
                extract_enabled
                and decision.urls_to_extract
                and tool_calls_used < max_searches
            ):
                urls_to_fetch = decision.urls_to_extract[:extract_max_per_iter]
                extract_added = await self._topic_extract(
                    urls=urls_to_fetch,
                    sub_query=sub_query,
                    state=state,
                    seen_urls=seen_urls,
                    seen_titles=seen_titles,
                    state_lock=state_lock,
                    semaphore=semaphore,
                    timeout=timeout,
                )
                tool_calls_used += 1  # count extraction batch as 1 tool call
                result.sources_found += extract_added
                if extract_added > 0:
                    result.reflection_notes.append(
                        f"[extract] Fetched {extract_added} source(s) from {len(urls_to_fetch)} URL(s)"
                    )

            # Check for explicit research completion signal
            if decision.research_complete:
                result.early_completion = True
                result.completion_rationale = rationale
                break

            # Check if reflection says to stop searching
            if not decision.continue_searching:
                result.completion_rationale = (
                    rationale or "Reflection decided to stop searching"
                )
                break

            # Check budget after extraction
            if tool_calls_used >= max_searches:
                logger.info(
                    "Topic %r hit max_tool_calls cap after extraction (%d/%d)",
                    sub_query.id,
                    tool_calls_used,
                    max_searches,
                )
                break

            # --- Think step (between reflect and next search) ---
            # Articulates: what was found, what angle to try next, why it
            # matters.  Grounds the query refinement in explicit reasoning.
            think_output = await self._topic_think(
                original_query=sub_query.query,
                current_query=current_query,
                reflection_rationale=rationale,
                refined_query_suggestion=decision.refined_query,
                sources_found=result.sources_found,
                iteration=iteration + 1,
                state=state,
            )
            local_tokens_used += think_output.get("tokens_used", 0)
            if think_output.get("reasoning"):
                result.reflection_notes.append(
                    f"[think] {think_output['reasoning']}"
                )

            # --- Refine step ---
            # Prefer the think step's refined query (more grounded in explicit
            # gap analysis) over the reflection's suggestion.
            think_refined = think_output.get("next_query")
            refined_query = think_refined or decision.refined_query
            if refined_query and refined_query != current_query:
                current_query = refined_query
                result.refined_queries.append(refined_query)
            elif sources_added == 0 and iteration == 0:
                # Fallback: strip quotes if present, otherwise broaden
                broadened = sub_query.query.replace('"', "").strip()
                if broadened != current_query:
                    current_query = broadened
                    result.refined_queries.append(broadened)
                else:
                    # No refinement possible and no sources — stop
                    break
            else:
                # No meaningful refinement possible, stop
                break

        # --- Compile per-topic summary ---
        # Merge accumulated tokens under lock (avoids race on state.total_tokens_used)
        async with state_lock:
            state.total_tokens_used += local_tokens_used
            result.source_ids = list(sub_query.source_ids)

        # mark_completed/mark_failed are called outside the lock. This is safe
        # because each sub_query is owned by exactly one topic coroutine — no
        # other coroutine reads or writes to this sub_query instance. The lock
        # above only protects shared state (total_tokens_used, source_ids list).
        if result.sources_found > 0:
            sub_query.mark_completed(
                findings=f"Topic research found {result.sources_found} sources "
                f"in {result.searches_performed} search(es)"
            )
        else:
            sub_query.mark_failed("No sources found after topic research loop")

        # --- Inline per-topic compression ---
        # Compress this topic's findings immediately so the supervision phase
        # can assess actual content coverage (not just source counts).
        # Gated by config flag; non-fatal on failure.
        inline_compression_enabled = getattr(
            self.config, "deep_research_inline_compression", True
        )
        if (
            inline_compression_enabled
            and result.sources_found > 0
            and result.compressed_findings is None
        ):
            try:
                comp_input, comp_output, comp_ok = await self._compress_single_topic_async(
                    topic_result=result,
                    state=state,
                    timeout=timeout,
                )
                if comp_ok:
                    logger.info(
                        "Inline compression for topic %r: %d tokens",
                        sub_query.id,
                        comp_input + comp_output,
                    )
                else:
                    logger.warning(
                        "Inline compression failed for topic %r, supervision will use metadata-only assessment",
                        sub_query.id,
                    )
                async with state_lock:
                    state.total_tokens_used += comp_input + comp_output
            except Exception as comp_exc:
                logger.warning(
                    "Inline compression exception for topic %r: %s. Non-fatal, continuing.",
                    sub_query.id,
                    comp_exc,
                )

        self._write_audit_event(
            state,
            "topic_research_complete",
            data={
                "sub_query_id": sub_query.id,
                "sub_query": sub_query.query,
                "searches_performed": result.searches_performed,
                "tool_calls_used": tool_calls_used,
                "sources_found": result.sources_found,
                "refined_queries": result.refined_queries,
                "reflection_notes": result.reflection_notes,
                "early_completion": result.early_completion,
                "completion_rationale": result.completion_rationale,
                "inline_compressed": result.compressed_findings is not None,
                "extract_enabled": extract_enabled,
            },
        )

        return result

    # ------------------------------------------------------------------
    # Search step (scoped to one sub-query)
    # ------------------------------------------------------------------

    async def _topic_search(
        self,
        query: str,
        sub_query: SubQuery,
        state: DeepResearchState,
        available_providers: list[Any],
        max_sources_per_provider: int | None,
        timeout: float,
        seen_urls: set[str],
        seen_titles: dict[str, str],
        state_lock: asyncio.Lock,
        semaphore: asyncio.Semaphore,
    ) -> int:
        """Execute search for a single query across all available providers.

        Args:
            query: Search query string
            sub_query: The SubQuery being researched
            state: Current research state
            available_providers: Search provider instances
            max_sources_per_provider: Max results per provider call (budget-split).
                Falls back to state.max_sources_per_query when None.
            timeout: Per-provider search timeout
            seen_urls: Shared URL dedup set
            seen_titles: Shared title dedup dict
            state_lock: Lock for thread-safe state mutations
            semaphore: Semaphore bounding concurrent search calls

        Returns the number of new (deduplicated) sources added to state.
        """
        from foundry_mcp.core.research.providers import SearchProviderError

        effective_max_results = (
            max_sources_per_provider if max_sources_per_provider is not None else state.max_sources_per_query
        )
        added = 0

        async with semaphore:
            for provider in available_providers:
                provider_name = provider.get_provider_name()

                try:
                    self._check_cancellation(state)

                    search_kwargs: dict[str, Any] = {
                        "query": query,
                        "max_results": effective_max_results,
                        "sub_query_id": sub_query.id,
                    }

                    # Add provider-specific kwargs
                    if provider_name == "tavily":
                        search_kwargs.update(self._get_tavily_search_kwargs(state))
                    elif provider_name == "perplexity":
                        search_kwargs.update(self._get_perplexity_search_kwargs(state))
                        search_kwargs["include_raw_content"] = state.follow_links
                    elif provider_name == "semantic_scholar":
                        search_kwargs.update(self._get_semantic_scholar_search_kwargs(state))
                        search_kwargs["include_raw_content"] = state.follow_links
                    else:
                        search_kwargs["include_raw_content"] = state.follow_links

                    sources = await asyncio.wait_for(
                        provider.search(**search_kwargs),
                        timeout=timeout,
                    )

                    for source in sources:
                        async with state_lock:
                            # URL-based deduplication
                            if source.url and source.url in seen_urls:
                                continue

                            # Title-based deduplication
                            normalized_title = _normalize_title(source.title)
                            if normalized_title and len(normalized_title) > 20:
                                if normalized_title in seen_titles:
                                    continue
                                seen_titles[normalized_title] = source.url or ""

                            # Content-similarity deduplication (Phase 5.3)
                            # Check if this source's content is substantially
                            # similar to an existing source (mirror/syndicated).
                            content_dedup_enabled = getattr(
                                self.config, "deep_research_enable_content_dedup", True
                            )
                            dedup_threshold = getattr(
                                self.config, "deep_research_content_dedup_threshold", 0.8
                            )
                            if (
                                content_dedup_enabled
                                and source.content
                                and len(source.content) > 100
                            ):
                                is_content_dup = False
                                for existing_src in state.sources:
                                    if (
                                        existing_src.content
                                        and len(existing_src.content) > 100
                                    ):
                                        sim = content_similarity(
                                            source.content, existing_src.content
                                        )
                                        if sim >= dedup_threshold:
                                            logger.debug(
                                                "Content dedup: %r (%.2f similar to %r)",
                                                source.url or source.title,
                                                sim,
                                                existing_src.url or existing_src.title,
                                            )
                                            is_content_dup = True
                                            break
                                if is_content_dup:
                                    continue

                            if source.url:
                                seen_urls.add(source.url)
                                if source.quality == SourceQuality.UNKNOWN:
                                    source.quality = get_domain_quality(source.url, state.research_mode)

                            # Add source to state (centralised citation assignment)
                            state.append_source(source)
                            sub_query.source_ids.append(source.id)
                            added += 1

                    # Track search provider query count
                    async with state_lock:
                        state.search_provider_stats[provider_name] = (
                            state.search_provider_stats.get(provider_name, 0) + 1
                        )

                except SearchProviderError as e:
                    logger.warning(
                        "Topic search provider %s error for query %r: %s",
                        provider_name,
                        query[:50],
                        e,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Topic search provider %s timed out for query %r",
                        provider_name,
                        query[:50],
                    )
                except Exception as e:
                    logger.warning(
                        "Topic search provider %s unexpected error for query %r: %s",
                        provider_name,
                        query[:50],
                        e,
                    )

        return added

    # ------------------------------------------------------------------
    # Reflect step (fast LLM evaluates search results)
    # ------------------------------------------------------------------

    async def _topic_reflect(
        self,
        original_query: str,
        current_query: str,
        sources_found: int,
        iteration: int,
        max_iterations: int,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Mandatory LLM reflection on topic search results.

        Called after every search iteration to evaluate research progress
        and produce a structured decision about next steps. The response
        includes the raw LLM text (``raw_response``) for structured
        parsing via ``parse_reflection_decision()``.

        Returns:
            Dict with keys: assessment (str), raw_response (str),
            tokens_used (int). The caller uses ``parse_reflection_decision``
            on ``raw_response`` to get the structured decision.
        """
        # Resolve provider and model via role-based hierarchy (Phase 6).
        # Falls back to phase-specific config, then global default.
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            resolve_phase_provider,
            safe_resolve_model_for_role,
        )

        provider_id, reflection_model = safe_resolve_model_for_role(self.config, "topic_reflection")
        if provider_id is None:
            provider_id = resolve_phase_provider(self.config, "topic_reflection", "reflection")

        # Collect source quality distribution for context
        source_qualities: dict[str, int] = {}
        for src in state.sources:
            if hasattr(src, "quality") and src.quality:
                quality_name = str(src.quality.value) if hasattr(src.quality, "value") else str(src.quality)
                source_qualities[quality_name] = source_qualities.get(quality_name, 0) + 1
        quality_summary = ", ".join(f"{k}={v}" for k, v in sorted(source_qualities.items())) or "none yet"

        # Collect distinct domains for coverage assessment
        distinct_domains: set[str] = set()
        for src in state.sources:
            if src.url:
                try:
                    from urllib.parse import urlparse

                    domain = urlparse(src.url).netloc.lower()
                    # Strip www. prefix for cleaner domain grouping
                    if domain.startswith("www."):
                        domain = domain[4:]
                    if domain:
                        distinct_domains.add(domain)
                except Exception:
                    pass
        domain_count = len(distinct_domains)

        system_prompt = (
            "You are a research assistant evaluating search results for a specific sub-topic. "
            "After each search iteration, you MUST assess whether the research is sufficient "
            "and decide the next action.\n\n"
            "Respond with valid JSON using this exact schema:\n"
            "{\n"
            '  "continue_searching": true/false,\n'
            '  "refined_query": "new search query if continuing" or null,\n'
            '  "research_complete": true/false,\n'
            '  "rationale": "explain your assessment and what specific gap remains if continuing",\n'
            '  "urls_to_extract": ["url1", "url2"] or null\n'
            "}\n\n"
            "Decision rules (follow strictly, in priority order):\n"
            "1. STOP IMMEDIATELY — set research_complete=true when you have 3+ sources "
            "from 2+ distinct domains AND at least one high-quality source. "
            "This is a hard stop; additional searches will not meaningfully improve coverage.\n"
            "2. STOP — set research_complete=true when you have 3+ relevant sources "
            "from distinct domains, even if none are explicitly high-quality.\n"
            "3. ADEQUATE — set continue_searching=false (without research_complete) if you "
            "have 2+ sources but they are from the same domain or cover only one perspective. "
            "This stops the loop without signalling full completion.\n"
            "4. CONTINUE — set continue_searching=true ONLY if fewer than 2 relevant sources "
            "have been found. You MUST provide a refined_query and your rationale MUST "
            "identify the specific information gap that another search would fill.\n"
            "5. NO RESULTS — if zero sources were found, set continue_searching=true and "
            "provide a broader or alternative refined_query.\n"
            "6. The rationale field is REQUIRED and must never be empty. It must explain "
            "your reasoning, not just restate the decision.\n\n"
            "URL extraction (urls_to_extract):\n"
            "- If a search result snippet suggests rich content behind a URL (e.g., detailed "
            "technical documentation, comparison tables, research papers, API specs), recommend "
            "extracting it by including the URL in urls_to_extract.\n"
            "- Only recommend extraction for URLs where the snippet indicates valuable detail "
            "you cannot get from the snippet alone.\n"
            "- Limit to at most 2 URLs per reflection. Set to null if no extraction is needed.\n\n"
            "IMPORTANT: Err on the side of stopping early. Each additional search iteration "
            "has diminishing returns. If you already have diverse, relevant sources, STOP.\n"
            "- Keep refined queries focused on the original research topic.\n"
            "- Return ONLY valid JSON, no additional text."
        )

        user_prompt = (
            f"Original research sub-topic: {original_query}\n"
            f"Current search query: {current_query}\n"
            f"Sources found so far: {sources_found}\n"
            f"Distinct source domains: {domain_count}\n"
            f"Source quality distribution: {quality_summary}\n"
            f"Search iteration: {iteration}/{max_iterations}\n\n"
            "Assess the research progress for this sub-topic and decide the next action."
        )

        try:
            result = await self._execute_provider_async(
                prompt=user_prompt,
                provider_id=provider_id,
                model=reflection_model,
                system_prompt=system_prompt,
                timeout=self.config.deep_research_reflection_timeout,
                temperature=0.2,
                phase="topic_reflection",
                fallback_providers=[],
                max_retries=1,
                retry_delay=2.0,
            )

            if not result.success:
                return {
                    "assessment": "Reflection call failed, proceeding",
                    "raw_response": json.dumps({
                        "continue_searching": False,
                        "research_complete": False,
                        "rationale": "Reflection call failed",
                    }),
                    "tokens_used": 0,
                }

            tokens_used = result.tokens_used or 0
            raw_content = result.content or ""

            # Extract assessment from the raw response for backward compat
            assessment = ""
            json_str = extract_json(raw_content)
            if json_str:
                try:
                    data = json.loads(json_str)
                    assessment = str(data.get("rationale", ""))
                except (json.JSONDecodeError, TypeError):
                    assessment = "Reflection JSON parse warning"

            return {
                "assessment": assessment,
                "raw_response": raw_content,
                "tokens_used": tokens_used,
            }

        except (asyncio.TimeoutError, OSError, ValueError, RuntimeError) as exc:
            logger.warning("Topic reflection failed: %s. Treating as stop.", exc)

        return {
            "assessment": "Reflection unavailable",
            "raw_response": json.dumps({
                "continue_searching": False,
                "research_complete": False,
                "rationale": "Reflection unavailable due to error",
            }),
            "tokens_used": 0,
        }

    # ------------------------------------------------------------------
    # Cost-aware early-exit heuristic
    # ------------------------------------------------------------------

    async def _check_early_exit(
        self,
        sub_query: SubQuery,
        state: DeepResearchState,
        state_lock: asyncio.Lock,
    ) -> bool:
        """Check if this topic has sufficient high-quality diverse sources.

        Returns True when: sources_found >= 3 AND distinct_domains >= 2
        AND at least one HIGH quality source among this sub-query's sources.

        This is a cheap heuristic check (no LLM call) that allows skipping
        both the reflection and any subsequent search iterations.
        """
        async with state_lock:
            # Collect sources belonging to this sub-query
            topic_source_ids = set(sub_query.source_ids)
            topic_sources = [s for s in state.sources if s.id in topic_source_ids]

        if len(topic_sources) < 3:
            return False

        # Count distinct domains
        distinct_domains: set[str] = set()
        has_high_quality = False
        for src in topic_sources:
            if src.url:
                try:
                    from urllib.parse import urlparse

                    domain = urlparse(src.url).netloc.lower()
                    if domain.startswith("www."):
                        domain = domain[4:]
                    if domain:
                        distinct_domains.add(domain)
                except Exception:
                    pass
            if hasattr(src, "quality") and src.quality == SourceQuality.HIGH:
                has_high_quality = True

        return len(distinct_domains) >= 2 and has_high_quality

    # ------------------------------------------------------------------
    # Extract step (fetch full content from promising URLs)
    # ------------------------------------------------------------------

    async def _topic_extract(
        self,
        urls: list[str],
        sub_query: SubQuery,
        state: DeepResearchState,
        seen_urls: set[str],
        seen_titles: dict[str, str],
        state_lock: asyncio.Lock,
        semaphore: asyncio.Semaphore,
        timeout: float = 60.0,
    ) -> int:
        """Extract full content from promising URLs found in search results.

        Uses Tavily Extract API to fetch and parse page content. Extracted
        sources are deduplicated against existing sources and added to state.

        Args:
            urls: URLs to extract (pre-validated, max per config)
            sub_query: The sub-query being researched (for source association)
            state: Current research state
            seen_urls: Shared URL dedup set
            seen_titles: Shared title dedup dict
            state_lock: Lock for thread-safe state mutations
            semaphore: Semaphore for concurrency control
            timeout: Per-extraction timeout

        Returns:
            Number of new sources added from extraction
        """
        import os

        from foundry_mcp.core.research.providers.tavily_extract import (
            TavilyExtractProvider,
        )

        if not urls:
            return 0

        api_key = getattr(self.config, "tavily_api_key", None) or os.environ.get(
            "TAVILY_API_KEY"
        )
        if not api_key:
            logger.debug("Tavily API key not available for topic extract")
            return 0

        added = 0
        async with semaphore:
            try:
                provider = TavilyExtractProvider(api_key=api_key)
                extract_depth = getattr(self.config, "tavily_extract_depth", "basic")

                extracted_sources = await asyncio.wait_for(
                    provider.extract(
                        urls=urls,
                        extract_depth=extract_depth,
                        format="markdown",
                        query=sub_query.query,
                    ),
                    timeout=timeout,
                )

                for source in extracted_sources:
                    async with state_lock:
                        # URL dedup
                        if source.url and source.url in seen_urls:
                            continue

                        # Title dedup
                        normalized_title = _normalize_title(source.title)
                        if normalized_title and len(normalized_title) > 20:
                            if normalized_title in seen_titles:
                                continue
                            seen_titles[normalized_title] = source.url or ""

                        if source.url:
                            seen_urls.add(source.url)
                            if source.quality == SourceQuality.UNKNOWN:
                                source.quality = get_domain_quality(
                                    source.url, state.research_mode
                                )

                        # Tag as extracted source
                        source.sub_query_id = sub_query.id
                        source.metadata["extract_source"] = True

                        state.append_source(source)
                        sub_query.source_ids.append(source.id)
                        added += 1

                logger.info(
                    "Topic extract for %r: %d/%d URLs yielded new sources",
                    sub_query.id,
                    added,
                    len(urls),
                )

            except asyncio.TimeoutError:
                logger.warning(
                    "Topic extract timed out for %r after %.1fs",
                    sub_query.id,
                    timeout,
                )
            except Exception as exc:
                logger.warning(
                    "Topic extract failed for %r: %s. Non-fatal, continuing.",
                    sub_query.id,
                    exc,
                )

        return added

    # ------------------------------------------------------------------
    # Think step (within-loop deliberation)
    # ------------------------------------------------------------------

    async def _topic_think(
        self,
        original_query: str,
        current_query: str,
        reflection_rationale: str,
        refined_query_suggestion: str | None,
        sources_found: int,
        iteration: int,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Deliberate about what was found and what angle to try next.

        This is the "think-tool" equivalent for per-topic research — a
        lightweight LLM call that articulates explicit reasoning about:
        1. What information has been gathered so far
        2. What specific gap remains
        3. What search angle would best fill that gap
        4. A refined query targeting that angle

        Uses the reflection (cheap) model to minimize cost.

        Returns:
            Dict with keys: reasoning (str), next_query (str or None),
            tokens_used (int).
        """
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            resolve_phase_provider,
            safe_resolve_model_for_role,
        )

        provider_id, think_model = safe_resolve_model_for_role(
            self.config, "topic_reflection"
        )
        if provider_id is None:
            provider_id = resolve_phase_provider(
                self.config, "topic_reflection", "reflection"
            )

        system_prompt = (
            "You are a research strategist. Given the current state of a "
            "topic investigation, think through what has been found and what "
            "specific angle would most improve coverage.\n\n"
            "Respond with valid JSON:\n"
            "{\n"
            '  "reasoning": "Your analysis of what was found, what gap '
            'remains, and why the next query would help",\n'
            '  "next_query": "A specific, targeted search query to fill '
            'the identified gap" or null\n'
            "}\n\n"
            "Guidelines:\n"
            "- Focus on WHAT IS MISSING, not what was already found\n"
            "- The next_query should target a different angle, perspective, "
            "or information type than previous searches\n"
            "- If the reflection's suggested query is good, you may refine "
            "it further or return null to accept it as-is\n"
            "- Keep reasoning concise (2-3 sentences)\n"
            "- Return ONLY valid JSON, no additional text."
        )

        user_prompt = (
            f"Original research topic: {original_query}\n"
            f"Current search query: {current_query}\n"
            f"Sources found so far: {sources_found}\n"
            f"Search iteration: {iteration}\n"
            f"Reflection assessment: {reflection_rationale}\n"
        )
        if refined_query_suggestion:
            user_prompt += (
                f"Reflection's suggested next query: {refined_query_suggestion}\n"
            )
        user_prompt += (
            "\nThink about what specific information gap remains and what "
            "search angle would best address it."
        )

        try:
            result = await self._execute_provider_async(
                prompt=user_prompt,
                provider_id=provider_id,
                model=think_model,
                system_prompt=system_prompt,
                timeout=self.config.deep_research_reflection_timeout,
                temperature=0.3,
                phase="topic_think",
                fallback_providers=[],
                max_retries=1,
                retry_delay=2.0,
            )

            if not result.success:
                return {"reasoning": "", "next_query": None, "tokens_used": 0}

            tokens_used = result.tokens_used or 0
            raw_content = result.content or ""

            # Parse JSON response
            json_str = extract_json(raw_content)
            if json_str:
                try:
                    data = json.loads(json_str)
                    return {
                        "reasoning": str(data.get("reasoning", "")),
                        "next_query": data.get("next_query"),
                        "tokens_used": tokens_used,
                    }
                except (json.JSONDecodeError, TypeError):
                    pass

            return {
                "reasoning": raw_content[:200],
                "next_query": None,
                "tokens_used": tokens_used,
            }

        except (asyncio.TimeoutError, OSError, ValueError, RuntimeError) as exc:
            logger.warning("Topic think step failed: %s. Continuing without.", exc)

        return {"reasoning": "", "next_query": None, "tokens_used": 0}
