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
from foundry_mcp.core.research.workflows.deep_research._helpers import extract_json
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
        """Execute a single-topic ReAct research loop.

        The loop runs: search → reflect → (refine query → search)* →
        compile summary. Each iteration searches, then reflects on whether
        enough information was found. If gaps remain and budget allows,
        the query is refined and another search is performed.

        Args:
            sub_query: The sub-query to research
            state: Current research state (for config access and source storage)
            available_providers: List of initialized search providers
            max_searches: Maximum search iterations for this topic
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
        async with state_lock:
            sub_query.status = "executing"

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
            result.sources_found += sources_added

            # If this is the last allowed iteration, skip reflection
            if iteration >= max_searches - 1:
                break

            # If no sources found at all, try a refined query via reflection
            if sources_added == 0 and iteration == 0:
                result.reflection_notes.append(
                    f"No sources found for query: {current_query!r}. Requesting LLM refinement."
                )
                # Use the reflect step to get a properly refined query
                zero_reflection = await self._topic_reflect(
                    original_query=sub_query.query,
                    current_query=current_query,
                    sources_found=0,
                    iteration=1,
                    max_iterations=max_searches,
                    state=state,
                )
                local_tokens_used += zero_reflection.get("tokens_used", 0)
                refined_query = zero_reflection.get("refined_query")
                if refined_query and refined_query != current_query:
                    current_query = refined_query
                    result.refined_queries.append(refined_query)
                else:
                    # Fallback: strip quotes if present, otherwise broaden
                    broadened = sub_query.query.replace('"', "").strip()
                    if broadened != current_query:
                        current_query = broadened
                        result.refined_queries.append(broadened)
                continue

            # --- Reflect step ---
            reflection = await self._topic_reflect(
                original_query=sub_query.query,
                current_query=current_query,
                sources_found=result.sources_found,
                iteration=iteration + 1,
                max_iterations=max_searches,
                state=state,
            )
            local_tokens_used += reflection.get("tokens_used", 0)

            result.reflection_notes.append(reflection.get("assessment", ""))

            if reflection.get("sufficient", True):
                # Enough information gathered for this topic
                break

            # --- Refine step ---
            refined_query = reflection.get("refined_query")
            if refined_query and refined_query != current_query:
                current_query = refined_query
                result.refined_queries.append(refined_query)
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

        self._write_audit_event(
            state,
            "topic_research_complete",
            data={
                "sub_query_id": sub_query.id,
                "sub_query": sub_query.query,
                "searches_performed": result.searches_performed,
                "sources_found": result.sources_found,
                "refined_queries": result.refined_queries,
                "reflection_notes": result.reflection_notes,
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
            max_sources_per_provider
            if max_sources_per_provider is not None
            else state.max_sources_per_query
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
        """Fast LLM reflection on topic search results.

        Evaluates whether enough information has been gathered for this
        topic and optionally suggests a refined query.

        Returns:
            Dict with keys: sufficient (bool), assessment (str),
            refined_query (optional str)
        """
        from foundry_mcp.core.research.workflows.deep_research._helpers import resolve_phase_provider

        provider_id = resolve_phase_provider(
            self.config, "topic_reflection", "reflection"
        )

        system_prompt = (
            "You are a research assistant evaluating search results for a specific sub-topic. "
            "Determine if enough information has been gathered or if the search query should be refined.\n\n"
            "Respond with valid JSON:\n"
            '{"sufficient": true/false, "assessment": "brief assessment", '
            '"refined_query": "optional refined search query if not sufficient"}\n\n'
            "Rules:\n"
            "- Set sufficient=true if at least 2-3 relevant sources were found\n"
            "- Set sufficient=true if this is already a refined query and sources were found\n"
            "- If insufficient, suggest a refined_query that is more specific or uses different terms\n"
            "- Keep refined queries focused on the original topic\n"
            "- Return ONLY valid JSON"
        )

        user_prompt = (
            f"Original research sub-topic: {original_query}\n"
            f"Current search query: {current_query}\n"
            f"Sources found so far: {sources_found}\n"
            f"Search iteration: {iteration}/{max_iterations}\n\n"
            "Is the information gathered sufficient for this sub-topic, "
            "or should the search query be refined?"
        )

        try:
            result = await self._execute_provider_async(
                prompt=user_prompt,
                provider_id=provider_id,
                model=None,
                system_prompt=system_prompt,
                timeout=self.config.deep_research_reflection_timeout,
                temperature=0.2,
                phase="topic_reflection",
                fallback_providers=[],
                max_retries=1,
                retry_delay=2.0,
            )

            if not result.success:
                return {"sufficient": True, "assessment": "Reflection call failed, proceeding",
                        "tokens_used": 0}

            tokens_used = result.tokens_used or 0

            json_str = extract_json(result.content)
            if json_str:
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as exc:
                    logger.warning("Topic reflection JSON parse failed: %s", exc)
                    return {"sufficient": True, "assessment": "Reflection JSON invalid",
                            "tokens_used": tokens_used}
                return {
                    "sufficient": bool(data.get("sufficient", True)),
                    "assessment": str(data.get("assessment", "")),
                    "refined_query": data.get("refined_query"),
                    "tokens_used": tokens_used,
                }

            return {"sufficient": True, "assessment": "No JSON in reflection response",
                    "tokens_used": tokens_used}

        except (asyncio.TimeoutError, OSError, ValueError, RuntimeError) as exc:
            logger.warning("Topic reflection failed: %s. Treating as sufficient.", exc)

        return {"sufficient": True, "assessment": "Reflection unavailable",
                "tokens_used": 0}
