"""Per-topic ReAct research mixin for DeepResearchWorkflow.

Implements parallel sub-topic researcher agents that each run an
independent tool-calling ReAct loop for a single sub-query. The
researcher LLM decides which tools to call (web_search, extract_content,
think, research_complete) via structured JSON responses, replacing the
prior fixed search → reflect → think → refine sequence.

This merges the separate reflect and think LLM calls into a single
call per turn, reducing LLM calls from 2 per iteration to 1 per turn.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    ExtractContentTool,
    ResearcherToolCall,
    ThinkTool,
    TopicResearchResult,
    WebSearchTool,
    parse_researcher_response,
)
from foundry_mcp.core.research.models.sources import SourceQuality, SubQuery
from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    _normalize_title,
    get_domain_quality,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Researcher system prompt template
# ------------------------------------------------------------------

_RESEARCHER_SYSTEM_PROMPT = """\
You are a focused research agent. Your task is to thoroughly research a specific topic by using the tools available to you.

## Available Tools

### web_search
Search the web for information.
Arguments: {{"query": "search query string", "max_results": 5}}
Returns: Search results with titles, URLs, and content summaries.

### extract_content
Extract full page content from promising URLs found in search results.
Arguments: {{"urls": ["url1", "url2"]}}  (max 2 URLs per call)
Returns: Full page content in markdown format.
Only available when extraction is enabled. If unavailable, focus on web_search.

### think
Pause and reflect on your research progress. Use this to assess what you've found, identify gaps, and plan your next steps.
Arguments: {{"reasoning": "your analysis of findings and gaps"}}
Returns: Acknowledgment. Does NOT count against your tool call budget.

### research_complete
Signal that your research is complete and summarize your findings.
Arguments: {{"summary": "comprehensive summary addressing the research question"}}
Returns: Confirmation. Call this when you are confident your findings address the research question.

## Response Format

Respond with a JSON object containing your tool calls for this turn:

```json
{{
  "tool_calls": [
    {{"tool": "web_search", "arguments": {{"query": "...", "max_results": 5}}}},
    {{"tool": "think", "arguments": {{"reasoning": "..."}}}}
  ]
}}
```

You may include multiple tool calls per turn. When calling think alongside other tools, think will be executed first.

## Research Strategy

- Start with broader searches, then narrow based on what you find.
- Use think to pause and assess your findings before deciding next steps.
- Prefer primary sources, official documentation, and peer-reviewed content.
- Seek diverse perspectives — multiple domains and viewpoints.
- Call research_complete when you are confident the findings address the research question, or when additional searches yield diminishing returns.
- Simple factual queries: 2-3 searches are usually sufficient.
- Complex multi-dimensional topics: use up to your budget limit.

## Budget

You have {remaining} of {total} tool calls remaining (web_search and extract_content count against budget; think and research_complete do not).

## Context

Today's date is {date}.
"""


def _build_researcher_system_prompt(
    *,
    budget_total: int,
    budget_remaining: int,
    extract_enabled: bool,
    date_str: str | None = None,
) -> str:
    """Build the researcher system prompt with budget and context.

    Args:
        budget_total: Total tool call budget for this researcher.
        budget_remaining: Remaining tool calls.
        extract_enabled: Whether extract_content tool is available.
        date_str: Today's date string. Defaults to UTC today.

    Returns:
        Formatted system prompt string.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = _RESEARCHER_SYSTEM_PROMPT.format(
        remaining=budget_remaining,
        total=budget_total,
        date=date_str,
    )

    if not extract_enabled:
        # Remove extract_content tool documentation
        prompt = prompt.replace(
            "### extract_content\n"
            "Extract full page content from promising URLs found in search results.\n"
            'Arguments: {"urls": ["url1", "url2"]}  (max 2 URLs per call)\n'
            "Returns: Full page content in markdown format.\n"
            "Only available when extraction is enabled. If unavailable, focus on web_search.\n\n",
            "",
        )

    return prompt


def _build_react_user_prompt(
    topic: str,
    message_history: list[dict[str, str]],
    budget_remaining: int,
    budget_total: int,
) -> str:
    """Build the user prompt for a ReAct turn from message history.

    Encodes the conversation history (previous tool calls and results)
    into a structured prompt for the next LLM call.

    Args:
        topic: The research topic/question.
        message_history: List of message dicts with role/content keys.
        budget_remaining: Remaining tool call budget.
        budget_total: Total tool call budget.

    Returns:
        Formatted user prompt string.
    """
    parts: list[str] = [f"<research_topic>\n{topic}\n</research_topic>"]

    if message_history:
        parts.append("\n<conversation_history>")
        for i, msg in enumerate(message_history):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "assistant":
                parts.append(f"\n<turn number=\"{i + 1}\" role=\"assistant\">\n{content}\n</turn>")
            elif role == "tool":
                tool_name = msg.get("tool", "unknown")
                parts.append(f"\n<turn number=\"{i + 1}\" role=\"tool\" tool=\"{tool_name}\">\n{content}\n</turn>")
        parts.append("\n</conversation_history>")

    parts.append(
        f"\n<budget>You have {budget_remaining} of {budget_total} tool calls remaining.</budget>"
    )
    parts.append(
        "\nRespond with your next action as a JSON object containing tool_calls. "
        "Return ONLY valid JSON, no additional text."
    )

    return "\n".join(parts)


class TopicResearchMixin:
    """Per-topic ReAct research methods. Mixed into DeepResearchWorkflow.

    Provides ``_execute_topic_research_async`` which runs a tool-calling
    ReAct loop for a single sub-query: the LLM decides which tools to
    call (web_search, extract_content, think, research_complete) and the
    loop dispatches them to existing infrastructure.

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
    # Single-topic ReAct loop (tool-calling researcher)
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
        """Execute a single-topic tool-calling ReAct research loop.

        The researcher LLM decides which tools to call each turn:
        web_search, extract_content, think, or research_complete.
        This replaces the prior fixed search → reflect → think → refine
        sequence, reducing LLM calls from 2 per iteration to 1 per turn.

        Args:
            sub_query: The sub-query to research.
            state: Current research state (for config access and source storage).
            available_providers: List of initialized search providers.
            max_searches: Maximum tool call budget for this topic (hard cap).
                Only web_search and extract_content count against this.
            max_sources_per_provider: Max results to request from each provider
                per search call. When None, falls back to state.max_sources_per_query.
            timeout: Timeout per search operation.
            seen_urls: Shared set of already-seen URLs (for deduplication).
            seen_titles: Shared dict of normalized titles (for deduplication).
            state_lock: Lock for thread-safe state mutations.
            semaphore: Semaphore for concurrency control.

        Returns:
            TopicResearchResult with per-topic findings.
        """
        result = TopicResearchResult(sub_query_id=sub_query.id)
        # Accumulate tokens locally and merge under lock after the loop
        local_tokens_used = 0
        # Track tool calls (web_search + extract_content) toward budget
        tool_calls_used = 0
        budget_remaining = max_searches

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

        # Resolve provider and model for the researcher LLM
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            resolve_phase_provider,
            safe_resolve_model_for_role,
        )

        provider_id, researcher_model = safe_resolve_model_for_role(
            self.config, "topic_reflection"
        )
        if provider_id is None:
            provider_id = resolve_phase_provider(
                self.config, "topic_reflection", "reflection"
            )

        # Build researcher system prompt
        system_prompt = _build_researcher_system_prompt(
            budget_total=max_searches,
            budget_remaining=budget_remaining,
            extract_enabled=extract_enabled,
        )

        # Message history for multi-turn conversation
        message_history: list[dict[str, str]] = []

        # Maximum turns to prevent infinite loops (safety net)
        max_turns = max_searches * 3  # generous: 3x budget allows think steps

        for turn in range(max_turns):
            self._check_cancellation(state)

            if budget_remaining <= 0:
                logger.info(
                    "Topic %r budget exhausted (%d/%d tool calls used), stopping",
                    sub_query.id,
                    tool_calls_used,
                    max_searches,
                )
                break

            # Build user prompt with conversation history
            user_prompt = _build_react_user_prompt(
                topic=sub_query.query,
                message_history=message_history,
                budget_remaining=budget_remaining,
                budget_total=max_searches,
            )

            # One LLM call per turn
            try:
                llm_result = await self._execute_provider_async(
                    prompt=user_prompt,
                    provider_id=provider_id,
                    model=researcher_model,
                    system_prompt=system_prompt,
                    timeout=self.config.deep_research_reflection_timeout,
                    temperature=0.3,
                    phase="topic_research",
                    fallback_providers=[],
                    max_retries=1,
                    retry_delay=2.0,
                )

                if not llm_result.success:
                    logger.warning(
                        "Topic %r researcher LLM call failed on turn %d: %s",
                        sub_query.id,
                        turn + 1,
                        llm_result.error,
                    )
                    break

                local_tokens_used += llm_result.tokens_used or 0

            except (asyncio.TimeoutError, OSError, ValueError, RuntimeError) as exc:
                logger.warning(
                    "Topic %r researcher LLM call exception on turn %d: %s",
                    sub_query.id,
                    turn + 1,
                    exc,
                )
                break

            # Parse tool calls from LLM response
            raw_content = llm_result.content or ""
            response = parse_researcher_response(raw_content)

            # No tool calls = model chose to stop
            if not response.tool_calls:
                logger.info(
                    "Topic %r researcher returned no tool calls on turn %d, stopping",
                    sub_query.id,
                    turn + 1,
                )
                break

            # Record the assistant's response in message history
            message_history.append({"role": "assistant", "content": raw_content})

            # Sort tool calls: Think first (before action tools), then others
            think_calls = [tc for tc in response.tool_calls if tc.tool == "think"]
            action_calls = [tc for tc in response.tool_calls if tc.tool != "think"]
            ordered_calls = think_calls + action_calls

            # Process each tool call
            loop_should_break = False
            for tool_call in ordered_calls:
                if tool_call.tool == "think":
                    tool_result_text = self._handle_think_tool(
                        tool_call=tool_call,
                        sub_query=sub_query,
                        result=result,
                    )
                    message_history.append({
                        "role": "tool",
                        "tool": "think",
                        "content": tool_result_text,
                    })

                elif tool_call.tool == "web_search":
                    if budget_remaining <= 0:
                        message_history.append({
                            "role": "tool",
                            "tool": "web_search",
                            "content": "Budget exhausted. No more searches allowed.",
                        })
                        continue

                    tool_result_text = await self._handle_web_search_tool(
                        tool_call=tool_call,
                        sub_query=sub_query,
                        state=state,
                        result=result,
                        available_providers=available_providers,
                        max_sources_per_provider=max_sources_per_provider,
                        timeout=timeout,
                        seen_urls=seen_urls,
                        seen_titles=seen_titles,
                        state_lock=state_lock,
                        semaphore=semaphore,
                    )
                    tool_calls_used += 1
                    budget_remaining -= 1
                    result.searches_performed += 1
                    message_history.append({
                        "role": "tool",
                        "tool": "web_search",
                        "content": tool_result_text,
                    })

                elif tool_call.tool == "extract_content":
                    if not extract_enabled:
                        message_history.append({
                            "role": "tool",
                            "tool": "extract_content",
                            "content": "Content extraction is not available.",
                        })
                        continue

                    if budget_remaining <= 0:
                        message_history.append({
                            "role": "tool",
                            "tool": "extract_content",
                            "content": "Budget exhausted. No more extractions allowed.",
                        })
                        continue

                    tool_result_text = await self._handle_extract_tool(
                        tool_call=tool_call,
                        sub_query=sub_query,
                        state=state,
                        result=result,
                        seen_urls=seen_urls,
                        seen_titles=seen_titles,
                        state_lock=state_lock,
                        semaphore=semaphore,
                        timeout=timeout,
                        extract_max=extract_max_per_iter,
                    )
                    tool_calls_used += 1
                    budget_remaining -= 1
                    message_history.append({
                        "role": "tool",
                        "tool": "extract_content",
                        "content": tool_result_text,
                    })

                elif tool_call.tool == "research_complete":
                    summary = tool_call.arguments.get("summary", "")
                    result.early_completion = True
                    result.completion_rationale = summary
                    message_history.append({
                        "role": "tool",
                        "tool": "research_complete",
                        "content": "Research complete. Findings recorded.",
                    })
                    loop_should_break = True
                    break

                else:
                    logger.warning(
                        "Topic %r researcher called unknown tool %r, ignoring",
                        sub_query.id,
                        tool_call.tool,
                    )

            if loop_should_break:
                break

        # --- Compile per-topic summary ---
        # Merge accumulated tokens under lock
        async with state_lock:
            state.total_tokens_used += local_tokens_used
            result.source_ids = list(sub_query.source_ids)

        if result.sources_found > 0:
            sub_query.mark_completed(
                findings=f"Topic research found {result.sources_found} sources "
                f"in {result.searches_performed} search(es)"
            )
        else:
            sub_query.mark_failed("No sources found after topic research loop")

        # --- Inline per-topic compression ---
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
                "turns_used": len([m for m in message_history if m.get("role") == "assistant"]),
            },
        )

        return result

    # ------------------------------------------------------------------
    # Tool dispatch handlers
    # ------------------------------------------------------------------

    def _handle_think_tool(
        self,
        tool_call: ResearcherToolCall,
        sub_query: SubQuery,
        result: TopicResearchResult,
    ) -> str:
        """Handle a Think tool call: log reasoning, return acknowledgment.

        Args:
            tool_call: The think tool call with reasoning argument.
            sub_query: The sub-query being researched (for logging).
            result: TopicResearchResult to update with reflection notes.

        Returns:
            Tool result string.
        """
        try:
            think_args = ThinkTool.model_validate(tool_call.arguments)
            reasoning = think_args.reasoning
        except Exception:
            reasoning = tool_call.arguments.get("reasoning", "")

        logger.info(
            "Topic %r think: %s",
            sub_query.id,
            reasoning[:200] if reasoning else "(empty)",
        )
        result.reflection_notes.append(f"[think] {reasoning}")
        return "Reflection recorded."

    async def _handle_web_search_tool(
        self,
        tool_call: ResearcherToolCall,
        sub_query: SubQuery,
        state: DeepResearchState,
        result: TopicResearchResult,
        available_providers: list[Any],
        max_sources_per_provider: int | None,
        timeout: float,
        seen_urls: set[str],
        seen_titles: dict[str, str],
        state_lock: asyncio.Lock,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Handle a WebSearch tool call: dispatch to search providers.

        Args:
            tool_call: The web_search tool call with query argument.
            sub_query: The sub-query being researched.
            state: Current research state.
            result: TopicResearchResult to update.
            available_providers: Search provider instances.
            max_sources_per_provider: Per-provider result cap.
            timeout: Search timeout.
            seen_urls: Shared URL dedup set.
            seen_titles: Shared title dedup dict.
            state_lock: Lock for thread-safe state mutations.
            semaphore: Semaphore for concurrency control.

        Returns:
            Formatted tool result string with search results.
        """
        try:
            search_args = WebSearchTool.model_validate(tool_call.arguments)
            query = search_args.query
        except Exception:
            query = tool_call.arguments.get("query", sub_query.query)

        # Track refined queries
        if query != sub_query.query:
            result.refined_queries.append(query)

        sources_added = await self._topic_search(
            query=query,
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
        result.sources_found += sources_added

        # Format search results for message history
        if sources_added == 0:
            return f'Search for "{query}" returned no new sources.'

        # Build formatted source listing from this search
        topic_source_ids = set(sub_query.source_ids)
        topic_sources = [s for s in state.sources if s.id in topic_source_ids]
        # Show the most recent sources (from this search)
        recent_sources = topic_sources[-sources_added:]

        lines: list[str] = [f"Found {sources_added} new source(s):"]
        for idx, src in enumerate(recent_sources, 1):
            lines.append(f"\n--- SOURCE {idx}: {src.title} ---")
            if src.url:
                lines.append(f"URL: {src.url}")
            if src.metadata.get("summarized") and src.content:
                lines.append(f"\nSUMMARY:\n{src.content}")
            elif src.snippet:
                lines.append(f"\nSNIPPET:\n{src.snippet}")
            elif src.content:
                truncated = src.content[:500]
                if len(src.content) > 500:
                    truncated += "..."
                lines.append(f"\nCONTENT:\n{truncated}")

        return "\n".join(lines)

    async def _handle_extract_tool(
        self,
        tool_call: ResearcherToolCall,
        sub_query: SubQuery,
        state: DeepResearchState,
        result: TopicResearchResult,
        seen_urls: set[str],
        seen_titles: dict[str, str],
        state_lock: asyncio.Lock,
        semaphore: asyncio.Semaphore,
        timeout: float,
        extract_max: int = 2,
    ) -> str:
        """Handle an ExtractContent tool call: fetch full page content.

        Args:
            tool_call: The extract_content tool call with URLs argument.
            sub_query: The sub-query being researched.
            state: Current research state.
            result: TopicResearchResult to update.
            seen_urls: Shared URL dedup set.
            seen_titles: Shared title dedup dict.
            state_lock: Lock for thread-safe state mutations.
            semaphore: Semaphore for concurrency control.
            timeout: Extraction timeout.
            extract_max: Maximum URLs per extraction call.

        Returns:
            Formatted tool result string with extracted content.
        """
        try:
            extract_args = ExtractContentTool.model_validate(tool_call.arguments)
            urls = extract_args.urls[:extract_max]
        except Exception:
            raw_urls = tool_call.arguments.get("urls", [])
            if isinstance(raw_urls, list):
                urls = [str(u) for u in raw_urls if isinstance(u, str)][:extract_max]
            else:
                return "Invalid URLs argument."

        if not urls:
            return "No valid URLs provided for extraction."

        extract_added = await self._topic_extract(
            urls=urls,
            sub_query=sub_query,
            state=state,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
            state_lock=state_lock,
            semaphore=semaphore,
            timeout=timeout,
        )
        result.sources_found += extract_added

        if extract_added > 0:
            result.reflection_notes.append(
                f"[extract] Fetched {extract_added} source(s) from {len(urls)} URL(s)"
            )
            return f"Extracted content from {extract_added} of {len(urls)} URL(s)."
        else:
            return f"Extraction from {len(urls)} URL(s) yielded no new content."

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
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            content_similarity,
        )

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

                            # Content-similarity deduplication
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
