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
import re
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
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    sanitize_external_content,
)
from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    _extract_domain,
    _normalize_title,
    get_domain_quality,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Search result formatting helpers (Phase 4 ODR alignment)
# ------------------------------------------------------------------


def _format_source_block(
    idx: int,
    src: Any,
    novelty_tag: Any,
) -> str:
    """Format a single source into a structured, citation-friendly block.

    Produces a numbered-source layout that makes it easy for the researcher
    to reference specific sources and for the compression step to preserve
    citations — matching the ODR presentation pattern.

    Args:
        idx: 1-based source index within the search batch.
        src: ResearchSource object.
        novelty_tag: NoveltyTag for this source.

    Returns:
        Formatted multi-line string for one source.
    """
    # Sanitize web-derived fields before interpolation into LLM prompts
    safe_title = sanitize_external_content(src.title)
    safe_snippet = sanitize_external_content(src.snippet) if src.snippet else ""
    safe_content = sanitize_external_content(src.content) if src.content else ""

    lines: list[str] = []
    lines.append(f"--- SOURCE {idx}: {safe_title} ---")
    if src.url:
        lines.append(f"URL: {src.url}")
    lines.append(f"NOVELTY: {novelty_tag.tag}")

    # Content presentation: prefer structured summary with separate excerpts,
    # fall back to snippet, then truncated raw content.
    if src.metadata.get("summarized") and safe_content:
        # Extract the executive summary from the XML-tagged content.
        # The raw excerpts list is stored in metadata by _summarize_search_results.
        summary_text = safe_content
        # If we have structured excerpts in metadata, present them separately
        # rather than embedded in XML tags within the content.
        excerpts = src.metadata.get("excerpts")
        if excerpts:
            # Strip the <key_excerpts>...</key_excerpts> from the content to
            # avoid duplication — the summary tag is left for the SUMMARY block.
            summary_text = re.sub(
                r"\n*<key_excerpts>.*?</key_excerpts>",
                "",
                summary_text,
                flags=re.DOTALL,
            ).strip()
            # Also strip the <summary> tags for cleaner presentation
            summary_text = re.sub(
                r"</?summary>", "", summary_text
            ).strip()
            lines.append(f"\nSUMMARY:\n{summary_text}")
            safe_excerpts = [sanitize_external_content(e) for e in excerpts]
            excerpt_lines = "\n".join(f'- "{e}"' for e in safe_excerpts)
            lines.append(f"\nKEY EXCERPTS:\n{excerpt_lines}")
        else:
            # No separate excerpts — show the full summarized content
            summary_text = re.sub(
                r"</?summary>", "", summary_text
            ).strip()
            summary_text = re.sub(
                r"</?key_excerpts>", "", summary_text
            ).strip()
            lines.append(f"\nSUMMARY:\n{summary_text}")
    elif safe_snippet:
        lines.append(f"\nSNIPPET:\n{safe_snippet}")
    elif safe_content:
        truncated = safe_content[:500]
        if len(safe_content) > 500:
            truncated += "..."
        lines.append(f"\nCONTENT:\n{truncated}")

    return "\n".join(lines)


def _format_search_results_batch(
    sources: list[Any],
    novelty_tags: list[Any],
    novelty_header: str,
) -> str:
    """Format a batch of search results with header and numbered sources.

    Produces a structured, citation-friendly presentation matching the
    ODR pattern: batch header with domain summary, then numbered source
    blocks with novelty annotations.

    Args:
        sources: List of ResearchSource objects from this search.
        novelty_tags: Parallel list of NoveltyTag objects.
        novelty_header: Pre-built novelty summary string.

    Returns:
        Complete formatted string for the tool response message.
    """
    sources_count = len(sources)

    # Compute unique domains for the batch header (Phase 4b)
    domains: set[str] = set()
    for src in sources:
        domain = _extract_domain(src.url) if src.url else None
        if domain:
            domains.add(domain)
    domain_count = len(domains)

    # Build batch header
    header = (
        f"Found {sources_count} new source(s) from {domain_count} domain(s).\n"
        f"{novelty_header}"
    )

    # Build per-source blocks
    blocks: list[str] = [header]
    for idx, (src, ntag) in enumerate(zip(sources, novelty_tags), 1):
        blocks.append(_format_source_block(idx, src, ntag))

    return "\n\n".join(blocks)


# ------------------------------------------------------------------
# Researcher system prompt template
# ------------------------------------------------------------------

_RESEARCHER_SYSTEM_PROMPT = """\
You are a focused research agent. Your task is to thoroughly research a specific topic by using the tools available to you.

## Available Tools

### web_search
Search the web for information. Supports single or batch queries.
Arguments: {{"query": "search query string", "max_results": 5}}
  — OR for batch searches —
Arguments: {{"queries": ["query 1", "query 2", ...], "max_results": 5}}
Returns: Search results with titles, URLs, and content summaries. Batch queries return one consolidated, deduplicated result set. Each query in a batch counts against your tool call budget.

### extract_content
Extract full page content from promising URLs found in search results.
Arguments: {{"urls": ["url1", "url2"]}}  (max 2 URLs per call)
Returns: Full page content in markdown format.
Only available when extraction is enabled. If unavailable, focus on web_search.

### think
Pause and reflect on your research progress. Research quality improves when you periodically assess what you've found vs. what's still missing, rather than firing searches reactively.
Arguments: {{"reasoning": "your analysis of findings and gaps"}}
Returns: Acknowledgment. Does NOT count against your tool call budget — it's free precisely to encourage this reflection.
After each web_search or extract_content, call think as your next action before issuing another search. Use the `queries` parameter to search multiple angles at once for initial broad coverage.

### research_complete
Signal that your research is complete and summarize your findings.
Arguments: {{"summary": "comprehensive summary addressing the research question"}}
Returns: Confirmation. Call this when you are confident your findings address the research question.

## Response Format

Respond with a JSON object containing your tool calls for this turn:

```json
{{
  "tool_calls": [
    {{"tool": "web_search", "arguments": {{"query": "...", "max_results": 5}}}}
  ]
}}
```

Generally include one tool call per turn. For broad initial coverage, use the batch `queries` parameter instead of multiple tool calls.

## Research Strategy

- Start with broader searches, then narrow based on what you find. Broad-first avoids premature narrowing and reveals unexpected angles the user didn't anticipate; narrowing too early risks missing entire dimensions of the topic.
- Prefer primary sources, official documentation, and peer-reviewed content. Secondary sources introduce interpretation drift, may be outdated, and can't be independently verified — primary sources let downstream synthesis draw its own conclusions.
- Seek diverse perspectives — multiple domains and viewpoints. Single-perspective research produces blind spots that undermine user trust when they later encounter contradicting information.
- Simple factual queries: 2-3 searches are usually sufficient — factual queries converge quickly (additional searches return the same answer).
- Complex multi-dimensional topics: use up to your budget limit — these need coverage across distinct facets.

## Stop Immediately When

Call `research_complete` as soon as ANY of the following conditions is true:

1. **Comprehensive answer available**: You can fully and confidently answer the research question with what you have already found.
2. **3+ high-quality relevant sources**: You have found 3 or more high-quality, directly relevant sources that substantiate your answer. Three is the minimum for triangulation — if three independent sources agree, the finding is robust enough for confident reporting.
3. **Diminishing returns**: Your last 2 searches returned substantially similar information — this signals topic saturation, so further searches will likely resurface the same content, wasting budget.
4. **Futility stop**: Always call `research_complete` after 5 search tool calls if you have not found adequate sources — some topics are poorly covered online, and continuing past 5 attempts risks burning the entire budget on a dry well while other topics await research. Report what you found and note the gaps.

Do NOT exhaust your budget just because you can. Past saturation, additional searches add noise without new signal — the token budget is better spent on synthesis quality than marginal search results. Stop early when one of these conditions is met.

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

        # Reflection enforcement tracking (Phase 2)
        previous_turn_had_search = False
        search_turn_count = 0  # for first-turn exception
        reflection_injections = 0  # count of synthetic reflection prompts

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

            # Parse tool calls from LLM response, retrying on parse failure
            raw_content = llm_result.content or ""
            response = parse_researcher_response(raw_content)

            # Retry on parse failure: re-prompt the LLM with a clarifying suffix
            # up to 2 times (matching ODR's stop_after_attempt=3 total attempts).
            parse_retries = 0
            while response.parse_failed and parse_retries < 2:
                parse_retries += 1
                result.tool_parse_failures += 1
                logger.warning(
                    "Topic %r researcher returned unparseable JSON on turn %d "
                    "(retry %d/2), re-prompting with format clarification",
                    sub_query.id,
                    turn + 1,
                    parse_retries,
                )
                # Append the failed response + clarification to history
                message_history.append({"role": "assistant", "content": raw_content})
                message_history.append({
                    "role": "tool",
                    "tool": "system",
                    "content": (
                        "Your previous response was not valid JSON. Please respond "
                        "with ONLY a JSON object in the exact format:\n"
                        '{"tool_calls": [{"tool": "tool_name", "arguments": {...}}]}\n'
                        "Do not include any text outside the JSON object."
                    ),
                })
                retry_user_prompt = _build_react_user_prompt(
                    topic=sub_query.query,
                    message_history=message_history,
                    budget_remaining=budget_remaining,
                    budget_total=max_searches,
                )
                try:
                    retry_result = await self._execute_provider_async(
                        prompt=retry_user_prompt,
                        provider_id=provider_id,
                        model=researcher_model,
                        system_prompt=system_prompt,
                        timeout=self.config.deep_research_reflection_timeout,
                        temperature=0.2,  # lower temp for format compliance
                        phase="topic_research",
                        fallback_providers=[],
                        max_retries=1,
                        retry_delay=2.0,
                    )
                    if not retry_result.success:
                        break
                    local_tokens_used += retry_result.tokens_used or 0
                    raw_content = retry_result.content or ""
                    response = parse_researcher_response(raw_content)
                except (asyncio.TimeoutError, OSError, ValueError, RuntimeError):
                    break

            # No tool calls = model chose to stop (or all retries exhausted)
            if not response.tool_calls:
                logger.info(
                    "Topic %r researcher returned no tool calls on turn %d%s, stopping",
                    sub_query.id,
                    turn + 1,
                    f" (after {parse_retries} parse retries)" if parse_retries > 0 else "",
                )
                break

            # --- Reflection enforcement (Phase 2) ---
            # After the first search turn, require think between searches.
            current_has_search = any(
                tc.tool in ("web_search", "extract_content")
                for tc in response.tool_calls
            )
            current_has_think = any(
                tc.tool == "think" for tc in response.tool_calls
            )

            if (
                previous_turn_had_search
                and current_has_search
                and not current_has_think
                and search_turn_count > 0  # first search turn is exempt
            ):
                logger.warning(
                    "Topic %r: researcher skipped reflection after search on turn %d, "
                    "injecting synthetic think prompt",
                    sub_query.id,
                    turn + 1,
                )
                reflection_injections += 1
                message_history.append({"role": "assistant", "content": raw_content})
                message_history.append({
                    "role": "tool",
                    "tool": "system",
                    "content": (
                        "REFLECTION REQUIRED: You must call `think` to reflect on your "
                        "previous search results before issuing another search. Assess "
                        "what you found, identify gaps, and plan your next step. "
                        "Respond with ONLY a `think` tool call."
                    ),
                })
                continue

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

                    tool_result_text, queries_charged = await self._handle_web_search_tool(
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
                        budget_remaining=budget_remaining,
                    )
                    tool_calls_used += queries_charged
                    budget_remaining -= queries_charged
                    result.searches_performed += queries_charged
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

            # Update reflection tracking
            if current_has_search:
                search_turn_count += 1
            previous_turn_had_search = current_has_search

            if loop_should_break:
                break

        # --- Persist message history for downstream compression ---
        # The raw conversation gives compression the researcher's full
        # reasoning chain, failed attempts, and iterative refinements —
        # significantly better than structured metadata alone.
        result.message_history = list(message_history)

        # --- Build raw_notes from message history (Phase 1 ODR alignment) ---
        # Capture unprocessed concatenation of all tool-result and assistant
        # messages before compression. This provides a fallback when compression
        # degrades or fails, and ground-truth evidence for the evaluator.
        raw_notes_parts: list[str] = []
        for msg in message_history:
            if msg.get("role") in ("assistant", "tool"):
                content = msg.get("content", "")
                if content:
                    raw_notes_parts.append(content)
        if raw_notes_parts:
            raw_notes_text = "\n".join(raw_notes_parts)
            try:
                max_raw_notes_len = int(
                    getattr(self.config, "deep_research_max_content_length", 50_000)
                )
            except (TypeError, ValueError):
                max_raw_notes_len = 50_000
            if len(raw_notes_text) > max_raw_notes_len:
                raw_notes_text = raw_notes_text[:max_raw_notes_len]
            result.raw_notes = raw_notes_text

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
                "reflection_injections": reflection_injections,
                "tool_parse_failures": result.tool_parse_failures,
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
        return (
            "Reflection recorded. Before your next search, check the stop criteria:\n"
            "- Do I have 3+ high-quality relevant sources?\n"
            "- Did my last 2 searches return substantially similar information?\n"
            "- Check novelty tags: if most recent results are [RELATED] or [DUPLICATE], "
            "additional searches are unlikely to yield new insights.\n"
            "- Can I answer the research question comprehensively now?\n"
            "If YES to any, call research_complete instead of searching again."
        )

    async def _summarize_search_results(
        self,
        sources: list[Any],
        state: DeepResearchState,
        state_lock: asyncio.Lock,
    ) -> None:
        """Summarize newly added search results that have long raw content.

        Uses SourceSummarizer to produce structured summaries at fetch time,
        reducing context consumption in the researcher's message history by
        60-70%.  Sources already summarized (e.g. by the provider layer) are
        skipped.

        On timeout or failure, the source retains its raw content — matching
        ODR's fallback pattern (utils.py:175-213).

        Args:
            sources: List of ResearchSource objects to consider for summarization.
            state: Current research state (for token accounting).
            state_lock: Lock for thread-safe state mutations.
        """
        from foundry_mcp.core.research.providers.shared import SourceSummarizer
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            resolve_phase_provider,
            safe_resolve_model_for_role,
        )

        min_content_length = int(
            getattr(self.config, "deep_research_summarization_min_content_length", 300)
        )
        per_result_timeout = float(
            getattr(self.config, "deep_research_summarization_timeout", 60)
        )
        # Cap per-result timeout for inline summarization (30s default)
        per_result_timeout = min(per_result_timeout, 30.0)

        # Filter to sources that need summarization
        candidates = [
            src for src in sources
            if not src.metadata.get("summarized")
            and src.content
            and len(src.content) > min_content_length
        ]
        if not candidates:
            return

        # Resolve summarization provider/model
        role_provider, role_model = safe_resolve_model_for_role(
            self.config, "summarization"
        )
        provider_id = role_provider or resolve_phase_provider(
            self.config, "summarization"
        )

        summarizer = SourceSummarizer(
            provider_id=provider_id,
            model=role_model,
            timeout=per_result_timeout,
            max_concurrent=3,
            max_content_length=getattr(
                self.config, "deep_research_max_content_length", 50_000
            ),
        )

        results = await summarizer.summarize_sources(candidates)

        tokens_used = 0
        for src in candidates:
            if src.id in results:
                summary_result = results[src.id]
                # Preserve original content for downstream compression fidelity
                src.raw_content = src.content
                # Replace with structured summary
                src.content = SourceSummarizer.format_summarized_content(
                    summary_result.executive_summary,
                    summary_result.key_excerpts,
                )
                src.metadata["summarized"] = True
                src.metadata["excerpts"] = summary_result.key_excerpts
                src.metadata["summarization_input_tokens"] = summary_result.input_tokens
                src.metadata["summarization_output_tokens"] = summary_result.output_tokens
                tokens_used += (summary_result.input_tokens + summary_result.output_tokens)

        if tokens_used > 0:
            async with state_lock:
                state.total_tokens_used += tokens_used

        logger.info(
            "Inline summarization: %d/%d sources summarized (%d tokens)",
            len(results),
            len(candidates),
            tokens_used,
        )

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
        budget_remaining: int = 1,
    ) -> tuple[str, int]:
        """Handle a WebSearch tool call: dispatch to search providers.

        Supports both single-query and batch-query forms. When multiple
        queries are provided, they are dispatched in parallel via
        ``asyncio.gather`` over ``_topic_search``. Cross-query dedup is
        automatic via the shared ``seen_urls``/``seen_titles`` sets.

        Args:
            tool_call: The web_search tool call with query/queries argument.
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
            budget_remaining: Remaining tool call budget; batch is capped
                to this value so the researcher cannot overspend.

        Returns:
            Tuple of (formatted tool result string, queries_charged).
        """
        # --- Parse queries (batch or single) ---
        try:
            search_args = WebSearchTool.model_validate(tool_call.arguments)
            queries = list(search_args.queries)  # type: ignore[arg-type]
        except Exception:
            raw_query = tool_call.arguments.get("query", sub_query.query)
            raw_queries = tool_call.arguments.get("queries")
            if isinstance(raw_queries, list) and raw_queries:
                queries = [str(q) for q in raw_queries]
            else:
                queries = [str(raw_query)]

        # Cap batch to budget_remaining so researcher can't overspend
        queries = queries[:max(budget_remaining, 1)]
        queries_charged = len(queries)

        # Track refined queries
        for q in queries:
            if q != sub_query.query:
                result.refined_queries.append(q)

        # --- Dispatch searches (parallel for batch) ---
        if len(queries) == 1:
            # Fast path: single query, no gather overhead
            sources_added = await self._topic_search(
                query=queries[0],
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
        else:
            # Batch path: parallel dispatch via asyncio.gather
            per_query_results = await asyncio.gather(
                *(
                    self._topic_search(
                        query=q,
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
                    for q in queries
                ),
                return_exceptions=True,
            )
            sources_added = 0
            for i, r in enumerate(per_query_results):
                if isinstance(r, Exception):
                    logger.warning(
                        "Batch query %r failed: %s", queries[i], r,
                    )
                elif isinstance(r, int):
                    sources_added += r

        result.sources_found += sources_added

        # Format search results for message history
        query_label = (
            f'"{queries[0]}"'
            if len(queries) == 1
            else f"{len(queries)} queries ({', '.join(repr(q) for q in queries)})"
        )
        if sources_added == 0:
            return (
                f"Search for {query_label} returned no new sources.",
                queries_charged,
            )

        # Build formatted source listing from this search
        topic_source_ids = set(sub_query.source_ids)
        topic_sources = [s for s in state.sources if s.id in topic_source_ids]
        # Show the most recent sources (from this search)
        recent_sources = topic_sources[-sources_added:]

        # Per-result summarization at search time (Phase 1 ODR alignment).
        # Summarize sources with long raw content to reduce context consumption.
        # Sources already summarized by the provider layer are skipped.
        try:
            await self._summarize_search_results(
                sources=recent_sources,
                state=state,
                state_lock=state_lock,
            )
        except Exception as summ_exc:
            logger.warning(
                "Batch summarization failed for %s: %s. Using raw content.",
                query_label,
                summ_exc,
            )

        # --- Novelty scoring (Phase 3 ODR alignment) ---
        # Compare each new source against existing sources for this sub-query
        # to give the researcher explicit signals for stop decisions.
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            NoveltyTag,
            build_novelty_summary,
            compute_novelty_tag,
        )

        # Build existing source tuples (content, title, url) for comparison
        # Only include sources already known *before* this search batch
        pre_existing_sources: list[tuple[str, str, str | None]] = []
        pre_existing_ids = {s.id for s in recent_sources}
        for s in topic_sources:
            if s.id not in pre_existing_ids:
                pre_existing_sources.append(
                    (s.content or s.snippet or "", s.title, s.url)
                )

        novelty_tags: list[NoveltyTag] = []
        for src in recent_sources:
            tag = compute_novelty_tag(
                new_content=src.content or src.snippet or "",
                new_url=src.url,
                existing_sources=pre_existing_sources,
            )
            novelty_tags.append(tag)
            # Store tag in source metadata for downstream consumers
            src.metadata["novelty_tag"] = tag.category
            src.metadata["novelty_similarity"] = tag.similarity

        # Format results with novelty annotations (Phase 4 ODR alignment)
        novelty_header = build_novelty_summary(novelty_tags)
        return (
            _format_search_results_batch(
                sources=recent_sources,
                novelty_tags=novelty_tags,
                novelty_header=novelty_header,
            ),
            queries_charged,
        )

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

        After successful extraction, summarizes content via SourceSummarizer
        and applies novelty scoring — mirroring the search result pipeline so
        the researcher LLM can reason about extracted content.

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
            Formatted tool result string with extracted content summaries
            and novelty annotations.
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

        # 1c: Snapshot pre-existing source IDs so we only process newly
        # extracted sources (guard against redundant summarization).
        pre_extract_source_ids = set(sub_query.source_ids)

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

        if extract_added == 0:
            return f"Extraction from {len(urls)} URL(s) yielded no new content."

        result.reflection_notes.append(
            f"[extract] Fetched {extract_added} source(s) from {len(urls)} URL(s)"
        )
        confirmation = f"Extracted content from {extract_added} of {len(urls)} URL(s)."

        # 1c: Identify only sources added by *this* extraction call.
        new_source_ids = set(sub_query.source_ids) - pre_extract_source_ids
        newly_extracted = [s for s in state.sources if s.id in new_source_ids]

        if not newly_extracted:
            return confirmation

        # 1a: Summarize extracted content (same pipeline as search results).
        try:
            await self._summarize_search_results(
                sources=newly_extracted,
                state=state,
                state_lock=state_lock,
            )
        except Exception as summ_exc:
            logger.warning(
                "Summarization failed for extracted sources: %s. Using raw content.",
                summ_exc,
            )

        # 1b: Novelty scoring against pre-existing sources for this sub-query.
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            NoveltyTag,
            build_novelty_summary,
            compute_novelty_tag,
        )

        pre_existing_sources: list[tuple[str, str, str | None]] = [
            (s.content or s.snippet or "", s.title, s.url)
            for s in state.sources
            if s.id in pre_extract_source_ids
        ]

        novelty_tags: list[NoveltyTag] = []
        for src in newly_extracted:
            tag = compute_novelty_tag(
                new_content=src.content or src.snippet or "",
                new_url=src.url,
                existing_sources=pre_existing_sources,
            )
            novelty_tags.append(tag)
            src.metadata["novelty_tag"] = tag.category
            src.metadata["novelty_similarity"] = tag.similarity

        # Format with novelty annotations (mirroring search result presentation).
        novelty_header = build_novelty_summary(novelty_tags)
        blocks: list[str] = [f"{confirmation}\n{novelty_header}"]
        for idx, (src, ntag) in enumerate(zip(newly_extracted, novelty_tags), 1):
            blocks.append(_format_source_block(idx, src, ntag))

        return "\n\n".join(blocks)

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
