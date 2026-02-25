"""Per-topic compression mixin for DeepResearchWorkflow.

Compresses each topic's raw sources into citation-rich summaries before
analysis.  Aligned with open_deep_research's ``compress_research()``
approach: the compression prompt receives the full topic research context
(reflections, refined queries, completion rationale) — not re-truncated
raw source snippets — and the system prompt instructs verbatim
preservation with structured output.

Extracted from GatheringPhaseMixin (Phase 3 PA.2) to isolate the
compression logic as an independently testable unit.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from foundry_mcp.core.research.models.deep_research import DeepResearchState, TopicResearchResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Prompt construction helpers for _compress_single_topic_async
# ------------------------------------------------------------------


def _build_message_history_prompt(
    *,
    query_text: str,
    message_history: list[dict[str, str]],
    topic_sources: list[Any],
    max_content_length: int,
) -> str:
    """Build compression user prompt from raw ReAct message history.

    Passes the researcher's full conversation — tool calls, results,
    reasoning — to the compression model, matching open_deep_research's
    ``compress_research`` approach.  Oldest messages are truncated first
    when the total exceeds *max_content_length*.

    Args:
        query_text: The research sub-query text.
        message_history: Raw ReAct conversation (role + content dicts).
        topic_sources: Sources belonging to this topic (for source list).
        max_content_length: Character cap for the message history block.

    Returns:
        Formatted user prompt string.
    """
    # Format each message in chronological order
    history_lines: list[str] = []
    for msg in message_history:
        role = msg.get("role", "unknown")
        tool_name = msg.get("tool", "")
        content = msg.get("content", "")

        if role == "assistant":
            history_lines.append(f"[Assistant]\n{content}")
        elif role == "tool":
            label = f"[Tool: {tool_name}]" if tool_name else "[Tool Result]"
            history_lines.append(f"{label}\n{content}")
        else:
            history_lines.append(f"[{role}]\n{content}")

    history_block = "\n\n".join(history_lines)

    # Truncate oldest messages first if over budget
    if len(history_block) > max_content_length:
        # Walk forward from the end, keeping the most recent messages
        kept: list[str] = []
        remaining = max_content_length
        for line in reversed(history_lines):
            entry_len = len(line) + 2  # +2 for "\n\n" separator
            if remaining >= entry_len:
                kept.append(line)
                remaining -= entry_len
            else:
                break
        kept.reverse()
        if kept:
            history_block = "\n\n".join(kept)
        else:
            # Single very large message — hard-truncate from end
            history_block = history_block[-max_content_length:]

    # Build source reference list so the model can map citations
    source_ref_lines: list[str] = []
    for idx, src in enumerate(topic_sources, 1):
        url_part = f": {src.url}" if src.url else ""
        source_ref_lines.append(f"[{idx}] {src.title}{url_part}")
    source_ref = "\n".join(source_ref_lines)

    return (
        f"Research sub-query: {query_text}\n\n"
        f"Below is the full researcher conversation (tool calls and "
        f"responses) for this topic:\n\n"
        f"{history_block}\n\n"
        f"Sources ({len(topic_sources)} total):\n"
        f"{source_ref}\n\n"
        f"All above messages are about research conducted by an AI "
        f"Researcher. Please clean up these findings.\n\n"
        f"DO NOT summarize the information. I want the raw information "
        f"returned, just in a cleaner format. Make sure all relevant "
        f"information is preserved - you can rewrite findings verbatim."
    )


def _build_structured_metadata_prompt(
    *,
    query_text: str,
    topic_result: "TopicResearchResult",
    topic_sources: list[Any],
    max_content_length: int,
) -> str:
    """Build compression user prompt from structured metadata (fallback).

    Used when ``message_history`` is empty — e.g., legacy results or
    results loaded from saved state that predate the message history field.

    Args:
        query_text: The research sub-query text.
        topic_result: Topic result with structured metadata fields.
        topic_sources: Sources belonging to this topic.
        max_content_length: Character cap for source content.

    Returns:
        Formatted user prompt string.
    """
    # Search iteration history
    iteration_lines: list[str] = []

    iteration_lines.append(
        f'  1. Query: "{query_text}" '
        f"-> {topic_result.sources_found} sources found"
    )
    if topic_result.reflection_notes:
        iteration_lines.append(
            f"     Reflection: {topic_result.reflection_notes[0]}"
        )

    for i, refined_q in enumerate(topic_result.refined_queries):
        iter_num = i + 2
        reflection = (
            topic_result.reflection_notes[i + 1]
            if i + 1 < len(topic_result.reflection_notes)
            else ""
        )
        iteration_lines.append(f'  {iter_num}. Query: "{refined_q}"')
        if reflection:
            iteration_lines.append(
                f"     Reflection: {reflection}"
            )

    if topic_result.early_completion and topic_result.completion_rationale:
        iteration_lines.append(
            f"  Completion: {topic_result.completion_rationale}"
        )

    iterations_block = "\n".join(iteration_lines)

    # Source block with full content
    source_lines: list[str] = []
    for idx, src in enumerate(topic_sources, 1):
        source_lines.append(f"[{idx}] Title: {src.title}")
        if src.url:
            source_lines.append(f"    URL: {src.url}")
        content = src.content or src.snippet or ""
        if content:
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            source_lines.append(f"    Content: {content}")
        source_lines.append("")

    sources_block = "\n".join(source_lines)

    return (
        f"Research sub-query: {query_text}\n\n"
        f"Search iterations:\n{iterations_block}\n\n"
        f"Sources ({len(topic_sources)} total):\n\n"
        f"{sources_block}\n"
        "Clean up these findings preserving all relevant "
        "information with inline citations."
    )


# ------------------------------------------------------------------
# Message-boundary-aware truncation for compression retries (Phase 5)
# ------------------------------------------------------------------

# Number of most-recent think message pairs to unconditionally preserve
# during compression retry truncation.
_PRESERVE_LAST_N_THINKS: int = 2


def _classify_message_pair(
    assistant_msg: dict[str, str],
    tool_msg: dict[str, str] | None,
) -> str:
    """Classify a message pair by the tool used.

    Returns one of: "think", "search", "research_complete", "other".
    """
    tool_name = ""
    if tool_msg:
        tool_name = tool_msg.get("tool", "")
    if not tool_name and assistant_msg.get("content"):
        # Try to infer from the assistant's tool call content
        content = assistant_msg["content"]
        if '"research_complete"' in content:
            return "research_complete"
        if '"think"' in content:
            return "think"
        if '"web_search"' in content:
            return "search"

    if tool_name == "think":
        return "think"
    elif tool_name == "web_search":
        return "search"
    elif tool_name == "research_complete":
        return "research_complete"
    return "other"


def _group_message_pairs(
    message_history: list[dict[str, str]],
) -> list[tuple[str, list[dict[str, str]]]]:
    """Group message history into (type, messages) pairs.

    Groups consecutive assistant+tool messages into logical pairs.
    Messages that don't fit the pattern are grouped as singles with
    type "other".

    Returns:
        List of (message_type, message_list) tuples in chronological order.
    """
    pairs: list[tuple[str, list[dict[str, str]]]] = []
    i = 0
    while i < len(message_history):
        msg = message_history[i]
        if msg.get("role") == "assistant" and i + 1 < len(message_history):
            next_msg = message_history[i + 1]
            if next_msg.get("role") == "tool":
                pair_type = _classify_message_pair(msg, next_msg)
                pairs.append((pair_type, [msg, next_msg]))
                i += 2
                continue
        # Single message (no pair partner)
        pair_type = _classify_message_pair(msg, None)
        pairs.append((pair_type, [msg]))
        i += 1
    return pairs


def truncate_message_history_for_retry(
    message_history: list[dict[str, str]],
    attempt: int,
    max_attempts: int = 3,
) -> tuple[list[dict[str, str]], int]:
    """Drop oldest complete message pairs for compression retry.

    Implements ODR-inspired message-boundary-aware truncation: instead of
    blanket percentage-based string truncation, this identifies logical
    message pairs (assistant + tool response) and drops the oldest
    non-protected pairs first.

    **Protected messages** (never dropped):
    - The most recent ``_PRESERVE_LAST_N_THINKS`` think message pairs
    - The most recent search result pair (latest findings)
    - The research_complete pair (if present, usually last)

    Progressive removal per attempt:
    - Attempt 1: Drop oldest 1/3 of droppable pairs
    - Attempt 2: Drop oldest 2/3 of droppable pairs
    - Attempt 3: Keep only protected pairs

    Args:
        message_history: Raw ReAct conversation messages.
        attempt: Current retry attempt (1-based).
        max_attempts: Maximum retry attempts (default 3).

    Returns:
        (truncated_history, messages_dropped) — the truncated message
        list and the count of individual messages removed.
    """
    if not message_history or attempt < 1:
        return message_history, 0

    pairs = _group_message_pairs(message_history)
    if len(pairs) <= 1:
        return message_history, 0

    # --- Identify protected pair indices ---
    protected: set[int] = set()

    # Protect the research_complete pair (usually the last one)
    for idx in range(len(pairs) - 1, -1, -1):
        if pairs[idx][0] == "research_complete":
            protected.add(idx)
            break

    # Protect the most recent N think pairs
    think_indices = [i for i, (t, _) in enumerate(pairs) if t == "think"]
    for idx in think_indices[-_PRESERVE_LAST_N_THINKS:]:
        protected.add(idx)

    # Protect the most recent search pair
    search_indices = [i for i, (t, _) in enumerate(pairs) if t == "search"]
    if search_indices:
        protected.add(search_indices[-1])

    # --- Identify droppable pairs (not protected, ordered oldest first) ---
    droppable = [i for i in range(len(pairs)) if i not in protected]

    if not droppable:
        return message_history, 0

    # Progressive removal: attempt 1 → 1/3, attempt 2 → 2/3, attempt 3 → all
    fraction = min(attempt / max_attempts, 1.0)
    num_to_drop = max(1, int(len(droppable) * fraction))
    pairs_to_drop: set[int] = set(droppable[:num_to_drop])

    # --- Rebuild message list ---
    result: list[dict[str, str]] = []
    messages_dropped = 0
    for idx, (_, msgs) in enumerate(pairs):
        if idx in pairs_to_drop:
            messages_dropped += len(msgs)
        else:
            result.extend(msgs)

    return result, messages_dropped


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
    # Single-topic compression (reusable helper)
    # ------------------------------------------------------------------

    async def _compress_single_topic_async(
        self,
        topic_result: TopicResearchResult,
        state: DeepResearchState,
        timeout: float,
    ) -> tuple[int, int, bool]:
        """Compress a single topic's sources into a citation-rich summary.

        Reusable helper callable from both the batch compression path
        (``_compress_topic_findings_async``) and inline during gathering
        (``_execute_topic_research_async``).

        Builds a compression prompt using the **full topic research context**
        — including reflection notes, refined queries, and completion rationale
        from the ReAct loop — and asks the LLM to reformat the raw findings
        with inline citations, preserving all relevant information.

        Args:
            topic_result: The topic's research result (sources, reflections, etc.).
            state: Current research state (for source lookup and cancellation).
            timeout: Per-compression LLM call timeout in seconds.

        Returns:
            (input_tokens, output_tokens, success) tuple.
            On success, ``topic_result.compressed_findings`` is populated.
        """
        from foundry_mcp.core.research.workflows.base import WorkflowResult
        from foundry_mcp.core.research.workflows.deep_research._helpers import safe_resolve_model_for_role
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_llm_call,
        )

        self._check_cancellation(state)

        # Collect sources belonging to this topic
        topic_sources = [
            s for s in state.sources if s.id in topic_result.source_ids
        ]
        if not topic_sources:
            return (0, 0, True)

        # Look up the sub-query text for context
        sub_query = state.get_sub_query(topic_result.sub_query_id)
        query_text = sub_query.query if sub_query else "Unknown query"

        # Resolve compression provider/model via role-based hierarchy.
        role_provider, role_model = safe_resolve_model_for_role(self.config, "compression")
        compression_provider: str = role_provider or self.config.default_provider
        compression_model: str | None = role_model

        # Source content char limit — configurable, defaults to 50,000
        max_content_length: int = getattr(
            self.config, "deep_research_compression_max_content_length", 50_000
        )

        # ---------------------------------------------------------
        # System prompt — aligned with open_deep_research's
        # compress_research_system_prompt structure
        # ---------------------------------------------------------

        system_prompt = (
            "You are a research assistant that has conducted research on "
            "a topic by calling several tools and web searches. Your job "
            "is now to clean up the findings, but preserve all of the "
            "relevant statements and information that the researcher has "
            "gathered.\n\n"
            "<Task>\n"
            "You need to clean up information gathered from tool calls "
            "and web searches in the existing messages. All relevant "
            "information should be repeated and rewritten verbatim, but "
            "in a cleaner format. The purpose of this step is just to "
            "remove any obviously irrelevant or duplicative information. "
            "For example, if three sources all say the same thing, you "
            "could consolidate them with appropriate citations.\n"
            "Only these fully comprehensive cleaned findings are going to "
            "be returned to the user, so it's crucial that you don't "
            "lose any information.\n"
            "</Task>\n\n"
            "<Guidelines>\n"
            "1. Your output findings should be fully comprehensive and "
            "include ALL of the information and sources that the "
            "researcher has gathered from tool calls and web searches. "
            "It is expected that you repeat key information verbatim.\n"
            "2. This report can be as long as necessary to return ALL of "
            "the information that the researcher has gathered.\n"
            "3. In your report, you should return inline citations for "
            "each source that the researcher found.\n"
            "4. You should include a Sources section at the end of the "
            "report that lists all of the sources the researcher found "
            "with corresponding citations, cited against statements in "
            "the report.\n"
            "5. Make sure to include ALL of the sources that the "
            "researcher gathered in the report, and how they were used "
            "to answer the question!\n"
            "6. It's really important not to lose any sources. A later "
            "LLM will be used to merge this report with others, so "
            "having all of the sources is critical.\n"
            "</Guidelines>\n\n"
            "<Output Format>\n"
            "The report should be structured like this:\n"
            "**Queries and Tool Calls Made**\n"
            "**Fully Comprehensive Findings**\n"
            "**Sources**\n"
            "</Output Format>\n\n"
            "<Citation Rules>\n"
            "- Assign each unique URL a single citation number in your "
            "text\n"
            "- End with ### Sources that lists each source with "
            "corresponding numbers\n"
            "- IMPORTANT: Number sources sequentially without gaps "
            "(1, 2, 3, 4, ...) in the final list regardless of which "
            "sources you choose\n"
            "- Example format:\n"
            "  [1] Source Title: URL\n"
            "  [2] Source Title: URL\n"
            "</Citation Rules>\n\n"
            "Critical Reminder: It is extremely important that any "
            "information that is even remotely relevant to the research "
            "topic is preserved verbatim (e.g. don't rewrite it, don't "
            "summarize it, don't paraphrase it)."
        )

        # ---------------------------------------------------------
        # Build user prompt — prefer raw message history when available
        # ---------------------------------------------------------

        has_message_history = bool(topic_result.message_history)

        if has_message_history:
            user_prompt = _build_message_history_prompt(
                query_text=query_text,
                message_history=topic_result.message_history,
                topic_sources=topic_sources,
                max_content_length=max_content_length,
            )
            # Record original message count for truncation metadata
            topic_result.compression_original_message_count = len(
                topic_result.message_history
            )
        else:
            # Fallback: build from structured metadata (backward compat)
            user_prompt = _build_structured_metadata_prompt(
                query_text=query_text,
                topic_result=topic_result,
                topic_sources=topic_sources,
                max_content_length=max_content_length,
            )

        # ---------------------------------------------------------
        # LLM call with phase-specific outer retry on token-limit
        # errors.  When message_history is available, retries use
        # message-boundary-aware truncation (dropping oldest
        # complete message pairs while preserving the most recent
        # think messages, search results, and research_complete
        # summary).  Otherwise falls back to percentage-based
        # prompt truncation.
        # ---------------------------------------------------------
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            MAX_PHASE_TOKEN_RETRIES,
            _is_context_window_exceeded,
            truncate_prompt_for_retry,
        )

        current_user_prompt = user_prompt
        outer_retries = 0
        total_messages_dropped = 0

        for outer_attempt in range(MAX_PHASE_TOKEN_RETRIES + 1):  # 0 = initial
            call_result = await execute_llm_call(
                workflow=self,
                state=state,
                phase_name="compression",
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                provider_id=compression_provider,
                model=compression_model,
                temperature=0.2,
                timeout=timeout,
                role="compression",
            )

            if isinstance(call_result, WorkflowResult):
                # Check if this is a context-window error we can retry
                if (
                    _is_context_window_exceeded(call_result)
                    and outer_attempt < MAX_PHASE_TOKEN_RETRIES
                ):
                    outer_retries += 1

                    if has_message_history:
                        # Message-boundary-aware truncation: drop
                        # oldest complete message pairs, preserving
                        # the most recent thinks, search results,
                        # and research_complete summary.
                        truncated_history, _ = (
                            truncate_message_history_for_retry(
                                topic_result.message_history,
                                outer_retries,
                                MAX_PHASE_TOKEN_RETRIES,
                            )
                        )
                        total_messages_dropped = (
                            len(topic_result.message_history)
                            - len(truncated_history)
                        )
                        current_user_prompt = _build_message_history_prompt(
                            query_text=query_text,
                            message_history=truncated_history,
                            topic_sources=topic_sources,
                            max_content_length=max_content_length,
                        )
                        logger.warning(
                            "Compression outer retry %d/%d for topic %s: "
                            "message-boundary truncation dropped %d messages "
                            "(%d remaining of %d original)",
                            outer_retries,
                            MAX_PHASE_TOKEN_RETRIES,
                            topic_result.sub_query_id,
                            total_messages_dropped,
                            len(truncated_history),
                            len(topic_result.message_history),
                        )
                    else:
                        # Fallback: percentage-based prompt truncation
                        # for structured metadata path
                        current_user_prompt = truncate_prompt_for_retry(
                            user_prompt, outer_retries, MAX_PHASE_TOKEN_RETRIES,
                        )
                        logger.warning(
                            "Compression outer retry %d/%d for topic %s: "
                            "pre-truncating user prompt by %d%%",
                            outer_retries,
                            MAX_PHASE_TOKEN_RETRIES,
                            topic_result.sub_query_id,
                            int((0.1 + outer_retries * 0.1) * 100),
                        )
                    continue

                # Non-retryable error or retries exhausted
                logger.error(
                    "Compression failed for topic %s (outer_retries=%d): %s",
                    topic_result.sub_query_id,
                    outer_retries,
                    call_result.error,
                )
                # Record truncation metadata even on failure
                topic_result.compression_retry_count = outer_retries
                topic_result.compression_messages_dropped = total_messages_dropped
                if outer_retries > 0:
                    self._write_audit_event(
                        state,
                        "compression_retry_exhausted",
                        data={
                            "sub_query_id": topic_result.sub_query_id,
                            "outer_retries": outer_retries,
                            "messages_dropped": total_messages_dropped,
                            "original_message_count": topic_result.compression_original_message_count,
                            "error": call_result.error,
                        },
                        level="warning",
                    )
                return (0, 0, False)

            # Success path
            result = call_result.result
            if result.success and result.content:
                topic_result.compressed_findings = result.content
                # Record truncation metadata
                topic_result.compression_retry_count = outer_retries
                topic_result.compression_messages_dropped = total_messages_dropped
                if outer_retries > 0:
                    self._write_audit_event(
                        state,
                        "compression_retry_succeeded",
                        data={
                            "sub_query_id": topic_result.sub_query_id,
                            "outer_retries": outer_retries,
                            "messages_dropped": total_messages_dropped,
                            "original_message_count": topic_result.compression_original_message_count,
                        },
                    )
                return (result.input_tokens or 0, result.output_tokens or 0, True)
            return (0, 0, False)

        # Should not reach here, but safety fallback
        return (0, 0, False)

    # ------------------------------------------------------------------
    # Batch per-topic compression
    # ------------------------------------------------------------------

    async def _compress_topic_findings_async(
        self,
        state: DeepResearchState,
        max_concurrent: int,
        timeout: float,
    ) -> dict[str, Any]:
        """Compress each topic's sources into citation-rich summaries.

        Runs after all topic researchers complete.  For each
        ``TopicResearchResult`` that has sources, builds a compression
        prompt using the **full topic research context** — including
        reflection notes, refined queries, and completion rationale from
        the ReAct loop — and asks the LLM to reformat the raw findings
        with inline citations, preserving all relevant information.

        Aligned with open_deep_research's ``compress_research()`` approach:
        the compression prompt receives the full research context (not
        re-truncated snippets), and the system prompt instructs verbatim
        preservation with structured output.

        Features:
        - Full ReAct context in prompt (reflections, refined queries, rationale).
        - Configurable source content limit (default 50,000 chars, matching
          open_deep_research's ``max_content_length``).
        - Progressive token-limit handling via ``execute_llm_call``.
        - Parallel compression across topics, bounded by *max_concurrent*.
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
        results_to_compress = [
            tr for tr in state.topic_research_results
            if tr.source_ids and tr.compressed_findings is None
        ]

        if not results_to_compress:
            return {"topics_compressed": 0, "topics_failed": 0, "total_compression_tokens": 0}

        semaphore = asyncio.Semaphore(max_concurrent)

        async def compress_one(
            topic_result: TopicResearchResult,
        ) -> tuple[int, int, bool]:
            """Compress a single topic's findings under semaphore."""
            async with semaphore:
                return await self._compress_single_topic_async(
                    topic_result=topic_result,
                    state=state,
                    timeout=timeout,
                )

        # Run compression tasks in parallel
        tasks = [compress_one(tr) for tr in results_to_compress]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results after gather completes (no nonlocal mutation).
        # Note: per-topic PhaseMetrics and token tracking are handled by
        # execute_llm_call.  We aggregate here only for the audit event
        # and return statistics.
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

        total_compression_tokens = total_input_tokens + total_output_tokens

        self._write_audit_event(
            state,
            "topic_compression_complete",
            data={
                "topics_compressed": topics_compressed,
                "topics_failed": topics_failed,
                "total_compression_tokens": total_compression_tokens,
                "mode": "batch",
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
    # Global cross-topic compression
    # ------------------------------------------------------------------

    async def _execute_global_compression_async(
        self,
        state: DeepResearchState,
        provider_id: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        """Compress all per-topic findings into a unified research digest.

        This runs as the COMPRESSION phase between ANALYSIS and SYNTHESIS.
        Unlike per-topic compression (which preserves everything verbatim),
        global compression actively deduplicates cross-topic findings,
        merges related themes, and flags contradictions — producing a
        single digest that synthesis can consume directly.

        The digest includes:
        - Deduplicated findings with consistent cross-topic citation numbers
        - Thematic grouping of related findings across topics
        - Explicitly flagged cross-topic contradictions
        - A unified source reference list

        Skipped when:
        - Only one topic was researched (no cross-topic dedup value)
        - Global compression is disabled via config
        - No per-topic compressed findings or analysis findings exist

        Args:
            state: Current research state with per-topic compressed findings
                and analysis output (findings, contradictions, gaps).
            provider_id: LLM provider to use (research-tier recommended).
            timeout: LLM call timeout in seconds.

        Returns:
            Dict with compression statistics (success, tokens_used, etc.).
        """
        from foundry_mcp.core.research.workflows.base import WorkflowResult
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            safe_resolve_model_for_role,
        )
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            execute_llm_call,
        )

        # Skip for single-topic research — no cross-topic dedup value
        topic_count = len(state.topic_research_results)
        if topic_count <= 1:
            logger.info(
                "Skipping global compression: only %d topic(s) for research %s",
                topic_count,
                state.id,
            )
            return {"skipped": True, "reason": "single_topic", "tokens_used": 0}

        # Collect per-topic compressed findings (or fall back to per-topic summaries)
        topic_sections: list[str] = []
        for tr in state.topic_research_results:
            sub_query = state.get_sub_query(tr.sub_query_id)
            query_text = sub_query.query if sub_query else "Unknown query"

            content = tr.compressed_findings or tr.per_topic_summary
            if not content:
                # Last resort: list the source titles for this topic
                topic_sources = [s for s in state.sources if s.id in tr.source_ids]
                if topic_sources:
                    source_lines = [f"  - [{s.citation_number}] {s.title}" for s in topic_sources if s.citation_number]
                    content = "Sources found:\n" + "\n".join(source_lines)
                else:
                    continue

            topic_sections.append(
                f"### Topic: {query_text}\n\n{content}"
            )

        if not topic_sections:
            logger.info("Skipping global compression: no topic content for research %s", state.id)
            return {"skipped": True, "reason": "no_content", "tokens_used": 0}

        # Build analysis findings section
        findings_section = ""
        if state.findings:
            id_to_citation = state.source_id_to_citation()
            finding_lines: list[str] = []
            for f in state.findings:
                citation_refs = [
                    f"[{id_to_citation[sid]}]"
                    for sid in f.source_ids
                    if sid in id_to_citation
                ]
                refs_str = ", ".join(citation_refs) if citation_refs else "no sources"
                category = f.category or "General"
                finding_lines.append(f"- [{category}] {f.content} (Sources: {refs_str})")
            findings_section = (
                "\n\n## Analysis Findings\n" + "\n".join(finding_lines)
            )

        # Build contradictions section
        contradictions_section = ""
        if state.contradictions:
            contra_lines: list[str] = []
            for c in state.contradictions:
                contra_lines.append(f"- [{c.severity.upper()}] {c.description}")
                if c.resolution:
                    contra_lines.append(f"  Resolution: {c.resolution}")
            contradictions_section = (
                "\n\n## Detected Contradictions\n" + "\n".join(contra_lines)
            )

        # Build gaps section
        gaps_section = ""
        if state.gaps:
            gap_lines = [
                f"- [{'resolved' if g.resolved else 'unresolved'}] {g.description}"
                for g in state.gaps
            ]
            gaps_section = "\n\n## Knowledge Gaps\n" + "\n".join(gap_lines)

        # Build full source reference
        source_ref_lines: list[str] = []
        for s in state.sources:
            if s.citation_number is not None:
                source_ref_lines.append(
                    f"[{s.citation_number}] {s.title}"
                    + (f" - {s.url}" if s.url else "")
                )
        source_reference = "\n\n## Source Reference\n" + "\n".join(source_ref_lines)

        # Assemble user prompt
        topics_block = "\n\n".join(topic_sections)
        user_prompt = (
            f"# Research Query\n{state.original_query}\n\n"
            f"# Per-Topic Research Findings ({topic_count} topics)\n\n"
            f"{topics_block}"
            f"{findings_section}"
            f"{contradictions_section}"
            f"{gaps_section}"
            f"{source_reference}\n\n"
            "Produce a unified research digest following the instructions."
        )

        system_prompt = (
            "You are a research synthesizer. You have received per-topic "
            "research findings from parallel researchers who each investigated "
            "a different sub-query of the same research question.\n\n"
            "<Task>\n"
            "Merge all per-topic findings into a single, unified research "
            "digest. Your goals:\n"
            "1. DEDUPLICATE: Identify findings that appear across multiple "
            "topics (same fact from different sources) and merge them, "
            "keeping all relevant citation numbers.\n"
            "2. MERGE THEMES: Group related findings into coherent thematic "
            "sections, regardless of which topic they came from.\n"
            "3. FLAG CONTRADICTIONS: When different topics report conflicting "
            "information, explicitly flag the contradiction with the "
            "conflicting citation numbers.\n"
            "4. PRESERVE CITATIONS: Maintain the original citation numbers "
            "[N] exactly as provided. Do NOT renumber.\n"
            "5. PRESERVE ALL UNIQUE INFORMATION: Every unique fact, data "
            "point, or insight must appear in the digest. Only truly "
            "duplicated content should be merged.\n"
            "</Task>\n\n"
            "<Output Format>\n"
            "## Research Digest\n"
            "Brief overview of the research scope and key themes found.\n\n"
            "## Thematic Findings\n"
            "### [Theme 1]\n"
            "Merged findings with inline citations [N].\n"
            "### [Theme 2]\n"
            "...\n\n"
            "## Cross-Topic Contradictions\n"
            "List any conflicting information found across topics, with "
            "citations for both sides.\n\n"
            "## Knowledge Gaps\n"
            "Areas where research was insufficient or inconclusive.\n\n"
            "## Source Summary\n"
            "Brief note on source coverage: total sources, diversity of "
            "domains, any quality concerns.\n"
            "</Output Format>\n\n"
            "<Guidelines>\n"
            "- Use the ORIGINAL citation numbers [N] — do NOT renumber.\n"
            "- When merging duplicate findings, combine citation numbers: "
            "e.g., 'AI adoption is growing rapidly [1][4][7]'.\n"
            "- Be comprehensive — this digest replaces the raw findings "
            "for the synthesis phase.\n"
            "- Organize by theme, not by original topic.\n"
            "- The digest should be self-contained: a reader should understand "
            "the full research landscape without seeing the original topics.\n"
            "</Guidelines>"
        )

        # Resolve provider/model via role hierarchy
        role_provider, role_model = safe_resolve_model_for_role(
            self.config, "global_compression"
        )
        resolved_provider = provider_id or role_provider or self.config.default_provider
        resolved_model = role_model

        self._check_cancellation(state)

        call_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="compression",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=resolved_provider,
            model=resolved_model,
            temperature=0.3,
            timeout=timeout,
            error_metadata={
                "topic_count": topic_count,
                "findings_count": len(state.findings),
                "guidance": "Try reducing topic count or findings included",
            },
            role="global_compression",
        )

        if isinstance(call_result, WorkflowResult):
            logger.error(
                "Global compression failed for research %s: %s",
                state.id,
                call_result.error,
            )
            self._write_audit_event(
                state,
                "global_compression_failed",
                data={"error": call_result.error},
                level="error",
            )
            return {"success": False, "error": call_result.error, "tokens_used": 0}

        result = call_result.result
        if result.success and result.content:
            state.compressed_digest = result.content
            tokens_used = (result.input_tokens or 0) + (result.output_tokens or 0)

            self._write_audit_event(
                state,
                "global_compression_complete",
                data={
                    "topic_count": topic_count,
                    "findings_count": len(state.findings),
                    "contradictions_count": len(state.contradictions),
                    "digest_length": len(result.content),
                    "tokens_used": tokens_used,
                    "provider_id": resolved_provider,
                    "model_used": resolved_model,
                },
            )

            logger.info(
                "Global compression complete: %d topics → %d char digest, %d tokens",
                topic_count,
                len(result.content),
                tokens_used,
            )

            return {
                "success": True,
                "digest_length": len(result.content),
                "tokens_used": tokens_used,
                "topic_count": topic_count,
            }

        logger.warning(
            "Global compression produced empty result for research %s",
            state.id,
        )
        return {"success": False, "error": "empty_result", "tokens_used": 0}
