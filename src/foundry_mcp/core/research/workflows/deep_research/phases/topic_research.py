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
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig
    from foundry_mcp.core.research.memory import ResearchMemory

from foundry_mcp.core.errors.provider import ContextWindowError
from foundry_mcp.core.research.models.deep_research import (
    CitationSearchTool,
    DeepResearchState,
    ExtractContentTool,
    ExtractPDFTool,
    RelatedPapersTool,
    ResearcherToolCall,
    ThinkTool,
    TopicResearchResult,
    WebSearchTool,
    parse_researcher_response,
)
from foundry_mcp.core.research.models.sources import SourceQuality, SubQuery
from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
    sanitize_external_content,
)
from foundry_mcp.core.research.workflows.deep_research.phases.compression import (
    _compression_output_is_valid,
)
from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    _extract_domain,
    _normalize_title,
    get_domain_quality,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# PDF URL detection helper (PLAN-4 Item 1c)
# ------------------------------------------------------------------

# URL patterns that strongly indicate a PDF resource.
_PDF_URL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\.pdf(\?|#|$)", re.IGNORECASE),
    re.compile(r"arxiv\.org/pdf/", re.IGNORECASE),
    re.compile(r"arxiv\.org/ftp/", re.IGNORECASE),
]


def _is_pdf_url(url: str) -> bool:
    """Detect whether a URL likely points to a PDF document.

    Checks URL patterns (*.pdf, arxiv.org/pdf/*) without making
    any network requests.  Content-Type based detection is handled
    downstream by the extraction layer.

    Args:
        url: URL string to check.

    Returns:
        True if the URL matches a known PDF pattern.
    """
    return any(pattern.search(url) for pattern in _PDF_URL_PATTERNS)


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

    **Sanitization contract:** All web-derived fields (title, snippet,
    content, URL) are passed through ``sanitize_external_content()``
    before interpolation into the prompt.  Callers do not need to
    pre-sanitize source objects.

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
        safe_url = sanitize_external_content(src.url)
        lines.append(f"URL: {safe_url}")
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
            summary_text = re.sub(r"</?summary>", "", summary_text).strip()
            lines.append(f"\nSUMMARY:\n{summary_text}")
            safe_excerpts = [sanitize_external_content(e) for e in excerpts]
            excerpt_lines = "\n".join(f'- "{e}"' for e in safe_excerpts)
            lines.append(f"\nKEY EXCERPTS:\n{excerpt_lines}")
        else:
            # No separate excerpts — show the full summarized content
            summary_text = re.sub(r"</?summary>", "", summary_text).strip()
            summary_text = re.sub(r"</?key_excerpts>", "", summary_text).strip()
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
    header = f"Found {sources_count} new source(s) from {domain_count} domain(s).\n{novelty_header}"

    # Build per-source blocks
    blocks: list[str] = [header]
    for idx, (src, ntag) in enumerate(zip(sources, novelty_tags, strict=False), 1):
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

# Additional tool documentation injected when citation tools are enabled
_CITATION_TOOLS_PROMPT = """
### citation_search
Search for papers that cite a given paper (forward citation search / snowball sampling).
Arguments: {{"paper_id": "DOI or Semantic Scholar paper ID", "max_results": 10}}
Returns: List of citing papers with metadata. Use this for forward snowball sampling — start from a seminal paper and trace who built on it.

### related_papers
Find papers similar to a given paper (lateral discovery).
Arguments: {{"paper_id": "DOI or Semantic Scholar paper ID", "max_results": 5}}
Returns: List of related papers based on content similarity. Use this to discover work the initial search may have missed — especially useful for finding papers that use different terminology for the same concept.
"""

# Strategic research guidance injected alongside citation tools
_STRATEGIC_RESEARCH_PROMPT = """
## Research Strategies

Apply these strategies deliberately. State which strategy you are using in your `think` calls so your reasoning is transparent.

### BROADEN — Expand coverage when initial results are narrow
- Reformulate with alternative terminology, synonyms, or related concepts.
- Use `related_papers` to discover adjacent work that uses different vocabulary.
- Search across disciplinary boundaries (e.g., a clinical concept may have a parallel in public health or social science).
- **When to use**: Your first 1-2 searches returned results from a single perspective or subdomain.

### DEEPEN — Drill into a promising thread
- Use `citation_search` on a seminal or highly-cited paper to trace subsequent developments.
- Search for methodological variations, replications, or extensions of key findings.
- Extract full text from the most relevant sources to capture nuance that abstracts miss.
- **When to use**: You found a key paper and need to understand the lineage of work it spawned.

### VALIDATE — Corroborate or challenge a finding
- Search explicitly for contradictory evidence, failed replications, or critical commentary.
- Use `citation_search` to find papers that cite a controversial claim — citing papers often include critique.
- Cross-check across providers: a finding corroborated by independent sources is far more reliable.
- **When to use**: A finding seems too clean, comes from a single group, or is central to your answer.

### SATURATE — Recognize when coverage is sufficient
- If >50% of new results are duplicates or near-duplicates of sources you already have, coverage is likely saturated.
- Check novelty tags: a pattern of [RELATED] and [DUPLICATE] signals diminishing returns.
- Call `research_complete` — additional searches at saturation add noise without new signal.
- **When to use**: Your last 2 tool calls returned mostly familiar material.
"""

# PDF extraction tool documentation (injected when profile enables PDF extraction)
_EXTRACT_PDF_PROMPT = """
### extract_pdf
Extract full text from an open-access academic paper PDF. Use this for papers where the abstract is insufficient — methods sections, detailed results, and supplementary data are only available in the full text.
Arguments: {"url": "direct PDF URL (e.g. https://arxiv.org/pdf/2301.00001.pdf)", "max_pages": 30}
Returns: Full paper text with section structure (Abstract, Methods, Results, etc.) and page numbers. Only works with open-access PDFs; paywalled papers will fail gracefully.
Use this when `extract_content` returns limited content for an academic paper and you have a direct PDF link (often ending in .pdf or from arxiv.org/pdf/).
"""


# ------------------------------------------------------------------
# Shared dedup helper for _topic_search and _topic_extract
# ------------------------------------------------------------------


async def _dedup_and_add_source(
    source: Any,
    sub_query: SubQuery,
    state: "DeepResearchState",
    seen_urls: set[str],
    seen_titles: dict[str, str],
    state_lock: asyncio.Lock,
    content_dedup_enabled: bool = True,
    dedup_threshold: float = 0.8,
) -> tuple[bool, Optional[str]]:
    """Deduplicate a source and add it to state if novel.

    Performs URL dedup, title dedup, and optional content-similarity dedup
    using the three-phase lock pattern (fast check → unlocked comparison →
    final commit). Used by both ``_topic_search`` and ``_topic_extract``.

    Returns:
        Tuple of (was_added, dedup_reason). dedup_reason is None when the
        source was added, or one of "url_match", "title_match",
        "content_similarity" when deduplicated.
    """
    from foundry_mcp.core.research.workflows.deep_research._content_dedup import (
        content_similarity,
    )

    # --- Phase 1: fast checks under lock (URL + title dedup) ---
    async with state_lock:
        if source.url and source.url in seen_urls:
            return False, "url_match"

        normalized_title = _normalize_title(source.title)
        if normalized_title and len(normalized_title) > 20:
            if normalized_title in seen_titles:
                return False, "title_match"

        # Snapshot existing sources for content-similarity check
        # outside the lock to avoid O(n^2) contention.
        if content_dedup_enabled and source.content and len(source.content) > 100:
            sources_snapshot = list(state.sources)
        else:
            sources_snapshot = None

    # --- Phase 2: content-similarity dedup outside the lock ---
    if sources_snapshot is not None:
        for existing_src in sources_snapshot:
            if existing_src.content and len(existing_src.content) > 100:
                sim = content_similarity(source.content, existing_src.content)
                if sim >= dedup_threshold:
                    logger.debug(
                        "Content dedup: %r (%.2f similar to %r)",
                        source.url or source.title,
                        sim,
                        existing_src.url or existing_src.title,
                    )
                    return False, "content_similarity"

    # --- Phase 3: final add-or-skip under lock (re-check for races) ---
    async with state_lock:
        if source.url and source.url in seen_urls:
            return False, "url_match"

        normalized_title = _normalize_title(source.title)
        if normalized_title and len(normalized_title) > 20:
            if normalized_title in seen_titles:
                return False, "title_match"
            seen_titles[normalized_title] = source.url or ""

        if source.url:
            seen_urls.add(source.url)
            if source.quality == SourceQuality.UNKNOWN:
                source.quality = get_domain_quality(source.url, state.research_mode)

        state.append_source(source)
        sub_query.source_ids.append(source.id)
        return True, None


def _build_researcher_system_prompt(
    *,
    budget_total: int,
    budget_remaining: int,
    extract_enabled: bool,
    citation_tools_enabled: bool = False,
    pdf_extraction_enabled: bool = False,
    date_str: str | None = None,
) -> str:
    """Build the researcher system prompt with budget and context.

    Args:
        budget_total: Total tool call budget for this researcher.
        budget_remaining: Remaining tool calls.
        extract_enabled: Whether extract_content tool is available.
        citation_tools_enabled: Whether citation_search and related_papers
            tools are available (gated by research profile).
        pdf_extraction_enabled: Whether extract_pdf tool is available
            (gated by research profile enable_pdf_extraction).
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

    if citation_tools_enabled:
        # Insert citation tool documentation and strategic guidance
        # before the Response Format section
        prompt = prompt.replace(
            "## Response Format",
            f"{_CITATION_TOOLS_PROMPT}\n{_STRATEGIC_RESEARCH_PROMPT}\n## Response Format",
        )

    if pdf_extraction_enabled:
        # Insert extract_pdf tool documentation before the Response Format section
        prompt = prompt.replace(
            "## Response Format",
            f"{_EXTRACT_PDF_PROMPT}\n## Response Format",
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
    parts: list[str] = [f"<research_topic>\n{sanitize_external_content(topic)}\n</research_topic>"]

    if message_history:
        parts.append("\n<conversation_history>")
        for i, msg in enumerate(message_history):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "assistant":
                safe_content = sanitize_external_content(content)
                parts.append(f'\n<turn number="{i + 1}" role="assistant">\n{safe_content}\n</turn>')
            elif role == "tool":
                tool_name = msg.get("tool", "unknown")
                # Tool results contain web-sourced content — sanitize before
                # interpolating into the researcher prompt.
                safe_content = sanitize_external_content(content)
                parts.append(f'\n<turn number="{i + 1}" role="tool" tool="{tool_name}">\n{safe_content}\n</turn>')
        parts.append("\n</conversation_history>")

    parts.append(f"\n<budget>You have {budget_remaining} of {budget_total} tool calls remaining.</budget>")
    parts.append(
        "\nRespond with your next action as a JSON object containing tool_calls. "
        "Return ONLY valid JSON, no additional text."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Researcher history truncation
# ---------------------------------------------------------------------------

# Reserve this fraction of the model's context window for the researcher's
# conversation history.  The rest is needed for the system prompt, research
# topic, and the LLM's response tokens.
_RESEARCHER_HISTORY_BUDGET_FRACTION: float = 0.35

# Import canonical constant; keep the private alias for local usage.
from foundry_mcp.core.research.workflows.deep_research._token_budget import (
    CHARS_PER_TOKEN as _CHARS_PER_TOKEN,
)

# Minimum number of recent turns to always preserve regardless of budget.
_MIN_PRESERVE_RECENT_TURNS: int = 4


def _truncate_researcher_history(
    message_history: list[dict[str, str]],
    model: str | None,
) -> list[dict[str, str]]:
    """Truncate researcher conversation history to fit model context window.

    Estimates the token budget from the model's context window and drops the
    oldest turns when the history exceeds the budget, always preserving at
    least ``_MIN_PRESERVE_RECENT_TURNS`` recent messages.

    Args:
        message_history: The researcher's message history (assistant/tool turns).
        model: Model identifier for context-window lookup.

    Returns:
        The (possibly truncated) message list.  Returns the original list
        unchanged if it fits within budget.
    """
    if len(message_history) <= _MIN_PRESERVE_RECENT_TURNS:
        return message_history

    from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
        estimate_token_limit_for_model,
    )
    from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
        get_model_token_limits,
    )

    max_tokens = estimate_token_limit_for_model(model, get_model_token_limits())
    if max_tokens is None:
        max_tokens = 128_000  # conservative fallback

    budget_chars = int(max_tokens * _RESEARCHER_HISTORY_BUDGET_FRACTION * _CHARS_PER_TOKEN)

    total_chars = sum(len(msg.get("content", "")) for msg in message_history)
    if total_chars <= budget_chars:
        return message_history

    # Drop oldest turns until within budget, preserving the most recent ones
    # Use index + slice instead of O(n²) pop(0) loop
    result = list(message_history)
    max_droppable = len(result) - _MIN_PRESERVE_RECENT_TURNS
    drop_count = 0
    while drop_count < max_droppable and total_chars > budget_chars:
        total_chars -= len(result[drop_count].get("content", ""))
        drop_count += 1
    if drop_count:
        result = result[drop_count:]

    if len(result) < len(message_history):
        dropped = len(message_history) - len(result)
        logger.info(
            "Truncated researcher history: dropped %d oldest turns (%d -> %d), budget=%d chars, model=%s",
            dropped,
            len(message_history),
            len(result),
            budget_chars,
            model,
        )

    return result


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

    See ``DeepResearchWorkflowProtocol`` in ``_protocols.py`` for the
    full structural contract.
    """

    config: ResearchConfig
    memory: ResearchMemory
    _search_providers: dict[str, Any]

    # Stubs for Pyright — canonical signatures live in _protocols.py
    if TYPE_CHECKING:

        def _write_audit_event(
            self,
            state: DeepResearchState | None,
            event_name: str,
            *,
            data: dict[str, Any] | None = ...,
            level: str = ...,
        ) -> None: ...
        def _check_cancellation(self, state: DeepResearchState) -> None: ...
        def _get_search_provider(self, provider_name: str) -> Any: ...
        def _get_tavily_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]: ...
        def _get_perplexity_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]: ...
        def _get_semantic_scholar_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]: ...
        async def _execute_provider_async(self, **kwargs: Any) -> Any: ...
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

        The researcher LLM decides which tools to call each turn
        (web_search, extract_content, think, or research_complete).
        Delegates to extracted helpers for LLM calls, parse retries,
        reflection enforcement, tool dispatch, and finalization.
        """
        result = TopicResearchResult(sub_query_id=sub_query.id)
        local_tokens_used = 0
        tool_calls_used = 0
        budget_remaining = max_searches

        async with state_lock:
            sub_query.status = "executing"

        extract_enabled, extract_max_per_iter, provider_id, researcher_model = self._resolve_topic_research_config()

        # Resolve profile-gated tool flags
        profile = state.research_profile
        citation_tools_enabled = getattr(profile, "enable_citation_tools", False) if profile else False
        pdf_extraction_enabled = getattr(profile, "enable_pdf_extraction", False) if profile else False

        message_history: list[dict[str, str]] = []
        max_turns = max_searches * 3  # generous: 3x budget allows think steps

        # Reflection enforcement tracking (Phase 2)
        previous_turn_had_search = False
        search_turn_count = 0
        reflection_injections = 0

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

            # Execute one LLM call for this turn
            llm_result = await self._execute_researcher_llm_call(
                sub_query=sub_query,
                message_history=message_history,
                researcher_model=researcher_model,
                provider_id=provider_id,
                budget_remaining=budget_remaining,
                max_searches=max_searches,
                extract_enabled=extract_enabled,
                citation_tools_enabled=citation_tools_enabled,
                pdf_extraction_enabled=pdf_extraction_enabled,
                turn=turn,
            )
            if llm_result is None:
                break
            local_tokens_used += llm_result.tokens_used or 0

            # Parse tool calls, retrying on parse failure
            raw_content = llm_result.content or ""
            system_prompt = _build_researcher_system_prompt(
                budget_total=max_searches,
                budget_remaining=budget_remaining,
                extract_enabled=extract_enabled,
                citation_tools_enabled=citation_tools_enabled,
                pdf_extraction_enabled=pdf_extraction_enabled,
            )
            response, tokens_delta = await self._parse_with_retry_async(
                raw_content=raw_content,
                message_history=message_history,
                sub_query=sub_query,
                result=result,
                turn=turn,
                provider_id=provider_id,
                researcher_model=researcher_model,
                system_prompt=system_prompt,
                budget_remaining=budget_remaining,
                max_searches=max_searches,
            )
            local_tokens_used += tokens_delta

            if not response.tool_calls:
                logger.info(
                    "Topic %r researcher returned no tool calls on turn %d, stopping",
                    sub_query.id,
                    turn + 1,
                )
                break

            # Reflection enforcement: inject synthetic think prompt if needed
            needs_reflection, current_has_search = self._check_reflection_needed(
                response=response,
                previous_turn_had_search=previous_turn_had_search,
                search_turn_count=search_turn_count,
            )
            if needs_reflection:
                logger.warning(
                    "Topic %r: researcher skipped reflection after search on turn %d, injecting synthetic think prompt",
                    sub_query.id,
                    turn + 1,
                )
                reflection_injections += 1
                self._inject_reflection_prompt(message_history, raw_content)
                if current_has_search:
                    search_turn_count += 1
                previous_turn_had_search = current_has_search
                continue

            # Record the assistant's response and dispatch tool calls
            message_history.append({"role": "assistant", "content": raw_content})
            loop_should_break, calls_delta, budget_delta = await self._dispatch_tool_calls(
                response=response,
                sub_query=sub_query,
                state=state,
                result=result,
                message_history=message_history,
                available_providers=available_providers,
                max_sources_per_provider=max_sources_per_provider,
                timeout=timeout,
                seen_urls=seen_urls,
                seen_titles=seen_titles,
                state_lock=state_lock,
                semaphore=semaphore,
                budget_remaining=budget_remaining,
                extract_enabled=extract_enabled,
                extract_max_per_iter=extract_max_per_iter,
                citation_tools_enabled=citation_tools_enabled,
                pdf_extraction_enabled=pdf_extraction_enabled,
            )
            tool_calls_used += calls_delta
            budget_remaining -= budget_delta

            if current_has_search:
                search_turn_count += 1
            previous_turn_had_search = current_has_search

            if loop_should_break:
                break

        return await self._finalize_topic_result(
            result=result,
            sub_query=sub_query,
            state=state,
            state_lock=state_lock,
            local_tokens_used=local_tokens_used,
            tool_calls_used=tool_calls_used,
            message_history=message_history,
            reflection_injections=reflection_injections,
            extract_enabled=extract_enabled,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Extracted helpers for _execute_topic_research_async
    # ------------------------------------------------------------------

    def _resolve_topic_research_config(
        self,
    ) -> tuple[bool, int, str | None, str | None]:
        """Resolve extract and LLM config once for the research loop.

        Returns:
            Tuple of (extract_enabled, extract_max_per_iter,
            provider_id, researcher_model).
        """
        import os as _os

        from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
            resolve_phase_provider,
            safe_resolve_model_for_role,
        )

        extract_enabled = self.config.deep_research_enable_extract and bool(
            self.config.tavily_api_key or _os.environ.get("TAVILY_API_KEY")
        )
        extract_max_per_iter = getattr(self.config, "deep_research_extract_max_per_iteration", 2)

        provider_id, researcher_model = safe_resolve_model_for_role(self.config, "topic_reflection")
        if provider_id is None:
            provider_id = resolve_phase_provider(self.config, "topic_reflection", "reflection")

        return extract_enabled, extract_max_per_iter, provider_id, researcher_model

    async def _execute_researcher_llm_call(
        self,
        sub_query: SubQuery,
        message_history: list[dict[str, str]],
        researcher_model: str | None,
        provider_id: str | None,
        budget_remaining: int,
        max_searches: int,
        extract_enabled: bool,
        turn: int,
        citation_tools_enabled: bool = False,
        pdf_extraction_enabled: bool = False,
    ) -> Any | None:
        """Execute a single researcher LLM call for one ReAct turn.

        Builds the system and user prompts, calls the provider, and
        returns the LLM result. Returns ``None`` on failure or exception
        to signal the main loop should break.

        Args:
            sub_query: The sub-query being researched.
            message_history: Conversation history for prompt building.
            researcher_model: Model identifier for the researcher LLM.
            provider_id: Provider ID for the researcher LLM.
            budget_remaining: Remaining tool call budget.
            max_searches: Total tool call budget.
            extract_enabled: Whether extract_content is available.
            turn: Current turn index (for logging).
            citation_tools_enabled: Whether citation_search and related_papers
                tools are available.
            pdf_extraction_enabled: Whether extract_pdf tool is available.

        Returns:
            LLM result object on success, None on failure.
        """
        system_prompt = _build_researcher_system_prompt(
            budget_total=max_searches,
            budget_remaining=budget_remaining,
            extract_enabled=extract_enabled,
            citation_tools_enabled=citation_tools_enabled,
            pdf_extraction_enabled=pdf_extraction_enabled,
        )
        truncated_history = _truncate_researcher_history(message_history, researcher_model)
        user_prompt = _build_react_user_prompt(
            topic=sub_query.query,
            message_history=truncated_history,
            budget_remaining=budget_remaining,
            budget_total=max_searches,
        )

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
                return None
            return llm_result
        except ContextWindowError as exc:
            logger.warning(
                "Topic %r researcher hit context window on turn %d, truncating history and retrying once: %s",
                sub_query.id,
                turn + 1,
                exc,
            )
            # Aggressively truncate and retry once
            truncated = _truncate_researcher_history(message_history, researcher_model)
            retry_prompt = _build_react_user_prompt(
                topic=sub_query.query,
                message_history=truncated,
                budget_remaining=budget_remaining,
                budget_total=max_searches,
            )
            try:
                llm_result = await self._execute_provider_async(
                    prompt=retry_prompt,
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
                        "Topic %r researcher retry after truncation failed on turn %d: %s",
                        sub_query.id,
                        turn + 1,
                        llm_result.error,
                    )
                    return None
                return llm_result
            except Exception as retry_exc:
                logger.warning(
                    "Topic %r researcher retry after truncation exception on turn %d: %s",
                    sub_query.id,
                    turn + 1,
                    retry_exc,
                )
                return None
        except (asyncio.TimeoutError, OSError, ValueError, RuntimeError) as exc:
            # Check if a generic exception is actually a context-window error
            # from a provider that doesn't raise ContextWindowError directly.
            from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
                _is_context_window_error,
            )

            if _is_context_window_error(exc):
                logger.warning(
                    "Topic %r researcher hit provider-specific context window error on turn %d, "
                    "truncating history and retrying once: %s",
                    sub_query.id,
                    turn + 1,
                    exc,
                )
                truncated = _truncate_researcher_history(message_history, researcher_model)
                retry_prompt = _build_react_user_prompt(
                    topic=sub_query.query,
                    message_history=truncated,
                    budget_remaining=budget_remaining,
                    budget_total=max_searches,
                )
                try:
                    llm_result = await self._execute_provider_async(
                        prompt=retry_prompt,
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
                        return None
                    return llm_result
                except Exception:
                    return None
            logger.warning(
                "Topic %r researcher LLM call exception on turn %d: %s",
                sub_query.id,
                turn + 1,
                exc,
            )
            return None

    @staticmethod
    def _inject_reflection_prompt(
        message_history: list[dict[str, str]],
        raw_content: str,
    ) -> None:
        """Inject a synthetic reflection-required prompt into history.

        Called when the researcher skips ``think`` between consecutive
        search turns. Appends the assistant's response and a system
        message requesting reflection.

        Args:
            message_history: Conversation history (mutated in-place).
            raw_content: The assistant's raw response to record.
        """
        message_history.append({"role": "assistant", "content": raw_content})
        message_history.append(
            {
                "role": "tool",
                "tool": "system",
                "content": (
                    "REFLECTION REQUIRED: You must call `think` to reflect on your "
                    "previous search results before issuing another search. Assess "
                    "what you found, identify gaps, and plan your next step. "
                    "Respond with ONLY a `think` tool call."
                ),
            }
        )

    async def _parse_with_retry_async(
        self,
        raw_content: str,
        message_history: list[dict[str, str]],
        sub_query: SubQuery,
        result: TopicResearchResult,
        turn: int,
        provider_id: str | None,
        researcher_model: str | None,
        system_prompt: str,
        budget_remaining: int,
        max_searches: int,
    ) -> tuple[Any, int]:
        """Parse tool calls from LLM response, retrying on parse failure.

        Re-prompts the LLM up to 2 times on unparseable JSON, matching
        ODR's stop_after_attempt=3 total attempts. Appends clarification
        messages to ``message_history`` on each retry (mutates in-place).

        Args:
            raw_content: Raw LLM response text to parse.
            message_history: Conversation history (mutated on retries).
            sub_query: The sub-query being researched (for logging).
            result: TopicResearchResult (parse failure counter updated).
            turn: Current turn index (for logging).
            provider_id: LLM provider ID for retry calls.
            researcher_model: LLM model for retry calls.
            system_prompt: System prompt for retry calls.
            budget_remaining: Remaining tool call budget (for prompt).
            max_searches: Total tool call budget (for prompt).

        Returns:
            Tuple of (parsed response, tokens_used_delta). The response
            may have empty tool_calls if all retries were exhausted.
        """
        response = parse_researcher_response(raw_content)
        tokens_delta = 0

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
            message_history.append(
                {
                    "role": "tool",
                    "tool": "system",
                    "content": (
                        "Your previous response was not valid JSON. Please respond "
                        "with ONLY a JSON object in the exact format:\n"
                        '{"tool_calls": [{"tool": "tool_name", "arguments": {...}}]}\n'
                        "Do not include any text outside the JSON object."
                    ),
                }
            )
            retry_user_prompt = _build_react_user_prompt(
                topic=sub_query.query,
                message_history=_truncate_researcher_history(message_history, researcher_model),
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
                tokens_delta += retry_result.tokens_used or 0
                raw_content = retry_result.content or ""
                response = parse_researcher_response(raw_content)
            except (asyncio.TimeoutError, OSError, ValueError, RuntimeError):
                break

        return response, tokens_delta

    @staticmethod
    def _check_reflection_needed(
        response: Any,
        previous_turn_had_search: bool,
        search_turn_count: int,
    ) -> tuple[bool, bool]:
        """Check whether a synthetic reflection prompt should be injected.

        After the first search turn, the researcher must include a ``think``
        call between consecutive search turns. If the researcher skips
        reflection, this method signals that the turn should be retried
        with a reflection-required prompt.

        Args:
            response: Parsed researcher response with tool_calls.
            previous_turn_had_search: Whether the prior turn had a search.
            search_turn_count: Number of search turns so far (first exempt).

        Returns:
            Tuple of (needs_reflection, current_has_search).
        """
        current_has_search = any(tc.tool in ("web_search", "extract_content") for tc in response.tool_calls)
        current_has_think = any(tc.tool == "think" for tc in response.tool_calls)

        needs_reflection = (
            previous_turn_had_search
            and current_has_search
            and not current_has_think
            and search_turn_count > 0  # first search turn is exempt
        )
        return needs_reflection, current_has_search

    async def _dispatch_tool_calls(
        self,
        response: Any,
        sub_query: SubQuery,
        state: DeepResearchState,
        result: TopicResearchResult,
        message_history: list[dict[str, str]],
        available_providers: list[Any],
        max_sources_per_provider: int | None,
        timeout: float,
        seen_urls: set[str],
        seen_titles: dict[str, str],
        state_lock: asyncio.Lock,
        semaphore: asyncio.Semaphore,
        budget_remaining: int,
        extract_enabled: bool,
        extract_max_per_iter: int,
        citation_tools_enabled: bool = False,
        pdf_extraction_enabled: bool = False,
    ) -> tuple[bool, int, int]:
        """Dispatch all tool calls from a single researcher turn.

        Processes tool calls in order (think first, then action tools).
        Appends tool-result messages to ``message_history`` and updates
        ``result`` in-place.

        Returns:
            Tuple of (loop_should_break, tool_calls_delta, budget_delta).
        """
        # Sort tool calls: Think first (before action tools), then others
        think_calls = [tc for tc in response.tool_calls if tc.tool == "think"]
        action_calls = [tc for tc in response.tool_calls if tc.tool != "think"]
        ordered_calls = think_calls + action_calls

        tool_calls_delta = 0
        budget_delta = 0

        for tool_call in ordered_calls:
            if tool_call.tool == "think":
                tool_result_text = self._handle_think_tool(
                    tool_call=tool_call,
                    sub_query=sub_query,
                    result=result,
                    state=state,
                )
                message_history.append(
                    {
                        "role": "tool",
                        "tool": "think",
                        "content": tool_result_text,
                    }
                )

            elif tool_call.tool == "web_search":
                if budget_remaining - budget_delta <= 0:
                    message_history.append(
                        {
                            "role": "tool",
                            "tool": "web_search",
                            "content": "Budget exhausted. No more searches allowed.",
                        }
                    )
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
                    budget_remaining=budget_remaining - budget_delta,
                )
                tool_calls_delta += queries_charged
                budget_delta += queries_charged
                result.searches_performed += queries_charged
                message_history.append(
                    {
                        "role": "tool",
                        "tool": "web_search",
                        "content": tool_result_text,
                    }
                )

            elif tool_call.tool == "extract_content":
                if not extract_enabled:
                    message_history.append(
                        {
                            "role": "tool",
                            "tool": "extract_content",
                            "content": "Content extraction is not available.",
                        }
                    )
                    continue

                if budget_remaining - budget_delta <= 0:
                    message_history.append(
                        {
                            "role": "tool",
                            "tool": "extract_content",
                            "content": "Budget exhausted. No more extractions allowed.",
                        }
                    )
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
                tool_calls_delta += 1
                budget_delta += 1
                message_history.append(
                    {
                        "role": "tool",
                        "tool": "extract_content",
                        "content": tool_result_text,
                    }
                )

            elif tool_call.tool == "extract_pdf":
                if not pdf_extraction_enabled:
                    message_history.append(
                        {
                            "role": "tool",
                            "tool": "extract_pdf",
                            "content": "PDF extraction is not available for this research profile.",
                        }
                    )
                    continue

                if budget_remaining - budget_delta <= 0:
                    message_history.append(
                        {
                            "role": "tool",
                            "tool": "extract_pdf",
                            "content": "Budget exhausted. No more extractions allowed.",
                        }
                    )
                    continue

                tool_result_text = await self._handle_extract_pdf_tool(
                    tool_call=tool_call,
                    sub_query=sub_query,
                    state=state,
                    result=result,
                    seen_urls=seen_urls,
                    seen_titles=seen_titles,
                    state_lock=state_lock,
                    semaphore=semaphore,
                    timeout=timeout,
                )
                tool_calls_delta += 1
                budget_delta += 1
                message_history.append(
                    {
                        "role": "tool",
                        "tool": "extract_pdf",
                        "content": tool_result_text,
                    }
                )

            elif tool_call.tool == "citation_search":
                if not citation_tools_enabled:
                    message_history.append(
                        {
                            "role": "tool",
                            "tool": "citation_search",
                            "content": "Citation search is not available for this research profile.",
                        }
                    )
                    continue

                if budget_remaining - budget_delta <= 0:
                    message_history.append(
                        {
                            "role": "tool",
                            "tool": "citation_search",
                            "content": "Budget exhausted. No more searches allowed.",
                        }
                    )
                    continue

                tool_result_text = await self._handle_citation_search_tool(
                    tool_call=tool_call,
                    sub_query=sub_query,
                    state=state,
                    result=result,
                    seen_urls=seen_urls,
                    seen_titles=seen_titles,
                    state_lock=state_lock,
                    semaphore=semaphore,
                    timeout=timeout,
                )
                tool_calls_delta += 1
                budget_delta += 1
                result.searches_performed += 1
                message_history.append(
                    {
                        "role": "tool",
                        "tool": "citation_search",
                        "content": tool_result_text,
                    }
                )

            elif tool_call.tool == "related_papers":
                if not citation_tools_enabled:
                    message_history.append(
                        {
                            "role": "tool",
                            "tool": "related_papers",
                            "content": "Related papers search is not available for this research profile.",
                        }
                    )
                    continue

                if budget_remaining - budget_delta <= 0:
                    message_history.append(
                        {
                            "role": "tool",
                            "tool": "related_papers",
                            "content": "Budget exhausted. No more searches allowed.",
                        }
                    )
                    continue

                tool_result_text = await self._handle_related_papers_tool(
                    tool_call=tool_call,
                    sub_query=sub_query,
                    state=state,
                    result=result,
                    seen_urls=seen_urls,
                    seen_titles=seen_titles,
                    state_lock=state_lock,
                    semaphore=semaphore,
                    timeout=timeout,
                )
                tool_calls_delta += 1
                budget_delta += 1
                result.searches_performed += 1
                message_history.append(
                    {
                        "role": "tool",
                        "tool": "related_papers",
                        "content": tool_result_text,
                    }
                )

            elif tool_call.tool == "research_complete":
                summary = tool_call.arguments.get("summary", "")
                result.early_completion = True
                result.completion_rationale = summary
                message_history.append(
                    {
                        "role": "tool",
                        "tool": "research_complete",
                        "content": "Research complete. Findings recorded.",
                    }
                )
                return True, tool_calls_delta, budget_delta

            else:
                logger.warning(
                    "Topic %r researcher called unknown tool %r, ignoring",
                    sub_query.id,
                    tool_call.tool,
                )

        return False, tool_calls_delta, budget_delta

    def _build_raw_notes_from_history(
        self,
        message_history: list[dict[str, str]],
        result: TopicResearchResult,
    ) -> None:
        """Build raw_notes from message history (Phase 1 ODR alignment).

        Captures unprocessed concatenation of all tool-result and assistant
        messages before compression. This provides a fallback when compression
        degrades or fails, and ground-truth evidence for the evaluator.

        Truncates to ``deep_research_max_content_length`` config value.
        Mutates ``result.raw_notes`` in-place.

        Args:
            message_history: The researcher's conversation history.
            result: TopicResearchResult to update with raw_notes.
        """
        raw_notes_parts: list[str] = []
        for msg in message_history:
            if msg.get("role") in ("assistant", "tool"):
                content = msg.get("content", "")
                if content:
                    raw_notes_parts.append(content)
        if raw_notes_parts:
            raw_notes_text = "\n".join(raw_notes_parts)
            try:
                max_raw_notes_len = int(self.config.deep_research_max_content_length)
            except (TypeError, ValueError):
                max_raw_notes_len = 50_000
            if len(raw_notes_text) > max_raw_notes_len:
                raw_notes_text = raw_notes_text[:max_raw_notes_len]
            result.raw_notes = raw_notes_text

    async def _apply_inline_compression_async(
        self,
        result: TopicResearchResult,
        sub_query: SubQuery,
        state: DeepResearchState,
        state_lock: asyncio.Lock,
        timeout: float,
    ) -> None:
        """Apply inline per-topic compression if enabled and applicable.

        Checks config, calls ``_compress_single_topic_async``, and accounts
        for compression tokens under lock. Errors are non-fatal: on failure,
        supervision falls back to metadata-only assessment.

        Mutates ``result.compressed_findings`` and ``state.total_tokens_used``
        in-place.

        Args:
            result: TopicResearchResult to compress.
            sub_query: The sub-query (for logging).
            state: Current research state (for token accounting).
            state_lock: Lock for thread-safe state mutations.
            timeout: Compression timeout.
        """
        inline_compression_enabled = getattr(self.config, "deep_research_inline_compression", True)
        if not (inline_compression_enabled and result.sources_found > 0 and result.compressed_findings is None):
            return

        try:
            comp_input, comp_output, comp_ok = await self._compress_single_topic_async(
                topic_result=result,
                state=state,
                timeout=timeout,
            )
            if comp_ok:
                if _compression_output_is_valid(
                    result.compressed_findings,
                    result.message_history,
                    sub_query.id,
                ):
                    # compressed_findings now captures the essential content —
                    # free the raw message_history to bound state memory growth.
                    result.message_history.clear()
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

    async def _finalize_topic_result(
        self,
        result: TopicResearchResult,
        sub_query: SubQuery,
        state: DeepResearchState,
        state_lock: asyncio.Lock,
        local_tokens_used: int,
        tool_calls_used: int,
        message_history: list[dict[str, str]],
        reflection_injections: int,
        extract_enabled: bool,
        timeout: float,
    ) -> TopicResearchResult:
        """Finalize topic research: persist history, compress, write audit.

        Called after the ReAct loop completes. Handles:
        1. Persisting message history on the result.
        2. Building raw_notes from history.
        3. Merging accumulated tokens under lock.
        4. Marking the sub_query completed or failed.
        5. Running inline compression (non-fatal on error).
        6. Writing the audit event.

        Args:
            result: TopicResearchResult to finalize.
            sub_query: The sub-query that was researched.
            state: Current research state.
            state_lock: Lock for thread-safe state mutations.
            local_tokens_used: Tokens consumed by LLM calls.
            tool_calls_used: Tool calls charged against budget.
            message_history: The researcher's conversation history.
            reflection_injections: Count of synthetic reflection prompts.
            extract_enabled: Whether extract_content was available.
            timeout: Timeout for inline compression.

        Returns:
            The finalized TopicResearchResult.
        """
        # Persist message history for downstream compression
        result.message_history = list(message_history)

        # Build raw_notes from message history
        self._build_raw_notes_from_history(message_history, result)

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

        # Inline per-topic compression
        await self._apply_inline_compression_async(
            result=result,
            sub_query=sub_query,
            state=state,
            state_lock=state_lock,
            timeout=timeout,
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

    # Strategy keywords recognized in think tool output for provenance logging
    _STRATEGY_KEYWORDS = ("BROADEN", "DEEPEN", "VALIDATE", "SATURATE")

    def _handle_think_tool(
        self,
        tool_call: ResearcherToolCall,
        sub_query: SubQuery,
        result: TopicResearchResult,
        state: DeepResearchState | None = None,
    ) -> str:
        """Handle a Think tool call: log reasoning, return acknowledgment.

        Args:
            tool_call: The think tool call with reasoning argument.
            sub_query: The sub-query being researched (for logging).
            result: TopicResearchResult to update with reflection notes.
            state: Optional research state for provenance logging.

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

        # Detect strategy keywords and log in provenance
        if reasoning and state is not None and state.provenance is not None:
            reasoning_upper = reasoning.upper()
            detected = [kw for kw in self._STRATEGY_KEYWORDS if kw in reasoning_upper]
            if detected:
                state.provenance.append(
                    phase="supervision",
                    event_type="strategy_detected",
                    summary=f"Strategy {', '.join(detected)} used in topic {sub_query.id}",
                    strategies=detected,
                    sub_query_id=sub_query.id,
                    reasoning_excerpt=reasoning[:200],
                )

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
        from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
            resolve_phase_provider,
            safe_resolve_model_for_role,
        )

        min_content_length = int(self.config.deep_research_summarization_min_content_length)
        per_result_timeout = float(self.config.deep_research_summarization_timeout)
        # Cap per-result timeout for inline summarization (30s default)
        per_result_timeout = min(per_result_timeout, 30.0)

        # Filter to sources that need summarization
        candidates = [
            src
            for src in sources
            if not src.metadata.get("summarized") and src.content and len(src.content) > min_content_length
        ]
        if not candidates:
            return

        # Resolve summarization provider/model
        role_provider, role_model = safe_resolve_model_for_role(self.config, "summarization")
        provider_id = role_provider or resolve_phase_provider(self.config, "summarization")

        summarizer = SourceSummarizer(
            provider_id=provider_id,
            model=role_model,
            timeout=per_result_timeout,
            max_concurrent=3,
            max_content_length=getattr(self.config, "deep_research_max_content_length", 50_000),
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
                tokens_used += summary_result.input_tokens + summary_result.output_tokens

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
        queries = queries[: max(budget_remaining, 1)]
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
                        "Batch query %r failed: %s",
                        queries[i],
                        r,
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
        from foundry_mcp.core.research.workflows.deep_research._content_dedup import (
            NoveltyTag,
            compute_novelty_tag,
        )
        from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
            build_novelty_summary,
        )

        # Build existing source tuples (content, title, url) for comparison
        # Only include sources already known *before* this search batch
        pre_existing_sources: list[tuple[str, str, str | None]] = []
        pre_existing_ids = {s.id for s in recent_sources}
        for s in topic_sources:
            if s.id not in pre_existing_ids:
                pre_existing_sources.append((s.content or s.snippet or "", s.title, s.url))

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
        from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
            validate_extract_url,
        )

        try:
            extract_args = ExtractContentTool.model_validate(tool_call.arguments)
            urls = extract_args.urls[:extract_max]
        except Exception:
            raw_urls = tool_call.arguments.get("urls", [])
            if isinstance(raw_urls, list):
                urls = [str(u) for u in raw_urls if isinstance(u, str)][:extract_max]
            else:
                return "Invalid URLs argument."

        # SSRF protection: filter out private/internal URLs
        urls = [u for u in urls if validate_extract_url(u, resolve_dns=True)]

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

        result.reflection_notes.append(f"[extract] Fetched {extract_added} source(s) from {len(urls)} URL(s)")
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
        from foundry_mcp.core.research.workflows.deep_research._content_dedup import (
            NoveltyTag,
            compute_novelty_tag,
        )
        from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
            build_novelty_summary,
        )

        pre_existing_sources: list[tuple[str, str, str | None]] = [
            (s.content or s.snippet or "", s.title, s.url) for s in state.sources if s.id in pre_extract_source_ids
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
        for idx, (src, ntag) in enumerate(zip(newly_extracted, novelty_tags, strict=False), 1):
            blocks.append(_format_source_block(idx, src, ntag))

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # PDF extraction handler (PLAN-4 Item 1d)
    # ------------------------------------------------------------------

    async def _handle_extract_pdf_tool(
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
    ) -> str:
        """Handle an extract_pdf tool call: fetch and extract a PDF.

        Uses PDFExtractor to download and extract text from an open-access
        PDF URL, then applies section detection and prioritized extraction.
        Creates a ResearchSource with page boundary metadata.

        Args:
            tool_call: The extract_pdf tool call with url argument.
            sub_query: The sub-query being researched.
            state: Current research state.
            result: TopicResearchResult to update.
            seen_urls: Shared URL dedup set.
            seen_titles: Shared title dedup dict.
            state_lock: Lock for thread-safe state mutations.
            semaphore: Semaphore for concurrency control.
            timeout: Extraction timeout.

        Returns:
            Formatted tool result string with extracted content summary.
        """
        from foundry_mcp.core.research.models.sources import ResearchSource, SourceType
        from foundry_mcp.core.research.pdf_extractor import PDFExtractor
        from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
            validate_extract_url,
        )

        # Parse arguments
        try:
            args = ExtractPDFTool.model_validate(tool_call.arguments)
            url = args.url
            max_pages = args.max_pages
        except Exception:
            url = tool_call.arguments.get("url", "")
            max_pages = int(tool_call.arguments.get("max_pages", 30))

        if not url:
            return "Error: url is required for extract_pdf."

        # SSRF protection
        if not validate_extract_url(url, resolve_dns=True):
            return "Error: URL failed security validation."

        # Use config for max_pages if available
        config_max_pages = getattr(self.config, "deep_research_pdf_max_pages", 50)
        max_pages = min(max_pages, config_max_pages)

        priority_sections = getattr(
            self.config, "deep_research_pdf_priority_sections", None
        ) or ["methods", "results", "discussion"]

        async with semaphore:
            try:
                extractor = PDFExtractor(max_pages=max_pages, timeout=timeout)
                pdf_result = await asyncio.wait_for(
                    extractor.extract_from_url(url),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return f"PDF extraction timed out after {timeout:.0f}s for {url}"
            except Exception as exc:
                logger.warning("PDF extraction failed for %s: %s", url, exc)
                return f"PDF extraction failed: {exc}"

        if not pdf_result.success:
            warnings_str = "; ".join(pdf_result.warnings[:3]) if pdf_result.warnings else "no text extracted"
            return f"PDF extraction yielded no text ({warnings_str}). The PDF may be scanned or paywalled."

        # Section detection and prioritized extraction
        sections = extractor.detect_sections(pdf_result)
        content = extractor.extract_prioritized(
            pdf_result,
            max_chars=50000,
            priority_sections=list(priority_sections),
        )

        # Build section summary for the researcher
        section_names = list(sections.keys()) if sections else []
        section_info = f"Sections detected: {', '.join(section_names)}" if section_names else "No standard sections detected"

        # Build page boundary metadata for downstream digest pipeline
        page_boundaries = [
            (i + 1, start, end)
            for i, (start, end) in enumerate(pdf_result.page_offsets)
        ]

        # Create source with PDF metadata
        source = ResearchSource(
            url=url,
            title=f"PDF: {url.split('/')[-1][:80]}",
            source_type=SourceType.ACADEMIC,
            content=content,
            snippet=f"Full-text PDF ({pdf_result.extracted_page_count} pages). {section_info}",
            sub_query_id=sub_query.id,
            metadata={
                "pdf_extraction": True,
                "page_count": pdf_result.page_count,
                "extracted_page_count": pdf_result.extracted_page_count,
                "sections": section_names,
                "page_boundaries": page_boundaries,
                "warnings": pdf_result.warnings[:5],
            },
        )

        was_added, dedup_reason = await _dedup_and_add_source(
            source=source,
            sub_query=sub_query,
            state=state,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
            state_lock=state_lock,
        )

        if not was_added:
            return f"PDF content from {url} was deduplicated ({dedup_reason})."

        result.sources_found += 1

        # Build response for the researcher
        content_preview = sanitize_external_content(
            content[:500] + "..." if len(content) > 500 else content
        )
        response_parts = [
            f"Extracted {pdf_result.extracted_page_count} pages from PDF ({len(content)} chars).",
            section_info,
            f"\nCONTENT PREVIEW:\n{content_preview}",
        ]
        if pdf_result.warnings:
            response_parts.append(f"\nWarnings: {'; '.join(pdf_result.warnings[:3])}")

        return "\n".join(response_parts)

    # ------------------------------------------------------------------
    # Citation and related papers handlers
    # ------------------------------------------------------------------

    async def _handle_citation_search_tool(
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
    ) -> str:
        """Handle a citation_search tool call: find papers citing a given paper.

        Tries Semantic Scholar first, falls back to OpenAlex. Runs dedup
        and novelty scoring on results following the web_search handler pattern.

        Args:
            tool_call: The citation_search tool call with paper_id argument.
            sub_query: The sub-query being researched.
            state: Current research state.
            result: TopicResearchResult to update.
            seen_urls: Shared URL dedup set.
            seen_titles: Shared title dedup dict.
            state_lock: Lock for thread-safe state mutations.
            semaphore: Semaphore for concurrency control.
            timeout: Request timeout.

        Returns:
            Formatted tool result string.
        """
        # Parse arguments
        try:
            args = CitationSearchTool.model_validate(tool_call.arguments)
            paper_id = args.paper_id
            max_results = args.max_results
        except Exception:
            paper_id = tool_call.arguments.get("paper_id", "")
            max_results = int(tool_call.arguments.get("max_results", 10))

        if not paper_id:
            return "Error: paper_id is required for citation_search."

        # Try Semantic Scholar first, then OpenAlex as fallback
        sources: list[Any] = []
        provider_name = "unknown"
        async with semaphore:
            for name in ("semantic_scholar", "openalex"):
                provider = self._get_search_provider(name)
                if provider is None:
                    continue
                try:
                    self._check_cancellation(state)
                    provider_name = name
                    sources = await asyncio.wait_for(
                        provider.get_citations(paper_id, max_results),
                        timeout=timeout,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "Citation search via %s failed for %r: %s",
                        name,
                        paper_id[:50],
                        e,
                    )
                    continue

        return await self._process_academic_tool_results(
            sources=sources,
            provider_name=provider_name,
            tool_name="citation_search",
            paper_id=paper_id,
            sub_query=sub_query,
            state=state,
            result=result,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
            state_lock=state_lock,
        )

    async def _handle_related_papers_tool(
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
    ) -> str:
        """Handle a related_papers tool call: find papers similar to a given paper.

        Tries Semantic Scholar recommendations first, falls back to OpenAlex
        get_related. Runs dedup and novelty scoring on results.

        Args:
            tool_call: The related_papers tool call with paper_id argument.
            sub_query: The sub-query being researched.
            state: Current research state.
            result: TopicResearchResult to update.
            seen_urls: Shared URL dedup set.
            seen_titles: Shared title dedup dict.
            state_lock: Lock for thread-safe state mutations.
            semaphore: Semaphore for concurrency control.
            timeout: Request timeout.

        Returns:
            Formatted tool result string.
        """
        # Parse arguments
        try:
            args = RelatedPapersTool.model_validate(tool_call.arguments)
            paper_id = args.paper_id
            max_results = args.max_results
        except Exception:
            paper_id = tool_call.arguments.get("paper_id", "")
            max_results = int(tool_call.arguments.get("max_results", 5))

        if not paper_id:
            return "Error: paper_id is required for related_papers."

        # Try Semantic Scholar recommendations first, then OpenAlex related
        sources: list[Any] = []
        provider_name = "unknown"
        async with semaphore:
            # Semantic Scholar recommendations
            s2_provider = self._get_search_provider("semantic_scholar")
            if s2_provider is not None:
                try:
                    self._check_cancellation(state)
                    provider_name = "semantic_scholar"
                    sources = await asyncio.wait_for(
                        s2_provider.get_recommendations(paper_id, max_results),
                        timeout=timeout,
                    )
                except Exception as e:
                    logger.warning(
                        "Related papers via semantic_scholar failed for %r: %s",
                        paper_id[:50],
                        e,
                    )

            # Fallback to OpenAlex if S2 failed or returned nothing
            if not sources:
                oa_provider = self._get_search_provider("openalex")
                if oa_provider is not None:
                    try:
                        self._check_cancellation(state)
                        provider_name = "openalex"
                        sources = await asyncio.wait_for(
                            oa_provider.get_related(paper_id, max_results),
                            timeout=timeout,
                        )
                    except Exception as e:
                        logger.warning(
                            "Related papers via openalex failed for %r: %s",
                            paper_id[:50],
                            e,
                        )

        return await self._process_academic_tool_results(
            sources=sources,
            provider_name=provider_name,
            tool_name="related_papers",
            paper_id=paper_id,
            sub_query=sub_query,
            state=state,
            result=result,
            seen_urls=seen_urls,
            seen_titles=seen_titles,
            state_lock=state_lock,
        )

    async def _process_academic_tool_results(
        self,
        sources: list[Any],
        provider_name: str,
        tool_name: str,
        paper_id: str,
        sub_query: SubQuery,
        state: DeepResearchState,
        result: TopicResearchResult,
        seen_urls: set[str],
        seen_titles: dict[str, str],
        state_lock: asyncio.Lock,
    ) -> str:
        """Process results from citation_search or related_papers tools.

        Handles dedup, novelty scoring, provenance logging, and formatting —
        shared logic for both academic tool handlers.

        Args:
            sources: ResearchSource objects from the provider.
            provider_name: Which provider returned the results.
            tool_name: Tool name for provenance logging.
            paper_id: The seed paper ID.
            sub_query: The sub-query being researched.
            state: Current research state.
            result: TopicResearchResult to update.
            seen_urls: Shared URL dedup set.
            seen_titles: Shared title dedup dict.
            state_lock: Lock for thread-safe state mutations.

        Returns:
            Formatted tool result string.
        """
        if not sources:
            return f"No results found for {tool_name} with paper_id={paper_id!r}."

        # Dedup sources
        content_dedup_enabled = getattr(self.config, "deep_research_enable_content_dedup", True)
        dedup_threshold = getattr(self.config, "deep_research_content_dedup_threshold", 0.8)

        added = 0
        added_source_ids: list[str] = []
        for source in sources:
            was_added, dedup_reason = await _dedup_and_add_source(
                source=source,
                sub_query=sub_query,
                state=state,
                seen_urls=seen_urls,
                seen_titles=seen_titles,
                state_lock=state_lock,
                content_dedup_enabled=content_dedup_enabled,
                dedup_threshold=dedup_threshold,
            )
            if was_added:
                added += 1
                added_source_ids.append(source.id)
            elif dedup_reason and state.provenance is not None:
                state.provenance.append(
                    phase="supervision",
                    event_type="source_deduplicated",
                    summary=f"Source deduplicated ({dedup_reason}): {(source.title or '')[:80]}",
                    source_url=source.url or "",
                    source_title=(source.title or "")[:120],
                    reason=dedup_reason,
                )

        # Log provenance
        if state.provenance is not None:
            state.provenance.append(
                phase="supervision",
                event_type="provider_query",
                summary=f"{tool_name} via {provider_name}: {len(sources)} results for paper_id={paper_id!r}",
                provider=provider_name,
                query=f"{tool_name}:{paper_id}",
                result_count=len(sources),
                source_ids=added_source_ids,
            )

        # Track provider stats
        async with state_lock:
            state.search_provider_stats[provider_name] = (
                state.search_provider_stats.get(provider_name, 0) + 1
            )

        result.sources_found += added

        if added == 0:
            return f"{tool_name} for paper_id={paper_id!r} returned {len(sources)} result(s), but all were duplicates of existing sources."

        # Novelty scoring
        from foundry_mcp.core.research.workflows.deep_research._content_dedup import (
            NoveltyTag,
            compute_novelty_tag,
        )
        from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
            build_novelty_summary,
        )

        topic_source_ids = set(sub_query.source_ids)
        topic_sources = [s for s in state.sources if s.id in topic_source_ids]
        recent_sources = topic_sources[-added:]

        pre_existing_ids = {s.id for s in recent_sources}
        pre_existing_sources: list[tuple[str, str, str | None]] = []
        for s in topic_sources:
            if s.id not in pre_existing_ids:
                pre_existing_sources.append((s.content or s.snippet or "", s.title, s.url))

        novelty_tags: list[NoveltyTag] = []
        for src in recent_sources:
            tag = compute_novelty_tag(
                new_content=src.content or src.snippet or "",
                new_url=src.url,
                existing_sources=pre_existing_sources,
            )
            novelty_tags.append(tag)
            src.metadata["novelty_tag"] = tag.category
            src.metadata["novelty_similarity"] = tag.similarity

        novelty_header = build_novelty_summary(novelty_tags)
        return _format_search_results_batch(
            sources=recent_sources,
            novelty_tags=novelty_tags,
            novelty_header=novelty_header,
        )

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

                    # Read dedup config once outside the per-source loop
                    content_dedup_enabled = getattr(self.config, "deep_research_enable_content_dedup", True)
                    dedup_threshold = getattr(self.config, "deep_research_content_dedup_threshold", 0.8)

                    added_source_ids: list[str] = []
                    for source in sources:
                        was_added, dedup_reason = await _dedup_and_add_source(
                            source=source,
                            sub_query=sub_query,
                            state=state,
                            seen_urls=seen_urls,
                            seen_titles=seen_titles,
                            state_lock=state_lock,
                            content_dedup_enabled=content_dedup_enabled,
                            dedup_threshold=dedup_threshold,
                        )
                        if was_added:
                            added += 1
                            added_source_ids.append(source.id)
                        elif dedup_reason and state.provenance is not None:
                            state.provenance.append(
                                phase="supervision",
                                event_type="source_deduplicated",
                                summary=f"Source deduplicated ({dedup_reason}): {(source.title or '')[:80]}",
                                source_url=source.url or "",
                                source_title=(source.title or "")[:120],
                                reason=dedup_reason,
                            )

                    # Log provider_query provenance event
                    if state.provenance is not None:
                        state.provenance.append(
                            phase="supervision",
                            event_type="provider_query",
                            summary=f"Queried {provider_name}: {len(sources)} results for '{query[:80]}'",
                            provider=provider_name,
                            query=query,
                            result_count=len(sources),
                            source_ids=added_source_ids,
                        )

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

        Uses Tavily Extract API for web pages and PDFExtractor for PDF URLs.
        PDF URLs are detected by pattern (*.pdf, arxiv.org/pdf/*) and routed
        to the PDF extraction pipeline with section detection.

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

        # Split PDF URLs from regular web URLs for routing
        pdf_urls = [u for u in urls if _is_pdf_url(u)]
        web_urls = [u for u in urls if not _is_pdf_url(u)]

        added = 0

        # --- Route PDF URLs to PDFExtractor ---
        if pdf_urls:
            added += await self._extract_pdf_urls(
                pdf_urls=pdf_urls,
                sub_query=sub_query,
                state=state,
                seen_urls=seen_urls,
                seen_titles=seen_titles,
                state_lock=state_lock,
                semaphore=semaphore,
                timeout=timeout,
            )

        # --- Route web URLs to Tavily Extract ---
        if not web_urls:
            return added

        api_key = self.config.tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            logger.debug("Tavily API key not available for topic extract")
            return added

        async with semaphore:
            try:
                provider = TavilyExtractProvider(api_key=api_key)
                extract_depth = self.config.tavily_extract_depth

                extracted_sources = await asyncio.wait_for(
                    provider.extract(
                        urls=web_urls,
                        extract_depth=extract_depth,
                        format="markdown",
                        query=sub_query.query,
                    ),
                    timeout=timeout,
                )

                # Read dedup config
                content_dedup_enabled = getattr(self.config, "deep_research_enable_content_dedup", True)
                dedup_threshold = getattr(self.config, "deep_research_content_dedup_threshold", 0.8)

                for source in extracted_sources:
                    # Tag as extracted source before dedup (metadata preserved)
                    source.sub_query_id = sub_query.id
                    source.metadata["extract_source"] = True

                    was_added, dedup_reason = await _dedup_and_add_source(
                        source=source,
                        sub_query=sub_query,
                        state=state,
                        seen_urls=seen_urls,
                        seen_titles=seen_titles,
                        state_lock=state_lock,
                        content_dedup_enabled=content_dedup_enabled,
                        dedup_threshold=dedup_threshold,
                    )
                    if was_added:
                        added += 1
                    elif dedup_reason and state.provenance is not None:
                        state.provenance.append(
                            phase="supervision",
                            event_type="source_deduplicated",
                            summary=f"Source deduplicated ({dedup_reason}): {(source.title or '')[:80]}",
                            source_url=source.url or "",
                            source_title=(source.title or "")[:120],
                            reason=dedup_reason,
                        )

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
    # PDF URL extraction helper (PLAN-4 Item 1c)
    # ------------------------------------------------------------------

    async def _extract_pdf_urls(
        self,
        pdf_urls: list[str],
        sub_query: SubQuery,
        state: DeepResearchState,
        seen_urls: set[str],
        seen_titles: dict[str, str],
        state_lock: asyncio.Lock,
        semaphore: asyncio.Semaphore,
        timeout: float = 60.0,
    ) -> int:
        """Extract content from PDF URLs via PDFExtractor.

        Called by ``_topic_extract`` when URLs match PDF patterns.
        Creates ResearchSource objects with page boundary metadata
        for downstream page-aware digest locators.

        Args:
            pdf_urls: PDF URLs to extract (pre-validated).
            sub_query: The sub-query being researched.
            state: Current research state.
            seen_urls: Shared URL dedup set.
            seen_titles: Shared title dedup dict.
            state_lock: Lock for thread-safe state mutations.
            semaphore: Semaphore for concurrency control.
            timeout: Per-extraction timeout.

        Returns:
            Number of new sources added from PDF extraction.
        """
        from foundry_mcp.core.research.models.sources import ResearchSource, SourceType
        from foundry_mcp.core.research.pdf_extractor import PDFExtractor

        config_max_pages = getattr(self.config, "deep_research_pdf_max_pages", 50)
        priority_sections = getattr(
            self.config, "deep_research_pdf_priority_sections", None
        ) or ["methods", "results", "discussion"]

        added = 0
        for url in pdf_urls:
            try:
                extractor = PDFExtractor(max_pages=config_max_pages, timeout=timeout)
                async with semaphore:
                    pdf_result = await asyncio.wait_for(
                        extractor.extract_from_url(url),
                        timeout=timeout,
                    )

                if not pdf_result.success:
                    logger.debug("PDF extraction yielded no text for %s", url)
                    continue

                # Section-aware content extraction
                sections = extractor.detect_sections(pdf_result)
                content = extractor.extract_prioritized(
                    pdf_result,
                    max_chars=50000,
                    priority_sections=list(priority_sections),
                )
                section_names = list(sections.keys()) if sections else []

                # Page boundaries for digest locators
                page_boundaries = [
                    (i + 1, start, end)
                    for i, (start, end) in enumerate(pdf_result.page_offsets)
                ]

                source = ResearchSource(
                    url=url,
                    title=f"PDF: {url.split('/')[-1][:80]}",
                    source_type=SourceType.ACADEMIC,
                    content=content,
                    snippet=f"Full-text PDF ({pdf_result.extracted_page_count} pages)",
                    sub_query_id=sub_query.id,
                    metadata={
                        "pdf_extraction": True,
                        "extract_source": True,
                        "page_count": pdf_result.page_count,
                        "extracted_page_count": pdf_result.extracted_page_count,
                        "sections": section_names,
                        "page_boundaries": page_boundaries,
                        "warnings": pdf_result.warnings[:5],
                    },
                )

                was_added, dedup_reason = await _dedup_and_add_source(
                    source=source,
                    sub_query=sub_query,
                    state=state,
                    seen_urls=seen_urls,
                    seen_titles=seen_titles,
                    state_lock=state_lock,
                )
                if was_added:
                    added += 1
                    logger.info(
                        "PDF extracted for %r: %s (%d pages, %d chars)",
                        sub_query.id,
                        url[:80],
                        pdf_result.extracted_page_count,
                        len(content),
                    )
                elif dedup_reason and state.provenance is not None:
                    state.provenance.append(
                        phase="supervision",
                        event_type="source_deduplicated",
                        summary=f"PDF source deduplicated ({dedup_reason}): {url[:80]}",
                        source_url=url,
                        source_title=f"PDF: {url.split('/')[-1][:80]}",
                        reason=dedup_reason,
                    )

            except asyncio.TimeoutError:
                logger.warning("PDF extraction timed out for %s after %.1fs", url, timeout)
            except Exception as exc:
                logger.warning("PDF extraction failed for %s: %s. Falling back to Tavily.", url, exc)

        return added
