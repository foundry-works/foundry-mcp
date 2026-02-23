"""Prompt-building mixin for the analysis phase.

Constructs system and user prompts for the analysis LLM call.
Split from ``analysis.py`` to keep each module focused.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from foundry_mcp.core.research.context_budget import AllocationResult
from foundry_mcp.core.research.document_digest import deserialize_payload
from foundry_mcp.core.research.models.deep_research import DeepResearchState

logger = logging.getLogger(__name__)


class AnalysisPromptsMixin:
    """Prompt construction methods for analysis. Mixed into AnalysisPhaseMixin.

    At runtime, ``self`` is a DeepResearchWorkflow instance.
    """

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

    def _build_analysis_user_prompt(
        self,
        state: DeepResearchState,
        allocation_result: Optional[AllocationResult] = None,
    ) -> str:
        """Build user prompt with source summaries for analysis.

        When per-topic compressed findings are available (from Phase 3
        compression), uses those as pre-organized input — reducing
        token pressure and providing citation-rich, structured content.
        Falls through to raw source listing when compressed findings
        are absent.

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
        ]

        # Check if we have compressed findings from per-topic compression
        compressed_topics = [
            tr for tr in state.topic_research_results
            if tr.compressed_findings
        ]

        if compressed_topics:
            # Use compressed findings as primary analysis input
            prompt_parts.append("Per-Topic Research Summaries:")
            prompt_parts.append("(Pre-organized with inline citations per topic)")
            prompt_parts.append("")

            # Build a source ID → citation number map for reference
            source_id_to_citation = state.source_id_to_citation()

            for topic_idx, topic_result in enumerate(compressed_topics, 1):
                sub_query = state.get_sub_query(topic_result.sub_query_id)
                query_text = sub_query.query if sub_query else "Unknown query"

                prompt_parts.append(f"--- Topic {topic_idx}: {query_text} ---")
                prompt_parts.append(topic_result.compressed_findings or "")
                prompt_parts.append("")

                # Include source ID mapping so the LLM can reference them
                if topic_result.source_ids:
                    prompt_parts.append("  Source ID mapping:")
                    for src_id in topic_result.source_ids:
                        source = state.get_source(src_id)
                        if source:
                            citation = source_id_to_citation.get(src_id, "?")
                            prompt_parts.append(
                                f"    {src_id} [citation {citation}]: {source.title}"
                            )
                    prompt_parts.append("")

            # If some topics lack compressed findings, include their raw sources
            uncompressed_source_ids: set[str] = set()
            for tr in state.topic_research_results:
                if not tr.compressed_findings:
                    uncompressed_source_ids.update(tr.source_ids)

            if uncompressed_source_ids:
                prompt_parts.append("Additional Sources (not yet compressed):")
                prompt_parts.append("")
                self._append_raw_sources(
                    prompt_parts,
                    state,
                    allocation_result,
                    source_filter=uncompressed_source_ids,
                )
        else:
            # No compressed findings — use raw source listing (existing behavior)
            prompt_parts.append("Sources to Analyze:")
            prompt_parts.append("")
            self._append_raw_sources(prompt_parts, state, allocation_result)

        prompt_parts.extend(
            [
                "Please analyze these sources and:",
                "1. Extract 2-5 key findings relevant to the research query",
                "2. Assess confidence levels based on source agreement and authority",
                "3. Identify any knowledge gaps or unanswered questions",
                "4. Assess the quality of each source",
                "",
                "Return your analysis as JSON.",
            ]
        )

        return "\n".join(prompt_parts)

    def _append_raw_sources(
        self,
        prompt_parts: list[str],
        state: DeepResearchState,
        allocation_result: Optional[AllocationResult] = None,
        source_filter: Optional[set[str]] = None,
    ) -> None:
        """Append raw source listings to prompt parts.

        Extracted from ``_build_analysis_user_prompt`` to support both
        the full-source and filtered-source (uncompressed remainder) paths.

        Args:
            prompt_parts: List to append prompt lines to (mutated in place).
            state: Current research state.
            allocation_result: Optional budget allocation result.
            source_filter: If provided, only include sources whose ID is in this set.
        """
        # Build source lookup for allocation info
        allocated_map: dict[str, Any] = {}
        if allocation_result:
            for item in allocation_result.items:
                allocated_map[item.id] = item

        # Add source summaries based on allocation
        sources_to_include: list[tuple[Any, Any]] = []
        if allocation_result:
            for item in allocation_result.items:
                if source_filter and item.id not in source_filter:
                    continue
                source = next((s for s in state.sources if s.id == item.id), None)
                if source:
                    sources_to_include.append((source, item))
        else:
            for source in state.sources[:20]:
                if source_filter and source.id not in source_filter:
                    continue
                sources_to_include.append((source, None))

        for i, (source, alloc_item) in enumerate(sources_to_include, 1):
            prompt_parts.append(f"Source {i} (ID: {source.id}):")
            prompt_parts.append(f"  Title: {source.title}")
            if source.url:
                prompt_parts.append(f"  URL: {source.url}")

            # Determine content limit based on allocation
            if alloc_item and alloc_item.needs_summarization:
                char_limit = max(100, alloc_item.allocated_tokens * 4)
                snippet_limit = min(500, char_limit // 3)
                content_limit = min(1000, char_limit - snippet_limit)
            else:
                snippet_limit = 500
                content_limit = 1000

            if source.snippet:
                snippet = source.snippet[:snippet_limit]
                if len(source.snippet) > snippet_limit:
                    snippet += "..."
                prompt_parts.append(f"  Snippet: {snippet}")

            if source.content:
                if source.is_digest:
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
                                prompt_parts.append(f'    - "{ev.text[:200]}" [{ev.locator}]')
                    except Exception:
                        content = source.content[:content_limit]
                        prompt_parts.append(f"  Content: {content}")
                else:
                    content = source.content[:content_limit]
                    if len(source.content) > content_limit:
                        content += "..."
                    prompt_parts.append(f"  Content: {content}")

            prompt_parts.append("")
