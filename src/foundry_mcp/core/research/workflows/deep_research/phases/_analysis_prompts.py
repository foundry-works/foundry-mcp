"""Prompt-building mixin for the analysis phase.

Constructs system and user prompts for the analysis LLM call.
Split from ``analysis.py`` to keep each module focused.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.context_budget import AllocationResult
from foundry_mcp.core.research.document_digest import deserialize_payload
from foundry_mcp.core.research.models.deep_research import DeepResearchState

if TYPE_CHECKING:
    from foundry_mcp.core.research.workflows.deep_research.core import (
        DeepResearchWorkflow,
    )

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
        self: DeepResearchWorkflow,
        state: DeepResearchState,
        allocation_result: Optional[AllocationResult] = None,
    ) -> str:
        """Build user prompt with source summaries for analysis.

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
            "Sources to Analyze:",
            "",
        ]

        # Build source lookup for allocation info
        allocated_map: dict[str, Any] = {}
        if allocation_result:
            for item in allocation_result.items:
                allocated_map[item.id] = item

        # Add source summaries based on allocation
        sources_to_include = []
        if allocation_result:
            # Use allocated sources in priority order
            for item in allocation_result.items:
                source = next((s for s in state.sources if s.id == item.id), None)
                if source:
                    sources_to_include.append((source, item))
        else:
            # Fallback: use first 20 sources (legacy behavior)
            for source in state.sources[:20]:
                sources_to_include.append((source, None))

        for i, (source, alloc_item) in enumerate(sources_to_include, 1):
            prompt_parts.append(f"Source {i} (ID: {source.id}):")
            prompt_parts.append(f"  Title: {source.title}")
            if source.url:
                prompt_parts.append(f"  URL: {source.url}")

            # Determine content limit based on allocation
            if alloc_item and alloc_item.needs_summarization:
                # Use allocated tokens to estimate character limit (~4 chars/token)
                char_limit = max(100, alloc_item.allocated_tokens * 4)
                snippet_limit = min(500, char_limit // 3)
                content_limit = min(1000, char_limit - snippet_limit)
            else:
                # Full fidelity: use default limits
                snippet_limit = 500
                content_limit = 1000

            if source.snippet:
                snippet = source.snippet[:snippet_limit]
                if len(source.snippet) > snippet_limit:
                    snippet += "..."
                prompt_parts.append(f"  Snippet: {snippet}")

            if source.content:
                # Check if source contains a digest payload
                if source.is_digest:
                    # Parse digest and use evidence snippets for citations
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
                                prompt_parts.append(f"    - \"{ev.text[:200]}\" [{ev.locator}]")
                    except Exception:
                        # Fallback to raw content if parsing fails
                        content = source.content[:content_limit]
                        prompt_parts.append(f"  Content: {content}")
                else:
                    content = source.content[:content_limit]
                    if len(source.content) > content_limit:
                        content += "..."
                    prompt_parts.append(f"  Content: {content}")

            prompt_parts.append("")

        prompt_parts.extend([
            "Please analyze these sources and:",
            "1. Extract 2-5 key findings relevant to the research query",
            "2. Assess confidence levels based on source agreement and authority",
            "3. Identify any knowledge gaps or unanswered questions",
            "4. Assess the quality of each source",
            "",
            "Return your analysis as JSON.",
        ])

        return "\n".join(prompt_parts)
