"""Synthesis phase mixin for DeepResearchWorkflow.

Generates a comprehensive markdown report from analyzed findings and sources.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from foundry_mcp.core.observability import get_metrics
from foundry_mcp.core.providers import ContextWindowError
from foundry_mcp.core.research.context_budget import AllocationResult
from foundry_mcp.core.research.models import (
    DeepResearchState,
    PhaseMetrics,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._budgeting import (
    allocate_synthesis_budget,
    final_fit_validate,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (
    SYNTHESIS_OUTPUT_RESERVED,
)
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    fidelity_level_from_score,
)

if TYPE_CHECKING:
    from foundry_mcp.core.research.workflows.deep_research.core import (
        DeepResearchWorkflow,
    )

logger = logging.getLogger(__name__)


class SynthesisPhaseMixin:
    """Synthesis phase methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)
    """

    async def _execute_synthesis_async(
        self: DeepResearchWorkflow,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute synthesis phase: generate comprehensive report from findings.

        This phase:
        1. Builds a synthesis prompt with all findings grouped by theme
        2. Includes source references for citation
        3. Generates a structured markdown report with:
           - Executive summary
           - Key findings organized by theme
           - Source citations
           - Knowledge gaps and limitations
           - Conclusions with actionable insights
        4. Stores the report in state.report

        Args:
            state: Current research state with findings from analysis
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with synthesis outcome
        """
        if not state.findings:
            logger.warning("No findings to synthesize")
            # Generate a minimal report even without findings
            state.report = self._generate_empty_report(state)
            self._write_audit_event(
                state,
                "synthesis_result",
                data={
                    "provider_id": None,
                    "model_used": None,
                    "tokens_used": None,
                    "duration_ms": None,
                    "system_prompt": None,
                    "user_prompt": None,
                    "raw_response": None,
                    "report": state.report,
                    "empty_report": True,
                },
                level="warning",
            )
            return WorkflowResult(
                success=True,
                content=state.report,
                metadata={
                    "research_id": state.id,
                    "finding_count": 0,
                    "empty_report": True,
                },
            )

        logger.info(
            "Starting synthesis phase: %d findings, %d sources",
            len(state.findings),
            len(state.sources),
        )

        # Emit phase.started audit event
        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "synthesis",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        # Allocate token budget for findings and sources
        allocation_result = allocate_synthesis_budget(
            state=state,
            provider_id=provider_id,
        )

        # Update state with allocation metadata
        # Store overall fidelity in metadata (content_fidelity is now per-item dict)
        state.dropped_content_ids = allocation_result.dropped_ids
        allocation_dict = allocation_result.to_dict()
        allocation_dict["overall_fidelity_level"] = fidelity_level_from_score(
            allocation_result.fidelity
        )
        state.content_allocation_metadata = allocation_dict

        logger.info(
            "Synthesis budget allocation: %d items allocated, %d dropped, fidelity=%.1f%%",
            len(allocation_result.items),
            len(allocation_result.dropped_ids),
            allocation_result.fidelity * 100,
        )

        # Build the synthesis prompt with allocated content
        system_prompt = self._build_synthesis_system_prompt(state)
        user_prompt = self._build_synthesis_user_prompt(state, allocation_result)

        # Final-fit validation before provider dispatch
        valid, _preflight, system_prompt, user_prompt = final_fit_validate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.synthesis_provider,
            model=state.synthesis_model,
            output_reserved=SYNTHESIS_OUTPUT_RESERVED,
            phase="synthesis",
        )

        if not valid:
            logger.warning(
                "Synthesis phase final-fit validation failed, proceeding with truncated prompts"
            )

        # Check for cancellation before making provider call
        self._check_cancellation(state)

        # Execute LLM call with context window error handling and timeout protection
        effective_provider = provider_id or state.synthesis_provider
        llm_call_start_time = time.perf_counter()
        # Update heartbeat and persist interim state for progress visibility
        state.last_heartbeat_at = datetime.now(timezone.utc)
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "llm.call.started",
            data={
                "provider": effective_provider,
                "task_id": state.id,
                "phase": "synthesis",
            },
        )
        try:
            result = await self._execute_provider_async(
                prompt=user_prompt,
                provider_id=effective_provider,
                model=state.synthesis_model,
                system_prompt=system_prompt,
                timeout=timeout,
                temperature=0.5,  # Balanced for coherent but varied writing
                phase="synthesis",
                fallback_providers=self.config.get_phase_fallback_providers("synthesis"),
                max_retries=self.config.deep_research_max_retries,
                retry_delay=self.config.deep_research_retry_delay,
            )
        except ContextWindowError as e:
            llm_call_duration_ms = (time.perf_counter() - llm_call_start_time) * 1000
            self._write_audit_event(
                state,
                "llm.call.completed",
                data={
                    "provider": effective_provider,
                    "task_id": state.id,
                    "duration_ms": llm_call_duration_ms,
                    "status": "error",
                    "error_type": "context_window_exceeded",
                },
            )
            get_metrics().histogram(
                "foundry_mcp_research_llm_call_duration_seconds",
                llm_call_duration_ms / 1000.0,
                labels={"provider": effective_provider or "unknown", "status": "error"},
            )
            logger.error(
                "Synthesis phase context window exceeded: prompt_tokens=%s, "
                "max_tokens=%s, truncation_needed=%s, provider=%s, finding_count=%d",
                e.prompt_tokens,
                e.max_tokens,
                e.truncation_needed,
                e.provider,
                len(state.findings),
            )
            return WorkflowResult(
                success=False,
                content="",
                error=str(e),
                metadata={
                    "research_id": state.id,
                    "phase": "synthesis",
                    "error_type": "context_window_exceeded",
                    "prompt_tokens": e.prompt_tokens,
                    "max_tokens": e.max_tokens,
                    "truncation_needed": e.truncation_needed,
                    "finding_count": len(state.findings),
                    "guidance": "Try reducing the number of findings or source content included",
                },
            )

        # Emit llm.call.completed audit event
        llm_call_duration_ms = (time.perf_counter() - llm_call_start_time) * 1000
        llm_call_status = "success" if result.success else "error"
        llm_call_provider: str = result.provider_id or effective_provider or "unknown"
        self._write_audit_event(
            state,
            "llm.call.completed",
            data={
                "provider": llm_call_provider,
                "task_id": state.id,
                "duration_ms": llm_call_duration_ms,
                "status": llm_call_status,
            },
        )
        get_metrics().histogram(
            "foundry_mcp_research_llm_call_duration_seconds",
            llm_call_duration_ms / 1000.0,
            labels={"provider": llm_call_provider, "status": llm_call_status},
        )

        if not result.success:
            # Check if this was a timeout
            if result.metadata and result.metadata.get("timeout"):
                logger.error(
                    "Synthesis phase timed out after exhausting all providers: %s",
                    result.metadata.get("providers_tried", []),
                )
            else:
                logger.error("Synthesis phase LLM call failed: %s", result.error)
            return result

        # Track token usage
        if result.tokens_used:
            state.total_tokens_used += result.tokens_used

        # Track phase metrics for audit
        state.phase_metrics.append(
            PhaseMetrics(
                phase="synthesis",
                duration_ms=result.duration_ms or 0.0,
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
                cached_tokens=result.cached_tokens or 0,
                provider_id=result.provider_id,
                model_used=result.model_used,
            )
        )

        # Extract the markdown report from the response
        report = self._extract_markdown_report(result.content)

        if not report:
            logger.warning("Failed to extract report from synthesis response")
            # Use raw content as fallback
            report = result.content

        # Store report in state
        state.report = report

        # Save state
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "synthesis_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": result.content,
                "report": state.report,
                "report_length": len(state.report),
            },
        )

        logger.info(
            "Synthesis phase complete: report length %d chars",
            len(state.report),
        )

        # Emit phase.completed audit event
        phase_duration_ms = (time.perf_counter() - phase_start_time) * 1000
        self._write_audit_event(
            state,
            "phase.completed",
            data={
                "phase_name": "synthesis",
                "iteration": state.iteration,
                "task_id": state.id,
                "duration_ms": phase_duration_ms,
            },
        )

        # Emit phase duration metric
        get_metrics().histogram(
            "foundry_mcp_research_phase_duration_seconds",
            phase_duration_ms / 1000.0,
            labels={"phase_name": "synthesis", "status": "success"},
        )

        return WorkflowResult(
            success=True,
            content=state.report,
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "finding_count": len(state.findings),
                "source_count": len(state.sources),
                "report_length": len(state.report),
                "iteration": state.iteration,
            },
        )

    def _build_synthesis_system_prompt(self: DeepResearchWorkflow, state: DeepResearchState) -> str:
        """Build system prompt for report synthesis.

        Args:
            state: Current research state (reserved for future state-aware prompts)

        Returns:
            System prompt string
        """
        # state is reserved for future state-aware prompt customization
        _ = state
        return """You are a research synthesizer. Your task is to create a comprehensive, well-structured research report from analyzed findings.

Generate a markdown-formatted report with the following structure:

# Research Report: [Topic]

## Executive Summary
A 2-3 paragraph overview of the key insights and conclusions.

## Key Findings

### [Theme/Category 1]
- Finding with supporting evidence and source citations [Source ID]
- Related findings grouped together

### [Theme/Category 2]
- Continue for each major theme...

## Analysis

### Supporting Evidence
Discussion of well-supported findings with high confidence.

### Conflicting Information
Note any contradictions or disagreements between sources (if present).

### Limitations
Acknowledge gaps in the research and areas needing further investigation.

## Sources
List sources as markdown links with their IDs: **[src-xxx]** [Title](URL)

## Conclusions
Actionable insights and recommendations based on the findings.

---

Guidelines:
- Organize findings thematically rather than listing them sequentially
- Cite source IDs in brackets when referencing specific information [src-xxx]
- Distinguish between high-confidence findings (well-supported) and lower-confidence insights
- Be specific and actionable in conclusions
- Keep the report focused on the original research query
- Use clear, professional language
- Include all relevant findings - don't omit information

IMPORTANT: Return ONLY the markdown report, no preamble or meta-commentary."""

    def _build_synthesis_user_prompt(
        self: DeepResearchWorkflow,
        state: DeepResearchState,
        allocation_result: Optional[AllocationResult] = None,
    ) -> str:
        """Build user prompt with findings and sources for synthesis.

        Args:
            state: Current research state
            allocation_result: Optional budget allocation result for token-aware prompts

        Returns:
            User prompt string
        """
        prompt_parts = [
            f"# Research Query\n{state.original_query}",
            "",
            f"## Research Brief\n{state.research_brief or 'Direct research on the query'}",
            "",
            "## Findings to Synthesize",
            "",
        ]

        # Group findings by category if available
        categorized: dict[str, list] = {}

        for finding in state.findings:
            category = finding.category or "General"
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(finding)

        # Add findings by category - findings are protected, always included at full fidelity
        for category, findings in categorized.items():
            prompt_parts.append(f"### {category}")
            for f in findings:
                confidence_label = f.confidence.value if hasattr(f.confidence, 'value') else str(f.confidence)
                source_refs = ", ".join(f.source_ids) if f.source_ids else "no sources"
                prompt_parts.append(f"- [{confidence_label.upper()}] {f.content}")
                prompt_parts.append(f"  Sources: {source_refs}")
            prompt_parts.append("")

        # Add knowledge gaps
        if state.gaps:
            prompt_parts.append("## Knowledge Gaps Identified")
            for gap in state.gaps:
                status = "addressed" if gap.resolved else "unresolved"
                prompt_parts.append(f"- [{status}] {gap.description}")
            prompt_parts.append("")

        # Add source reference list - use allocation-aware content
        prompt_parts.append("## Source Reference")

        if allocation_result:
            # Use allocated sources in priority order, applying token limits
            for item in allocation_result.items:
                # Skip findings (they're in the findings section)
                if not item.id.startswith("src-"):
                    continue

                source = next((s for s in state.sources if s.id == item.id), None)
                if not source:
                    continue

                quality = source.quality.value if hasattr(source.quality, 'value') else str(source.quality)
                prompt_parts.append(f"- **{source.id}**: {source.title} [{quality}]")
                if source.url:
                    prompt_parts.append(f"  URL: {source.url}")

                # Apply token-aware content limit for snippets
                if item.needs_summarization:
                    # Compressed: use allocated tokens to estimate character limit (~4 chars/token)
                    char_limit = max(50, item.allocated_tokens * 4)
                    if source.snippet:
                        snippet = source.snippet[:char_limit]
                        if len(source.snippet) > char_limit:
                            snippet += "..."
                        prompt_parts.append(f"  Snippet: {snippet}")
                else:
                    # Full fidelity: include snippet up to 200 chars
                    if source.snippet:
                        snippet = source.snippet[:200]
                        if len(source.snippet) > 200:
                            snippet += "..."
                        prompt_parts.append(f"  Snippet: {snippet}")

            # Note dropped sources if any
            if allocation_result.dropped_ids:
                dropped_sources = [sid for sid in allocation_result.dropped_ids if sid.startswith("src-")]
                if dropped_sources:
                    prompt_parts.append(f"\n*Note: {len(dropped_sources)} additional source(s) omitted for context limits*")
        else:
            # Fallback: use first 30 sources (legacy behavior)
            for source in state.sources[:30]:
                quality = source.quality.value if hasattr(source.quality, 'value') else str(source.quality)
                prompt_parts.append(f"- {source.id}: {source.title} [{quality}]")
                if source.url:
                    prompt_parts.append(f"  URL: {source.url}")

        prompt_parts.append("")

        # Add synthesis instructions
        prompt_parts.extend([
            "## Instructions",
            f"Generate a comprehensive research report addressing the query: '{state.original_query}'",
            "",
            f"This is iteration {state.iteration} of {state.max_iterations}.",
            f"Total findings: {len(state.findings)}",
            f"Total sources: {len(state.sources)}",
            f"Unresolved gaps: {len(state.unresolved_gaps())}",
            "",
            "Create a well-structured markdown report following the format specified.",
        ])

        return "\n".join(prompt_parts)

    def _extract_markdown_report(self: DeepResearchWorkflow, content: str) -> Optional[str]:
        """Extract markdown report from LLM response.

        The response should be pure markdown, but this handles cases where
        the LLM wraps it in code blocks or adds preamble.

        Args:
            content: Raw LLM response content

        Returns:
            Extracted markdown report or None if extraction fails
        """
        if not content:
            return None

        # If content starts with markdown heading, it's likely clean
        if content.strip().startswith("#"):
            return content.strip()

        # Check for markdown code block wrapper
        if "```markdown" in content or "```md" in content:
            # Extract content between code blocks
            pattern = r'```(?:markdown|md)?\s*([\s\S]*?)```'
            matches = re.findall(pattern, content)
            if matches:
                return matches[0].strip()

        # Check for generic code block
        if "```" in content:
            pattern = r'```\s*([\s\S]*?)```'
            matches = re.findall(pattern, content)
            for match in matches:
                # Check if it looks like markdown (has headings)
                if match.strip().startswith("#") or "##" in match:
                    return match.strip()

        # Look for first heading and take everything from there
        heading_match = re.search(r'^(#[^\n]+)', content, re.MULTILINE)
        if heading_match:
            start_pos = heading_match.start()
            return content[start_pos:].strip()

        # If nothing else works, return the trimmed content
        return content.strip() if len(content.strip()) > 50 else None

    def _generate_empty_report(self: DeepResearchWorkflow, state: DeepResearchState) -> str:
        """Generate a minimal report when no findings are available.

        Args:
            state: Current research state

        Returns:
            Minimal markdown report
        """
        return f"""# Research Report

## Executive Summary

Research was conducted on the query: "{state.original_query}"

Unfortunately, the analysis phase did not yield extractable findings from the gathered sources. This may indicate:
- The sources lacked relevant information
- The query may need refinement
- Additional research iterations may be needed

## Research Query

{state.original_query}

## Research Brief

{state.research_brief or "No research brief generated."}

## Sources Examined

{len(state.sources)} source(s) were examined during this research session.

## Recommendations

1. Consider refining the research query for more specific results
2. Try additional research iterations if available
3. Review the gathered sources manually for relevant information

---

*Report generated with no extractable findings. Iteration {state.iteration}/{state.max_iterations}.*
"""
