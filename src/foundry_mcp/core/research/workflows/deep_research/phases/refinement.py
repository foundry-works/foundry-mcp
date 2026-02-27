"""Refinement phase mixin for DeepResearchWorkflow.

Analyzes knowledge gaps and generates follow-up queries for the next iteration.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._budgeting import (
    compute_refinement_budget,
    final_fit_validate,
    summarize_report_for_refinement,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (
    REFINEMENT_OUTPUT_RESERVED,
)
from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
    sanitize_external_content,
)
from foundry_mcp.core.research.workflows.deep_research._json_parsing import (
    extract_json,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    execute_llm_call,
    finalize_phase,
)

logger = logging.getLogger(__name__)


class RefinementPhaseMixin:
    """Refinement phase methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)
    """

    config: Any
    memory: Any

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _check_cancellation(self, *args: Any, **kwargs: Any) -> None: ...

    async def _execute_refinement_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute refinement phase: analyze gaps and generate follow-up queries.

        This phase:
        1. Reviews the current report and identified gaps
        2. Uses LLM to assess gap severity and addressability
        3. Generates follow-up queries for unresolved gaps
        4. Converts high-priority gaps to new sub-queries for next iteration
        5. Respects max_iterations limit for workflow termination

        Args:
            state: Current research state with report and gaps
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with refinement outcome
        """
        unresolved_gaps = state.unresolved_gaps()

        # Check iteration limit
        if state.iteration >= state.max_iterations:
            logger.info(
                "Refinement: max iterations (%d) reached, no further refinement",
                state.max_iterations,
            )
            self._write_audit_event(
                state,
                "refinement_result",
                data={
                    "reason": "max_iterations_reached",
                    "unresolved_gaps": len(unresolved_gaps),
                    "iteration": state.iteration,
                },
                level="warning",
            )
            return WorkflowResult(
                success=True,
                content="Max iterations reached, refinement complete",
                metadata={
                    "research_id": state.id,
                    "iteration": state.iteration,
                    "max_iterations": state.max_iterations,
                    "unresolved_gaps": len(unresolved_gaps),
                    "reason": "max_iterations_reached",
                },
            )

        if not unresolved_gaps:
            logger.info("Refinement: no unresolved gaps, research complete")
            self._write_audit_event(
                state,
                "refinement_result",
                data={
                    "reason": "no_gaps",
                    "unresolved_gaps": 0,
                    "iteration": state.iteration,
                },
            )
            return WorkflowResult(
                success=True,
                content="No unresolved gaps, research complete",
                metadata={
                    "research_id": state.id,
                    "iteration": state.iteration,
                    "reason": "no_gaps",
                },
            )

        logger.info(
            "Starting refinement phase: %d unresolved gaps, iteration %d/%d",
            len(unresolved_gaps),
            state.iteration,
            state.max_iterations,
        )

        # Emit phase.started audit event
        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "refinement",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        # Compute budget allocation to prevent unbounded context growth
        _phase_budget, report_budget, remaining_budget = compute_refinement_budget(provider_id, state)

        # Summarize report if needed to fit within budget
        report_summary = ""
        report_fidelity = "full"
        if state.report:
            report_summary, report_fidelity = summarize_report_for_refinement(state.report, report_budget)

        # Update state fidelity tracking for refinement phase
        # Note: We update fidelity in metadata if we actually summarized
        if report_fidelity != "full":
            state.content_allocation_metadata["refinement_report_fidelity"] = report_fidelity
            logger.info(
                "Refinement phase using summarized context: report_fidelity=%s",
                report_fidelity,
            )

        # Build the refinement prompt with budget-aware content
        system_prompt = self._build_refinement_system_prompt(state)
        user_prompt = self._build_refinement_user_prompt(
            state,
            report_summary=report_summary,
            remaining_budget=remaining_budget,
        )

        # Final-fit validation before provider dispatch
        valid, _preflight, system_prompt, user_prompt = final_fit_validate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.refinement_provider,
            model=state.refinement_model,
            output_reserved=REFINEMENT_OUTPUT_RESERVED,
            phase="refinement",
        )

        if not valid:
            logger.warning("Refinement phase final-fit validation failed, proceeding with truncated prompts")

        # Check for cancellation before making provider call
        self._check_cancellation(state)

        # Execute LLM call with lifecycle instrumentation
        call_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="refinement",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.refinement_provider,
            model=state.refinement_model,
            temperature=0.4,  # Lower temperature for focused analysis
            timeout=timeout,
        )
        if isinstance(call_result, WorkflowResult):
            return call_result  # Error path
        result = call_result.result

        # Parse the response
        parsed = self._parse_refinement_response(result.content, state)

        if not parsed["success"]:
            logger.warning("Failed to parse refinement response, using existing gap suggestions")
            # Fallback: use existing gap suggestions as follow-up queries
            follow_up_queries = self._extract_fallback_queries(state)
        else:
            follow_up_queries = parsed["follow_up_queries"]

            # Mark gaps as resolved if specified
            for gap_id in parsed.get("addressed_gap_ids", []):
                gap = state.get_gap(gap_id)
                if gap:
                    gap.resolved = True

        # Convert follow-up queries to new sub-queries for next iteration
        new_sub_queries = 0
        for query_data in follow_up_queries[: state.max_sub_queries]:
            # Add as new sub-query
            state.add_sub_query(
                query=query_data["query"],
                rationale=query_data.get("rationale", "Follow-up from gap analysis"),
                priority=query_data.get("priority", 1),
            )
            new_sub_queries += 1

        # Save state
        self.memory.save_deep_research(state)
        self._write_audit_event(
            state,
            "refinement_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": result.content,
                "parse_success": parsed["success"],
                "gap_analysis": parsed.get("gap_analysis", []),
                "follow_up_queries": follow_up_queries,
                "addressed_gap_ids": parsed.get("addressed_gap_ids", []),
                "should_iterate": parsed.get("should_iterate", True),
            },
        )

        logger.info(
            "Refinement phase complete: %d follow-up queries generated",
            new_sub_queries,
        )

        finalize_phase(self, state, "refinement", phase_start_time)

        return WorkflowResult(
            success=True,
            content=f"Generated {new_sub_queries} follow-up queries from {len(unresolved_gaps)} gaps",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "iteration": state.iteration,
                "unresolved_gaps": len(unresolved_gaps),
                "follow_up_queries": new_sub_queries,
                "gaps_addressed": len(parsed.get("addressed_gap_ids", [])),
            },
        )

    def _build_refinement_system_prompt(self, state: DeepResearchState) -> str:
        """Build system prompt for gap analysis and refinement.

        Args:
            state: Current research state (reserved for future state-aware prompts)

        Returns:
            System prompt string
        """
        # state is reserved for future state-aware prompt customization
        _ = state
        return """You are a research refiner. Your task is to analyze knowledge gaps identified during research and generate focused follow-up queries to address them.

Your response MUST be valid JSON with this exact structure:
{
    "gap_analysis": [
        {
            "gap_id": "gap-xxx",
            "severity": "critical|moderate|minor",
            "addressable": true,
            "rationale": "Why this gap matters and whether it can be addressed"
        }
    ],
    "follow_up_queries": [
        {
            "query": "A specific, focused search query to address the gap",
            "target_gap_id": "gap-xxx",
            "rationale": "How this query will fill the gap",
            "priority": 1
        }
    ],
    "addressed_gap_ids": ["gap-xxx"],
    "iteration_recommendation": {
        "should_iterate": true,
        "rationale": "Why iteration is or isn't recommended"
    }
}

Guidelines:
- Assess each gap's severity: "critical" (blocks conclusions), "moderate" (affects confidence), "minor" (nice to have). Severity determines whether a follow-up iteration is worth the budget — only critical and moderate gaps justify the cost of another research cycle.
- Only mark gaps as addressable if follow-up research can realistically fill them. Some gaps exist because the information genuinely doesn't exist online (proprietary data, future events, classified information) — marking these as addressable wastes iteration budget on impossible searches.
- Generate 1-3 highly focused follow-up queries per addressable gap
- Priority 1 is highest priority
- Mark gaps as addressed if the current report already covers them adequately
- Recommend iteration only if there are addressable critical/moderate gaps AND value exceeds research cost

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."""

    def _build_refinement_user_prompt(
        self,
        state: DeepResearchState,
        report_summary: Optional[str] = None,
        remaining_budget: Optional[int] = None,
    ) -> str:
        """Build user prompt with gaps and report context for refinement.

        Args:
            state: Current research state
            report_summary: Pre-summarized report content (for budget-aware prompts)
            remaining_budget: Token budget for gaps and findings

        Returns:
            User prompt string
        """
        prompt_parts = [
            f"# Research Query\n{sanitize_external_content(state.original_query)}",
            "",
            "## Research Status",
            f"- Iteration: {state.iteration}/{state.max_iterations}",
            f"- Sources examined: {len(state.sources)}",
            f"- Findings extracted: {len(state.findings)}",
            f"- Unresolved gaps: {len(state.unresolved_gaps())}",
            "",
        ]

        # Add report summary - use provided summary or fallback to legacy truncation
        if report_summary:
            prompt_parts.append("## Current Report Summary")
            prompt_parts.append(report_summary)
            prompt_parts.append("")
        elif state.report:
            # Legacy fallback: simple truncation at 2000 chars
            report_excerpt = state.report[:2000]
            if len(state.report) > 2000:
                report_excerpt += "\n\n[Report truncated...]"
            prompt_parts.append("## Current Report Summary")
            prompt_parts.append(report_excerpt)
            prompt_parts.append("")

        # Calculate character budget for gaps and findings
        # Default to ~2000 chars for gaps, ~1000 for findings if no budget specified
        if remaining_budget:
            gap_char_budget = int(remaining_budget * 4 * 0.6)  # 60% for gaps
            finding_char_budget = int(remaining_budget * 4 * 0.4)  # 40% for findings
        else:
            gap_char_budget = 8000
            finding_char_budget = 4000

        # Add unresolved gaps with budget awareness
        prompt_parts.append("## Unresolved Knowledge Gaps")
        gaps_chars_used = 0
        gaps_included = 0
        for gap in state.unresolved_gaps():
            gap_text = f"\n### Gap: {gap.id}\nDescription: {gap.description}\nPriority: {gap.priority}"
            if gap.suggested_queries:
                gap_text += "\nSuggested queries from analysis:"
                for sq in gap.suggested_queries[:3]:
                    gap_text += f"\n  - {sq}"

            if gaps_chars_used + len(gap_text) <= gap_char_budget:
                prompt_parts.append(gap_text)
                gaps_chars_used += len(gap_text)
                gaps_included += 1
            else:
                # Budget exceeded - note remaining gaps
                remaining_gaps = len(state.unresolved_gaps()) - gaps_included
                if remaining_gaps > 0:
                    prompt_parts.append(f"\n*[{remaining_gaps} additional gap(s) omitted for context limits]*")
                break
        prompt_parts.append("")

        # Add high-confidence findings for context with budget awareness
        high_conf_findings = [
            f for f in state.findings if hasattr(f.confidence, "value") and f.confidence.value in ("high", "confirmed")
        ]
        if high_conf_findings:
            prompt_parts.append("## High-Confidence Findings Already Established")
            findings_chars_used = 0
            findings_included = 0
            for f in high_conf_findings:
                # Limit individual finding content
                content_limit = min(200, finding_char_budget // max(1, len(high_conf_findings)))
                finding_text = f"- {f.content[:content_limit]}"
                if len(f.content) > content_limit:
                    finding_text += "..."

                if findings_chars_used + len(finding_text) <= finding_char_budget:
                    prompt_parts.append(finding_text)
                    findings_chars_used += len(finding_text)
                    findings_included += 1
                else:
                    remaining = len(high_conf_findings) - findings_included
                    if remaining > 0:
                        prompt_parts.append(f"*[{remaining} additional finding(s) omitted]*")
                    break
            prompt_parts.append("")

        # Add instructions
        prompt_parts.extend(
            [
                "## Instructions",
                "1. Analyze each gap for severity and addressability",
                "2. Generate focused follow-up queries for addressable gaps",
                "3. Mark any gaps that are actually addressed by existing findings",
                "4. Recommend whether iteration is worthwhile given remaining gaps",
                "",
                "Return your analysis as JSON.",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_refinement_response(
        self,
        content: str,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Parse LLM response into structured refinement data.

        Args:
            content: Raw LLM response content
            state: Current research state (reserved for context-aware parsing)

        Returns:
            Dict with 'success', 'follow_up_queries', 'addressed_gap_ids', etc.
        """
        # state is reserved for future context-aware parsing
        _ = state
        result = {
            "success": False,
            "gap_analysis": [],
            "follow_up_queries": [],
            "addressed_gap_ids": [],
            "should_iterate": True,
        }

        if not content:
            return result

        # Try to extract JSON from the response
        json_str = extract_json(content)
        if not json_str:
            logger.warning("No JSON found in refinement response")
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from refinement response: %s", e)
            return result

        # Parse gap analysis
        raw_analysis = data.get("gap_analysis", [])
        if isinstance(raw_analysis, list):
            for ga in raw_analysis:
                if not isinstance(ga, dict):
                    continue
                result["gap_analysis"].append(
                    {
                        "gap_id": ga.get("gap_id", ""),
                        "severity": ga.get("severity", "moderate"),
                        "addressable": ga.get("addressable", True),
                        "rationale": ga.get("rationale", ""),
                    }
                )

        # Parse follow-up queries
        raw_queries = data.get("follow_up_queries", [])
        if isinstance(raw_queries, list):
            for fq in raw_queries:
                if not isinstance(fq, dict):
                    continue
                query = fq.get("query", "").strip()
                if not query:
                    continue
                result["follow_up_queries"].append(
                    {
                        "query": query,
                        "target_gap_id": fq.get("target_gap_id", ""),
                        "rationale": fq.get("rationale", ""),
                        "priority": min(max(int(fq.get("priority", 1)), 1), 10),
                    }
                )

        # Parse addressed gaps
        raw_addressed = data.get("addressed_gap_ids", [])
        if isinstance(raw_addressed, list):
            result["addressed_gap_ids"] = [gid for gid in raw_addressed if isinstance(gid, str)]

        # Parse iteration recommendation
        iter_rec = data.get("iteration_recommendation", {})
        if isinstance(iter_rec, dict):
            result["should_iterate"] = iter_rec.get("should_iterate", True)

        # Mark success if we got at least one follow-up query
        result["success"] = len(result["follow_up_queries"]) > 0

        return result

    def _extract_fallback_queries(self, state: DeepResearchState) -> list[dict[str, Any]]:
        """Extract follow-up queries from existing gap suggestions as fallback.

        Used when LLM parsing fails but we still want to progress.

        Args:
            state: Current research state with gaps

        Returns:
            List of follow-up query dictionaries
        """
        queries = []
        for gap in state.unresolved_gaps():
            for sq in gap.suggested_queries[:2]:  # Max 2 per gap
                queries.append(
                    {
                        "query": sq,
                        "target_gap_id": gap.id,
                        "rationale": f"Suggested query from gap: {gap.description[:50]}",
                        "priority": gap.priority,
                    }
                )
        return queries[: state.max_sub_queries]  # Respect limit
