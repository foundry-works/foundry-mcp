"""Brief phase mixin for DeepResearchWorkflow.

Transforms the raw user query into a structured research brief that
maximises specificity, fills unstated dimensions as open-ended, specifies
source preferences, and defines scope boundaries.  The brief drives all
downstream phases (planning, supervision, synthesis).

Adapted from open_deep_research's ``write_research_brief`` pattern.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    ResearchBriefOutput,
    parse_brief_output,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    execute_structured_llm_call,
    finalize_phase,
)

logger = logging.getLogger(__name__)


class BriefPhaseMixin:
    """Brief phase methods. Mixed into DeepResearchWorkflow.

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

    async def _execute_brief_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute brief phase: enrich the raw query into a research brief.

        Transforms the user's original query (plus any clarification
        constraints) into a detailed, structured research brief paragraph
        that maximises specificity and drives all downstream phases.

        Non-fatal: if brief generation fails, logs a warning and proceeds
        with the original query as-is.

        Args:
            state: Current research state
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with brief generation outcome
        """
        logger.info("Starting brief phase for query: %s", state.original_query[:100])

        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "brief",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        system_prompt = self._build_brief_system_prompt()
        user_prompt = self._build_brief_user_prompt(state)

        self._check_cancellation(state)

        # Use structured output parsing with automatic retry on parse failure.
        # parse_brief_output() accepts both JSON and plain-text responses
        # for backward compatibility.
        call_result = await execute_structured_llm_call(
            workflow=self,
            state=state,
            phase_name="brief",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id,
            model=None,  # Resolved by role
            temperature=0.4,  # Moderate creativity — enrichment, not invention
            timeout=timeout,
            parse_fn=parse_brief_output,
            role="brief",  # Research-tier via _ROLE_RESOLUTION_CHAIN
        )

        if isinstance(call_result, WorkflowResult):
            # Brief generation failed — non-fatal, proceed with original query
            logger.warning(
                "Brief generation LLM call failed for research %s, "
                "proceeding with original query: %s",
                state.id,
                call_result.error,
            )
            self._write_audit_event(
                state,
                "brief_generation_failed",
                data={
                    "error": call_result.error,
                    "fallback": "original_query",
                },
                level="warning",
            )
            # Do NOT set state.research_brief — leave it as None so
            # planning's inline refinement sub-step can still run as fallback
            finalize_phase(self, state, "brief", phase_start_time)
            return WorkflowResult(
                success=True,  # Non-fatal failure
                content="Brief generation failed, using original query",
                metadata={
                    "research_id": state.id,
                    "brief_generated": False,
                    "fallback": "original_query",
                },
            )

        result = call_result.result

        # Extract brief text from structured parse or raw content
        if call_result.parsed is not None:
            brief_output: ResearchBriefOutput = call_result.parsed
            brief_text = brief_output.research_brief.strip()
        else:
            brief_text = (result.content or "").strip()

        if brief_text:
            state.research_brief = brief_text
            logger.info(
                "Brief generation complete (%d chars) for research %s",
                len(brief_text),
                state.id,
            )
        else:
            logger.warning(
                "Brief generation returned empty content for research %s, "
                "proceeding with original query",
                state.id,
            )

        # Persist state with the generated brief
        self.memory.save_deep_research(state)

        self._write_audit_event(
            state,
            "brief_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "brief_length": len(brief_text),
                "brief_generated": bool(brief_text),
                "research_brief": state.research_brief,
            },
        )

        finalize_phase(self, state, "brief", phase_start_time)

        return WorkflowResult(
            success=True,
            content=state.research_brief or "Brief phase complete",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "brief_generated": bool(brief_text),
                "brief_length": len(brief_text),
            },
        )

    def _build_brief_system_prompt(self) -> str:
        """Build system prompt for research brief generation.

        Adapts open_deep_research's ``transform_messages_into_research_topic_prompt``
        approach: maximise specificity, prefer primary sources, fill unstated
        dimensions as open-ended, avoid unwarranted assumptions.

        Returns:
            System prompt string
        """
        return (
            "You are a research brief writer. Your task is to transform a "
            "user's research request into a detailed, structured research "
            "brief that will drive a multi-phase deep research workflow.\n\n"
            "Your brief MUST:\n"
            "1. **Maximise specificity**: Extract every concrete detail the "
            "user provided (names, dates, versions, quantities, geographic "
            "scope) and foreground them prominently.\n"
            "2. **Fill unstated dimensions as open-ended**: When the user "
            "leaves a dimension unspecified (time period, geography, "
            "methodology), explicitly note it as an open question the "
            "research should address — do NOT assume a value.\n"
            "3. **Specify source preferences**: Bias toward primary and "
            "official sources (specifications, documentation, peer-reviewed "
            "work, government datasets, original research papers) over "
            "aggregators or secondary commentary.\n"
            "4. **Define scope boundaries**: State what the research should "
            "include AND what it should exclude to prevent scope creep.\n\n"
            "Output your response as a JSON object with this schema:\n"
            "{\n"
            '  "research_brief": "Complete brief as one or two well-structured paragraphs",\n'
            '  "scope_boundaries": "What the research should include and exclude" or null,\n'
            '  "source_preferences": "Preferred source types" or null\n'
            "}\n\n"
            "IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."
        )

    def _build_brief_user_prompt(self, state: DeepResearchState) -> str:
        """Build user prompt for research brief generation.

        Includes the original query, any clarification constraints from the
        CLARIFICATION phase, the current date for temporal context, and
        optional system prompt context.

        Args:
            state: Current research state

        Returns:
            User prompt string
        """
        parts: list[str] = [
            f"Research request:\n{state.original_query}",
        ]

        if state.system_prompt:
            parts.append(
                f"\nAdditional context provided by the user:\n{state.system_prompt}"
            )

        if state.clarification_constraints:
            parts.append(
                "\nClarification constraints (already confirmed with the user):"
            )
            for key, value in state.clarification_constraints.items():
                parts.append(f"- {key}: {value}")

        # Temporal context helps the brief scope time-sensitive research
        parts.append(
            f"\nCurrent date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        )

        return "\n".join(parts)
