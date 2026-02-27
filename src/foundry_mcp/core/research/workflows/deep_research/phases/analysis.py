"""Analysis phase mixin for DeepResearchWorkflow.

Extracts findings from gathered sources via LLM analysis.

Sub-modules:
- ``_analysis_prompts``: AnalysisPromptsMixin (system/user prompt construction)
- ``_analysis_parsing``: AnalysisParsingMixin (LLM response parsing)

Note: The digest pipeline (DigestStepMixin) and contradiction detection
have been removed — both are dead code since the pipeline was collapsed
to BRIEF → SUPERVISION → SYNTHESIS.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.document_digest import DocumentDigestor  # noqa: F401  # re-export for test patch targets
from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.models.sources import SourceQuality
from foundry_mcp.core.research.pdf_extractor import PDFExtractor  # noqa: F401  # re-export for test patch targets
from foundry_mcp.core.research.summarization import ContentSummarizer  # noqa: F401  # re-export for test patch targets
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._budgeting import (
    allocate_source_budget,
    final_fit_validate,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (
    ANALYSIS_OUTPUT_RESERVED,
)
from foundry_mcp.core.research.workflows.deep_research._token_budget import (
    fidelity_level_from_score,
)
from foundry_mcp.core.research.workflows.deep_research.phases._analysis_parsing import (
    AnalysisParsingMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases._analysis_prompts import (
    AnalysisPromptsMixin,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    execute_llm_call,
    finalize_phase,
)

logger = logging.getLogger(__name__)


class AnalysisPhaseMixin(AnalysisPromptsMixin, AnalysisParsingMixin):
    """Analysis phase methods. Mixed into DeepResearchWorkflow.

    Inherits from:
    - AnalysisPromptsMixin: ``_build_analysis_system_prompt``, ``_build_analysis_user_prompt``
    - AnalysisParsingMixin: ``_parse_analysis_response``

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)

    Note: This phase is retained for legacy resume compatibility.
    New workflows use BRIEF → SUPERVISION → SYNTHESIS.
    """

    config: Any
    memory: Any

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _check_cancellation(self, *args: Any, **kwargs: Any) -> None: ...
        async def _execute_provider_async(self, *args: Any, **kwargs: Any) -> Any: ...

    async def _execute_analysis_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute analysis phase: extract findings from sources.

        This phase:
        1. Builds prompt with gathered source summaries
        2. Uses LLM to extract key findings
        3. Assesses confidence levels for each finding
        4. Identifies knowledge gaps requiring follow-up
        5. Updates source quality assessments

        Args:
            state: Current research state with gathered sources
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with analysis outcome
        """
        if not state.sources:
            logger.warning("No sources to analyze")
            return WorkflowResult(
                success=True,
                content="No sources to analyze",
                metadata={"research_id": state.id, "finding_count": 0},
            )

        logger.info(
            "Starting analysis phase: %d sources to analyze",
            len(state.sources),
        )

        # Emit phase.started audit event
        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "analysis",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        # Allocate token budget for sources
        allocation_result = allocate_source_budget(
            state=state,
            provider_id=provider_id,
        )

        # Update state with allocation metadata
        # Store overall fidelity in metadata (content_fidelity is now per-item dict)
        state.dropped_content_ids = allocation_result.dropped_ids
        allocation_dict = allocation_result.to_dict()
        allocation_dict["overall_fidelity_level"] = fidelity_level_from_score(allocation_result.fidelity)
        state.content_allocation_metadata = allocation_dict

        logger.info(
            "Budget allocation: %d sources allocated, %d dropped, fidelity=%.1f%%",
            len(allocation_result.items),
            len(allocation_result.dropped_ids),
            allocation_result.fidelity * 100,
        )

        # Build the analysis prompt with allocated sources
        system_prompt = self._build_analysis_system_prompt(state)
        user_prompt = self._build_analysis_user_prompt(state, allocation_result)

        # Final-fit validation before provider dispatch
        valid, _preflight, system_prompt, user_prompt = final_fit_validate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.analysis_provider,
            model=state.analysis_model,
            output_reserved=ANALYSIS_OUTPUT_RESERVED,
            phase="analysis",
        )

        if not valid:
            logger.warning("Analysis phase final-fit validation failed, proceeding with truncated prompts")

        # Check for cancellation before making provider call
        self._check_cancellation(state)

        # Execute LLM call with lifecycle instrumentation
        call_result = await execute_llm_call(
            workflow=self,
            state=state,
            phase_name="analysis",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.analysis_provider,
            model=state.analysis_model,
            temperature=0.3,  # Lower temperature for analytical tasks
            timeout=timeout,
            error_metadata={
                "source_count": len(state.sources),
                "guidance": "Try reducing max_sources_per_query or processing sources in batches",
            },
            role="research",
        )
        if isinstance(call_result, WorkflowResult):
            return call_result  # Error path
        result = call_result.result

        # Parse the response
        parsed = self._parse_analysis_response(result.content, state)

        if not parsed["success"]:
            logger.warning("Failed to parse analysis response")
            audit_data_fail: dict[str, Any] = {
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "parse_success": False,
                "findings": [],
                "gaps": [],
                "quality_updates": [],
            }
            if self.config.audit_verbosity == "full":
                audit_data_fail["system_prompt"] = system_prompt
                audit_data_fail["user_prompt"] = user_prompt
                audit_data_fail["raw_response"] = result.content
            else:
                audit_data_fail["system_prompt_length"] = len(system_prompt)
                audit_data_fail["user_prompt_length"] = len(user_prompt)
                audit_data_fail["raw_response_length"] = len(result.content)
            self._write_audit_event(
                state,
                "analysis_result",
                data=audit_data_fail,
                level="warning",
            )
            # Still mark as success but with no findings
            return WorkflowResult(
                success=True,
                content="Analysis completed but no findings extracted",
                metadata={
                    "research_id": state.id,
                    "finding_count": 0,
                    "parse_error": True,
                },
            )

        # Add findings to state
        for finding_data in parsed["findings"]:
            state.add_finding(
                content=finding_data["content"],
                confidence=finding_data["confidence"],
                source_ids=finding_data.get("source_ids", []),
                category=finding_data.get("category"),
            )

        # Add gaps to state
        for gap_data in parsed["gaps"]:
            state.add_gap(
                description=gap_data["description"],
                suggested_queries=gap_data.get("suggested_queries", []),
                priority=gap_data.get("priority", 1),
            )

        # Update source quality assessments
        for quality_update in parsed.get("quality_updates", []):
            source = state.get_source(quality_update["source_id"])
            if source:
                try:
                    source.quality = SourceQuality(quality_update["quality"])
                except ValueError:
                    pass  # Invalid quality value, skip

        # Save state
        self.memory.save_deep_research(state)
        audit_data_ok: dict[str, Any] = {
            "provider_id": result.provider_id,
            "model_used": result.model_used,
            "tokens_used": result.tokens_used,
            "duration_ms": result.duration_ms,
            "parse_success": True,
            "findings": parsed["findings"],
            "gaps": parsed["gaps"],
            "quality_updates": parsed.get("quality_updates", []),
        }
        if self.config.audit_verbosity == "full":
            audit_data_ok["system_prompt"] = system_prompt
            audit_data_ok["user_prompt"] = user_prompt
            audit_data_ok["raw_response"] = result.content
        else:
            audit_data_ok["system_prompt_length"] = len(system_prompt)
            audit_data_ok["user_prompt_length"] = len(user_prompt)
            audit_data_ok["raw_response_length"] = len(result.content)
        self._write_audit_event(
            state,
            "analysis_result",
            data=audit_data_ok,
        )

        logger.info(
            "Analysis phase complete: %d findings, %d gaps identified",
            len(parsed["findings"]),
            len(parsed["gaps"]),
        )

        finalize_phase(self, state, "analysis", phase_start_time)

        return WorkflowResult(
            success=True,
            content=f"Extracted {len(parsed['findings'])} findings and identified {len(parsed['gaps'])} gaps",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "finding_count": len(parsed["findings"]),
                "gap_count": len(parsed["gaps"]),
                "source_count": len(state.sources),
                "contradiction_count": len(state.contradictions),
                "parse_method": parsed.get("parse_method"),
            },
        )

