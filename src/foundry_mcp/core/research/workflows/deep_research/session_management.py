"""Session management mixin for DeepResearchWorkflow.

Handles listing, deleting, resuming, and validating deep research sessions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.workflows.base import WorkflowResult

logger = logging.getLogger(__name__)


class SessionManagementMixin:
    """Session management methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - memory (inherited from ResearchWorkflowBase)
    - _execute_workflow_async() (orchestration loop on core)
    """

    memory: Any

    if TYPE_CHECKING:

        async def _execute_workflow_async(self, *args: Any, **kwargs: Any) -> Any: ...

    def list_sessions(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
        completed_only: bool = False,
    ) -> list[dict[str, Any]]:
        """List deep research sessions.

        Args:
            limit: Maximum sessions to return
            cursor: Pagination cursor (research_id to start after)
            completed_only: Only return completed sessions

        Returns:
            List of session summaries
        """
        sessions = self.memory.list_deep_research(
            limit=limit,
            cursor=cursor,
            completed_only=completed_only,
        )

        return [
            {
                "id": s.id,
                "query": s.original_query,
                "phase": s.phase.value,
                "iteration": s.iteration,
                "source_count": len(s.sources),
                "finding_count": len(s.findings),
                "is_complete": s.completed_at is not None,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]

    def delete_session(self, research_id: str) -> bool:
        """Delete a research session.

        Args:
            research_id: ID of session to delete

        Returns:
            True if deleted, False if not found
        """
        return self.memory.delete_deep_research(research_id)

    def resume_research(
        self,
        research_id: str,
        provider_id: Optional[str] = None,
        timeout_per_operation: float = 120.0,
        max_concurrent: int = 3,
    ) -> WorkflowResult:
        """Resume an interrupted deep research workflow from persisted state.

        Loads the DeepResearchState from persistence, validates it, and resumes
        execution from the current phase. Handles edge cases like corrupted
        state or missing sources gracefully.

        Args:
            research_id: ID of the research session to resume
            provider_id: Optional provider override for LLM operations
            timeout_per_operation: Timeout per operation in seconds
            max_concurrent: Maximum concurrent operations

        Returns:
            WorkflowResult with resumed research outcome or error
        """
        logger.info("Attempting to resume research session: %s", research_id)

        # Load existing state
        state = self.memory.load_deep_research(research_id)

        if state is None:
            logger.warning("Research session '%s' not found in persistence", research_id)
            return WorkflowResult(
                success=False,
                content="",
                error=f"Research session '{research_id}' not found. It may have expired or been deleted.",
                metadata={"research_id": research_id, "error_type": "not_found"},
            )

        # Check if already completed
        if state.completed_at is not None:
            logger.info(
                "Research session '%s' already completed at %s",
                research_id,
                state.completed_at.isoformat(),
            )
            return WorkflowResult(
                success=True,
                content=state.report or "Research already completed",
                metadata={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "is_complete": True,
                    "completed_at": state.completed_at.isoformat(),
                    "resumed": False,
                },
            )

        # Validate state integrity
        validation_result = self._validate_state_for_resume(state)
        if not validation_result["valid"]:
            logger.error(
                "Research session '%s' failed validation: %s",
                research_id,
                validation_result["error"],
            )
            return WorkflowResult(
                success=False,
                content="",
                error=validation_result["error"],
                metadata={
                    "research_id": research_id,
                    "error_type": "validation_failed",
                    "phase": state.phase.value,
                    "issues": validation_result.get("issues", []),
                },
            )

        # Log resumption context
        logger.info(
            "Resuming research '%s': phase=%s, iteration=%d/%d, "
            "sub_queries=%d (completed=%d), sources=%d, findings=%d, gaps=%d",
            research_id,
            state.phase.value,
            state.iteration,
            state.max_iterations,
            len(state.sub_queries),
            len(state.completed_sub_queries()),
            len(state.sources),
            len(state.findings),
            len(state.unresolved_gaps()),
        )

        # Resume workflow execution
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._execute_workflow_async(
                            state=state,
                            provider_id=provider_id,
                            timeout_per_operation=timeout_per_operation,
                            max_concurrent=max_concurrent,
                        ),
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    self._execute_workflow_async(
                        state=state,
                        provider_id=provider_id,
                        timeout_per_operation=timeout_per_operation,
                        max_concurrent=max_concurrent,
                    )
                )
        except RuntimeError:
            result = asyncio.run(
                self._execute_workflow_async(
                    state=state,
                    provider_id=provider_id,
                    timeout_per_operation=timeout_per_operation,
                    max_concurrent=max_concurrent,
                )
            )

        # Add resumption metadata
        if result.metadata is None:
            result.metadata = {}
        result.metadata["resumed"] = True
        result.metadata["resumed_from_phase"] = state.phase.value

        return result

    def _validate_state_for_resume(self, state: DeepResearchState) -> dict[str, Any]:
        """Validate a DeepResearchState for safe resumption.

        Checks for common corruption issues and missing required data.

        Args:
            state: The state to validate

        Returns:
            Dict with 'valid' bool and 'error'/'issues' if invalid
        """
        issues = []

        # Check required fields
        if not state.original_query:
            issues.append("Missing original_query")

        if not state.id:
            issues.append("Missing research ID")

        # Phase-specific validation
        if state.phase.value in ("gathering", "analysis", "synthesis", "refinement"):
            # These phases require sub-queries from planning
            if not state.sub_queries:
                issues.append(f"No sub-queries found for {state.phase.value} phase")

        if state.phase.value in ("analysis", "synthesis"):
            # These phases require sources from gathering
            if not state.sources and state.phase.value == "analysis":
                # Only warn for analysis - synthesis can work with findings
                issues.append("No sources found for analysis phase")

        if state.phase.value == "synthesis":
            # Synthesis requires findings from analysis
            if not state.findings:
                issues.append("No findings found for synthesis phase")

        # Note: Pydantic's default_factory=list guarantees collections are never None,
        # so explicit None checks are unnecessary. Corrupted data would fail Pydantic
        # validation during deserialization.

        if issues:
            return {
                "valid": False,
                "error": f"State validation failed: {'; '.join(issues)}",
                "issues": issues,
            }

        return {"valid": True}

    def list_resumable_sessions(self) -> list[dict[str, Any]]:
        """List all in-progress research sessions that can be resumed.

        Scans persistence for sessions that are not completed and can be resumed.

        Returns:
            List of session summaries with resumption context
        """
        sessions = self.memory.list_deep_research(completed_only=False)

        resumable = []
        for state in sessions:
            if state.completed_at is not None:
                continue  # Skip completed

            validation = self._validate_state_for_resume(state)

            resumable.append(
                {
                    "id": state.id,
                    "query": state.original_query[:100] + ("..." if len(state.original_query) > 100 else ""),
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                    "max_iterations": state.max_iterations,
                    "sub_queries": len(state.sub_queries),
                    "completed_queries": len(state.completed_sub_queries()),
                    "sources": len(state.sources),
                    "findings": len(state.findings),
                    "gaps": len(state.unresolved_gaps()),
                    "can_resume": validation["valid"],
                    "issues": validation.get("issues", []),
                    "created_at": state.created_at.isoformat(),
                    "updated_at": state.updated_at.isoformat(),
                }
            )

        return resumable
