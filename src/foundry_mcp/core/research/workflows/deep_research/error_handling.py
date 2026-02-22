"""Error recording and orchestrator transition handling for deep research.

Provides structured error capture to the persistent error store and
safe orchestrator phase transitions with exception logging.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from foundry_mcp.core.error_collection import ErrorRecord
from foundry_mcp.core.error_store import FileErrorStore
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)

logger = logging.getLogger(__name__)


class ErrorHandlingMixin:
    """Mixin providing error recording and safe orchestrator transitions.

    Requires the composing class to provide:
    - self.orchestrator: SupervisorOrchestrator
    - self.hooks: SupervisorHooks
    - self._write_audit_event(): from AuditMixin
    """

    orchestrator: Any
    hooks: Any

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...

    def _record_workflow_error(
        self,
        error: Exception,
        state: DeepResearchState,
        context: str,
    ) -> None:
        """Record error to the persistent error store.

        Args:
            error: The exception that occurred
            state: Current research state
            context: Context string (e.g., "background_task", "orchestrator")
        """
        try:
            error_store = FileErrorStore(Path.home() / ".foundry-mcp" / "errors")
            record = ErrorRecord(
                id=f"err_{uuid4().hex[:12]}",
                fingerprint=f"deep-research:{context}:{type(error).__name__}",
                error_code="WORKFLOW_ERROR",
                error_type="internal",
                tool_name=f"deep-research:{context}",
                correlation_id=state.id,
                message=str(error),
                exception_type=type(error).__name__,
                stack_trace=traceback.format_exc(),
                input_summary={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                },
            )
            error_store.append(record)
        except Exception as store_err:
            logger.error("Failed to record error to store: %s", store_err)

    def _safe_orchestrator_transition(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
    ) -> None:
        """Safely execute orchestrator phase transition with error logging.

        This wraps orchestrator calls with exception handling to ensure any
        failures are properly logged and recorded before re-raising.

        Args:
            state: Current research state
            phase: The phase that just completed

        Raises:
            Exception: Re-raises any exception after logging
        """
        try:
            self.orchestrator.evaluate_phase_completion(state, phase)
            prompt = self.orchestrator.get_reflection_prompt(state, phase)
            self.hooks.think_pause(state, prompt)
            self.orchestrator.record_to_state(state)
            state.advance_phase()
        except Exception as exc:
            logger.exception(
                "Orchestrator transition failed for phase %s, research %s: %s",
                phase.value,
                state.id,
                exc,
            )
            self._write_audit_event(
                state,
                "orchestrator_error",
                data={
                    "phase": phase.value,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                level="error",
            )
            self._record_workflow_error(exc, state, f"orchestrator_{phase.value}")
            raise  # Re-raise to be caught by workflow exception handler
