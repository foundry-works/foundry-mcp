"""Status persistence throttling for deep research workflow.

Manages state persistence with throttle-based write reduction to minimize
disk I/O during frequent status checks and phase transitions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig
    from foundry_mcp.core.research.memory import ResearchMemory

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)

logger = logging.getLogger(__name__)


class PersistenceMixin:
    """Mixin providing state persistence with throttle-based write reduction.

    Requires the composing class to provide:
    - self.config: ResearchConfig
    - self.memory: ResearchMemory
    - self._last_persisted_at: datetime | None
    - self._last_persisted_phase: DeepResearchPhase | None
    - self._last_persisted_iteration: int | None

    See ``DeepResearchWorkflowProtocol`` in ``phases/_protocols.py`` for
    the full structural contract.
    """

    config: ResearchConfig
    memory: ResearchMemory
    _last_persisted_at: Any
    _last_persisted_phase: DeepResearchPhase | None
    _last_persisted_iteration: int | None

    def _sync_persistence_tracking_from_state(self, state: DeepResearchState) -> None:
        """Sync persistence tracking fields from state metadata if available.

        This ensures throttling works across workflow instances by loading
        the last persisted timestamp/phase/iteration from persisted state.
        """
        if (
            self._last_persisted_at is not None
            and self._last_persisted_phase is not None
            and self._last_persisted_iteration is not None
        ):
            return

        meta = state.metadata.get("_status_persistence")
        if not isinstance(meta, dict):
            return

        # Load last persisted timestamp
        if self._last_persisted_at is None:
            raw_ts = meta.get("last_persisted_at")
            if isinstance(raw_ts, datetime):
                ts = raw_ts
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                self._last_persisted_at = ts
            elif isinstance(raw_ts, str):
                try:
                    ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    self._last_persisted_at = ts
                except ValueError:
                    pass

        # Load last persisted phase
        if self._last_persisted_phase is None:
            raw_phase = meta.get("last_persisted_phase")
            if isinstance(raw_phase, DeepResearchPhase):
                self._last_persisted_phase = raw_phase
            elif isinstance(raw_phase, str):
                try:
                    self._last_persisted_phase = DeepResearchPhase(raw_phase)
                except ValueError:
                    pass

        # Load last persisted iteration
        if self._last_persisted_iteration is None:
            raw_iter = meta.get("last_persisted_iteration")
            if isinstance(raw_iter, int):
                self._last_persisted_iteration = raw_iter

    def _is_terminal_state(self, state: DeepResearchState) -> bool:
        """Check if state represents a terminal condition (completed or failed)."""
        if state.completed_at is not None:
            return True
        if state.metadata.get("failed"):
            return True
        return False

    def _should_persist_status(self, state: DeepResearchState) -> bool:
        """Determine if state should be persisted based on throttle rules.

        Priority (highest to lowest):
        1. Terminal state (completed/failed) - always persist
        2. Phase/iteration change - always persist
        3. Throttle interval elapsed - persist if interval exceeded

        A throttle_seconds of 0 means always persist (current behavior).

        Args:
            state: Current deep research state

        Returns:
            True if state should be persisted, False to skip
        """
        # Sync persisted tracking fields from state metadata if needed
        self._sync_persistence_tracking_from_state(state)

        # Priority 1: Terminal states always persist
        if self._is_terminal_state(state):
            return True

        # Priority 2: Phase or iteration change always persists
        if self._last_persisted_phase is not None and state.phase != self._last_persisted_phase:
            return True
        if self._last_persisted_iteration is not None and state.iteration != self._last_persisted_iteration:
            return True

        # Priority 3: Check throttle interval
        throttle_seconds = self.config.status_persistence_throttle_seconds

        # 0 means always persist (backwards compatibility)
        if throttle_seconds == 0:
            return True

        # No previous persistence - should persist
        if self._last_persisted_at is None:
            return True

        # Check if throttle interval has elapsed
        elapsed = (datetime.now(timezone.utc) - self._last_persisted_at).total_seconds()
        return elapsed >= throttle_seconds

    def _persist_state(self, state: DeepResearchState) -> None:
        """Persist state and update tracking fields.

        Updates _last_persisted_at, _last_persisted_phase, and
        _last_persisted_iteration after successful save.

        Args:
            state: State to persist
        """
        now = datetime.now(timezone.utc)
        state.metadata["_status_persistence"] = {
            "last_persisted_at": now.isoformat(),
            "last_persisted_phase": state.phase.value,
            "last_persisted_iteration": state.iteration,
        }
        self.memory.save_deep_research(state)
        logger.debug(
            "Status persisted: research_id=%s phase=%s iteration=%d",
            state.id,
            state.phase.value,
            state.iteration,
        )
        self._last_persisted_at = now
        self._last_persisted_phase = state.phase
        self._last_persisted_iteration = state.iteration

    def _persist_state_if_needed(self, state: DeepResearchState) -> bool:
        """Conditionally persist state based on throttle rules.

        Args:
            state: State to potentially persist

        Returns:
            True if state was persisted, False if skipped
        """
        if self._should_persist_status(state):
            try:
                self._persist_state(state)
                return True
            except Exception as exc:
                logger.debug("Failed to persist state: %s", exc)
                return False
        logger.debug(
            "Status persistence skipped (throttled): research_id=%s phase=%s iteration=%d",
            state.id,
            state.phase.value,
            state.iteration,
        )
        return False

    def _flush_state(self, state: DeepResearchState) -> None:
        """Force-persist state, bypassing throttle rules.

        Use this for workflow completion paths (success, failure, cancellation)
        to ensure final state is always saved regardless of throttle interval.

        This guarantees:
        - Token usage/cache data is persisted
        - Final status is captured
        - Completion timestamp is saved

        Args:
            state: State to persist
        """
        self._persist_state(state)
