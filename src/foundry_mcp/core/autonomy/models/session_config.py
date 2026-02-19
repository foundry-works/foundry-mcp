"""Session configuration models for autonomous execution."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class SessionCounters(BaseModel):
    """Session execution counters."""

    tasks_completed: int = Field(default=0, ge=0, description="Number of completed tasks")
    consecutive_errors: int = Field(default=0, ge=0, description="Consecutive error count")
    fidelity_review_cycles_in_active_phase: int = Field(
        default=0, ge=0, description="Fidelity review cycles in current phase"
    )


class SessionLimits(BaseModel):
    """Session execution limits and thresholds."""

    max_tasks_per_session: int = Field(default=100, gt=0, description="Max tasks per session")
    max_consecutive_errors: int = Field(default=3, gt=0, description="Max consecutive errors")
    context_threshold_pct: int = Field(default=85, ge=0, le=100, description="Context usage threshold percentage")
    heartbeat_stale_minutes: int = Field(default=10, gt=0, description="Heartbeat staleness threshold")
    heartbeat_grace_minutes: int = Field(default=5, gt=0, description="Initial heartbeat grace window")
    step_stale_minutes: int = Field(default=60, gt=0, description="Step staleness threshold")
    max_fidelity_review_cycles_per_phase: int = Field(
        default=3, ge=1, description="Max fidelity review cycles per phase"
    )
    avg_pct_per_step: int = Field(
        default=3, ge=1, le=20, description="Estimated context growth per step for Tier 3 estimation"
    )
    context_staleness_threshold: int = Field(
        default=5, ge=1, description="Consecutive identical context reports before staleness penalty"
    )
    context_staleness_penalty_pct: int = Field(
        default=5, ge=1, le=20, description="Penalty percentage added when context reports are stale"
    )
    context_reset_threshold_pct: int = Field(
        default=10, ge=0, le=50, description="Context usage below this threshold is treated as a possible /clear reset"
    )
    sidecar_max_age_seconds: int = Field(
        default=120, ge=10, le=600, description="Maximum age in seconds for a sidecar file to be considered fresh"
    )

    @model_validator(mode="after")
    def validate_heartbeat_ordering(self) -> "SessionLimits":
        """Assert heartbeat_grace_minutes < heartbeat_stale_minutes."""
        if self.heartbeat_grace_minutes >= self.heartbeat_stale_minutes:
            raise ValueError(
                f"heartbeat_grace_minutes ({self.heartbeat_grace_minutes}) must be "
                f"less than heartbeat_stale_minutes ({self.heartbeat_stale_minutes})"
            )
        return self


class StopConditions(BaseModel):
    """Configurable stop conditions for the session."""

    stop_on_phase_completion: bool = Field(default=False, description="Pause after phase gate passes")
    auto_retry_fidelity_gate: bool = Field(
        default=True, description="Auto-retry failed gates for strict/lenient policies"
    )


class SessionContext(BaseModel):
    """Session runtime context (caller-reported)."""

    context_usage_pct: int = Field(default=0, ge=0, le=100, description="Caller-reported context usage percentage")
    estimated_tokens_used: Optional[int] = Field(default=None, ge=0, description="Estimated tokens used")
    last_heartbeat_at: Optional[datetime] = Field(default=None, description="Last heartbeat timestamp")
    context_source: Optional[str] = Field(
        default=None, description="Source of context usage: sidecar, caller, or estimated"
    )
    last_context_report_at: Optional[datetime] = Field(default=None, description="When context usage was last reported")
    last_context_report_pct: Optional[int] = Field(
        default=None, ge=0, le=100, description="Previous context usage value for monotonicity check"
    )
    consecutive_same_reports: int = Field(
        default=0, ge=0, description="Counter for consecutive identical context reports"
    )
    steps_since_last_report: int = Field(default=0, ge=0, description="Steps since last context usage report")
