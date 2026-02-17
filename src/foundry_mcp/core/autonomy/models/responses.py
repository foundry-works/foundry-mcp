"""Response and summary models for autonomous sessions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .enums import (
    LoopSignal,
    PauseReason,
    PhaseGateStatus,
    SessionStatus,
    StepType,
)
from .session_config import SessionCounters, SessionLimits, StopConditions
from .steps import StepInstruction


class CompletedTaskSummary(BaseModel):
    """Summary of a completed task for resume context."""

    task_id: str = Field(..., description="Task ID")
    title: str = Field(..., description="Task title")
    phase_id: str = Field(..., description="Phase ID")
    files_touched: Optional[List[str]] = Field(None, description="Files touched")


class CompletedPhaseSummary(BaseModel):
    """Summary of a completed phase for resume context."""

    phase_id: str = Field(..., description="Phase ID")
    title: Optional[str] = Field(None, description="Phase title")
    gate_status: PhaseGateStatus = Field(..., description="Gate status")


class PendingTaskSummary(BaseModel):
    """Summary of a pending task for resume context."""

    task_id: str = Field(..., description="Task ID")
    title: str = Field(..., description="Task title")


class RecommendedAction(BaseModel):
    """Machine-readable action recommendation for escalations."""

    action: str = Field(..., description="Recommended operation identifier")
    description: str = Field(..., description="Human-readable remediation guidance")
    command: Optional[str] = Field(None, description="Canonical task command to run")


class NextStep(BaseModel):
    """Next step to execute in autonomous session."""

    step_id: str = Field(..., description="ULID-format step identifier")
    type: StepType = Field(..., description="Type of step")
    task_id: Optional[str] = Field(None, description="Task ID for implement_task")
    phase_id: Optional[str] = Field(None, description="Phase ID")
    task_title: Optional[str] = Field(None, description="Task title for implement_task")
    gate_attempt_id: Optional[str] = Field(None, description="Gate attempt ID for gate steps")
    instructions: Optional[List[StepInstruction]] = Field(
        None, description="Structured instructions for the step"
    )
    reason: Optional[PauseReason] = Field(None, description="Pause reason for pause steps")
    message: Optional[str] = Field(None, description="Human-readable message")
    step_proof: Optional[str] = Field(
        None, description="One-time proof token for this step (must match in report)"
    )


class ResumeContext(BaseModel):
    """Context for resuming a paused session."""

    spec_id: str = Field(..., description="Spec ID")
    spec_title: Optional[str] = Field(None, description="Spec title")
    active_phase_id: Optional[str] = Field(None, description="Active phase ID")
    active_phase_title: Optional[str] = Field(None, description="Active phase title")
    completed_task_count: int = Field(default=0, ge=0, description="Total completed tasks")
    recent_completed_tasks: List[CompletedTaskSummary] = Field(
        default_factory=list,
        max_length=10,
        description="Recently completed tasks (capped at 10)",
    )
    completed_phases: List[CompletedPhaseSummary] = Field(
        default_factory=list, description="Completed phases"
    )
    pending_tasks_in_phase: List[PendingTaskSummary] = Field(
        default_factory=list, description="Pending tasks in active phase"
    )
    last_pause_reason: Optional[PauseReason] = Field(None, description="Why session was paused")
    journal_available: bool = Field(default=False, description="Whether journal entries exist")
    journal_hint: Optional[str] = Field(None, description="Hint for accessing journal")


class RebaseResultDetail(BaseModel):
    """Result of a rebase operation."""

    result: str = Field(..., description="Rebase result: no_change, success, or completed_tasks_removed")
    added_phases: List[str] = Field(default_factory=list, description="Added phase IDs")
    removed_phases: List[str] = Field(default_factory=list, description="Removed phase IDs")
    added_tasks: List[str] = Field(default_factory=list, description="Added task IDs")
    removed_tasks: List[str] = Field(default_factory=list, description="Removed task IDs")
    completed_tasks_removed: Optional[int] = Field(None, description="Number of completed tasks removed when force=true")


class ActivePhaseProgress(BaseModel):
    """Progress summary for the currently active phase."""

    phase_id: str = Field(..., description="Active phase ID")
    phase_title: Optional[str] = Field(None, description="Active phase title")
    total_tasks: int = Field(default=0, ge=0, description="Total tasks in active phase")
    completed_tasks: int = Field(default=0, ge=0, description="Completed tasks in active phase")
    blocked_tasks: int = Field(default=0, ge=0, description="Blocked tasks in active phase")
    remaining_tasks: int = Field(default=0, ge=0, description="Remaining tasks in active phase")
    completion_pct: int = Field(default=0, ge=0, le=100, description="Completion percentage (0-100)")


class RetryCounters(BaseModel):
    """Retry and error counters for operator polling."""

    consecutive_errors: int = Field(default=0, ge=0, description="Current consecutive error count")
    fidelity_review_cycles_in_active_phase: int = Field(
        default=0,
        ge=0,
        description="Fidelity review cycles in the current active phase",
    )
    phase_retry_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Phase-scoped retry counters keyed by phase_id",
    )
    task_retry_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Task-scoped retry counters keyed by task_id (when available)",
    )


class SessionResponseData(BaseModel):
    """Response data for session lifecycle commands."""

    session_id: str = Field(..., description="Session ID")
    spec_id: str = Field(..., description="Spec ID")
    status: SessionStatus = Field(..., description="Session status")
    pause_reason: Optional[PauseReason] = Field(None, description="Why session is paused")
    counters: SessionCounters = Field(..., description="Session counters")
    limits: SessionLimits = Field(..., description="Session limits")
    stop_conditions: StopConditions = Field(..., description="Stop conditions")
    write_lock_enforced: bool = Field(..., description="Whether write lock is enforced")
    active_phase_id: Optional[str] = Field(None, description="Active phase ID")
    last_heartbeat_at: Optional[datetime] = Field(None, description="Last heartbeat timestamp")
    next_action_hint: Optional[str] = Field(None, description="Hint for next action")
    last_step_id: Optional[str] = Field(None, description="Most recently issued step ID")
    last_step_type: Optional[StepType] = Field(None, description="Most recently issued step type")
    current_task_id: Optional[str] = Field(
        None,
        description="Current task ID inferred from last issued step or session state",
    )
    active_phase_progress: Optional[ActivePhaseProgress] = Field(
        None,
        description="Progress summary for the current active phase",
    )
    retry_counters: Optional[RetryCounters] = Field(
        None,
        description="Retry/error counters for active phase and tasks",
    )
    session_signal: Optional[LoopSignal] = Field(
        None, description="Derived loop summary for quick supervisor polling"
    )
    resume_context: Optional[ResumeContext] = Field(None, description="Resume context if applicable")
    rebase_result: Optional[RebaseResultDetail] = Field(None, description="Rebase result if applicable")


class SessionStepResponseData(BaseModel):
    """Response data for session-step commands."""

    session_id: str = Field(..., description="Session ID")
    status: SessionStatus = Field(..., description="Session status")
    state_version: int = Field(..., description="Monotonic state version")
    next_step: Optional[NextStep] = Field(None, description="Next step to execute")
    # Gate invariant observability fields
    required_phase_gates: Optional[List[str]] = Field(
        None, description="IDs of required phase gates for this session"
    )
    satisfied_gates: Optional[List[str]] = Field(
        None, description="IDs of gates that are passed or waived"
    )
    missing_required_gates: Optional[List[str]] = Field(
        None, description="IDs of required gates that are pending or failed"
    )
    loop_signal: Optional[LoopSignal] = Field(
        None,
        description="Normalized loop outcome used by supervisors to decide continue/stop/escalate",
    )
    recommended_actions: Optional[List[RecommendedAction]] = Field(
        None, description="Actionable escalation guidance when loop_signal is non-success"
    )


class SessionSummary(BaseModel):
    """Summary of a session for list operations."""

    session_id: str = Field(..., description="Session ID")
    spec_id: str = Field(..., description="Spec ID")
    status: SessionStatus = Field(..., description="Session status")
    effective_status: Optional[SessionStatus] = Field(
        None, description="Effective status (derived for stale sessions)"
    )
    pause_reason: Optional[PauseReason] = Field(None, description="Why session is paused")
    stale_reason: Optional[str] = Field(None, description="Staleness reason if applicable")
    stale_detected_at: Optional[datetime] = Field(None, description="When staleness was detected")
    created_at: datetime = Field(..., description="When session was created")
    updated_at: datetime = Field(..., description="When session was last updated")
    active_phase_id: Optional[str] = Field(None, description="Active phase ID")
    tasks_completed: int = Field(default=0, description="Number of completed tasks")
