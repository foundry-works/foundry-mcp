"""
Pydantic models for autonomous session state.

This module contains all models for the autonomous execution subsystem
as defined in ADR-002. Models are designed for file-backed persistence
with schema versioning support.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Enums
# =============================================================================


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ENDED = "ended"


# Canonical set of terminal statuses — import from here instead of redefining.
TERMINAL_STATUSES = frozenset({SessionStatus.COMPLETED, SessionStatus.ENDED})


class PauseReason(str, Enum):
    """Reasons why a session is paused."""

    USER = "user"
    CONTEXT_LIMIT = "context_limit"
    ERROR_THRESHOLD = "error_threshold"
    BLOCKED = "blocked"
    GATE_FAILED = "gate_failed"
    GATE_REVIEW_REQUIRED = "gate_review_required"
    TASK_LIMIT = "task_limit"
    HEARTBEAT_STALE = "heartbeat_stale"
    STEP_STALE = "step_stale"
    PHASE_COMPLETE = "phase_complete"
    FIDELITY_CYCLE_LIMIT = "fidelity_cycle_limit"


class FailureReason(str, Enum):
    """Reasons why a session has failed."""

    SPEC_NOT_FOUND = "spec_not_found"
    SPEC_STRUCTURE_CHANGED = "spec_structure_changed"
    STATE_CORRUPT = "state_corrupt"
    MIGRATION_FAILED = "migration_failed"


class StepType(str, Enum):
    """Types of steps in autonomous execution."""

    IMPLEMENT_TASK = "implement_task"
    EXECUTE_VERIFICATION = "execute_verification"
    RUN_FIDELITY_GATE = "run_fidelity_gate"
    ADDRESS_FIDELITY_FEEDBACK = "address_fidelity_feedback"
    PAUSE = "pause"
    COMPLETE_SPEC = "complete_spec"


class GatePolicy(str, Enum):
    """Gate evaluation policies."""

    STRICT = "strict"
    LENIENT = "lenient"
    MANUAL = "manual"


class GateVerdict(str, Enum):
    """Gate evaluation verdicts."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


class PhaseGateStatus(str, Enum):
    """Status of a phase gate."""

    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WAIVED = "waived"


class StepOutcome(str, Enum):
    """Outcome of a step execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class OverrideReasonCode(str, Enum):
    """Structured reason codes for privileged override actions.

    Required for gate waivers and other privileged operations to ensure
    auditability and prevent casual bypasses.
    """

    STUCK_AGENT = "stuck_agent"
    CORRUPT_STATE = "corrupt_state"
    OPERATOR_OVERRIDE = "operator_override"
    INCIDENT_RESPONSE = "incident_response"
    TESTING = "testing"


# =============================================================================
# Sub-models
# =============================================================================


class LastStepIssued(BaseModel):
    """Record of the last step issued to the caller."""

    step_id: str = Field(..., description="ULID-format step identifier")
    type: StepType = Field(..., description="Type of step")
    task_id: Optional[str] = Field(None, description="Task ID if step involves a task")
    phase_id: Optional[str] = Field(None, description="Phase ID if step involves a phase")
    issued_at: datetime = Field(..., description="When the step was issued")
    step_proof: Optional[str] = Field(
        None, description="One-time proof token for this step (consumed on report)"
    )


class PendingGateEvidence(BaseModel):
    """Pending gate evidence awaiting consumption by next command."""

    gate_attempt_id: str = Field(..., description="Unique gate attempt identifier")
    step_id: str = Field(..., description="Step ID this evidence is bound to")
    phase_id: str = Field(..., description="Phase ID this gate is for")
    verdict: GateVerdict = Field(..., description="Gate verdict from review")
    issued_at: datetime = Field(..., description="When this evidence was created")
    integrity_checksum: Optional[str] = Field(
        None,
        description="HMAC-SHA256 integrity checksum keyed by server secret (gate_attempt_id + step_id + phase_id + verdict)",
    )


class PendingManualGateAck(BaseModel):
    """Pending manual gate acknowledgment required for resume."""

    gate_attempt_id: str = Field(..., description="Gate attempt that requires acknowledgment")
    phase_id: str = Field(..., description="Phase ID for this gate")
    issued_at: datetime = Field(..., description="When this acknowledgment was created")


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
    context_threshold_pct: int = Field(
        default=85, ge=0, le=100, description="Context usage threshold percentage"
    )
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


class StopConditions(BaseModel):
    """Configurable stop conditions for the session."""

    stop_on_phase_completion: bool = Field(
        default=False, description="Pause after phase gate passes"
    )
    auto_retry_fidelity_gate: bool = Field(
        default=True, description="Auto-retry failed gates for strict/lenient policies"
    )


class SessionContext(BaseModel):
    """Session runtime context (caller-reported)."""

    context_usage_pct: int = Field(
        default=0, ge=0, le=100, description="Caller-reported context usage percentage"
    )
    estimated_tokens_used: Optional[int] = Field(None, ge=0, description="Estimated tokens used")
    last_heartbeat_at: Optional[datetime] = Field(None, description="Last heartbeat timestamp")
    context_source: Optional[str] = Field(
        None, description="Source of context usage: sidecar, caller, or estimated"
    )
    last_context_report_at: Optional[datetime] = Field(
        None, description="When context usage was last reported"
    )
    last_context_report_pct: Optional[int] = Field(
        None, ge=0, le=100, description="Previous context usage value for monotonicity check"
    )
    consecutive_same_reports: int = Field(
        default=0, ge=0, description="Counter for consecutive identical context reports"
    )
    steps_since_last_report: int = Field(
        default=0, ge=0, description="Steps since last context usage report"
    )


class PhaseGateRecord(BaseModel):
    """Record of a phase gate evaluation."""

    required: bool = Field(default=True, description="Whether gate is required")
    status: PhaseGateStatus = Field(default=PhaseGateStatus.PENDING, description="Gate status")
    verdict: Optional[GateVerdict] = Field(None, description="Gate verdict")
    gate_attempt_id: Optional[str] = Field(None, description="Gate attempt ID")
    review_path: Optional[str] = Field(None, description="Path to review file")
    evaluated_at: Optional[datetime] = Field(None, description="When gate was evaluated")
    # Waiver metadata (set when status is WAIVED)
    waiver_reason_code: Optional[OverrideReasonCode] = Field(
        None, description="Structured reason code for gate waiver"
    )
    waiver_reason_detail: Optional[str] = Field(
        None, description="Free-text detail for waiver reason"
    )
    waived_at: Optional[datetime] = Field(None, description="When gate was waived")
    waived_by_role: Optional[str] = Field(None, description="Role that waived the gate")


class LastStepResult(BaseModel):
    """Result of a step execution reported by the caller."""

    step_id: str = Field(..., description="Step ID being reported")
    step_type: StepType = Field(..., description="Type of step")
    task_id: Optional[str] = Field(None, description="Task ID if step involved a task")
    phase_id: Optional[str] = Field(None, description="Phase ID if step involved a phase")
    outcome: StepOutcome = Field(..., description="Step outcome")
    note: Optional[str] = Field(None, description="Free-text note about outcome")
    files_touched: Optional[List[str]] = Field(None, description="Files modified during step")
    gate_attempt_id: Optional[str] = Field(None, description="Gate attempt ID for gate steps")
    step_proof: Optional[str] = Field(
        None, description="One-time proof token from the issued step (required for proof-enforced steps)"
    )
    verification_receipt: Optional[VerificationReceipt] = Field(
        None,
        description="Server-issued receipt for execute_verification steps (required when outcome='success')",
    )

    @model_validator(mode="after")
    def validate_fields_by_step_type(self) -> "LastStepResult":
        """Validate required fields per step type (ADR section 4)."""
        if self.step_type in (StepType.IMPLEMENT_TASK, StepType.EXECUTE_VERIFICATION):
            if not self.task_id:
                raise ValueError(
                    f"task_id is required for step type {self.step_type.value}"
                )
        if self.step_type == StepType.RUN_FIDELITY_GATE:
            if not self.phase_id:
                raise ValueError(
                    "phase_id is required for step type run_fidelity_gate"
                )
            if not self.gate_attempt_id:
                raise ValueError(
                    "gate_attempt_id is required for step type run_fidelity_gate"
                )
        if self.step_type == StepType.ADDRESS_FIDELITY_FEEDBACK:
            if not self.phase_id:
                raise ValueError(
                    "phase_id is required for step type address_fidelity_feedback"
                )
        return self


class StepInstruction(BaseModel):
    """A structured instruction for step execution."""

    tool: str = Field(..., description="Tool to use")
    action: str = Field(..., description="Action to perform")
    description: str = Field(..., description="Human-readable description")


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


# =============================================================================
# Step Proof Models (P1.1 - One-time step proof tokens)
# =============================================================================


class StepProofRecord(BaseModel):
    """Record of a consumed step proof for replay protection.

    Persisted atomically to enable replay-safe exactly-once semantics
    across process restarts.
    """

    step_proof: str = Field(..., description="One-time proof token")
    step_id: str = Field(..., description="Step ID this proof is bound to")
    payload_hash: str = Field(..., description="SHA-256 hash of the original request payload")
    consumed_at: datetime = Field(..., description="When this proof was consumed")
    grace_expires_at: datetime = Field(..., description="When the replay grace window expires")
    response_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the response for replay validation"
    )
    cached_response: Optional[Dict[str, Any]] = Field(
        None, description="Cached response for idempotent replay within grace window"
    )


# =============================================================================
# Verification Receipt Models (P1.2 - Proof-carrying verification receipts)
# =============================================================================


class VerificationReceipt(BaseModel):
    """Server-issued verification receipt for execute_verification steps.

    Contains cryptographic hashes of command, exit code, and output to enable
    tamper-evident verification of test execution results. The receipt is
    issued by the server when verification completes and must be included
    in last_step_result when reporting outcome='success'.
    """

    command_hash: str = Field(
        ..., description="SHA-256 hash of the verification command executed"
    )
    exit_code: int = Field(..., description="Exit code from the verification command")
    output_digest: str = Field(
        ..., description="SHA-256 hash of the combined stdout/stderr output"
    )
    issued_at: datetime = Field(..., description="When this receipt was issued")
    step_id: str = Field(..., description="Step ID this receipt is bound to")


class PendingVerificationReceipt(BaseModel):
    """Pending verification receipt data stored in session state.

    When the orchestrator issues an EXECUTE_VERIFICATION step, it stores
    the expected receipt data here. When the caller reports the step result,
    the receipt validation checks that the provided receipt matches the
    expected values.
    """

    step_id: str = Field(..., description="Step ID this pending receipt is bound to")
    task_id: str = Field(..., description="Verification task ID")
    expected_command_hash: str = Field(
        ..., description="Expected SHA-256 hash of the verification command"
    )
    issued_at: datetime = Field(..., description="When this pending receipt was created")


def issue_verification_receipt(
    step_id: str,
    command: str,
    exit_code: int,
    output: str,
) -> VerificationReceipt:
    """Issue a verification receipt for an execute_verification step.

    This function creates a tamper-evident receipt containing cryptographic
    hashes of the verification command, exit code, and output. The receipt
    should be included in last_step_result when reporting outcome='success'.

    Args:
        step_id: Step ID this receipt is bound to
        command: The verification command that was executed
        exit_code: Exit code from the verification command
        output: Combined stdout/stderr output from the command

    Returns:
        VerificationReceipt with command_hash, exit_code, output_digest

    Example:
        >>> receipt = issue_verification_receipt(
        ...     step_id="step_01HXYZ",
        ...     command="pytest tests/",
        ...     exit_code=0,
        ...     output="3 passed",
        ... )
        >>> # Include receipt in last_step_result
        >>> result.verification_receipt = receipt
    """
    import hashlib

    command_hash = hashlib.sha256(command.encode()).hexdigest()
    output_digest = hashlib.sha256(output.encode()).hexdigest()

    return VerificationReceipt(
        command_hash=command_hash,
        exit_code=exit_code,
        output_digest=output_digest,
        issued_at=datetime.now(timezone.utc),
        step_id=step_id,
    )


# =============================================================================
# Response Models
# =============================================================================


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


# =============================================================================
# Main Session State Model
# =============================================================================


class AutonomousSessionState(BaseModel):
    """
    Complete autonomous session state for persistence.

    This model represents the full state of an autonomous execution session,
    designed for file-backed persistence with schema versioning support.

    IMPORTANT: ``schema_version`` uses ``alias="_schema_version"`` so that
    persisted JSON contains the underscore-prefixed key expected by
    ``state_migrations``.  Always serialize with ``by_alias=True`` (the
    overridden ``model_dump`` defaults to this) to avoid breaking migration
    detection.
    """

    # Schema versioning (serialized as _schema_version for ADR compliance)
    schema_version: int = Field(default=3, alias="_schema_version", description="Schema version")

    # Core identifiers
    id: str = Field(..., description="ULID-format session identifier")
    spec_id: str = Field(..., description="Spec ID this session is for")
    idempotency_key: Optional[str] = Field(
        None, max_length=128, description="Client-provided idempotency key"
    )

    # Spec integrity tracking
    spec_structure_hash: str = Field(..., description="SHA-256 hash of spec structure")
    spec_file_mtime: Optional[float] = Field(None, description="Cached spec file mtime")
    spec_file_size: Optional[int] = Field(None, description="Cached spec file size")

    # Session lifecycle
    status: SessionStatus = Field(default=SessionStatus.RUNNING, description="Session status")
    created_at: datetime = Field(..., description="When session was created")
    updated_at: datetime = Field(..., description="When session was last updated")
    paused_at: Optional[datetime] = Field(None, description="When session was paused")
    pause_reason: Optional[PauseReason] = Field(None, description="Why session is paused")
    failure_reason: Optional[FailureReason] = Field(None, description="Why session failed")

    # Execution state
    active_phase_id: Optional[str] = Field(None, description="Currently active phase")
    last_task_id: Optional[str] = Field(None, description="Last task that was worked on")
    last_step_issued: Optional[LastStepIssued] = Field(None, description="Last step issued to caller")
    last_issued_response: Optional[Dict[str, Any]] = Field(
        None, description="Cached last response for replay-safe exactly-once semantics"
    )

    # Pending state
    pending_gate_evidence: Optional[PendingGateEvidence] = Field(
        None, description="Pending gate evidence awaiting consumption"
    )
    pending_manual_gate_ack: Optional[PendingManualGateAck] = Field(
        None, description="Pending manual gate acknowledgment"
    )
    pending_verification_receipt: Optional[PendingVerificationReceipt] = Field(
        None, description="Pending verification receipt for execute_verification steps"
    )

    # Counters and limits
    counters: SessionCounters = Field(default_factory=SessionCounters, description="Session counters")
    limits: SessionLimits = Field(default_factory=SessionLimits, description="Session limits")
    stop_conditions: StopConditions = Field(
        default_factory=StopConditions, description="Stop conditions"
    )

    # Configuration
    write_lock_enforced: bool = Field(default=True, description="Whether write lock is enforced")
    gate_policy: GatePolicy = Field(default=GatePolicy.STRICT, description="Gate evaluation policy")

    # Runtime context
    context: SessionContext = Field(default_factory=SessionContext, description="Runtime context")

    # Phase gates (phase_id -> PhaseGateRecord)
    phase_gates: Dict[str, PhaseGateRecord] = Field(
        default_factory=dict, description="Phase gate records"
    )

    # Required gate obligations and satisfaction tracking (v3 schema)
    # Maps phase_id -> list of required gate types (e.g., ["fidelity"])
    required_phase_gates: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Required gate types per phase (phase_id -> gate_types)",
    )
    # Maps phase_id -> list of satisfied gate types (passed or waived)
    satisfied_gates: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Satisfied gate types per phase (phase_id -> gate_types)",
    )

    # Completed tasks
    completed_task_ids: List[str] = Field(
        default_factory=list, description="IDs of completed tasks"
    )

    # State version for observability
    state_version: int = Field(default=1, ge=1, description="Monotonic state version")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate idempotency key format: alphanumeric, hyphens, underscores only."""
        if v is None:
            return v
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "idempotency_key must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    model_config = {
        "populate_by_name": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None,
        },
    }

    def model_dump(self, *, by_alias: bool = True, **kwargs: Any) -> Dict[str, Any]:
        """Override to default ``by_alias=True`` for correct _schema_version serialization."""
        return super().model_dump(by_alias=by_alias, **kwargs)


# =============================================================================
# Session Summary for List Operations
# =============================================================================


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


# =============================================================================
# Shared Utility Functions
# =============================================================================


def compute_effective_status(session: AutonomousSessionState) -> Optional[SessionStatus]:
    """Compute effective status considering staleness.

    If the session is nominally RUNNING but heartbeat or step timestamps
    indicate staleness, the effective status is PAUSED.

    Args:
        session: Session state

    Returns:
        Derived status (PAUSED) if stale, None if the actual status applies.
    """
    if session.status != SessionStatus.RUNNING:
        return None

    now = datetime.now(timezone.utc)

    # Check step staleness
    if session.last_step_issued:
        step_stale_threshold = timedelta(minutes=session.limits.step_stale_minutes)
        if now - session.last_step_issued.issued_at > step_stale_threshold:
            return SessionStatus.PAUSED

    # Check heartbeat staleness
    if session.context.last_heartbeat_at:
        heartbeat_stale_threshold = timedelta(minutes=session.limits.heartbeat_stale_minutes)
        if now - session.context.last_heartbeat_at > heartbeat_stale_threshold:
            return SessionStatus.PAUSED
    else:
        # Pre-first-heartbeat: use heartbeat_grace_minutes from created_at
        grace_threshold = timedelta(minutes=session.limits.heartbeat_grace_minutes)
        if now - session.created_at > grace_threshold:
            return SessionStatus.PAUSED

    return None
