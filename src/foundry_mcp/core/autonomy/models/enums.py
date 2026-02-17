"""Enums for the autonomous session subsystem (ADR-002)."""

from __future__ import annotations

from enum import Enum


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


class LoopSignal(str, Enum):
    """Normalized loop outcomes for supervisor branching."""

    PHASE_COMPLETE = "phase_complete"
    SPEC_COMPLETE = "spec_complete"
    PAUSED_NEEDS_ATTENTION = "paused_needs_attention"
    FAILED = "failed"
    BLOCKED_RUNTIME = "blocked_runtime"


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
