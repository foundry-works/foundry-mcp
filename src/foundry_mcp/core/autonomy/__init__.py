"""
Autonomous execution subsystem for Foundry MCP.

This package provides durable autonomous session management for spec execution,
including:
- Write-lock enforcement during active sessions
- Session state persistence and migration
- Step orchestration with fidelity gates
- Resume and rebase capabilities

Key modules:
- models: Pydantic models for session state
- state_migrations: Schema versioning and migrations
- write_lock: Write-lock enforcement helpers
"""

from .models import (
    # Enums
    SessionStatus,
    PauseReason,
    FailureReason,
    StepType,
    GatePolicy,
    GateVerdict,
    PhaseGateStatus,
    StepOutcome,
    # Main state model
    AutonomousSessionState,
    # Sub-models
    LastStepIssued,
    PendingGateEvidence,
    PendingManualGateAck,
    SessionCounters,
    SessionLimits,
    StopConditions,
    SessionContext,
    PhaseGateRecord,
    LastStepResult,
    NextStep,
    ResumeContext,
    RebaseResultDetail,
    SessionResponseData,
    SessionStepResponseData,
    SessionSummary,
    CompletedTaskSummary,
    CompletedPhaseSummary,
    PendingTaskSummary,
    StepInstruction,
)
from .state_migrations import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_VERSION_KEY,
    MigrationError,
    MigrationWarning,
    get_schema_version,
    set_schema_version,
    validate_state_version,
    needs_migration,
    migrate_state,
)
from .write_lock import (
    AUTONOMY_WRITE_LOCK_ACTIVE,
    WriteLockStatus,
    WriteLockResult,
    PROTECTED_TASK_ACTIONS,
    PROTECTED_LIFECYCLE_ACTIONS,
    READ_ONLY_TASK_ACTIONS,
    TERMINAL_SESSION_STATUSES,
    NON_TERMINAL_SESSION_STATUSES,
    is_protected_action,
    check_autonomy_write_lock,
    make_write_lock_error_response,
    check_and_enforce_write_lock,
)
from .spec_hash import (
    SpecFileMetadata,
    StructuralDiff,
    compute_spec_structure_hash,
    get_spec_file_metadata,
    compute_structural_diff,
)


__all__ = [
    # Models - Enums
    "SessionStatus",
    "PauseReason",
    "FailureReason",
    "StepType",
    "GatePolicy",
    "GateVerdict",
    "PhaseGateStatus",
    "StepOutcome",
    # Models - Main
    "AutonomousSessionState",
    # Models - Sub-models
    "LastStepIssued",
    "PendingGateEvidence",
    "PendingManualGateAck",
    "SessionCounters",
    "SessionLimits",
    "StopConditions",
    "SessionContext",
    "PhaseGateRecord",
    "LastStepResult",
    "NextStep",
    "ResumeContext",
    "RebaseResultDetail",
    "SessionResponseData",
    "SessionStepResponseData",
    "SessionSummary",
    "CompletedTaskSummary",
    "CompletedPhaseSummary",
    "PendingTaskSummary",
    "StepInstruction",
    # State migrations
    "CURRENT_SCHEMA_VERSION",
    "SCHEMA_VERSION_KEY",
    "MigrationError",
    "MigrationWarning",
    "get_schema_version",
    "set_schema_version",
    "validate_state_version",
    "needs_migration",
    "migrate_state",
    # Write lock module
    "AUTONOMY_WRITE_LOCK_ACTIVE",
    "WriteLockStatus",
    "WriteLockResult",
    "PROTECTED_TASK_ACTIONS",
    "PROTECTED_LIFECYCLE_ACTIONS",
    "READ_ONLY_TASK_ACTIONS",
    "TERMINAL_SESSION_STATUSES",
    "NON_TERMINAL_SESSION_STATUSES",
    "is_protected_action",
    "check_autonomy_write_lock",
    "make_write_lock_error_response",
    "check_and_enforce_write_lock",
    # Spec hash module
    "SpecFileMetadata",
    "StructuralDiff",
    "compute_spec_structure_hash",
    "get_spec_file_metadata",
    "compute_structural_diff",
]
