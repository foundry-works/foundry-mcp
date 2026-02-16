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
- orchestrator: 18-step orchestration engine
- state_migrations: Schema versioning and migrations
- write_lock: Write-lock enforcement helpers
"""

from .models import (
    # Constants
    TERMINAL_STATUSES,
    # Enums
    SessionStatus,
    PauseReason,
    FailureReason,
    StepType,
    GatePolicy,
    GateVerdict,
    PhaseGateStatus,
    StepOutcome,
    LoopSignal,
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
    RecommendedAction,
    derive_loop_signal,
    derive_recommended_actions,
)
from .orchestrator import (
    StepOrchestrator,
    OrchestrationResult,
    ERROR_STEP_RESULT_REQUIRED,
    ERROR_STEP_MISMATCH,
    ERROR_INVALID_GATE_EVIDENCE,
    ERROR_NO_ACTIVE_SESSION,
    ERROR_AMBIGUOUS_ACTIVE_SESSION,
    ERROR_SESSION_UNRECOVERABLE,
    ERROR_SPEC_REBASE_REQUIRED,
    ERROR_HEARTBEAT_STALE,
    ERROR_STEP_STALE,
    ERROR_ALL_TASKS_BLOCKED,
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
from .context_tracker import (
    ContextTracker,
    is_sandbox_mode,
)
from .memory import (
    AutonomyStorage,
)
from .spec_hash import (
    SpecFileMetadata,
    StructuralDiff,
    compute_spec_structure_hash,
    get_spec_file_metadata,
    compute_structural_diff,
)
from .audit import (
    AuditEventType,
    AuditEvent,
    AuditLedger,
    VerificationResult,
    append_event,
    verify_chain,
    get_ledger_path,
    GENESIS_HASH,
)


__all__ = [
    # Constants
    "TERMINAL_STATUSES",
    # Models - Enums
    "SessionStatus",
    "PauseReason",
    "FailureReason",
    "StepType",
    "GatePolicy",
    "GateVerdict",
    "PhaseGateStatus",
    "StepOutcome",
    "LoopSignal",
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
    "RecommendedAction",
    "derive_loop_signal",
    "derive_recommended_actions",
    # Orchestrator
    "StepOrchestrator",
    "OrchestrationResult",
    "ERROR_STEP_RESULT_REQUIRED",
    "ERROR_STEP_MISMATCH",
    "ERROR_INVALID_GATE_EVIDENCE",
    "ERROR_NO_ACTIVE_SESSION",
    "ERROR_AMBIGUOUS_ACTIVE_SESSION",
    "ERROR_SESSION_UNRECOVERABLE",
    "ERROR_SPEC_REBASE_REQUIRED",
    "ERROR_HEARTBEAT_STALE",
    "ERROR_STEP_STALE",
    "ERROR_ALL_TASKS_BLOCKED",
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
    # Context tracker
    "ContextTracker",
    "is_sandbox_mode",
    # Storage
    "AutonomyStorage",
    # Spec hash module
    "SpecFileMetadata",
    "StructuralDiff",
    "compute_spec_structure_hash",
    "get_spec_file_metadata",
    "compute_structural_diff",
    # Audit module
    "AuditEventType",
    "AuditEvent",
    "AuditLedger",
    "VerificationResult",
    "append_event",
    "verify_chain",
    "get_ledger_path",
    "GENESIS_HASH",
]
