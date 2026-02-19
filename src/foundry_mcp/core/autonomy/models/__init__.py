"""Autonomy models sub-package.

Re-exports all public symbols for backward-compatible imports via
``from foundry_mcp.core.autonomy.models import <Symbol>``.
"""

from .enums import (
    TERMINAL_STATUSES,
    FailureReason,
    GatePolicy,
    GateVerdict,
    LoopSignal,
    OverrideReasonCode,
    PauseReason,
    PhaseGateStatus,
    SessionStatus,
    StepOutcome,
    StepType,
)
from .gates import (
    PendingGateEvidence,
    PendingManualGateAck,
    PhaseGateRecord,
)
from .responses import (
    ActivePhaseProgress,
    CompletedPhaseSummary,
    CompletedTaskSummary,
    NextStep,
    PendingTaskSummary,
    RebaseResultDetail,
    RecommendedAction,
    ResumeContext,
    RetryCounters,
    SessionResponseData,
    SessionStepResponseData,
    SessionSummary,
)
from .session_config import (
    SessionContext,
    SessionCounters,
    SessionLimits,
    StopConditions,
)
from .signals import (
    _BLOCKED_RUNTIME_ERROR_CODES,
    _PAUSED_NEEDS_ATTENTION_REASONS,
    _normalize_signal_value,
    compute_effective_status,
    derive_loop_signal,
    derive_recommended_actions,
)
from .state import (
    AutonomousSessionState,
)
from .steps import (
    LastStepIssued,
    LastStepResult,
    StepInstruction,
    StepProofRecord,
)
from .verification import (
    PendingVerificationReceipt,
    VerificationReceipt,
    issue_verification_receipt,
)

__all__ = [
    # Enums
    "FailureReason",
    "GatePolicy",
    "GateVerdict",
    "LoopSignal",
    "OverrideReasonCode",
    "PauseReason",
    "PhaseGateStatus",
    "SessionStatus",
    "StepOutcome",
    "StepType",
    "TERMINAL_STATUSES",
    # Gates
    "PendingGateEvidence",
    "PendingManualGateAck",
    "PhaseGateRecord",
    # Verification
    "PendingVerificationReceipt",
    "VerificationReceipt",
    "issue_verification_receipt",
    # Steps
    "LastStepIssued",
    "LastStepResult",
    "StepInstruction",
    "StepProofRecord",
    # Session config
    "SessionContext",
    "SessionCounters",
    "SessionLimits",
    "StopConditions",
    # Response models
    "ActivePhaseProgress",
    "CompletedPhaseSummary",
    "CompletedTaskSummary",
    "NextStep",
    "PendingTaskSummary",
    "RebaseResultDetail",
    "RecommendedAction",
    "ResumeContext",
    "RetryCounters",
    "SessionResponseData",
    "SessionStepResponseData",
    "SessionSummary",
    # State
    "AutonomousSessionState",
    # Signals / utilities
    "compute_effective_status",
    "derive_loop_signal",
    "derive_recommended_actions",
    "_BLOCKED_RUNTIME_ERROR_CODES",
    "_PAUSED_NEEDS_ATTENTION_REASONS",
    "_normalize_signal_value",
]
