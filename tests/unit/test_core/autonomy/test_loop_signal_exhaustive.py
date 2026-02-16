"""T1: Parametrized loop signal mapping exhaustiveness test.

Covers every branch of derive_loop_signal() with one row per mapping.
"""

import pytest

from foundry_mcp.core.autonomy.models import (
    derive_loop_signal,
    LoopSignal,
    SessionStatus,
    PauseReason,
    _BLOCKED_RUNTIME_ERROR_CODES,
    _PAUSED_NEEDS_ATTENTION_REASONS,
)


# Build parametrize rows for all _BLOCKED_RUNTIME_ERROR_CODES
_BLOCKED_ROWS = [
    (f"blocked_{code.lower()}", dict(error_code=code), LoopSignal.BLOCKED_RUNTIME)
    for code in sorted(_BLOCKED_RUNTIME_ERROR_CODES)
]

# Build parametrize rows for all _PAUSED_NEEDS_ATTENTION_REASONS
_PAUSED_ROWS = [
    (f"paused_{reason}", dict(pause_reason=reason), LoopSignal.PAUSED_NEEDS_ATTENTION)
    for reason in sorted(_PAUSED_NEEDS_ATTENTION_REASONS)
]


@pytest.mark.parametrize(
    "test_id,kwargs,expected",
    [
        # --- PHASE_COMPLETE ---
        (
            "phase_complete",
            dict(pause_reason=PauseReason.PHASE_COMPLETE),
            LoopSignal.PHASE_COMPLETE,
        ),
        # --- SPEC_COMPLETE ---
        (
            "spec_complete_status",
            dict(status=SessionStatus.COMPLETED),
            LoopSignal.SPEC_COMPLETE,
        ),
        (
            "spec_complete_pause_reason",
            dict(pause_reason="spec_complete"),
            LoopSignal.SPEC_COMPLETE,
        ),
        # --- BLOCKED_RUNTIME (from error codes) ---
        *_BLOCKED_ROWS,
        # --- BLOCKED_RUNTIME (from repeated invalid gate evidence) ---
        (
            "blocked_invalid_gate_evidence_repeated",
            dict(error_code="INVALID_GATE_EVIDENCE", repeated_invalid_gate_evidence=True),
            LoopSignal.BLOCKED_RUNTIME,
        ),
        (
            "blocked_error_invalid_gate_evidence_repeated",
            dict(error_code="ERROR_INVALID_GATE_EVIDENCE", repeated_invalid_gate_evidence=True),
            LoopSignal.BLOCKED_RUNTIME,
        ),
        # --- FAILED ---
        (
            "failed_status",
            dict(status=SessionStatus.FAILED),
            LoopSignal.FAILED,
        ),
        (
            "failed_unrecoverable",
            dict(is_unrecoverable_error=True),
            LoopSignal.FAILED,
        ),
        (
            "failed_session_unrecoverable",
            dict(error_code="SESSION_UNRECOVERABLE"),
            LoopSignal.FAILED,
        ),
        (
            "failed_error_session_unrecoverable",
            dict(error_code="ERROR_SESSION_UNRECOVERABLE"),
            LoopSignal.FAILED,
        ),
        # --- PAUSED_NEEDS_ATTENTION (from pause reasons) ---
        *_PAUSED_ROWS,
        # --- None / default ---
        (
            "running_no_signal",
            dict(status=SessionStatus.RUNNING),
            None,
        ),
        (
            "user_pause_no_signal",
            dict(pause_reason=PauseReason.USER),
            None,
        ),
        (
            "invalid_gate_evidence_not_repeated",
            dict(error_code="INVALID_GATE_EVIDENCE", repeated_invalid_gate_evidence=False),
            None,
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_derive_loop_signal_exhaustive(test_id, kwargs, expected):
    """Each mapping table entry produces the correct loop_signal."""
    assert derive_loop_signal(**kwargs) == expected
