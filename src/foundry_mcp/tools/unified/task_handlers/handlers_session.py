"""Session handler re-export shim.

The actual implementations live in focused submodules:
- handlers_session_lifecycle: start, pause, resume, end, reset
- handlers_session_query: status, list, events, heartbeat
- handlers_session_rebase: rebase, gate_waiver
- _session_common: shared helpers, constants, utilities

This module re-exports all public symbols so that existing imports
from ``handlers_session`` continue to work without modification.
"""

# --- Lifecycle handlers ---
# --- Authorization imports (kept here so tests can mock-patch them on this module) ---
from foundry_mcp.core.authorization import Role, check_action_allowed, get_server_role  # noqa: F401

# --- Shared helpers (re-exported for any code that imports them from here) ---
from foundry_mcp.tools.unified.task_handlers._session_common import (  # noqa: F401
    ERROR_IDEMPOTENCY_MISMATCH,
    ERROR_INVALID_GATE_ACK,
    ERROR_INVALID_STATE_TRANSITION,
    ERROR_MANUAL_GATE_ACK_REQUIRED,
    ERROR_SESSION_ALREADY_EXISTS,
    # Constants
    ERROR_SESSION_NOT_FOUND,
    ERROR_SPEC_NOT_FOUND,
    ERROR_SPEC_STRUCTURE_CHANGED,
    _build_active_phase_progress,
    _build_resume_context,
    _build_retry_counters,
    _build_session_response,
    _compute_effective_status,
    _compute_required_gates_from_spec,
    _inject_audit_status,
    _invalid_transition_response,
    _load_spec_for_session,
    _save_with_version_check,
    _write_session_journal,
)

# --- Lifecycle-specific helpers ---
from foundry_mcp.tools.unified.task_handlers.handlers_session_lifecycle import (  # noqa: F401  # noqa: F401
    _handle_existing_session,
    _handle_session_end,
    _handle_session_pause,
    _handle_session_reset,
    _handle_session_resume,
    _handle_session_start,
    _load_spec_for_session_start,
    _resolve_session_config,
    _validate_posture_constraints,
    _verify_audit_chain_for_start,
)

# --- Query handlers ---
# --- Query-specific constants ---
from foundry_mcp.tools.unified.task_handlers.handlers_session_query import (  # noqa: F401  # noqa: F401
    SESSION_EVENTS_CURSOR_KIND,
    SESSION_EVENTS_DEFAULT_LIMIT,
    SESSION_EVENTS_MAX_LIMIT,
    _collect_session_events,
    _decode_session_events_cursor,
    _encode_session_events_cursor,
    _handle_session_events,
    _handle_session_heartbeat,
    _handle_session_list,
    _handle_session_status,
    _parse_journal_timestamp,
)

# --- Rebase / gate waiver handlers ---
# --- Rebase-specific constants and helpers ---
from foundry_mcp.tools.unified.task_handlers.handlers_session_rebase import (  # noqa: F401  # noqa: F401
    ERROR_GATE_ALREADY_PASSED,
    ERROR_GATE_ALREADY_WAIVED,
    ERROR_GATE_NOT_FOUND,
    ERROR_GATE_WAIVER_DISABLED,
    ERROR_GATE_WAIVER_UNAUTHORIZED,
    _find_backup_with_hash,
    _handle_gate_waiver,
    _handle_session_rebase,
    _reconcile_gates_on_rebase,
)
