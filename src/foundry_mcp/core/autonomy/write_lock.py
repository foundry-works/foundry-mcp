"""
Write-lock enforcement helpers for autonomous execution.

This module provides utilities to enforce write-lock protection during
autonomous sessions. When an autonomous session is active for a spec,
protected mutations (task status changes, lifecycle mutations) are
blocked unless explicitly bypassed.

Key functions:
- check_autonomy_write_lock(): Check if a write lock is active for a spec
- is_protected_action(): Determine if an action requires write-lock protection

Usage:
    from foundry_mcp.core.autonomy.write_lock import (
        check_autonomy_write_lock,
        is_protected_action,
        WriteLockStatus,
    )

    # Check if write lock is active
    result = check_autonomy_write_lock(
        spec_id="my-spec",
        workspace="/path/to/workspace",
        bypass_flag=False,
        bypass_reason=None,
    )

    if result.lock_active:
        # Return AUTONOMY_WRITE_LOCK_ACTIVE error
        ...
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Set

import logging

from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    ToolResponse,
)


logger = logging.getLogger(__name__)


# Error code for autonomy write lock
AUTONOMY_WRITE_LOCK_ACTIVE = "AUTONOMY_WRITE_LOCK_ACTIVE"


class WriteLockStatus(Enum):
    """Status of write lock check."""

    ALLOWED = "allowed"  # No lock active, operation allowed
    LOCKED = "locked"  # Lock active, operation blocked
    BYPASSED = "bypassed"  # Lock active but bypassed with reason


# Protected task actions that change execution/progress state
PROTECTED_TASK_ACTIONS: FrozenSet[str] = frozenset({
    # Task status mutations
    "start",
    "complete",
    "update-status",
    "block",
    "unblock",
    # Batch operations that mutate task state
    "start-batch",
    "complete-batch",
    # Dependency mutations (affect task execution flow)
    "add-dependency",
    "remove-dependency",
    # Task structure mutations during active session
    "create",
    "delete",
    "move",
    "update",
})

# Protected lifecycle actions that can skip orchestration sequencing
PROTECTED_LIFECYCLE_ACTIONS: FrozenSet[str] = frozenset({
    "move",
    "activate",
    "complete",
    "archive",
})

# Read-only task actions (not protected)
READ_ONLY_TASK_ACTIONS: FrozenSet[str] = frozenset({
    "info",
    "query",
    "progress",
    "list",
    "prepare",
    "next",
    "session-config",
    "session",
    "session-step",
    "dependencies",
    "history",
})

# Terminal session statuses - write lock not enforced for these
TERMINAL_SESSION_STATUSES: FrozenSet[str] = frozenset({
    "completed",
    "ended",
})

# Non-terminal session statuses - write lock IS enforced for these
NON_TERMINAL_SESSION_STATUSES: FrozenSet[str] = frozenset({
    "running",
    "paused",
    "failed",
})


@dataclass
class WriteLockResult:
    """
    Result of checking autonomy write lock status.

    Attributes:
        status: Whether the operation is allowed, locked, or bypassed
        lock_active: True if a non-terminal session exists for the spec
        session_id: ID of the active session (if any)
        session_status: Status of the active session (if any)
        bypass_logged: True if bypass was logged
        message: Human-readable status message
        metadata: Additional context for logging/debugging
    """
    status: WriteLockStatus
    lock_active: bool
    session_id: Optional[str] = None
    session_status: Optional[str] = None
    bypass_logged: bool = False
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def is_protected_action(action_name: str, action_category: str = "task") -> bool:
    """
    Determine if an action is protected by autonomy write-lock.

    Protected actions are those that mutate task execution state or
    lifecycle status in ways that could bypass autonomous orchestration.

    Args:
        action_name: The name of the action being performed (e.g., "start", "complete")
        action_category: The category of action ("task" or "lifecycle")

    Returns:
        True if the action is protected, False for read-only actions

    Examples:
        >>> is_protected_action("start", "task")
        True
        >>> is_protected_action("complete", "task")
        True
        >>> is_protected_action("info", "task")
        False
        >>> is_protected_action("move", "lifecycle")
        True
    """
    action_lower = action_name.lower() if action_name else ""

    if action_category == "task":
        # Check if it's explicitly protected
        if action_lower in PROTECTED_TASK_ACTIONS:
            return True
        # Check if it's explicitly read-only
        if action_lower in READ_ONLY_TASK_ACTIONS:
            return False
        # Unknown task actions - be conservative and protect
        # This ensures new mutation actions are protected by default
        return True

    if action_category == "lifecycle":
        return action_lower in PROTECTED_LIFECYCLE_ACTIONS

    # Unknown category - be conservative
    return True


def _get_autonomy_sessions_dir(workspace: Optional[str]) -> Optional[Path]:
    """
    Get the autonomy sessions directory for a workspace.

    Args:
        workspace: Path to workspace (uses default if None)

    Returns:
        Path to sessions directory, or None if not determinable
    """
    if workspace:
        ws_path = Path(workspace)
        sessions_dir = ws_path / "specs" / ".autonomy" / "sessions"
        if sessions_dir.exists():
            return sessions_dir

    # Fallback to default location
    default_dir = Path.home() / ".foundry-mcp" / "autonomy" / "sessions"
    if default_dir.exists():
        return default_dir

    return None


def _get_active_session_index_path(workspace: Optional[str]) -> Optional[Path]:
    """
    Get the active session index file path.

    Args:
        workspace: Path to workspace

    Returns:
        Path to index file, or None if not determinable
    """
    if workspace:
        ws_path = Path(workspace)
        index_path = ws_path / "specs" / ".autonomy" / "index" / "spec_active_index.json"
        if index_path.parent.exists():
            return index_path

    # Fallback to default location
    default_path = Path.home() / ".foundry-mcp" / "autonomy" / "index" / "spec_active_index.json"
    return default_path


def _find_active_session_for_spec(
    spec_id: str,
    workspace: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Find an active (non-terminal) session for a spec.

    Checks the active session index and validates session status.

    Args:
        spec_id: The spec ID to check
        workspace: Path to workspace

    Returns:
        Session data dict if active session found, None otherwise
    """
    import json

    # First check the index for fast lookup
    index_path = _get_active_session_index_path(workspace)
    if index_path and index_path.exists():
        try:
            with open(index_path, "r") as f:
                index_data = json.load(f)

            # Index maps spec_id to session_id
            session_id = index_data.get("specs", {}).get(spec_id)
            if session_id:
                # Validate the session still exists and is non-terminal
                sessions_dir = _get_autonomy_sessions_dir(workspace)
                if sessions_dir:
                    session_file = sessions_dir / f"{session_id}.json"
                    if session_file.exists():
                        try:
                            with open(session_file, "r") as f:
                                session_data = json.load(f)

                            status = session_data.get("status", "")
                            if status in NON_TERMINAL_SESSION_STATUSES:
                                return session_data
                        except (json.JSONDecodeError, OSError) as e:
                            logger.debug(
                                "Failed to read session file",
                                extra={"session_id": session_id, "error": str(e)}
                            )
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(
                "Failed to read session index",
                extra={"spec_id": spec_id, "error": str(e)}
            )

    # Fallback: scan sessions directory if index not available
    sessions_dir = _get_autonomy_sessions_dir(workspace)
    if not sessions_dir or not sessions_dir.exists():
        return None

    try:
        for session_file in sessions_dir.glob("auto_*.json"):
            try:
                with open(session_file, "r") as f:
                    session_data = json.load(f)

                # Check if this session is for our spec and is non-terminal
                if session_data.get("spec_id") == spec_id:
                    status = session_data.get("status", "")
                    if status in NON_TERMINAL_SESSION_STATUSES:
                        return session_data
            except (json.JSONDecodeError, OSError):
                continue
    except OSError:
        pass

    return None


def _log_bypass_warning(
    spec_id: str,
    session_id: str,
    bypass_reason: str,
    action_name: str,
    action_category: str,
) -> None:
    """
    Log a warning when write-lock is bypassed.

    Args:
        spec_id: The spec ID
        session_id: The active session ID
        bypass_reason: Reason provided for bypass
        action_name: The action being performed
        action_category: Category of action (task/lifecycle)
    """
    logger.warning(
        "Autonomy write-lock bypassed",
        extra={
            "spec_id": spec_id,
            "session_id": session_id,
            "bypass_reason": bypass_reason,
            "action_name": action_name,
            "action_category": action_category,
            "event_type": "autonomy_write_lock_bypass",
        }
    )


def _write_bypass_journal_entry(
    spec_id: str,
    session_id: str,
    bypass_reason: str,
    action_name: str,
    action_category: str,
    workspace: Optional[str],
) -> bool:
    """
    Write a journal entry for bypass usage.

    This provides an audit trail for write-lock bypasses.

    Args:
        spec_id: The spec ID
        session_id: The active session ID
        bypass_reason: Reason provided for bypass
        action_name: The action being performed
        action_category: Category of action
        workspace: Path to workspace

    Returns:
        True if journal entry was written successfully
    """
    try:
        from foundry_mcp.core.journal import add_journal_entry

        # Load spec data for journaling
        spec_path = _find_spec_path(spec_id, workspace)
        if not spec_path:
            logger.debug(
                "Spec not found for bypass journal entry",
                extra={"spec_id": spec_id}
            )
            return False

        import json
        with open(spec_path, "r") as f:
            spec_data = json.load(f)

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Add journal entry
        add_journal_entry(
            spec_data=spec_data,
            title="Autonomy write-lock bypassed",
            content=(
                f"Write-lock bypassed for {action_category} action '{action_name}'. "
                f"Active session: {session_id}. Reason: {bypass_reason}"
            ),
            entry_type="session",
            author="autonomy-write-lock",
            metadata={
                "session_id": session_id,
                "action_name": action_name,
                "action_category": action_category,
                "bypass_reason": bypass_reason,
                "event_type": "autonomy_write_lock_bypass",
            },
        )

        # Save the updated spec
        with open(spec_path, "w") as f:
            json.dump(spec_data, f, indent=2)

        return True

    except Exception as e:
        logger.warning(
            "Failed to write bypass journal entry",
            extra={
                "spec_id": spec_id,
                "session_id": session_id,
                "error": str(e),
            }
        )
        return False


def _find_spec_path(spec_id: str, workspace: Optional[str]) -> Optional[Path]:
    """
    Find the path to a spec file.

    Args:
        spec_id: The spec ID
        workspace: Path to workspace

    Returns:
        Path to spec file, or None if not found
    """
    search_dirs = []

    if workspace:
        ws_path = Path(workspace)
        for folder in ["active", "pending", "completed", "archived"]:
            search_dirs.append(ws_path / "specs" / folder)

    # Search for the spec
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Check for exact match
        spec_path = search_dir / f"{spec_id}.json"
        if spec_path.exists():
            return spec_path

        # Check for files containing the spec_id
        for spec_file in search_dir.glob("*.json"):
            try:
                import json
                with open(spec_file, "r") as f:
                    data = json.load(f)
                if data.get("spec_id") == spec_id:
                    return spec_file
            except (json.JSONDecodeError, OSError):
                continue

    return None


def check_autonomy_write_lock(
    spec_id: str,
    workspace: Optional[str],
    bypass_flag: bool = False,
    bypass_reason: Optional[str] = None,
    action_name: Optional[str] = None,
    action_category: str = "task",
) -> WriteLockResult:
    """
    Check if an autonomy write lock is active for a spec.

    This function checks whether a non-terminal autonomous session exists
    for the given spec. If a lock is active, protected mutations should
    be rejected unless explicitly bypassed.

    Args:
        spec_id: The spec ID to check
        workspace: Path to the workspace (uses default if None)
        bypass_flag: If True, allow bypassing the lock (requires bypass_reason)
        bypass_reason: Reason for bypassing the lock (required if bypass_flag is True)
        action_name: Name of the action being performed (for logging)
        action_category: Category of action ("task" or "lifecycle")

    Returns:
        WriteLockResult with lock status and metadata

    Examples:
        >>> # Check lock without bypass
        >>> result = check_autonomy_write_lock("my-spec", "/workspace")
        >>> if result.lock_active:
        ...     # Reject with AUTONOMY_WRITE_LOCK_ACTIVE error
        ...     pass

        >>> # Check lock with valid bypass
        >>> result = check_autonomy_write_lock(
        ...     "my-spec",
        ...     "/workspace",
        ...     bypass_flag=True,
        ...     bypass_reason="Manual intervention required",
        ... )
        >>> if result.status == WriteLockStatus.BYPASSED:
        ...     # Allow the operation
        ...     pass

        >>> # Invalid bypass (no reason)
        >>> result = check_autonomy_write_lock(
        ...     "my-spec",
        ...     "/workspace",
        ...     bypass_flag=True,
        ...     bypass_reason=None,
        ... )
        >>> # result.status == WriteLockStatus.LOCKED
    """
    # Find active session for this spec
    session_data = _find_active_session_for_spec(spec_id, workspace)

    if not session_data:
        # No active session - operation allowed
        return WriteLockResult(
            status=WriteLockStatus.ALLOWED,
            lock_active=False,
            message="No active autonomous session for spec",
        )

    session_id = session_data.get("id", "unknown")
    session_status = session_data.get("status", "unknown")

    # Check if write_lock_enforced is enabled for this session
    write_lock_enforced = session_data.get("write_lock_enforced", True)

    if not write_lock_enforced:
        # Session doesn't enforce write lock
        return WriteLockResult(
            status=WriteLockStatus.ALLOWED,
            lock_active=True,
            session_id=session_id,
            session_status=session_status,
            message="Active session does not enforce write lock",
        )

    # Lock is active - check for bypass
    if bypass_flag:
        # Bypass requires a reason
        if not bypass_reason or not bypass_reason.strip():
            return WriteLockResult(
                status=WriteLockStatus.LOCKED,
                lock_active=True,
                session_id=session_id,
                session_status=session_status,
                message="Bypass requires bypass_reason to be provided",
                metadata={
                    "error": "bypass_reason_required",
                    "session_id": session_id,
                },
            )

        # Log the bypass
        _log_bypass_warning(
            spec_id=spec_id,
            session_id=session_id,
            bypass_reason=bypass_reason,
            action_name=action_name or "unknown",
            action_category=action_category,
        )

        # Write journal entry
        journal_written = _write_bypass_journal_entry(
            spec_id=spec_id,
            session_id=session_id,
            bypass_reason=bypass_reason,
            action_name=action_name or "unknown",
            action_category=action_category,
            workspace=workspace,
        )

        return WriteLockResult(
            status=WriteLockStatus.BYPASSED,
            lock_active=True,
            session_id=session_id,
            session_status=session_status,
            bypass_logged=True,
            message=f"Write lock bypassed with reason: {bypass_reason}",
            metadata={
                "bypass_reason": bypass_reason,
                "journal_written": journal_written,
            },
        )

    # Lock active, no bypass requested
    return WriteLockResult(
        status=WriteLockStatus.LOCKED,
        lock_active=True,
        session_id=session_id,
        session_status=session_status,
        message=f"Autonomous session '{session_id}' is active for spec '{spec_id}'",
        metadata={
            "session_id": session_id,
            "session_status": session_status,
        },
    )


def make_write_lock_error_response(
    result: WriteLockResult,
    action_name: Optional[str] = None,
    request_id: Optional[str] = None,
) -> ToolResponse:
    """
    Create a standardized error response for write lock active condition.

    Args:
        result: The WriteLockResult from check_autonomy_write_lock
        action_name: Name of the action that was blocked
        request_id: Optional request correlation ID

    Returns:
        ToolResponse with AUTONOMY_WRITE_LOCK_ACTIVE error
    """
    action_desc = f" for action '{action_name}'" if action_name else ""

    return error_response(
        message=(
            f"Cannot perform operation{action_desc}: "
            f"autonomous session '{result.session_id}' is active for this spec. "
            f"Use bypass_autonomy_lock=true with bypass_reason to override."
        ),
        error_code=ErrorCode.FORBIDDEN,  # Reuse existing FORBIDDEN code
        error_type=ErrorType.AUTHORIZATION,
        data={
            "error_code": AUTONOMY_WRITE_LOCK_ACTIVE,
            "session_id": result.session_id,
            "session_status": result.session_status,
            "action_name": action_name,
            "remediation": (
                "Either wait for the autonomous session to complete, "
                "or use bypass_autonomy_lock=true with a bypass_reason to override. "
                "Bypasses are logged and journaled."
            ),
        },
        request_id=request_id,
    )


# Convenience function for common use pattern
def check_and_enforce_write_lock(
    spec_id: str,
    workspace: Optional[str],
    action_name: str,
    action_category: str = "task",
    bypass_flag: bool = False,
    bypass_reason: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Optional[ToolResponse]:
    """
    Check write lock and return error response if blocked.

    This is a convenience function that combines check_autonomy_write_lock
    with error response creation. Returns None if the operation is allowed.

    Args:
        spec_id: The spec ID to check
        workspace: Path to the workspace
        action_name: Name of the action being performed
        action_category: Category of action ("task" or "lifecycle")
        bypass_flag: If True, allow bypassing the lock
        bypass_reason: Reason for bypassing (required if bypass_flag is True)
        request_id: Optional request correlation ID

    Returns:
        None if operation is allowed, ToolResponse error if blocked

    Examples:
        >>> # In a task mutation handler
        >>> error = check_and_enforce_write_lock(
        ...     spec_id="my-spec",
        ...     workspace="/workspace",
        ...     action_name="complete",
        ...     action_category="task",
        ...     bypass_flag=bypass_autonomy_lock,
        ...     bypass_reason=bypass_reason,
        ... )
        >>> if error:
        ...     return error  # Return the error response
        >>> # Proceed with the operation
    """
    # First check if this action is even protected
    if not is_protected_action(action_name, action_category):
        return None  # Not a protected action, allow

    # Check the write lock
    result = check_autonomy_write_lock(
        spec_id=spec_id,
        workspace=workspace,
        bypass_flag=bypass_flag,
        bypass_reason=bypass_reason,
        action_name=action_name,
        action_category=action_category,
    )

    # If locked and not bypassed, return error
    if result.status == WriteLockStatus.LOCKED:
        return make_write_lock_error_response(
            result=result,
            action_name=action_name,
            request_id=request_id,
        )

    # Allowed or bypassed
    return None


# Export public API
__all__ = [
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
]
