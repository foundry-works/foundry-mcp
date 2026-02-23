"""Role-based authorization for foundry-mcp.

This module provides role-based access control for tool actions,
constraining what operations different roles can perform.

Roles:
- autonomy_runner: Session and step management actions only
- maintainer: Full mutation access (wildcard, default)
- observer: Read-only operations

Runtime Isolation (P2.2):
- workspace_root: Restricts runner file operations to a path prefix
- path_traversal_check: Denies .. in file paths for runner requests
- subprocess_isolation: Restricted PATH, stdin timeout for runner subprocesses

Usage:
    from foundry_mcp.core.authorization import (
        check_action_allowed,
        get_role_allowlist,
        get_server_role,
        set_server_role,
        AuthzResult,
        validate_runner_path,
        RunnerIsolationConfig,
    )

    # Check if an action is allowed
    result = check_action_allowed("autonomy_runner", "task", "complete")
    if not result.allowed:
        return error_response(f"Action denied: {result.denied_action}")
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Role Definitions
# =============================================================================


class Role(str, Enum):
    """Defined roles for authorization."""

    AUTONOMY_RUNNER = "autonomy_runner"
    MAINTAINER = "maintainer"
    OBSERVER = "observer"


# =============================================================================
# Role Allowlists
# =============================================================================

# Actions allowed for autonomy_runner role - session management and step control only
AUTONOMY_RUNNER_ALLOWLIST: FrozenSet[str] = frozenset(
    {
        # Spec resolution
        "spec-find",
        # Runtime capability preflight
        "server-capabilities",
        # Session lifecycle
        "session-start",
        "session-resume",
        "session-heartbeat",
        "session-rebase",
        "session-list",
        "session-status",
        # Session-step actions
        "session-step-next",
        "session-step-report",
        "session-step-replay",
        "session-step-heartbeat",
        # Fidelity gate
        "review-fidelity-gate",
        # Verification execution (required for proof-carrying receipts)
        "verification-execute",
        # Read-only task context (used by step handlers for scope understanding)
        "prepare",
    }
)

# Actions allowed for maintainer role - full mutation access (wildcard)
MAINTAINER_ALLOWLIST: FrozenSet[str] = frozenset({"*"})

# Actions allowed for observer role - read-only operations
OBSERVER_ALLOWLIST: FrozenSet[str] = frozenset(
    {
        # Read-only task actions
        "list",
        "get",
        "view",
        "status",
        "search",
        "progress",
        "info",
        "query",
        "prepare",
        # Read-only session actions
        "session-status",
        "session-events",
        "session-list",
        # Read-only spec actions
        "spec-list",
        "spec-info",
        # Read-only journal actions
        "journal-list",
        "journal-get",
    }
)

# Role to allowlist mapping
_ROLE_ALLOWLISTS: Dict[str, FrozenSet[str]] = {
    Role.AUTONOMY_RUNNER.value: AUTONOMY_RUNNER_ALLOWLIST,
    Role.MAINTAINER.value: MAINTAINER_ALLOWLIST,
    Role.OBSERVER.value: OBSERVER_ALLOWLIST,
}


# =============================================================================
# Context Variable for Server Role
# =============================================================================

server_role_var: ContextVar[Optional[str]] = ContextVar("server_role", default=None)
"""Process-level server role for authorization decisions."""
_configured_server_role: str = Role.MAINTAINER.value
_configured_server_role_lock = Lock()


def get_server_role() -> str:
    """Get the current server role from context.

    Returns:
        Current role string (default: "maintainer")
    """
    context_role = server_role_var.get()
    if isinstance(context_role, str) and context_role:
        return context_role

    with _configured_server_role_lock:
        return _configured_server_role


def set_server_role(role: str) -> None:
    """Set the server role in context.

    Args:
        role: Role to set (must be a valid Role enum value)
    """
    # Validate role
    try:
        Role(role)
    except ValueError:
        logger.error(
            "Invalid role '%s', falling back to observer (fail-closed). Valid roles: %s",
            role,
            ", ".join(r.value for r in Role),
        )
        role = Role.OBSERVER.value

    global _configured_server_role

    with _configured_server_role_lock:
        _configured_server_role = role

    server_role_var.set(role)
    logger.info("Server role set to: %s", role)


# =============================================================================
# Authorization Result
# =============================================================================


@dataclass
class AuthzResult:
    """Result of an authorization check.

    Attributes:
        allowed: Whether the action is allowed
        denied_action: The action that was denied (if any)
        configured_role: The role that was checked
        required_role: The minimum role required for this action (if denied)
    """

    allowed: bool
    denied_action: Optional[str] = None
    configured_role: str = Role.MAINTAINER.value
    required_role: Optional[str] = None


# =============================================================================
# Authorization Functions
# =============================================================================


def get_role_allowlist(role: str) -> Set[str]:
    """Get the set of allowed actions for a role.

    Args:
        role: Role name to look up

    Returns:
        Set of allowed action names (empty set for unknown roles)
    """
    allowlist = _ROLE_ALLOWLISTS.get(role)
    if allowlist is None:
        logger.debug("Unknown role '%s', returning empty allowlist", role)
        return set()
    return set(allowlist)


def _normalize_action(tool_name: Optional[str], action: Optional[str]) -> str:
    """Normalize tool/action into a single action string.

    Args:
        tool_name: Tool/router name (e.g., "task", "session")
        action: Action name (e.g., "complete", "start")

    Returns:
        Normalized action string (e.g., "task-complete" or just "complete")
    """
    if not action:
        return ""

    # Handle already-combined actions (e.g., "session-start")
    if "-" in str(action) and tool_name and action.startswith(str(tool_name)):
        return str(action).lower()

    # Combine tool and action
    if tool_name and action:
        return f"{tool_name.lower()}-{action.lower()}"

    return str(action).lower() if action else ""


def check_action_allowed(
    role: str,
    tool_name: Optional[str],
    action: Optional[str],
) -> AuthzResult:
    """Check if an action is allowed for the given role.

    Args:
        role: Role to check authorization for
        tool_name: Tool/router name (e.g., "task", "session")
        action: Action name (e.g., "complete", "start")

    Returns:
        AuthzResult with allowed status and details
    """
    normalized_action = _normalize_action(tool_name, action)

    # Get allowlist for role
    allowlist = get_role_allowlist(role)

    # Unknown role - default to observer (fail-closed)
    if not allowlist and role not in _ROLE_ALLOWLISTS:
        _log_authorization_denial(
            role=role,
            action=normalized_action or action or "unknown",
            denied_action=normalized_action or action or "unknown",
            required_role=Role.MAINTAINER.value,
            reason="unknown_role",
        )
        return AuthzResult(
            allowed=False,
            denied_action=normalized_action or action or "unknown",
            configured_role=role,
            required_role=Role.MAINTAINER.value,
        )

    # Wildcard allows everything
    if "*" in allowlist:
        return AuthzResult(
            allowed=True,
            configured_role=role,
        )

    # Check if action is in allowlist
    if normalized_action in allowlist:
        return AuthzResult(
            allowed=True,
            configured_role=role,
        )

    # Check raw action name as fallback
    raw_action = (action or "").lower()
    if raw_action in allowlist:
        return AuthzResult(
            allowed=True,
            configured_role=role,
        )

    # Action not allowed
    _log_authorization_denial(
        role=role,
        action=normalized_action or raw_action,
        denied_action=normalized_action or action or "unknown",
        required_role=_get_required_role_for_action(normalized_action or raw_action),
    )
    return AuthzResult(
        allowed=False,
        denied_action=normalized_action or action or "unknown",
        configured_role=role,
        required_role=_get_required_role_for_action(normalized_action or raw_action),
    )


def _log_authorization_denial(
    role: str,
    action: str,
    denied_action: str,
    required_role: Optional[str],
    reason: str = "action_not_in_allowlist",
) -> None:
    """Emit structured warning for authorization denials."""
    logger.warning(
        "AUTHORIZATION_DENIED: role=%s action=%s denied_action=%s required_role=%s reason=%s",
        role,
        action,
        denied_action,
        required_role,
        reason,
        extra={
            "event": "authorization_denied",
            "role": role,
            "action": action,
            "denied_action": denied_action,
            "required_role": required_role,
            "reason": reason,
        },
    )


def _get_required_role_for_action(action: str) -> str:
    """Determine the minimum role required for an action.

    Args:
        action: Action name to check

    Returns:
        Minimum required role name
    """
    # Check if it's an observer action
    if action in OBSERVER_ALLOWLIST:
        return Role.OBSERVER.value

    # Check if it's an autonomy_runner action
    if action in AUTONOMY_RUNNER_ALLOWLIST:
        return Role.AUTONOMY_RUNNER.value

    # Default to maintainer for mutation actions
    return Role.MAINTAINER.value


def initialize_role_from_config(config_role: Optional[str] = None) -> str:
    """Initialize server role from config or environment.

    Priority:
    1. FOUNDRY_MCP_ROLE environment variable
    2. config_role parameter (from config file)
    3. Default: "maintainer"

    Args:
        config_role: Role from config file

    Returns:
        The initialized role
    """
    # Check environment variable first (highest priority)
    env_role = os.environ.get("FOUNDRY_MCP_ROLE", "").strip().lower()
    if env_role:
        set_server_role(env_role)
        return get_server_role()

    # Use config role if provided
    if config_role:
        set_server_role(config_role)
        return get_server_role()

    # Default to maintainer (full interactive access)
    set_server_role(Role.MAINTAINER.value)
    return get_server_role()


# =============================================================================
# Rate Limiting for Authorization Denials
# =============================================================================


@dataclass
class RateLimitConfig:
    """Configuration for authorization denial rate limiting.

    Attributes:
        max_consecutive_denials: Maximum denials before rate limiting
        denial_window_seconds: Sliding window for counting denials
        retry_after_seconds: Cooldown period after rate limit triggered
        max_tracked_actions: Maximum number of action keys retained in memory
        stale_entry_ttl_seconds: TTL for inactive tracking state
        global_cleanup_interval_seconds: Minimum interval between global sweeps
    """

    max_consecutive_denials: int = 10
    denial_window_seconds: int = 60
    retry_after_seconds: int = 5
    max_tracked_actions: int = 1024
    stale_entry_ttl_seconds: int = 300
    global_cleanup_interval_seconds: int = 5


@dataclass
class _DenialRecord:
    """Internal record of a denial event."""

    timestamp: float
    action: str


class RateLimitTracker:
    """Tracks authorization denials per action for rate limiting.

    Uses a sliding window to track consecutive denials. When the threshold
    is exceeded, subsequent requests are rate-limited until the cooldown
    period expires.

    Thread-safe implementation using a lock for state mutations.
    State is in-memory only and resets on process restart.

    Usage:
        tracker = RateLimitTracker()

        # Check before authorization
        retry_after = tracker.check_rate_limit(action)
        if retry_after:
            return rate_limited_response(retry_after)

        # On denial, record it
        tracker.record_denial(action)

        # On success, reset the counter
        tracker.reset(action)
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize the rate limit tracker.

        Args:
            config: Rate limit configuration (uses defaults if not provided)
        """
        self._config = config or RateLimitConfig()
        self._denials: Dict[str, List[_DenialRecord]] = {}
        self._rate_limited_until: Dict[str, float] = {}
        self._last_seen: Dict[str, float] = {}
        self._last_global_cleanup: float = 0.0
        self._lock = Lock()

    def check_rate_limit(self, action: str) -> Optional[float]:
        """Check if an action is currently rate-limited.

        Args:
            action: The action to check

        Returns:
            None if not rate-limited, otherwise retry_after seconds
        """
        now = time.monotonic()

        with self._lock:
            self._touch_action(action, now)
            # Check if currently rate-limited
            limited_until = self._rate_limited_until.get(action, 0)
            if now < limited_until:
                remaining = limited_until - now
                self._run_global_maintenance(now)
                return remaining

            # Clean up expired entries
            self._cleanup_expired(action, now)
            self._run_global_maintenance(now)

            return None

    def record_denial(self, action: str) -> None:
        """Record an authorization denial for an action.

        Args:
            action: The action that was denied
        """
        now = time.monotonic()

        with self._lock:
            self._touch_action(action, now)
            # Clean up old entries first
            self._cleanup_expired(action, now)

            # Add new denial
            if action not in self._denials:
                self._denials[action] = []
            self._denials[action].append(_DenialRecord(timestamp=now, action=action))

            # Check if we should trigger rate limiting
            denial_count = len(self._denials[action])
            if denial_count >= self._config.max_consecutive_denials:
                self._rate_limited_until[action] = now + self._config.retry_after_seconds
                logger.warning(
                    "Rate limiting triggered for action '%s' after %d denials. Retry after %d seconds.",
                    action,
                    denial_count,
                    self._config.retry_after_seconds,
                )
                # Clear the denial list after triggering rate limit
                self._denials[action] = []

            self._run_global_maintenance(now)

    def reset(self, action: str) -> None:
        """Reset the denial counter for an action.

        Called after a successful authorization to clear the counter.

        Args:
            action: The action to reset
        """
        with self._lock:
            self._evict_action(action)

    def _cleanup_expired(self, action: str, now: float) -> None:
        """Remove expired denial records outside the window.

        Args:
            action: The action to clean up
            now: Current timestamp
        """
        if action not in self._denials:
            pass
        else:
            window_start = now - self._config.denial_window_seconds
            self._denials[action] = [d for d in self._denials[action] if d.timestamp >= window_start]
            if not self._denials[action]:
                del self._denials[action]

        # Also clean up expired rate limits
        if action in self._rate_limited_until:
            if now >= self._rate_limited_until[action]:
                del self._rate_limited_until[action]

    def _touch_action(self, action: str, now: float) -> None:
        """Record recent activity for bounded-LRU cleanup decisions."""
        self._last_seen[action] = now

    def _evict_action(self, action: str) -> None:
        """Remove all tracking state for an action key."""
        self._denials.pop(action, None)
        self._rate_limited_until.pop(action, None)
        self._last_seen.pop(action, None)

    def _run_global_maintenance(self, now: float) -> None:
        """Sweep stale state and enforce cardinality bounds globally."""
        interval = max(0, self._config.global_cleanup_interval_seconds)
        if interval > 0 and now - self._last_global_cleanup < interval:
            return
        self._last_global_cleanup = now

        for action in list(self._denials):
            self._cleanup_expired(action, now)
        for action in list(self._rate_limited_until):
            self._cleanup_expired(action, now)

        stale_ttl = max(0, self._config.stale_entry_ttl_seconds)
        for action, seen_at in list(self._last_seen.items()):
            if action in self._denials or action in self._rate_limited_until:
                continue
            if now - seen_at >= stale_ttl:
                self._last_seen.pop(action, None)

        max_actions = max(1, self._config.max_tracked_actions)
        tracked_keys = set(self._last_seen) | set(self._denials) | set(self._rate_limited_until)
        if len(tracked_keys) <= max_actions:
            return

        oldest_first = sorted(
            tracked_keys,
            key=lambda key: self._last_seen.get(key, 0.0),
        )
        overflow = len(tracked_keys) - max_actions
        for action in oldest_first[:overflow]:
            self._evict_action(action)

    def get_stats(self, action: str) -> Dict[str, int]:
        """Get current stats for an action (for debugging/monitoring).

        Args:
            action: The action to get stats for

        Returns:
            Dict with denial_count, is_limited, retry_after fields
        """
        now = time.monotonic()

        with self._lock:
            self._touch_action(action, now)
            self._cleanup_expired(action, now)
            self._run_global_maintenance(now)

            denial_count = len(self._denials.get(action, []))
            limited_until = self._rate_limited_until.get(action, 0)
            is_limited = now < limited_until
            retry_after = max(0, limited_until - now) if is_limited else 0

            return {
                "denial_count": denial_count,
                "is_limited": is_limited,
                "retry_after": int(retry_after),
            }


# Global rate limit tracker instance
_rate_limit_tracker: Optional[RateLimitTracker] = None
_rate_limit_lock = Lock()


def get_rate_limit_tracker(
    config: Optional[RateLimitConfig] = None,
) -> RateLimitTracker:
    """Get the global rate limit tracker instance.

    Args:
        config: Configuration to use (only applied on first call)

    Returns:
        The global RateLimitTracker instance
    """
    global _rate_limit_tracker

    with _rate_limit_lock:
        if _rate_limit_tracker is None:
            _rate_limit_tracker = RateLimitTracker(config)
        elif config is not None:
            # Explicit config re-initializes the singleton so startup can
            # apply configured thresholds deterministically.
            _rate_limit_tracker = RateLimitTracker(config)
        return _rate_limit_tracker


def reset_rate_limit_tracker() -> None:
    """Reset the global rate limit tracker (for testing)."""
    global _rate_limit_tracker

    with _rate_limit_lock:
        _rate_limit_tracker = None


# =============================================================================
# Runtime Isolation Profiles (P2.2)
# =============================================================================


@dataclass
class RunnerIsolationConfig:
    """Configuration for autonomy_runner runtime isolation.

    Application-level hardening for the runner role:
    - workspace_root: Restricts file operations to this path prefix
    - stdin_timeout_seconds: Kill subprocesses blocking on stdin after this time
    - restricted_path: PATH environment for runner subprocesses (comma-separated)
    - allow_path_traversal: If False, reject .. in file paths for runner requests

    This is application-level restriction, not OS-level sandboxing.
    """

    workspace_root: Optional[str] = None
    stdin_timeout_seconds: int = 30
    restricted_path: Optional[str] = None
    allow_path_traversal: bool = False

    def __post_init__(self) -> None:
        """Normalize workspace_root to absolute path."""
        if self.workspace_root:
            self.workspace_root = str(Path(self.workspace_root).resolve())


# Global isolation config for runner role
_runner_isolation_config: Optional[RunnerIsolationConfig] = None
_isolation_config_lock = Lock()


def get_runner_isolation_config() -> RunnerIsolationConfig:
    """Get the global runner isolation configuration.

    Returns:
        The global RunnerIsolationConfig instance (creates default if not set)
    """
    global _runner_isolation_config

    with _isolation_config_lock:
        if _runner_isolation_config is None:
            # Initialize from environment variables
            workspace_root = os.environ.get("FOUNDRY_MCP_WORKSPACE_ROOT", "")
            restricted_path = os.environ.get("FOUNDRY_MCP_RUNNER_PATH", "")

            _runner_isolation_config = RunnerIsolationConfig(
                workspace_root=workspace_root or None,
                stdin_timeout_seconds=int(os.environ.get("FOUNDRY_MCP_STDIN_TIMEOUT", "30")),
                restricted_path=restricted_path or None,
                allow_path_traversal=False,  # Always False for security
            )
        return _runner_isolation_config


def set_runner_isolation_config(config: RunnerIsolationConfig) -> None:
    """Set the global runner isolation configuration.

    Args:
        config: The RunnerIsolationConfig to use
    """
    global _runner_isolation_config

    with _isolation_config_lock:
        _runner_isolation_config = config
        logger.info(
            "Runner isolation configured: workspace_root=%s, stdin_timeout=%ds",
            config.workspace_root,
            config.stdin_timeout_seconds,
        )


def reset_runner_isolation_config() -> None:
    """Reset the runner isolation config (for testing)."""
    global _runner_isolation_config

    with _isolation_config_lock:
        _runner_isolation_config = None


# Error classes (canonical definitions in foundry_mcp.core.errors.authorization)
from foundry_mcp.core.errors.authorization import PathValidationError  # noqa: E402


def validate_runner_path(
    path: Union[str, Path],
    *,
    require_within_workspace: bool = True,
    reject_traversal: bool = True,
) -> Path:
    """Validate a path for autonomy_runner isolation constraints.

    This function enforces:
    1. Path traversal rejection (deny .. in paths)
    2. Workspace root boundary (path must be within workspace_root)

    Args:
        path: The path to validate
        require_within_workspace: If True, path must be within workspace_root
        reject_traversal: If True, reject .. in path components

    Returns:
        The resolved, validated Path object

    Raises:
        PathValidationError: If validation fails

    Usage:
        try:
            safe_path = validate_runner_path(user_provided_path)
        except PathValidationError as e:
            return error_response(f"Invalid path: {e.reason}")
    """
    if isinstance(path, str):
        path_str = path
        path_obj = Path(path)
    else:
        path_str = str(path)
        path_obj = path

    # Check for path traversal (..)
    if reject_traversal:
        # Check for .. in the path string
        if ".." in path_str:
            raise PathValidationError(
                path=path_str,
                reason="path_traversal_denied",
                detail="Path contains '..' which is not allowed for runner-scoped requests",
            )

        # Also check resolved path components (catches encoded variants)
        try:
            resolved = path_obj.resolve()
            parts = resolved.parts
            if ".." in parts:
                raise PathValidationError(
                    path=path_str,
                    reason="path_traversal_denied",
                    detail="Resolved path contains '..' component",
                )
        except (OSError, ValueError) as e:
            # Path resolution failed - could be invalid path
            raise PathValidationError(
                path=path_str,
                reason="invalid_path",
                detail=str(e),
            ) from e

    # Check workspace root boundary
    config = get_runner_isolation_config()
    full_path: Optional[Path] = None
    if require_within_workspace and config.workspace_root:
        workspace_root = Path(config.workspace_root).resolve()

        try:
            # Resolve the path relative to workspace if not absolute
            if not path_obj.is_absolute():
                full_path = (workspace_root / path_obj).resolve()
            else:
                full_path = path_obj.resolve()

            # Check if path is within workspace
            try:
                full_path.relative_to(workspace_root)
            except ValueError as e:
                raise PathValidationError(
                    path=path_str,
                    reason="outside_workspace",
                    detail=f"Path is outside configured workspace_root: {config.workspace_root}",
                ) from e
        except PathValidationError:
            raise
        except Exception as e:
            raise PathValidationError(
                path=path_str,
                reason="validation_error",
                detail=str(e),
            ) from e
        # When workspace boundary enforcement is active, always return the
        # normalized path that was validated against workspace_root.
        return full_path

    return path_obj.resolve()


def get_restricted_environment(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Get environment variables with PATH restriction for runner subprocesses.

    Args:
        env: Base environment (defaults to os.environ)

    Returns:
        Environment dict with restricted PATH for runner subprocesses
    """
    config = get_runner_isolation_config()
    result = dict(env or os.environ)

    if config.restricted_path:
        result["PATH"] = config.restricted_path
        logger.debug("Runner subprocess PATH restricted to: %s", config.restricted_path)

    return result


def is_runner_role() -> bool:
    """Check if the current server role is autonomy_runner.

    Returns:
        True if current role is autonomy_runner
    """
    return get_server_role() == Role.AUTONOMY_RUNNER.value


from foundry_mcp.core.errors.authorization import StdinTimeoutError  # noqa: E402, F811


def run_isolated_subprocess(
    command: Sequence[str],
    *,
    timeout: Optional[float] = None,
    stdin_timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Union[str, Path]] = None,
    capture_output: bool = True,
    text: bool = True,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with runner isolation constraints.

    For autonomy_runner role, applies:
    - Restricted PATH from RunnerIsolationConfig
    - Hard stdin timeout cap with process termination
    - Workspace validation for cwd

    Args:
        command: Command and arguments to run
        timeout: Overall command timeout (seconds)
        stdin_timeout: Timeout for stdin blocking (default: from config)
        env: Environment variables (merged with restricted PATH)
        cwd: Working directory (validated for runner role)
        capture_output: Capture stdout/stderr
        text: Return strings instead of bytes
        **kwargs: Additional subprocess.run arguments

    Returns:
        CompletedProcess result

    Raises:
        StdinTimeoutError: If stdin timeout cap is exceeded
        subprocess.TimeoutExpired: If overall timeout is exceeded
        PathValidationError: If cwd validation fails for runner role
    """
    config = get_runner_isolation_config()

    # Get stdin timeout from config if not specified
    effective_stdin_timeout: Optional[float]
    if stdin_timeout is not None:
        effective_stdin_timeout = stdin_timeout
    else:
        effective_stdin_timeout = config.stdin_timeout_seconds
    if effective_stdin_timeout is not None and effective_stdin_timeout <= 0:
        effective_stdin_timeout = None

    # Build environment with PATH restriction
    effective_env = get_restricted_environment(env)

    # Validate cwd for runner role
    effective_cwd = cwd
    if cwd and is_runner_role():
        try:
            effective_cwd = str(validate_runner_path(cwd))
        except PathValidationError:
            raise

    # For stdin timeout, we use a shorter timeout for stdin operations
    # and stdin=DEVNULL to prevent blocking on input
    stdin_setting = kwargs.pop("stdin", subprocess.DEVNULL)

    # Apply stdin timeout as a hard cap (shorter of explicit timeout and stdin cap).
    timeout_source = "overall"
    effective_timeout = timeout
    if effective_stdin_timeout is not None:
        if effective_timeout is None or effective_timeout > effective_stdin_timeout:
            effective_timeout = effective_stdin_timeout
            timeout_source = "stdin"

    try:
        result = subprocess.run(
            command,
            stdin=stdin_setting,
            capture_output=capture_output,
            text=text,
            timeout=effective_timeout,
            env=effective_env,
            cwd=effective_cwd,
            **kwargs,
        )
        return result
    except subprocess.TimeoutExpired as e:
        if timeout_source == "stdin" and effective_stdin_timeout is not None:
            logger.warning(
                "Subprocess stdin timeout: %s",
                " ".join(str(c) for c in command[:3]),
            )
            raise StdinTimeoutError(effective_stdin_timeout, command) from e
        raise


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Role",
    "AuthzResult",
    "AUTONOMY_RUNNER_ALLOWLIST",
    "MAINTAINER_ALLOWLIST",
    "OBSERVER_ALLOWLIST",
    "server_role_var",
    "get_server_role",
    "set_server_role",
    "get_role_allowlist",
    "check_action_allowed",
    "initialize_role_from_config",
    # Rate limiting
    "RateLimitConfig",
    "RateLimitTracker",
    "get_rate_limit_tracker",
    "reset_rate_limit_tracker",
    # Runtime isolation (P2.2)
    "RunnerIsolationConfig",
    "get_runner_isolation_config",
    "set_runner_isolation_config",
    "reset_runner_isolation_config",
    "PathValidationError",
    "validate_runner_path",
    "get_restricted_environment",
    "is_runner_role",
    "StdinTimeoutError",
    "run_isolated_subprocess",
]
