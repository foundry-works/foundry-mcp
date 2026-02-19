"""Shared helpers for unified tool routers.

Consolidates duplicated per-router boilerplate (request IDs, metric names,
validation errors, specs-dir resolution, dispatch error handling) into
parameterised functions that each router can call with its own tool name.

Imports only from ``foundry_mcp.core`` and the standard library.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from foundry_mcp.core.authorization import (
    check_action_allowed,
    get_rate_limit_tracker,
    get_server_role,
)
from foundry_mcp.core.context import (
    generate_correlation_id,
    get_client_id,
    get_correlation_id,
)
from foundry_mcp.core.errors.authorization import PathValidationError
from foundry_mcp.core.errors.execution import ActionRouterError
from foundry_mcp.core.observability import MetricsCollector
from foundry_mcp.core.responses.builders import error_response
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.core.spec import find_specs_directory
from foundry_mcp.tools.unified.router import ActionRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authorization policy
# ---------------------------------------------------------------------------


def _normalize_principal(value: Any) -> Optional[str]:
    """Return a sanitized principal value or ``None`` when unavailable."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"anonymous", "unknown", "none", "null"}:
        return None
    return normalized


def _resolve_rate_limit_scope() -> tuple[str, str]:
    """Resolve per-request rate-limit scope.

    Uses only trusted request-context identity and falls back to role scope.
    """
    context_client = _normalize_principal(get_client_id())
    if context_client:
        return f"client:{context_client}", "client"

    return "", "role"


def _build_rate_limit_key(
    *,
    tool_name: str,
    action: str,
    role: str,
) -> tuple[str, str]:
    """Build a scoped authorization-denial rate-limit key."""
    scope_value, scope_kind = _resolve_rate_limit_scope()
    action_key = f"{tool_name}.{action}"
    if scope_kind == "client":
        return f"{action_key}|{scope_value}", scope_kind
    return f"{action_key}|role:{role}", scope_kind


# ---------------------------------------------------------------------------
# 1. Request ID
# ---------------------------------------------------------------------------


def build_request_id(tool_name: str) -> str:
    """Return an existing correlation ID or generate one with *tool_name* prefix."""
    return get_correlation_id() or generate_correlation_id(prefix=tool_name)


# ---------------------------------------------------------------------------
# 2. Metric name
# ---------------------------------------------------------------------------


def make_metric_name(prefix: str, action: str) -> str:
    """Build a dot-separated metric key, normalising hyphens to underscores.

    Examples::

        make_metric_name("authoring", "phase-add")   -> "authoring.phase_add"
        make_metric_name("unified_tools.task", "add") -> "unified_tools.task.add"
    """
    return f"{prefix}.{action.replace('-', '_')}"


# ---------------------------------------------------------------------------
# 3. Specs-dir resolution
# ---------------------------------------------------------------------------


def resolve_specs_dir(
    config: Any,
    path_or_workspace: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[dict]]:
    """Resolve the specs directory from *config* and an optional path hint.

    Returns ``(specs_dir, None)`` on success or ``(None, error_dict)`` on
    failure so callers can short-circuit with a ready-made error envelope.
    """
    try:
        if path_or_workspace:
            specs_dir = find_specs_directory(path_or_workspace)
        else:
            candidate = getattr(config, "specs_dir", None)
            if isinstance(candidate, Path):
                specs_dir = candidate
            elif isinstance(candidate, str) and candidate.strip():
                specs_dir = Path(candidate)
            else:
                specs_dir = find_specs_directory()
    except Exception as exc:
        logger.exception(
            "Failed to resolve specs directory",
            extra={"path_or_workspace": path_or_workspace},
        )
        return None, asdict(
            error_response(
                f"Failed to resolve specs directory: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Verify specs_dir configuration or pass a workspace path",
            )
        )

    if not specs_dir:
        return None, asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Set FOUNDRY_SPECS_DIR or provide a workspace path",
            )
        )

    return specs_dir, None


# ---------------------------------------------------------------------------
# 4. Dispatch with standard errors
# ---------------------------------------------------------------------------


def dispatch_with_standard_errors(
    router: ActionRouter,
    tool_name: str,
    action: str,
    /,
    *,
    request_id: Optional[str] = None,
    include_details_in_router_error: bool = False,
    **kwargs: Any,
) -> dict:
    """Dispatch *action* through *router*, converting exceptions to envelopes.

    Performs action validation, then rate-limit/authorization checks,
    then dispatches to the handler.
    Error precedence:
    action validation -> RATE_LIMITED -> AUTHORIZATION -> argument validation.

    Catches :class:`ActionRouterError` (unsupported action) and generic
    ``Exception`` and returns a well-formed error response dict.
    """
    # First, validate that the action exists (action validation)
    # Use allowed_actions() to check if action is registered
    allowed = router.allowed_actions()
    action_lower = action.lower() if action else ""

    # Check if action (case-insensitive) is in allowed actions
    action_exists = any(a.lower() == action_lower for a in allowed)

    if not action_exists:
        rid = request_id or build_request_id(tool_name)
        allowed_str = ", ".join(sorted(allowed))
        details: Optional[Dict[str, Any]] = None
        if include_details_in_router_error:
            details = {"action": action, "allowed_actions": list(allowed)}
        return asdict(
            error_response(
                f"Unsupported {tool_name} action '{action}'. Allowed actions: {allowed_str}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed_str}",
                request_id=rid,
                details=details,
            )
        )

    current_role = get_server_role()
    rate_limit_key, rate_limit_scope = _build_rate_limit_key(
        tool_name=tool_name,
        action=action,
        role=current_role,
    )
    action_key = f"{tool_name}.{action}"

    # Rate limit check (before authorization check)
    rate_limit_tracker = get_rate_limit_tracker()
    retry_after = rate_limit_tracker.check_rate_limit(rate_limit_key)

    if retry_after is not None:
        rid = request_id or build_request_id(tool_name)
        logger.warning(
            "Rate limited: %s.%s (%s scope) - retry after %.1f seconds",
            tool_name,
            action,
            rate_limit_scope,
            retry_after,
        )

        # Emit authz.rate_limited metric
        metrics = MetricsCollector(prefix="authz")
        metrics.counter(
            "rate_limited",
            labels={
                "tool": tool_name,
                "action": action,
                "scope": rate_limit_scope,
            },
        )

        return asdict(
            error_response(
                f"Rate limited: too many authorization denials for '{tool_name}.{action}'",
                error_code=ErrorCode.RATE_LIMITED,
                error_type=ErrorType.RATE_LIMIT,
                remediation=f"Wait {int(retry_after)} seconds before retrying.",
                request_id=rid,
                details={
                    "action": action_key,
                    "scope": rate_limit_scope,
                    "retry_after": int(retry_after),
                },
            )
        )

    # Authorization check (after rate limit, before dispatch)
    authz_result = check_action_allowed(current_role, tool_name, action)

    if not authz_result.allowed:
        rid = request_id or build_request_id(tool_name)
        logger.warning(
            "Authorization denied for %s.%s: role=%s, required=%s",
            tool_name,
            action,
            current_role,
            authz_result.required_role,
        )

        # Record denial for rate limiting
        rate_limit_tracker.record_denial(rate_limit_key)

        # Emit authz.denied metric
        metrics = MetricsCollector(prefix="authz")
        metrics.counter(
            "denied",
            labels={
                "role": current_role,
                "tool": tool_name,
                "action": action,
                "scope": rate_limit_scope,
            },
        )

        # Build recovery action guidance
        if authz_result.required_role:
            recovery = (
                f"This action requires '{authz_result.required_role}' role. "
                f"Current role is '{current_role}'. "
                f"Set FOUNDRY_MCP_ROLE environment variable or configure role in settings."
            )
        else:
            recovery = f"Role '{current_role}' is not authorized for this action. Contact administrator for access."

        return asdict(
            error_response(
                f"Authorization denied: role '{current_role}' cannot perform '{tool_name}.{action}'",
                error_code=ErrorCode.AUTHORIZATION,
                error_type=ErrorType.AUTHORIZATION,
                remediation=recovery,
                request_id=rid,
                details={
                    "role": current_role,
                    "action": f"{tool_name}.{action}",
                    "required_role": authz_result.required_role,
                    "recovery_action": recovery,
                },
            )
        )

    # Authorized - reset rate limit counter and proceed with dispatch
    rate_limit_tracker.reset(rate_limit_key)

    try:
        return router.dispatch(action=action, **kwargs)
    except ActionRouterError as exc:
        rid = request_id or build_request_id(tool_name)
        allowed = ", ".join(exc.allowed_actions)
        details = None
        if include_details_in_router_error:
            details = {"action": action, "allowed_actions": list(exc.allowed_actions)}
        return asdict(
            error_response(
                f"Unsupported {tool_name} action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                request_id=rid,
                details=details,
            )
        )
    except PathValidationError as exc:
        rid = request_id or build_request_id(tool_name)
        logger.warning(
            "%s action '%s' rejected: path validation failed: %s",
            tool_name.capitalize(),
            action,
            exc.reason,
        )
        return asdict(
            error_response(
                f"Invalid workspace path: {exc.reason}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Provide an absolute path without '..' components.",
                request_id=rid,
                details={
                    "action": action,
                    "reason": exc.reason,
                    "path": exc.path,
                },
            )
        )
    except Exception as exc:
        logger.exception(
            "%s action '%s' failed with unexpected error: %s",
            tool_name.capitalize(),
            action,
            exc,
        )
        error_msg = str(exc) if str(exc) else exc.__class__.__name__
        return asdict(
            error_response(
                f"{tool_name.capitalize()} action '{action}' failed: {error_msg}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check configuration and logs for details.",
                details={"action": action, "error_type": exc.__class__.__name__},
            )
        )


# ---------------------------------------------------------------------------
# 5. Validation error factory
# ---------------------------------------------------------------------------


def make_validation_error_fn(
    tool_name: str,
    *,
    include_request_id: bool = True,
    default_code: ErrorCode = ErrorCode.VALIDATION_ERROR,
) -> Callable[..., dict]:
    """Return a validation-error builder pre-bound to *tool_name*.

    The returned callable has the signature::

        validation_error(
            *,
            field: str,
            action: str,
            message: str,
            request_id: str | None = None,
            code: ErrorCode = ErrorCode.VALIDATION_ERROR,
            remediation: str | None = None,
        ) -> dict

    When *include_request_id* is ``True`` (default) and no *request_id* is
    passed, one is generated automatically.
    """

    def _validation_error(
        *,
        field: str,
        action: str,
        message: str,
        request_id: Optional[str] = None,
        code: ErrorCode = default_code,
        remediation: Optional[str] = None,
    ) -> dict:
        effective_remediation = remediation or f"Provide a valid '{field}' value"
        rid = request_id
        if rid is None and include_request_id:
            rid = build_request_id(tool_name)
        return asdict(
            error_response(
                f"Invalid field '{field}' for {tool_name}.{action}: {message}",
                error_code=code,
                error_type=ErrorType.VALIDATION,
                remediation=effective_remediation,
                details={"field": field, "action": f"{tool_name}.{action}"},
                request_id=rid,
            )
        )

    return _validation_error
