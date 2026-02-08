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

from foundry_mcp.core.context import generate_correlation_id, get_correlation_id
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
)
from foundry_mcp.core.spec import find_specs_directory
from foundry_mcp.tools.unified.router import ActionRouter, ActionRouterError

logger = logging.getLogger(__name__)


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
                remediation="Set SDD_SPECS_DIR or provide a workspace path",
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

    Catches :class:`ActionRouterError` (unsupported action) and generic
    ``Exception`` and returns a well-formed error response dict.
    """
    try:
        return router.dispatch(action=action, **kwargs)
    except ActionRouterError as exc:
        rid = request_id or build_request_id(tool_name)
        allowed = ", ".join(exc.allowed_actions)
        details: Optional[Dict[str, Any]] = None
        if include_details_in_router_error:
            details = {"action": action, "allowed_actions": list(exc.allowed_actions)}
        return asdict(
            error_response(
                f"Unsupported {tool_name} action '{action}'. "
                f"Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                request_id=rid,
                details=details,
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
