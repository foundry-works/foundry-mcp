"""Unified health tool with action routing."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.health import (
    HealthStatus,
    check_health,
    check_liveness,
    check_readiness,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.prometheus import get_prometheus_exporter
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)


_ACTION_SUMMARY = {
    "liveness": "Fast liveness probe for orchestrators",
    "readiness": "Dependency-aware readiness probe",
    "check": "Full health report with dependency details",
}


def _status_to_int(status: HealthStatus) -> int:
    """Convert HealthStatus to Prometheus-friendly integer."""

    return {"unhealthy": 0, "degraded": 1, "healthy": 2}.get(status.value, 0)


def _record_batch_metrics(
    check_type: str,
    status: HealthStatus,
    duration: float,
    dependencies: Dict[str, bool] | None = None,
) -> None:
    exporter = get_prometheus_exporter()

    if dependencies is None:
        exporter.record_health_check(
            check_type=check_type,
            status=_status_to_int(status),
            duration_seconds=duration,
        )
        return

    exporter.record_health_check_batch(
        check_type=check_type,
        status=_status_to_int(status),
        dependencies=dependencies,
        duration_seconds=duration,
    )


def perform_health_liveness() -> dict:
    """Execute the liveness check and return serialized response."""

    start_time = time.perf_counter()
    try:
        result = check_liveness()
        duration = time.perf_counter() - start_time
        _record_batch_metrics("liveness", result.status, duration)

        return asdict(success_response(data=result.to_dict()))
    except Exception as exc:  # pragma: no cover - defensive safeguard
        logger.exception("Error during liveness check")
        return asdict(
            error_response(
                f"Liveness check failed: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check server logs and retry.",
                details={"check_type": "liveness"},
            )
        )


def perform_health_readiness() -> dict:
    """Execute the readiness check."""

    start_time = time.perf_counter()
    try:
        result = check_readiness()
        duration = time.perf_counter() - start_time
        deps = {dep.name: dep.healthy for dep in result.dependencies}
        _record_batch_metrics("readiness", result.status, duration, deps)

        return asdict(success_response(data=result.to_dict()))
    except Exception as exc:  # pragma: no cover - defensive safeguard
        logger.exception("Error during readiness check")
        return asdict(
            error_response(
                f"Readiness check failed: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check server logs and retry.",
                details={"check_type": "readiness"},
            )
        )


def perform_health_check(include_details: bool = True) -> dict:
    """Execute the full health check."""

    start_time = time.perf_counter()
    try:
        result = check_health()
        duration = time.perf_counter() - start_time
        deps = {dep.name: dep.healthy for dep in result.dependencies}
        _record_batch_metrics("health", result.status, duration, deps)

        data = result.to_dict()
        if not include_details:
            data.pop("dependencies", None)

        return asdict(success_response(data=data))
    except Exception as exc:  # pragma: no cover - defensive safeguard
        logger.exception("Error during health check")
        return asdict(
            error_response(
                f"Health check failed: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check server logs and retry.",
                details={"check_type": "check"},
            )
        )


def _handle_liveness(_: Any = None) -> dict:
    return perform_health_liveness()


def _handle_readiness(_: Any = None) -> dict:
    return perform_health_readiness()


def _handle_check(*, include_details: bool = True) -> dict:
    return perform_health_check(include_details=include_details)


def _build_router() -> ActionRouter:
    definitions = [
        ActionDefinition(
            name="liveness",
            handler=_handle_liveness,
            summary=_ACTION_SUMMARY["liveness"],
        ),
        ActionDefinition(
            name="readiness",
            handler=_handle_readiness,
            summary=_ACTION_SUMMARY["readiness"],
        ),
        ActionDefinition(
            name="check",
            handler=_handle_check,
            summary=_ACTION_SUMMARY["check"],
        ),
    ]
    return ActionRouter(tool_name="health", actions=definitions)


_HEALTH_ROUTER = _build_router()


def _dispatch_health_action(action: str, *, include_details: bool = True) -> dict:
    try:
        kwargs: Dict[str, Any] = {}
        if action.lower() == "check":
            kwargs["include_details"] = include_details
        return _HEALTH_ROUTER.dispatch(action=action, **kwargs)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported health action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                details={"action": action, "allowed_actions": exc.allowed_actions},
            )
        )


def register_unified_health_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated health tool."""

    @canonical_tool(
        mcp,
        canonical_name="health",
    )
    def health(action: str, include_details: bool = True) -> dict:
        """Run health checks via `action` parameter.

        Args:
            action: One of "liveness", "readiness", or "check".
            include_details: When action is "check", controls dependency output.
        """

        return _dispatch_health_action(action=action, include_details=include_details)

    logger.debug("Registered unified health tool")


__all__ = [
    "register_unified_health_tool",
]
