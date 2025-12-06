"""Health check tools for foundry-mcp.

Provides MCP tools for Kubernetes-style health probes (liveness, readiness, health).
These tools expose the health check system via MCP for monitoring and orchestration.
"""

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.health import (
    HealthStatus,
    check_health,
    check_liveness,
    check_readiness,
    get_health_manager,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.prometheus import get_prometheus_exporter
from foundry_mcp.core.responses import error_response, success_response

logger = logging.getLogger(__name__)


def _status_to_int(status: HealthStatus) -> int:
    """Convert HealthStatus to integer for Prometheus metrics.

    Returns:
        0 for unhealthy, 1 for degraded, 2 for healthy
    """
    return {"unhealthy": 0, "degraded": 1, "healthy": 2}.get(status.value, 0)


def register_health_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """Register health check tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="health-liveness",
    )
    def health_liveness() -> dict:
        """
        Check if the server is alive (liveness probe).

        Simple alive check that returns quickly. Used by Kubernetes/orchestrators
        to determine if the process needs to be restarted.

        WHEN TO USE:
        - Kubernetes liveness probes
        - Basic "is it running" checks
        - Process health monitoring

        Returns:
            JSON object with liveness status:
            - status: "healthy", "degraded", or "unhealthy"
            - is_healthy: Boolean indicating overall health
            - message: Human-readable status message
            - timestamp: Unix timestamp of check
        """
        try:
            start_time = time.perf_counter()
            result = check_liveness()
            duration = time.perf_counter() - start_time

            # Record metrics
            exporter = get_prometheus_exporter()
            exporter.record_health_check(
                check_type="liveness",
                status=_status_to_int(result.status),
                duration_seconds=duration,
            )

            return asdict(
                success_response(
                    data=result.to_dict(),
                )
            )

        except Exception as e:
            logger.exception("Error during liveness check")
            return asdict(
                error_response(
                    f"Liveness check failed: {e}",
                    error_code="HEALTH_CHECK_ERROR",
                    error_type="internal",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="health-readiness",
    )
    def health_readiness() -> dict:
        """
        Check if the server is ready to handle requests (readiness probe).

        Verifies critical dependencies are available (specs directory, disk space).
        Used by load balancers/orchestrators to determine if traffic should be routed.

        WHEN TO USE:
        - Kubernetes readiness probes
        - Load balancer health checks
        - Pre-flight checks before operations

        Returns:
            JSON object with readiness status:
            - status: "healthy", "degraded", or "unhealthy"
            - is_healthy: Boolean indicating overall health
            - message: Human-readable status message
            - timestamp: Unix timestamp of check
            - dependencies: List of dependency check results
        """
        try:
            start_time = time.perf_counter()
            result = check_readiness()
            duration = time.perf_counter() - start_time

            # Record metrics
            exporter = get_prometheus_exporter()

            # Build dependency map for metrics
            deps: Dict[str, bool] = {}
            for dep in result.dependencies:
                deps[dep.name] = dep.healthy

            exporter.record_health_check_batch(
                check_type="readiness",
                status=_status_to_int(result.status),
                dependencies=deps,
                duration_seconds=duration,
            )

            return asdict(
                success_response(
                    data=result.to_dict(),
                )
            )

        except Exception as e:
            logger.exception("Error during readiness check")
            return asdict(
                error_response(
                    f"Readiness check failed: {e}",
                    error_code="HEALTH_CHECK_ERROR",
                    error_type="internal",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="health-check",
    )
    def health_check(
        include_details: bool = True,
    ) -> dict:
        """
        Perform a full health check of all dependencies.

        Comprehensive health check that verifies all system components:
        - Specs directory accessibility
        - Disk space availability
        - OpenTelemetry status
        - Prometheus metrics status
        - AI provider availability

        WHEN TO USE:
        - Detailed system health inspection
        - Troubleshooting connectivity/configuration issues
        - Dashboard/monitoring integrations
        - Pre-deployment verification

        Args:
            include_details: Include detailed breakdown of each dependency (default: True)

        Returns:
            JSON object with full health status:
            - status: "healthy", "degraded", or "unhealthy"
            - is_healthy: Boolean indicating overall health
            - message: Human-readable status message
            - timestamp: Unix timestamp of check
            - dependencies: List of dependency check results (if include_details)
            - details: Aggregate counts (healthy, degraded, unhealthy)
        """
        try:
            start_time = time.perf_counter()
            result = check_health()
            duration = time.perf_counter() - start_time

            # Record metrics
            exporter = get_prometheus_exporter()

            # Build dependency map for metrics
            deps: Dict[str, bool] = {}
            for dep in result.dependencies:
                deps[dep.name] = dep.healthy

            exporter.record_health_check_batch(
                check_type="health",
                status=_status_to_int(result.status),
                dependencies=deps,
                duration_seconds=duration,
            )

            # Build response data
            data = result.to_dict()
            if not include_details:
                # Remove detailed dependency info for lighter response
                data.pop("dependencies", None)

            return asdict(
                success_response(
                    data=data,
                )
            )

        except Exception as e:
            logger.exception("Error during health check")
            return asdict(
                error_response(
                    f"Health check failed: {e}",
                    error_code="HEALTH_CHECK_ERROR",
                    error_type="internal",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="health-config",
    )
    def health_config() -> dict:
        """
        Get health check configuration.

        Returns the current health check settings including timeouts
        and thresholds.

        WHEN TO USE:
        - Debugging health check behavior
        - Verifying configuration
        - Understanding timeout settings

        Returns:
            JSON object with health configuration:
            - enabled: Whether health checks are enabled
            - liveness_timeout: Timeout for liveness checks (seconds)
            - readiness_timeout: Timeout for readiness checks (seconds)
            - health_timeout: Timeout for full health checks (seconds)
            - disk_space_threshold_mb: Critical disk space threshold
            - disk_space_warning_mb: Warning disk space threshold
        """
        try:
            manager = get_health_manager()
            config_data = {
                "enabled": manager.config.enabled,
                "liveness_timeout": manager.config.liveness_timeout,
                "readiness_timeout": manager.config.readiness_timeout,
                "health_timeout": manager.config.health_timeout,
                "disk_space_threshold_mb": manager.config.disk_space_threshold_mb,
                "disk_space_warning_mb": manager.config.disk_space_warning_mb,
            }

            return asdict(
                success_response(
                    data=config_data,
                )
            )

        except Exception as e:
            logger.exception("Error getting health config")
            return asdict(
                error_response(
                    f"Failed to get health config: {e}",
                    error_code="CONFIG_ERROR",
                    error_type="internal",
                )
            )
