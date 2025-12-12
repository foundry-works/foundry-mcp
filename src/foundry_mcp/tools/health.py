"""Health check tools for foundry-mcp.

Provides MCP tools for Kubernetes-style health probes (liveness, readiness, health).
These tools expose the health check system via MCP for monitoring and orchestration.
"""

import logging
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.health import get_health_manager
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.responses import error_response, success_response
from foundry_mcp.tools.unified.health import legacy_health_action

logger = logging.getLogger(__name__)


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
        return legacy_health_action(action="liveness")

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
        return legacy_health_action(action="readiness")

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
        return legacy_health_action(action="check", include_details=include_details)

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
