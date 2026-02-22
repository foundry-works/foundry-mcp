"""
Observability utilities for foundry-mcp.

Provides structured logging, metrics collection, and audit logging
for MCP tools and resources.

FastMCP Middleware Integration:
    The decorators in this module can be applied to FastMCP tool and resource
    handlers to provide consistent observability. Example:

        from fastmcp import FastMCP
        from foundry_mcp.core.observability import mcp_tool, audit_log

        mcp = FastMCP("foundry-mcp")

        @mcp.tool()
        @mcp_tool(tool_name="list_specs")
        async def list_specs(status: str = "all") -> str:
            audit_log("tool_invocation", tool="list_specs", status=status)
            # ... implementation
            return result

    For resources, use the mcp_resource decorator:

        @mcp.resource("specs://{spec_id}")
        @mcp_resource(resource_type="spec")
        async def get_spec(spec_id: str) -> str:
            # ... implementation
            return spec_data
"""

from foundry_mcp.core.observability.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    audit_log,
    get_audit_logger,
)
from foundry_mcp.core.observability.decorators import (
    mcp_resource,
    mcp_tool,
)
from foundry_mcp.core.observability.manager import (
    ObservabilityManager,
    get_observability_manager,
    get_observability_status,
)
from foundry_mcp.core.observability.metrics import (
    Metric,
    MetricsCollector,
    MetricType,
    get_metrics,
)
from foundry_mcp.core.observability.redaction import (
    SENSITIVE_PATTERNS,
    redact_for_logging,
    redact_sensitive_data,
)

__all__ = [
    # Manager
    "ObservabilityManager",
    "get_observability_manager",
    "get_observability_status",
    # Metrics
    "Metric",
    "MetricType",
    "MetricsCollector",
    "get_metrics",
    # Audit
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "audit_log",
    "get_audit_logger",
    # Redaction
    "SENSITIVE_PATTERNS",
    "redact_for_logging",
    "redact_sensitive_data",
    # Decorators
    "mcp_resource",
    "mcp_tool",
]
