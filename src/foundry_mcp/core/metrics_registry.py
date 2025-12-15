"""Metrics registry for foundry-mcp observability stack.

This module provides a centralized catalog of all metrics emitted by foundry-mcp,
including their definitions, types, labels, and documentation. Use this for:
- Consistent metric naming across the codebase
- Automatic documentation generation
- Metric discovery and introspection

Usage:
    from foundry_mcp.core.metrics_registry import (
        get_metrics_catalog,
        get_metrics_by_category,
        MetricCategory,
    )

    # Get all metrics
    catalog = get_metrics_catalog()

    # Get tool-related metrics
    tool_metrics = get_metrics_by_category(MetricCategory.TOOL)

    # Generate markdown documentation
    docs = export_catalog_as_markdown()
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MetricCategory(Enum):
    """Categories for grouping related metrics."""

    TOOL = "tool"
    RESOURCE = "resource"
    AUDIT = "audit"
    ERROR = "error"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    CLI = "cli"


class MetricType(Enum):
    """Types of metrics supported."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a metric including its metadata.

    Attributes:
        name: Full metric name (e.g., "foundry_mcp_tool_invocations_total")
        description: Human-readable description
        metric_type: Type of metric (counter, gauge, histogram, summary)
        category: Category for grouping
        labels: List of label names
        unit: Unit of measurement (optional)
        examples: Example label values for documentation
    """

    name: str
    description: str
    metric_type: MetricType
    category: MetricCategory
    labels: List[str] = field(default_factory=list)
    unit: Optional[str] = None
    examples: Dict[str, str] = field(default_factory=dict)

    def full_name(self, namespace: str = "foundry_mcp") -> str:
        """Get full metric name with namespace prefix."""
        return f"{namespace}_{self.name}"


# =============================================================================
# Metrics Catalog
# =============================================================================

METRICS_CATALOG: Dict[str, MetricDefinition] = {
    # Tool metrics
    "tool_invocations_total": MetricDefinition(
        name="tool_invocations_total",
        description="Total number of tool invocations",
        metric_type=MetricType.COUNTER,
        category=MetricCategory.TOOL,
        labels=["tool", "status"],
        examples={"tool": "list_specs", "status": "success"},
    ),
    "tool_duration_seconds": MetricDefinition(
        name="tool_duration_seconds",
        description="Tool execution duration in seconds",
        metric_type=MetricType.HISTOGRAM,
        category=MetricCategory.TOOL,
        labels=["tool"],
        unit="seconds",
        examples={"tool": "task_complete"},
    ),
    "tool_errors_total": MetricDefinition(
        name="tool_errors_total",
        description="Total number of tool errors",
        metric_type=MetricType.COUNTER,
        category=MetricCategory.TOOL,
        labels=["tool", "error_type"],
        examples={"tool": "spec", "error_type": "NotFoundError"},
    ),
    "active_operations": MetricDefinition(
        name="active_operations",
        description="Number of currently active operations",
        metric_type=MetricType.GAUGE,
        category=MetricCategory.TOOL,
        labels=["operation_type"],
        examples={"operation_type": "tool:list"},
    ),
    # Resource metrics
    "resource_access_total": MetricDefinition(
        name="resource_access_total",
        description="Total number of resource accesses",
        metric_type=MetricType.COUNTER,
        category=MetricCategory.RESOURCE,
        labels=["resource_type", "action"],
        examples={"resource_type": "spec", "action": "read"},
    ),
    "resource_latency_seconds": MetricDefinition(
        name="resource_latency_seconds",
        description="Resource access latency in seconds",
        metric_type=MetricType.HISTOGRAM,
        category=MetricCategory.RESOURCE,
        labels=["resource_type"],
        unit="seconds",
        examples={"resource_type": "journal"},
    ),
    # Error metrics
    "errors_total": MetricDefinition(
        name="errors_total",
        description="Total number of errors by type",
        metric_type=MetricType.COUNTER,
        category=MetricCategory.ERROR,
        labels=["error_type", "component"],
        examples={"error_type": "ValidationError", "component": "spec"},
    ),
    # System metrics
    "info": MetricDefinition(
        name="info",
        description="Server information gauge (always 1)",
        metric_type=MetricType.GAUGE,
        category=MetricCategory.SYSTEM,
        labels=["version", "python_version"],
        examples={"version": "0.2.0", "python_version": "3.11"},
    ),
    "startup_time_seconds": MetricDefinition(
        name="startup_time_seconds",
        description="Server startup time in seconds",
        metric_type=MetricType.GAUGE,
        category=MetricCategory.SYSTEM,
        unit="seconds",
    ),
    # CLI metrics
    "cli_command_invocations_total": MetricDefinition(
        name="cli_command_invocations_total",
        description="Total CLI command invocations",
        metric_type=MetricType.COUNTER,
        category=MetricCategory.CLI,
        labels=["command", "status"],
        examples={"command": "validate", "status": "success"},
    ),
    "cli_command_duration_seconds": MetricDefinition(
        name="cli_command_duration_seconds",
        description="CLI command execution duration in seconds",
        metric_type=MetricType.HISTOGRAM,
        category=MetricCategory.CLI,
        labels=["command"],
        unit="seconds",
        examples={"command": "test"},
    ),
    # Performance metrics
    "spec_validation_duration_seconds": MetricDefinition(
        name="spec_validation_duration_seconds",
        description="Spec validation duration in seconds",
        metric_type=MetricType.HISTOGRAM,
        category=MetricCategory.PERFORMANCE,
        labels=["spec_id"],
        unit="seconds",
    ),
    "ai_consultation_duration_seconds": MetricDefinition(
        name="ai_consultation_duration_seconds",
        description="AI consultation request duration in seconds",
        metric_type=MetricType.HISTOGRAM,
        category=MetricCategory.PERFORMANCE,
        labels=["provider", "workflow"],
        unit="seconds",
        examples={"provider": "gemini", "workflow": "fidelity_review"},
    ),
    "ai_consultation_tokens_total": MetricDefinition(
        name="ai_consultation_tokens_total",
        description="Total tokens used in AI consultations",
        metric_type=MetricType.COUNTER,
        category=MetricCategory.PERFORMANCE,
        labels=["provider", "token_type"],
        examples={"provider": "openai", "token_type": "completion"},
    ),
    # Audit metrics
    "audit_events_total": MetricDefinition(
        name="audit_events_total",
        description="Total audit events by type",
        metric_type=MetricType.COUNTER,
        category=MetricCategory.AUDIT,
        labels=["event_type"],
        examples={"event_type": "tool_invocation"},
    ),
}


# =============================================================================
# Public API
# =============================================================================


def get_metrics_catalog() -> Dict[str, MetricDefinition]:
    """Get the full metrics catalog.

    Returns:
        Dict mapping metric names to their definitions
    """
    return METRICS_CATALOG.copy()


def get_metrics_by_category(category: MetricCategory) -> Dict[str, MetricDefinition]:
    """Get metrics filtered by category.

    Args:
        category: Category to filter by

    Returns:
        Dict of metrics in the specified category
    """
    return {
        name: defn
        for name, defn in METRICS_CATALOG.items()
        if defn.category == category
    }


def get_metric(name: str) -> Optional[MetricDefinition]:
    """Get a specific metric definition by name.

    Args:
        name: Metric name (without namespace prefix)

    Returns:
        MetricDefinition if found, None otherwise
    """
    return METRICS_CATALOG.get(name)


def export_catalog_as_markdown(namespace: str = "foundry_mcp") -> str:
    """Export the metrics catalog as Markdown documentation.

    Args:
        namespace: Namespace prefix for full metric names

    Returns:
        Markdown-formatted documentation string
    """
    lines = [
        "# Foundry MCP Metrics Catalog",
        "",
        "This document lists all metrics emitted by foundry-mcp.",
        "",
    ]

    # Group by category
    for category in MetricCategory:
        category_metrics = get_metrics_by_category(category)
        if not category_metrics:
            continue

        lines.append(f"## {category.value.title()} Metrics")
        lines.append("")

        for name, defn in sorted(category_metrics.items()):
            full_name = defn.full_name(namespace)
            lines.append(f"### `{full_name}`")
            lines.append("")
            lines.append(f"**Type:** {defn.metric_type.value}")
            lines.append("")
            lines.append(defn.description)
            lines.append("")

            if defn.unit:
                lines.append(f"**Unit:** {defn.unit}")
                lines.append("")

            if defn.labels:
                lines.append("**Labels:**")
                for label in defn.labels:
                    example = defn.examples.get(label, "")
                    if example:
                        lines.append(f"- `{label}` (e.g., `{example}`)")
                    else:
                        lines.append(f"- `{label}`")
                lines.append("")

    return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "MetricCategory",
    "MetricType",
    # Types
    "MetricDefinition",
    # Data
    "METRICS_CATALOG",
    # Functions
    "get_metrics_catalog",
    "get_metrics_by_category",
    "get_metric",
    "export_catalog_as_markdown",
]
