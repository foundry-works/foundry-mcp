"""Domain-specific configuration dataclasses.

Contains small, focused configuration classes for distinct infrastructure
domains: git, observability, health checks, error collection, metrics
persistence, and test runners.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from foundry_mcp.config.parsing import _parse_bool


@dataclass
class GitSettings:
    """Git workflow preferences for CLI + MCP surfaces."""

    enabled: bool = False
    auto_commit: bool = False
    auto_push: bool = False
    auto_pr: bool = False
    commit_cadence: str = "manual"
    show_before_commit: bool = True


@dataclass
class ObservabilityConfig:
    """Configuration for observability stack (OTel + Prometheus).

    Attributes:
        enabled: Master switch for all observability features
        otel_enabled: Enable OpenTelemetry tracing and metrics
        otel_endpoint: OTLP exporter endpoint
        otel_service_name: Service name for traces
        otel_sample_rate: Trace sampling rate (0.0 to 1.0)
        prometheus_enabled: Enable Prometheus metrics
        prometheus_port: HTTP server port for /metrics (0 = no server)
        prometheus_host: HTTP server host
        prometheus_namespace: Metric namespace prefix
    """

    enabled: bool = False
    otel_enabled: bool = False
    otel_endpoint: str = "localhost:4317"
    otel_service_name: str = "foundry-mcp"
    otel_sample_rate: float = 1.0
    prometheus_enabled: bool = False
    prometheus_port: int = 0
    prometheus_host: str = "0.0.0.0"
    prometheus_namespace: str = "foundry_mcp"

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "ObservabilityConfig":
        """Create config from TOML dict (typically [observability] section).

        Args:
            data: Dict from TOML parsing

        Returns:
            ObservabilityConfig instance
        """
        return cls(
            enabled=_parse_bool(data.get("enabled", False)),
            otel_enabled=_parse_bool(data.get("otel_enabled", False)),
            otel_endpoint=str(data.get("otel_endpoint", "localhost:4317")),
            otel_service_name=str(data.get("otel_service_name", "foundry-mcp")),
            otel_sample_rate=float(data.get("otel_sample_rate", 1.0)),
            prometheus_enabled=_parse_bool(data.get("prometheus_enabled", False)),
            prometheus_port=int(data.get("prometheus_port", 0)),
            prometheus_host=str(data.get("prometheus_host", "0.0.0.0")),
            prometheus_namespace=str(data.get("prometheus_namespace", "foundry_mcp")),
        )


@dataclass
class HealthConfig:
    """Configuration for health checks and probes.

    Attributes:
        enabled: Whether health checks are enabled
        liveness_timeout: Timeout for liveness checks (seconds)
        readiness_timeout: Timeout for readiness checks (seconds)
        health_timeout: Timeout for full health checks (seconds)
        disk_space_threshold_mb: Minimum disk space (MB) before unhealthy
        disk_space_warning_mb: Minimum disk space (MB) before degraded
    """

    enabled: bool = True
    liveness_timeout: float = 1.0
    readiness_timeout: float = 5.0
    health_timeout: float = 10.0
    disk_space_threshold_mb: int = 100
    disk_space_warning_mb: int = 500

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "HealthConfig":
        """Create config from TOML dict (typically [health] section).

        Args:
            data: Dict from TOML parsing

        Returns:
            HealthConfig instance
        """
        return cls(
            enabled=_parse_bool(data.get("enabled", True)),
            liveness_timeout=float(data.get("liveness_timeout", 1.0)),
            readiness_timeout=float(data.get("readiness_timeout", 5.0)),
            health_timeout=float(data.get("health_timeout", 10.0)),
            disk_space_threshold_mb=int(data.get("disk_space_threshold_mb", 100)),
            disk_space_warning_mb=int(data.get("disk_space_warning_mb", 500)),
        )


@dataclass
class ErrorCollectionConfig:
    """Configuration for error data collection infrastructure.

    Attributes:
        enabled: Whether error collection is enabled
        storage_path: Directory path for error storage (default: ~/.foundry-mcp/errors)
        retention_days: Delete records older than this many days
        max_errors: Maximum number of error records to keep
        include_stack_traces: Whether to include stack traces in error records
        redact_inputs: Whether to redact sensitive data from input parameters
    """

    enabled: bool = True
    storage_path: str = ""  # Empty string means use default
    retention_days: int = 30
    max_errors: int = 10000
    include_stack_traces: bool = True
    redact_inputs: bool = True

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "ErrorCollectionConfig":
        """Create config from TOML dict (typically [error_collection] section).

        Args:
            data: Dict from TOML parsing

        Returns:
            ErrorCollectionConfig instance
        """
        return cls(
            enabled=_parse_bool(data.get("enabled", True)),
            storage_path=str(data.get("storage_path", "")),
            retention_days=int(data.get("retention_days", 30)),
            max_errors=int(data.get("max_errors", 10000)),
            include_stack_traces=_parse_bool(data.get("include_stack_traces", True)),
            redact_inputs=_parse_bool(data.get("redact_inputs", True)),
        )

    def get_storage_path(self) -> Path:
        """Get the resolved storage path.

        Returns:
            Path to error storage directory
        """
        if self.storage_path:
            return Path(self.storage_path).expanduser()
        return Path.home() / ".foundry-mcp" / "errors"


@dataclass
class MetricsPersistenceConfig:
    """Configuration for metrics persistence infrastructure.

    Persists time-series metrics to disk so they survive server restarts.
    Metrics are aggregated into time buckets before storage to reduce
    disk usage while maintaining useful historical data.

    Attributes:
        enabled: Whether metrics persistence is enabled
        storage_path: Directory path for metrics storage (default: ~/.foundry-mcp/metrics)
        retention_days: Delete records older than this many days
        max_records: Maximum number of metric data points to keep
        bucket_interval_seconds: Aggregation bucket interval (default: 60s = 1 minute)
        flush_interval_seconds: How often to flush buffer to disk (default: 30s)
        persist_metrics: List of metric names to persist (empty = persist all)
    """

    enabled: bool = False
    storage_path: str = ""  # Empty string means use default
    retention_days: int = 7
    max_records: int = 100000
    bucket_interval_seconds: int = 60
    flush_interval_seconds: int = 30
    persist_metrics: List[str] = field(
        default_factory=lambda: [
            "tool_invocations_total",
            "tool_duration_seconds",
            "tool_errors_total",
            "health_status",
        ]
    )

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "MetricsPersistenceConfig":
        """Create config from TOML dict (typically [metrics_persistence] section).

        Args:
            data: Dict from TOML parsing

        Returns:
            MetricsPersistenceConfig instance
        """
        persist_metrics = data.get(
            "persist_metrics",
            [
                "tool_invocations_total",
                "tool_duration_seconds",
                "tool_errors_total",
                "health_status",
            ],
        )
        # Handle both list and comma-separated string
        if isinstance(persist_metrics, str):
            persist_metrics = [m.strip() for m in persist_metrics.split(",") if m.strip()]

        return cls(
            enabled=_parse_bool(data.get("enabled", False)),
            storage_path=str(data.get("storage_path", "")),
            retention_days=int(data.get("retention_days", 7)),
            max_records=int(data.get("max_records", 100000)),
            bucket_interval_seconds=int(data.get("bucket_interval_seconds", 60)),
            flush_interval_seconds=int(data.get("flush_interval_seconds", 30)),
            persist_metrics=persist_metrics,
        )

    def get_storage_path(self) -> Path:
        """Get the resolved storage path.

        Returns:
            Path to metrics storage directory
        """
        if self.storage_path:
            return Path(self.storage_path).expanduser()
        return Path.home() / ".foundry-mcp" / "metrics"

    def should_persist_metric(self, metric_name: str) -> bool:
        """Check if a metric should be persisted.

        Args:
            metric_name: Name of the metric

        Returns:
            True if the metric should be persisted
        """
        # Empty list means persist all metrics
        if not self.persist_metrics:
            return True
        return metric_name in self.persist_metrics


@dataclass
class RunnerConfig:
    """Configuration for a test runner (pytest, go, npm, etc.).

    Attributes:
        command: Command to execute (e.g., ["go", "test"] or ["python", "-m", "pytest"])
        run_args: Additional arguments for running tests
        discover_args: Arguments for test discovery
        pattern: File pattern for test discovery (e.g., "*_test.go", "test_*.py")
        timeout: Default timeout in seconds
    """

    command: List[str] = field(default_factory=list)
    run_args: List[str] = field(default_factory=list)
    discover_args: List[str] = field(default_factory=list)
    pattern: str = "*"
    timeout: int = 300

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "RunnerConfig":
        """Create config from TOML dict.

        Args:
            data: Dict from TOML parsing

        Returns:
            RunnerConfig instance
        """
        command = data.get("command", [])
        # Handle string command (convert to list)
        if isinstance(command, str):
            command = command.split()

        run_args = data.get("run_args", [])
        if isinstance(run_args, str):
            run_args = run_args.split()

        discover_args = data.get("discover_args", [])
        if isinstance(discover_args, str):
            discover_args = discover_args.split()

        return cls(
            command=command,
            run_args=run_args,
            discover_args=discover_args,
            pattern=str(data.get("pattern", "*")),
            timeout=int(data.get("timeout", 300)),
        )


@dataclass
class TestConfig:
    """Configuration for test runners.

    Supports multiple test runners (pytest, go, npm, etc.) with configurable
    commands and arguments. Runners can be defined in TOML config and selected
    at runtime via the 'runner' parameter.

    Attributes:
        default_runner: Default runner to use when none specified
        runners: Dict of runner name to RunnerConfig
    """

    default_runner: str = "pytest"
    runners: Dict[str, RunnerConfig] = field(default_factory=dict)

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "TestConfig":
        """Create config from TOML dict (typically [test] section).

        Args:
            data: Dict from TOML parsing

        Returns:
            TestConfig instance
        """
        runners = {}
        runners_data = data.get("runners", {})
        for name, runner_data in runners_data.items():
            runners[name] = RunnerConfig.from_toml_dict(runner_data)

        return cls(
            default_runner=str(data.get("default_runner", "pytest")),
            runners=runners,
        )

    def get_runner(self, name: Optional[str] = None) -> Optional[RunnerConfig]:
        """Get runner config by name.

        Args:
            name: Runner name, or None to use default

        Returns:
            RunnerConfig if found, None otherwise
        """
        runner_name = name or self.default_runner
        return self.runners.get(runner_name)
