"""
Server configuration for foundry-mcp.

Supports configuration via:
1. Environment variables (highest priority)
2. TOML config file (foundry-mcp.toml)
3. Default values (lowest priority)

Environment variables:
- FOUNDRY_MCP_WORKSPACE_ROOTS: Comma-separated list of workspace root paths
- FOUNDRY_MCP_SPECS_DIR: Path to specs directory
- FOUNDRY_MCP_JOURNALS_PATH: Path to journals directory
- FOUNDRY_MCP_BIKELANE_DIR: Path to bikelane intake queue directory (default: specs/.bikelane)
- FOUNDRY_MCP_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
- FOUNDRY_MCP_API_KEYS: Comma-separated list of valid API keys (optional)
- FOUNDRY_MCP_REQUIRE_AUTH: Whether to require API key authentication (true/false)
- FOUNDRY_MCP_CONFIG_FILE: Path to TOML config file

API Key Security:
- Keys should be rotated regularly (recommended: every 90 days)
- To revoke a key: remove it from FOUNDRY_MCP_API_KEYS and restart server
- Keys are validated on every tool/resource request when auth is required
- Use tenant-scoped keys for multi-tenant deployments (prefix with tenant ID)
"""

import os
import logging
import functools
import time
from dataclasses import dataclass, field
from importlib.metadata import version as get_package_version, PackageNotFoundError
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, TypeVar

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback


logger = logging.getLogger(__name__)


def _get_version() -> str:
    """Get package version from metadata (single source of truth: pyproject.toml)."""
    try:
        return get_package_version("foundry-mcp")
    except PackageNotFoundError:
        return "0.5.0"  # Fallback for dev without install


_PACKAGE_VERSION = _get_version()

T = TypeVar("T")


@dataclass
class GitSettings:
    """Git workflow preferences for CLI + MCP surfaces."""

    enabled: bool = False
    auto_branch: bool = False
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
    persist_metrics: List[str] = field(default_factory=lambda: [
        "tool_invocations_total",
        "tool_duration_seconds",
        "tool_errors_total",
        "health_status",
    ])

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "MetricsPersistenceConfig":
        """Create config from TOML dict (typically [metrics_persistence] section).

        Args:
            data: Dict from TOML parsing

        Returns:
            MetricsPersistenceConfig instance
        """
        persist_metrics = data.get("persist_metrics", [
            "tool_invocations_total",
            "tool_duration_seconds",
            "tool_errors_total",
            "health_status",
        ])
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
class DashboardConfig:
    """Configuration for built-in web dashboard.

    The dashboard provides a web UI for viewing errors, metrics, and
    AI provider status without requiring external tools like Grafana.

    Attributes:
        enabled: Whether the dashboard server is enabled
        port: HTTP port for dashboard (default: 8080)
        host: Host to bind to (default: 127.0.0.1 for localhost only)
        auto_open_browser: Open browser when dashboard starts
        refresh_interval_ms: Auto-refresh interval in milliseconds
    """

    enabled: bool = False
    port: int = 8501  # Streamlit default port
    host: str = "127.0.0.1"
    auto_open_browser: bool = False
    refresh_interval_ms: int = 5000

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "DashboardConfig":
        """Create config from TOML dict (typically [dashboard] section).

        Args:
            data: Dict from TOML parsing

        Returns:
            DashboardConfig instance
        """
        return cls(
            enabled=_parse_bool(data.get("enabled", False)),
            port=int(data.get("port", 8501)),  # Streamlit default
            host=str(data.get("host", "127.0.0.1")),
            auto_open_browser=_parse_bool(data.get("auto_open_browser", False)),
            refresh_interval_ms=int(data.get("refresh_interval_ms", 5000)),
        )


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


_VALID_COMMIT_CADENCE = {"manual", "task", "phase"}


def _normalize_commit_cadence(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _VALID_COMMIT_CADENCE:
        logger.warning(
            "Invalid commit cadence '%s'. Falling back to 'manual'. Valid options: %s",
            value,
            ", ".join(sorted(_VALID_COMMIT_CADENCE)),
        )
        return "manual"
    return normalized


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


@dataclass
class ServerConfig:
    """Server configuration with support for env vars and TOML overrides."""

    # Workspace configuration
    workspace_roots: List[Path] = field(default_factory=list)
    specs_dir: Optional[Path] = None
    journals_path: Optional[Path] = None
    bikelane_dir: Optional[Path] = None  # Intake queue storage (default: specs/.bikelane)

    # Logging configuration
    log_level: str = "INFO"
    structured_logging: bool = True

    # Authentication configuration
    api_keys: List[str] = field(default_factory=list)
    require_auth: bool = False

    # Server configuration
    server_name: str = "foundry-mcp"
    server_version: str = field(default_factory=lambda: _PACKAGE_VERSION)

    # Git workflow configuration
    git: GitSettings = field(default_factory=GitSettings)

    # Observability configuration
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # Health check configuration
    health: HealthConfig = field(default_factory=HealthConfig)

    # Error collection configuration
    error_collection: ErrorCollectionConfig = field(default_factory=ErrorCollectionConfig)

    # Metrics persistence configuration
    metrics_persistence: MetricsPersistenceConfig = field(default_factory=MetricsPersistenceConfig)

    # Dashboard configuration
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    # Test runner configuration
    test: TestConfig = field(default_factory=TestConfig)

    @classmethod
    def from_env(cls, config_file: Optional[str] = None) -> "ServerConfig":
        """
        Create configuration from environment variables and optional TOML file.

        Priority (highest to lowest):
        1. Environment variables
        2. TOML config file
        3. Default values
        """
        config = cls()

        # Load TOML config if available
        toml_path = config_file or os.environ.get("FOUNDRY_MCP_CONFIG_FILE")
        if toml_path:
            config._load_toml(Path(toml_path))
        else:
            # Try default locations
            for default_path in ["foundry-mcp.toml", ".foundry-mcp.toml"]:
                if Path(default_path).exists():
                    config._load_toml(Path(default_path))
                    break

        # Override with environment variables
        config._load_env()

        return config

    def _load_toml(self, path: Path) -> None:
        """Load configuration from TOML file."""
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            # Workspace settings
            if "workspace" in data:
                ws = data["workspace"]
                if "roots" in ws:
                    self.workspace_roots = [Path(p) for p in ws["roots"]]
                if "specs_dir" in ws:
                    self.specs_dir = Path(ws["specs_dir"])
                if "journals_path" in ws:
                    self.journals_path = Path(ws["journals_path"])
                if "bikelane_dir" in ws:
                    self.bikelane_dir = Path(ws["bikelane_dir"])

            # Logging settings
            if "logging" in data:
                log = data["logging"]
                if "level" in log:
                    self.log_level = log["level"].upper()
                if "structured" in log:
                    self.structured_logging = log["structured"]

            # Auth settings
            if "auth" in data:
                auth = data["auth"]
                if "api_keys" in auth:
                    self.api_keys = auth["api_keys"]
                if "require_auth" in auth:
                    self.require_auth = auth["require_auth"]

            # Server settings
            if "server" in data:
                srv = data["server"]
                if "name" in srv:
                    self.server_name = srv["name"]
                if "version" in srv:
                    self.server_version = srv["version"]

            # Git workflow settings
            if "git" in data:
                git_cfg = data["git"]
                if "enabled" in git_cfg:
                    self.git.enabled = _parse_bool(git_cfg["enabled"])
                if "auto_branch" in git_cfg:
                    self.git.auto_branch = _parse_bool(git_cfg["auto_branch"])
                if "auto_commit" in git_cfg:
                    self.git.auto_commit = _parse_bool(git_cfg["auto_commit"])
                if "auto_push" in git_cfg:
                    self.git.auto_push = _parse_bool(git_cfg["auto_push"])
                if "auto_pr" in git_cfg:
                    self.git.auto_pr = _parse_bool(git_cfg["auto_pr"])
                if "show_before_commit" in git_cfg:
                    self.git.show_before_commit = _parse_bool(
                        git_cfg["show_before_commit"]
                    )
                if "commit_cadence" in git_cfg:
                    self.git.commit_cadence = _normalize_commit_cadence(
                        str(git_cfg["commit_cadence"])
                    )

            # Observability settings
            if "observability" in data:
                self.observability = ObservabilityConfig.from_toml_dict(
                    data["observability"]
                )

            # Health check settings
            if "health" in data:
                self.health = HealthConfig.from_toml_dict(data["health"])

            # Error collection settings
            if "error_collection" in data:
                self.error_collection = ErrorCollectionConfig.from_toml_dict(
                    data["error_collection"]
                )

            # Metrics persistence settings
            if "metrics_persistence" in data:
                self.metrics_persistence = MetricsPersistenceConfig.from_toml_dict(
                    data["metrics_persistence"]
                )

            # Dashboard settings
            if "dashboard" in data:
                self.dashboard = DashboardConfig.from_toml_dict(data["dashboard"])

            # Test runner settings
            if "test" in data:
                self.test = TestConfig.from_toml_dict(data["test"])

        except Exception as e:
            logger.error(f"Error loading config file {path}: {e}")

    def _load_env(self) -> None:
        """Load configuration from environment variables."""
        # Workspace roots
        if roots := os.environ.get("FOUNDRY_MCP_WORKSPACE_ROOTS"):
            self.workspace_roots = [Path(p.strip()) for p in roots.split(",")]

        # Specs directory
        if specs := os.environ.get("FOUNDRY_MCP_SPECS_DIR"):
            self.specs_dir = Path(specs)

        # Journals path
        if journals := os.environ.get("FOUNDRY_MCP_JOURNALS_PATH"):
            self.journals_path = Path(journals)

        # Bikelane directory (intake queue storage)
        if bikelane := os.environ.get("FOUNDRY_MCP_BIKELANE_DIR"):
            self.bikelane_dir = Path(bikelane)

        # Log level
        if level := os.environ.get("FOUNDRY_MCP_LOG_LEVEL"):
            self.log_level = level.upper()

        # API keys
        if keys := os.environ.get("FOUNDRY_MCP_API_KEYS"):
            self.api_keys = [k.strip() for k in keys.split(",") if k.strip()]

        # Require auth
        if require := os.environ.get("FOUNDRY_MCP_REQUIRE_AUTH"):
            self.require_auth = require.lower() in ("true", "1", "yes")

        # Git settings
        if git_enabled := os.environ.get("FOUNDRY_MCP_GIT_ENABLED"):
            self.git.enabled = _parse_bool(git_enabled)
        if git_auto_branch := os.environ.get("FOUNDRY_MCP_GIT_AUTO_BRANCH"):
            self.git.auto_branch = _parse_bool(git_auto_branch)
        if git_auto_commit := os.environ.get("FOUNDRY_MCP_GIT_AUTO_COMMIT"):
            self.git.auto_commit = _parse_bool(git_auto_commit)
        if git_auto_push := os.environ.get("FOUNDRY_MCP_GIT_AUTO_PUSH"):
            self.git.auto_push = _parse_bool(git_auto_push)
        if git_auto_pr := os.environ.get("FOUNDRY_MCP_GIT_AUTO_PR"):
            self.git.auto_pr = _parse_bool(git_auto_pr)
        if git_show_preview := os.environ.get("FOUNDRY_MCP_GIT_SHOW_PREVIEW"):
            self.git.show_before_commit = _parse_bool(git_show_preview)
        if git_cadence := os.environ.get("FOUNDRY_MCP_GIT_COMMIT_CADENCE"):
            self.git.commit_cadence = _normalize_commit_cadence(git_cadence)

        # Observability settings
        if obs_enabled := os.environ.get("FOUNDRY_MCP_OBSERVABILITY_ENABLED"):
            self.observability.enabled = _parse_bool(obs_enabled)
        if otel_enabled := os.environ.get("FOUNDRY_MCP_OTEL_ENABLED"):
            self.observability.otel_enabled = _parse_bool(otel_enabled)
        if otel_endpoint := os.environ.get("FOUNDRY_MCP_OTEL_ENDPOINT"):
            self.observability.otel_endpoint = otel_endpoint
        if otel_service := os.environ.get("FOUNDRY_MCP_OTEL_SERVICE_NAME"):
            self.observability.otel_service_name = otel_service
        if otel_sample := os.environ.get("FOUNDRY_MCP_OTEL_SAMPLE_RATE"):
            try:
                self.observability.otel_sample_rate = float(otel_sample)
            except ValueError:
                pass
        if prom_enabled := os.environ.get("FOUNDRY_MCP_PROMETHEUS_ENABLED"):
            self.observability.prometheus_enabled = _parse_bool(prom_enabled)
        if prom_port := os.environ.get("FOUNDRY_MCP_PROMETHEUS_PORT"):
            try:
                self.observability.prometheus_port = int(prom_port)
            except ValueError:
                pass
        if prom_host := os.environ.get("FOUNDRY_MCP_PROMETHEUS_HOST"):
            self.observability.prometheus_host = prom_host
        if prom_ns := os.environ.get("FOUNDRY_MCP_PROMETHEUS_NAMESPACE"):
            self.observability.prometheus_namespace = prom_ns

        # Health check settings
        if health_enabled := os.environ.get("FOUNDRY_MCP_HEALTH_ENABLED"):
            self.health.enabled = _parse_bool(health_enabled)
        if health_liveness_timeout := os.environ.get(
            "FOUNDRY_MCP_HEALTH_LIVENESS_TIMEOUT"
        ):
            try:
                self.health.liveness_timeout = float(health_liveness_timeout)
            except ValueError:
                pass
        if health_readiness_timeout := os.environ.get(
            "FOUNDRY_MCP_HEALTH_READINESS_TIMEOUT"
        ):
            try:
                self.health.readiness_timeout = float(health_readiness_timeout)
            except ValueError:
                pass
        if health_timeout := os.environ.get("FOUNDRY_MCP_HEALTH_TIMEOUT"):
            try:
                self.health.health_timeout = float(health_timeout)
            except ValueError:
                pass
        if disk_threshold := os.environ.get("FOUNDRY_MCP_DISK_SPACE_THRESHOLD_MB"):
            try:
                self.health.disk_space_threshold_mb = int(disk_threshold)
            except ValueError:
                pass
        if disk_warning := os.environ.get("FOUNDRY_MCP_DISK_SPACE_WARNING_MB"):
            try:
                self.health.disk_space_warning_mb = int(disk_warning)
            except ValueError:
                pass

        # Error collection settings
        if err_enabled := os.environ.get("FOUNDRY_MCP_ERROR_COLLECTION_ENABLED"):
            self.error_collection.enabled = _parse_bool(err_enabled)
        if err_storage := os.environ.get("FOUNDRY_MCP_ERROR_STORAGE_PATH"):
            self.error_collection.storage_path = err_storage
        if err_retention := os.environ.get("FOUNDRY_MCP_ERROR_RETENTION_DAYS"):
            try:
                self.error_collection.retention_days = int(err_retention)
            except ValueError:
                pass
        if err_max := os.environ.get("FOUNDRY_MCP_ERROR_MAX_ERRORS"):
            try:
                self.error_collection.max_errors = int(err_max)
            except ValueError:
                pass
        if err_stack := os.environ.get("FOUNDRY_MCP_ERROR_INCLUDE_STACK_TRACES"):
            self.error_collection.include_stack_traces = _parse_bool(err_stack)
        if err_redact := os.environ.get("FOUNDRY_MCP_ERROR_REDACT_INPUTS"):
            self.error_collection.redact_inputs = _parse_bool(err_redact)

        # Metrics persistence settings
        if metrics_enabled := os.environ.get("FOUNDRY_MCP_METRICS_PERSISTENCE_ENABLED"):
            self.metrics_persistence.enabled = _parse_bool(metrics_enabled)
        if metrics_storage := os.environ.get("FOUNDRY_MCP_METRICS_STORAGE_PATH"):
            self.metrics_persistence.storage_path = metrics_storage
        if metrics_retention := os.environ.get("FOUNDRY_MCP_METRICS_RETENTION_DAYS"):
            try:
                self.metrics_persistence.retention_days = int(metrics_retention)
            except ValueError:
                pass
        if metrics_max := os.environ.get("FOUNDRY_MCP_METRICS_MAX_RECORDS"):
            try:
                self.metrics_persistence.max_records = int(metrics_max)
            except ValueError:
                pass
        if metrics_bucket := os.environ.get("FOUNDRY_MCP_METRICS_BUCKET_INTERVAL"):
            try:
                self.metrics_persistence.bucket_interval_seconds = int(metrics_bucket)
            except ValueError:
                pass
        if metrics_flush := os.environ.get("FOUNDRY_MCP_METRICS_FLUSH_INTERVAL"):
            try:
                self.metrics_persistence.flush_interval_seconds = int(metrics_flush)
            except ValueError:
                pass
        if persist_list := os.environ.get("FOUNDRY_MCP_METRICS_PERSIST_METRICS"):
            self.metrics_persistence.persist_metrics = [
                m.strip() for m in persist_list.split(",") if m.strip()
            ]

        # Dashboard settings
        if dash_enabled := os.environ.get("FOUNDRY_MCP_DASHBOARD_ENABLED"):
            self.dashboard.enabled = _parse_bool(dash_enabled)
        if dash_port := os.environ.get("FOUNDRY_MCP_DASHBOARD_PORT"):
            try:
                self.dashboard.port = int(dash_port)
            except ValueError:
                pass
        if dash_host := os.environ.get("FOUNDRY_MCP_DASHBOARD_HOST"):
            self.dashboard.host = dash_host
        if dash_auto_open := os.environ.get("FOUNDRY_MCP_DASHBOARD_AUTO_OPEN"):
            self.dashboard.auto_open_browser = _parse_bool(dash_auto_open)
        if dash_refresh := os.environ.get("FOUNDRY_MCP_DASHBOARD_REFRESH_INTERVAL"):
            try:
                self.dashboard.refresh_interval_ms = int(dash_refresh)
            except ValueError:
                pass

    def validate_api_key(self, key: Optional[str]) -> bool:
        """
        Validate an API key.

        Args:
            key: API key to validate

        Returns:
            True if valid (or auth not required), False otherwise
        """
        if not self.require_auth:
            return True

        if not key:
            return False

        return key in self.api_keys

    def get_bikelane_dir(self, specs_dir: Optional[Path] = None) -> Path:
        """
        Get the resolved bikelane directory path.

        Priority:
        1. Explicitly configured bikelane_dir (from TOML or env var)
        2. Default: specs_dir/.bikelane (where specs_dir is resolved)

        Args:
            specs_dir: Optional specs directory to use for default path.
                      If not provided, uses self.specs_dir or "./specs"

        Returns:
            Path to bikelane directory
        """
        if self.bikelane_dir is not None:
            return self.bikelane_dir.expanduser()

        # Fall back to default: specs/.bikelane
        base_specs = specs_dir or self.specs_dir or Path("./specs")
        return base_specs / ".bikelane"

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        level = getattr(logging, self.log_level, logging.INFO)

        if self.structured_logging:
            # JSON-style structured logging
            formatter = logging.Formatter(
                '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
                '"logger":"%(name)s","message":"%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        root_logger = logging.getLogger("foundry_mcp")
        root_logger.setLevel(level)
        root_logger.addHandler(handler)


# Global configuration instance
_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = ServerConfig.from_env()
    return _config


def set_config(config: ServerConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


# Metrics and observability decorators


def log_call(
    logger_name: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to log function calls with structured data.

    Args:
        logger_name: Optional logger name (defaults to function module)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        log = logging.getLogger(logger_name or func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            log.debug(
                f"Calling {func.__name__}",
                extra={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )
            try:
                result = func(*args, **kwargs)
                log.debug(
                    f"Completed {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "success": True,
                    },
                )
                return result
            except Exception as e:
                log.error(
                    f"Error in {func.__name__}: {e}",
                    extra={
                        "function": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                raise

        return wrapper

    return decorator


def timed(
    metric_name: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to measure and log function execution time.

    Args:
        metric_name: Optional metric name (defaults to function name)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = metric_name or func.__name__
        log = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                log.info(
                    f"Timer: {name}",
                    extra={
                        "metric": name,
                        "duration_ms": round(elapsed * 1000, 2),
                        "success": True,
                    },
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                log.info(
                    f"Timer: {name}",
                    extra={
                        "metric": name,
                        "duration_ms": round(elapsed * 1000, 2),
                        "success": False,
                        "error": str(e),
                    },
                )
                raise

        return wrapper

    return decorator


def require_auth(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to require API key authentication for a function.

    The function must accept an 'api_key' keyword argument.
    Raises ValueError if authentication fails.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        config = get_config()
        api_key = kwargs.get("api_key")

        if not config.validate_api_key(api_key):
            raise ValueError("Invalid or missing API key")

        return func(*args, **kwargs)

    return wrapper
