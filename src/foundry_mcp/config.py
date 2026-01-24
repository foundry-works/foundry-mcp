"""
Server configuration for foundry-mcp.

Supports configuration via:
1. Environment variables (highest priority)
2. TOML config file (foundry-mcp.toml)
3. Default values (lowest priority)

Environment variables:
- FOUNDRY_MCP_WORKSPACE_ROOTS: Comma-separated list of workspace root paths
- FOUNDRY_MCP_SPECS_DIR: Path to specs directory
- FOUNDRY_MCP_NOTES_DIR: Path to notes intake queue directory (default: specs/.notes)
- FOUNDRY_MCP_RESEARCH_DIR: Path to research state directory (default: specs/.research)
- FOUNDRY_MCP_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
- FOUNDRY_MCP_API_KEYS: Comma-separated list of valid API keys (optional)
- FOUNDRY_MCP_REQUIRE_AUTH: Whether to require API key authentication (true/false)
- FOUNDRY_MCP_CONFIG_FILE: Path to TOML config file

Search Provider API Keys (for deep research workflow):
- TAVILY_API_KEY: API key for Tavily web search (https://tavily.com/)
- PERPLEXITY_API_KEY: API key for Perplexity Search (https://docs.perplexity.ai/)
- GOOGLE_API_KEY: API key for Google Custom Search (https://console.cloud.google.com/)
- GOOGLE_CSE_ID: Google Custom Search Engine ID (https://cse.google.com/)
- SEMANTIC_SCHOLAR_API_KEY: API key for Semantic Scholar academic search (optional for basic tier)

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
from typing import Optional, List, Dict, Any, Callable, TypeVar, Tuple

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


@dataclass
class ResearchConfig:
    """Configuration for research workflows (CHAT, CONSENSUS, THINKDEEP, IDEATE, DEEP_RESEARCH).

    Attributes:
        enabled: Master switch for research tools
        ttl_hours: Time-to-live for stored states in hours
        max_messages_per_thread: Maximum messages retained in a conversation thread
        default_provider: Default LLM provider for single-model workflows
        consensus_providers: List of provider IDs for CONSENSUS workflow
        thinkdeep_max_depth: Maximum investigation depth for THINKDEEP workflow
        ideate_perspectives: List of perspectives for IDEATE brainstorming
        default_timeout: Default timeout in seconds for provider calls (thinkdeep uses 2x)
        deep_research_max_iterations: Maximum refinement iterations for DEEP_RESEARCH
        deep_research_max_sub_queries: Maximum sub-queries for query decomposition
        deep_research_max_sources: Maximum sources per sub-query
        deep_research_follow_links: Whether to follow and extract content from links
        deep_research_timeout: Default timeout per operation in seconds
        deep_research_max_concurrent: Maximum concurrent operations
        deep_research_providers: Ordered list of search providers for deep research
        deep_research_audit_artifacts: Whether to write per-run audit artifacts
        search_rate_limit: Global rate limit for search APIs (requests per minute)
        max_concurrent_searches: Maximum concurrent search requests (for asyncio.Semaphore)
        per_provider_rate_limits: Per-provider rate limits in requests per minute
        tavily_api_key: API key for Tavily search provider (optional, reads from TAVILY_API_KEY env var)
        perplexity_api_key: API key for Perplexity Search (optional, reads from PERPLEXITY_API_KEY env var)
        google_api_key: API key for Google Custom Search (optional, reads from GOOGLE_API_KEY env var)
        google_cse_id: Google Custom Search Engine ID (optional, reads from GOOGLE_CSE_ID env var)
        semantic_scholar_api_key: API key for Semantic Scholar (optional, reads from SEMANTIC_SCHOLAR_API_KEY env var)
    """

    enabled: bool = True
    ttl_hours: int = 24
    max_messages_per_thread: int = 100
    default_provider: str = "gemini"
    consensus_providers: List[str] = field(
        default_factory=lambda: ["gemini", "claude"]
    )
    thinkdeep_max_depth: int = 5
    ideate_perspectives: List[str] = field(
        default_factory=lambda: ["technical", "creative", "practical", "visionary"]
    )
    default_timeout: float = 360.0  # 360 seconds default for AI CLI providers
    # Deep research configuration
    deep_research_max_iterations: int = 3
    deep_research_max_sub_queries: int = 5
    deep_research_max_sources: int = 5
    deep_research_follow_links: bool = True
    deep_research_timeout: float = 600.0  # Whole workflow timeout
    deep_research_max_concurrent: int = 3
    # Per-phase timeout overrides (seconds) - uses deep_research_timeout if not set
    deep_research_planning_timeout: float = 360.0
    deep_research_analysis_timeout: float = 360.0
    deep_research_synthesis_timeout: float = 600.0  # Synthesis may take longer
    deep_research_refinement_timeout: float = 360.0
    # Per-phase provider overrides - uses default_provider if not set
    deep_research_planning_provider: Optional[str] = None
    deep_research_analysis_provider: Optional[str] = None
    deep_research_synthesis_provider: Optional[str] = None
    deep_research_refinement_provider: Optional[str] = None
    # Per-phase fallback provider lists (for retry/fallback on failure)
    # On failure, tries next provider in the list until success or exhaustion
    deep_research_planning_providers: List[str] = field(default_factory=list)
    deep_research_analysis_providers: List[str] = field(default_factory=list)
    deep_research_synthesis_providers: List[str] = field(default_factory=list)
    deep_research_refinement_providers: List[str] = field(default_factory=list)
    # Retry settings for deep research phases
    deep_research_max_retries: int = 2  # Retry attempts per provider
    deep_research_retry_delay: float = 5.0  # Seconds between retries
    deep_research_providers: List[str] = field(
        default_factory=lambda: ["tavily", "google", "semantic_scholar"]
    )
    deep_research_audit_artifacts: bool = True
    # Research mode: "general" | "academic" | "technical"
    deep_research_mode: str = "general"
    # Search rate limiting configuration
    search_rate_limit: int = 60  # requests per minute (global default)
    max_concurrent_searches: int = 3  # for asyncio.Semaphore in gathering phase
    per_provider_rate_limits: Dict[str, int] = field(
        default_factory=lambda: {
            "tavily": 60,  # Tavily free tier: ~1 req/sec
            "perplexity": 60,  # Perplexity: ~1 req/sec (pricing: $5/1k requests)
            "google": 100,  # Google CSE: 100 queries/day free, ~100/min paid
            "semantic_scholar": 100,  # Semantic Scholar: 100 req/5min unauthenticated
        }
    )
    # Search provider API keys (all optional, read from env vars if not set)
    tavily_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "ResearchConfig":
        """Create config from TOML dict (typically [research] section).

        Args:
            data: Dict from TOML parsing

        Returns:
            ResearchConfig instance
        """
        # Parse consensus_providers - handle both string and list
        consensus_providers = data.get("consensus_providers", ["gemini", "claude"])
        if isinstance(consensus_providers, str):
            consensus_providers = [p.strip() for p in consensus_providers.split(",")]

        # Parse ideate_perspectives - handle both string and list
        ideate_perspectives = data.get(
            "ideate_perspectives", ["technical", "creative", "practical", "visionary"]
        )
        if isinstance(ideate_perspectives, str):
            ideate_perspectives = [p.strip() for p in ideate_perspectives.split(",")]

        # Parse deep_research_providers - handle both string and list
        deep_research_providers = data.get(
            "deep_research_providers", ["tavily", "google", "semantic_scholar"]
        )
        if isinstance(deep_research_providers, str):
            deep_research_providers = [
                p.strip() for p in deep_research_providers.split(",") if p.strip()
            ]

        # Parse per-phase fallback provider lists
        def _parse_provider_list(key: str) -> List[str]:
            val = data.get(key, [])
            if isinstance(val, str):
                return [p.strip() for p in val.split(",") if p.strip()]
            return list(val) if val else []

        deep_research_planning_providers = _parse_provider_list("deep_research_planning_providers")
        deep_research_analysis_providers = _parse_provider_list("deep_research_analysis_providers")
        deep_research_synthesis_providers = _parse_provider_list("deep_research_synthesis_providers")
        deep_research_refinement_providers = _parse_provider_list("deep_research_refinement_providers")

        # Parse per_provider_rate_limits - handle dict from TOML
        per_provider_rate_limits = data.get("per_provider_rate_limits", {
            "tavily": 60,
            "perplexity": 60,
            "google": 100,
            "semantic_scholar": 100,
        })
        if isinstance(per_provider_rate_limits, dict):
            # Convert values to int
            per_provider_rate_limits = {
                k: int(v) for k, v in per_provider_rate_limits.items()
            }

        return cls(
            enabled=_parse_bool(data.get("enabled", True)),
            ttl_hours=int(data.get("ttl_hours", 24)),
            max_messages_per_thread=int(data.get("max_messages_per_thread", 100)),
            default_provider=str(data.get("default_provider", "gemini")),
            consensus_providers=consensus_providers,
            thinkdeep_max_depth=int(data.get("thinkdeep_max_depth", 5)),
            ideate_perspectives=ideate_perspectives,
            default_timeout=float(data.get("default_timeout", 360.0)),
            # Deep research configuration
            deep_research_max_iterations=int(data.get("deep_research_max_iterations", 3)),
            deep_research_max_sub_queries=int(data.get("deep_research_max_sub_queries", 5)),
            deep_research_max_sources=int(data.get("deep_research_max_sources", 5)),
            deep_research_follow_links=_parse_bool(data.get("deep_research_follow_links", True)),
            deep_research_timeout=float(data.get("deep_research_timeout", 600.0)),
            deep_research_max_concurrent=int(data.get("deep_research_max_concurrent", 3)),
            # Per-phase timeout overrides (match class defaults)
            deep_research_planning_timeout=float(data.get("deep_research_planning_timeout", 360.0)),
            deep_research_analysis_timeout=float(data.get("deep_research_analysis_timeout", 360.0)),
            deep_research_synthesis_timeout=float(data.get("deep_research_synthesis_timeout", 600.0)),
            deep_research_refinement_timeout=float(data.get("deep_research_refinement_timeout", 360.0)),
            # Per-phase provider overrides
            deep_research_planning_provider=data.get("deep_research_planning_provider"),
            deep_research_analysis_provider=data.get("deep_research_analysis_provider"),
            deep_research_synthesis_provider=data.get("deep_research_synthesis_provider"),
            deep_research_refinement_provider=data.get("deep_research_refinement_provider"),
            # Per-phase fallback provider lists
            deep_research_planning_providers=deep_research_planning_providers,
            deep_research_analysis_providers=deep_research_analysis_providers,
            deep_research_synthesis_providers=deep_research_synthesis_providers,
            deep_research_refinement_providers=deep_research_refinement_providers,
            # Retry settings
            deep_research_max_retries=int(data.get("deep_research_max_retries", 2)),
            deep_research_retry_delay=float(data.get("deep_research_retry_delay", 5.0)),
            deep_research_providers=deep_research_providers,
            deep_research_audit_artifacts=_parse_bool(
                data.get("deep_research_audit_artifacts", True)
            ),
            # Research mode
            deep_research_mode=str(data.get("deep_research_mode", "general")),
            # Search rate limiting configuration
            search_rate_limit=int(data.get("search_rate_limit", 60)),
            max_concurrent_searches=int(data.get("max_concurrent_searches", 3)),
            per_provider_rate_limits=per_provider_rate_limits,
            # Search provider API keys (None means not set in TOML, will check env vars)
            tavily_api_key=data.get("tavily_api_key"),
            perplexity_api_key=data.get("perplexity_api_key"),
            google_api_key=data.get("google_api_key"),
            google_cse_id=data.get("google_cse_id"),
            semantic_scholar_api_key=data.get("semantic_scholar_api_key"),
        )

    def get_provider_rate_limit(self, provider: str) -> int:
        """Get rate limit for a specific provider.

        Returns the provider-specific rate limit if configured,
        otherwise falls back to the global search_rate_limit.

        Args:
            provider: Provider name (e.g., "tavily", "google", "semantic_scholar")

        Returns:
            Rate limit in requests per minute
        """
        return self.per_provider_rate_limits.get(provider, self.search_rate_limit)

    def get_phase_timeout(self, phase: str) -> float:
        """Get timeout for a specific deep research phase.

        Returns the phase-specific timeout if configured, otherwise
        falls back to deep_research_timeout.

        Args:
            phase: Phase name ("planning", "analysis", "synthesis", "refinement", "gathering")

        Returns:
            Timeout in seconds for the phase
        """
        phase_timeouts = {
            "planning": self.deep_research_planning_timeout,
            "analysis": self.deep_research_analysis_timeout,
            "synthesis": self.deep_research_synthesis_timeout,
            "refinement": self.deep_research_refinement_timeout,
            "gathering": self.deep_research_timeout,  # Gathering uses default
        }
        return phase_timeouts.get(phase.lower(), self.deep_research_timeout)

    def get_phase_provider(self, phase: str) -> str:
        """Get LLM provider ID for a specific deep research phase.

        Returns the phase-specific provider if configured, otherwise
        falls back to default_provider. Supports both simple names ("gemini")
        and ProviderSpec format ("[cli]gemini:pro").

        Args:
            phase: Phase name ("planning", "analysis", "synthesis", "refinement")

        Returns:
            Provider ID for the phase (e.g., "gemini", "opencode")
        """
        provider_id, _ = self.resolve_phase_provider(phase)
        return provider_id

    def resolve_phase_provider(self, phase: str) -> Tuple[str, Optional[str]]:
        """Resolve provider ID and model for a deep research phase.

        Parses ProviderSpec format ("[cli]gemini:pro") or simple names ("gemini").
        Returns (provider_id, model) tuple for use with the provider registry.

        Args:
            phase: Phase name ("planning", "analysis", "synthesis", "refinement")

        Returns:
            Tuple of (provider_id, model) where model may be None
        """
        phase_providers = {
            "planning": self.deep_research_planning_provider,
            "analysis": self.deep_research_analysis_provider,
            "synthesis": self.deep_research_synthesis_provider,
            "refinement": self.deep_research_refinement_provider,
        }
        configured = phase_providers.get(phase.lower())
        spec_str = configured or self.default_provider
        return _parse_provider_spec(spec_str)

    def get_phase_fallback_providers(self, phase: str) -> List[str]:
        """Get fallback provider list for a specific deep research phase.

        Returns the phase-specific fallback provider list if configured,
        otherwise returns an empty list (no fallback).

        Args:
            phase: Phase name ("planning", "analysis", "synthesis", "refinement")

        Returns:
            List of fallback provider IDs to try on failure
        """
        phase_fallbacks = {
            "planning": self.deep_research_planning_providers,
            "analysis": self.deep_research_analysis_providers,
            "synthesis": self.deep_research_synthesis_providers,
            "refinement": self.deep_research_refinement_providers,
        }
        return phase_fallbacks.get(phase.lower(), [])

    def get_search_provider_api_key(
        self,
        provider: str,
        required: bool = True,
    ) -> Optional[str]:
        """Get API key for a search provider with fallback to environment variables.

        Checks config value first, then falls back to environment variable.
        Raises ValueError with clear error message if required and not found.

        Args:
            provider: Provider name ("tavily", "google", "semantic_scholar")
            required: If True, raises ValueError when key is missing (default: True)

        Returns:
            API key string, or None if not required and not found

        Raises:
            ValueError: If required=True and no API key is found

        Example:
            # Get Tavily API key (will raise if missing)
            api_key = config.research.get_search_provider_api_key("tavily")

            # Get Semantic Scholar API key (optional, returns None if missing)
            api_key = config.research.get_search_provider_api_key(
                "semantic_scholar", required=False
            )
        """
        # Map provider names to config attributes and env vars
        provider_config = {
            "tavily": {
                "config_key": "tavily_api_key",
                "env_var": "TAVILY_API_KEY",
                "setup_url": "https://tavily.com/",
            },
            "perplexity": {
                "config_key": "perplexity_api_key",
                "env_var": "PERPLEXITY_API_KEY",
                "setup_url": "https://docs.perplexity.ai/",
            },
            "google": {
                "config_key": "google_api_key",
                "env_var": "GOOGLE_API_KEY",
                "setup_url": "https://console.cloud.google.com/apis/credentials",
            },
            "google_cse": {
                "config_key": "google_cse_id",
                "env_var": "GOOGLE_CSE_ID",
                "setup_url": "https://cse.google.com/",
            },
            "semantic_scholar": {
                "config_key": "semantic_scholar_api_key",
                "env_var": "SEMANTIC_SCHOLAR_API_KEY",
                "setup_url": "https://www.semanticscholar.org/product/api",
            },
        }

        provider_lower = provider.lower()
        if provider_lower not in provider_config:
            raise ValueError(
                f"Unknown search provider: '{provider}'. "
                f"Valid providers: {', '.join(provider_config.keys())}"
            )

        config_info = provider_config[provider_lower]
        config_key = config_info["config_key"]
        env_var = config_info["env_var"]

        # Check config value first
        api_key = getattr(self, config_key, None)

        # Fall back to environment variable
        if not api_key:
            api_key = os.environ.get(env_var)

        # Handle missing key
        if not api_key:
            if required:
                raise ValueError(
                    f"{provider.title()} API key not configured. "
                    f"Set via {env_var} environment variable or "
                    f"'research.{config_key}' in foundry-mcp.toml. "
                    f"Get an API key at: {config_info['setup_url']}"
                )
            return None

        return api_key

    def get_google_credentials(self, required: bool = True) -> tuple[Optional[str], Optional[str]]:
        """Get both Google API key and CSE ID for Google Custom Search.

        Convenience method that retrieves both required credentials for
        Google Custom Search API.

        Args:
            required: If True, raises ValueError when either credential is missing

        Returns:
            Tuple of (api_key, cse_id)

        Raises:
            ValueError: If required=True and either credential is missing
        """
        api_key = self.get_search_provider_api_key("google", required=required)
        cse_id = self.get_search_provider_api_key("google_cse", required=required)
        return api_key, cse_id

    def get_default_provider_spec(self) -> "ProviderSpec":
        """Parse default_provider into a ProviderSpec."""
        from foundry_mcp.core.llm_config import ProviderSpec
        return ProviderSpec.parse_flexible(self.default_provider)

    def get_consensus_provider_specs(self) -> List["ProviderSpec"]:
        """Parse consensus_providers into ProviderSpec list."""
        from foundry_mcp.core.llm_config import ProviderSpec
        return [ProviderSpec.parse_flexible(p) for p in self.consensus_providers]


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


def _parse_provider_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Parse a provider specification into (provider_id, model).

    Supports both simple names and ProviderSpec bracket notation:
    - "gemini" -> ("gemini", None)
    - "[cli]gemini:pro" -> ("gemini", "pro")
    - "[cli]opencode:openai/gpt-5.2" -> ("opencode", "openai/gpt-5.2")
    - "[api]openai/gpt-4.1" -> ("openai", "gpt-4.1")

    Args:
        spec: Provider specification string

    Returns:
        Tuple of (provider_id, model) where model may be None
    """
    spec = spec.strip()

    # Simple name (no brackets) - backward compatible
    if not spec.startswith("["):
        return (spec, None)

    # Try to parse with ProviderSpec
    try:
        from foundry_mcp.core.llm_config import ProviderSpec

        parsed = ProviderSpec.parse(spec)
        # Build model string with backend routing if present
        model = None
        if parsed.backend and parsed.model:
            model = f"{parsed.backend}/{parsed.model}"
        elif parsed.model:
            model = parsed.model
        return (parsed.provider, model)
    except (ValueError, ImportError) as e:
        logger.warning("Failed to parse provider spec '%s': %s", spec, e)
        # Fall back to treating as simple name (strip brackets)
        return (spec.split("]")[-1].split(":")[0], None)


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
    notes_dir: Optional[Path] = None  # Intake queue storage (default: specs/.notes)
    research_dir: Optional[Path] = None  # Research state storage (default: specs/.research)

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

    # Test runner configuration
    test: TestConfig = field(default_factory=TestConfig)

    # Research workflows configuration
    research: ResearchConfig = field(default_factory=ResearchConfig)

    # Tool registration control
    disabled_tools: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, config_file: Optional[str] = None) -> "ServerConfig":
        """
        Create configuration from environment variables and optional TOML file.

        Priority (highest to lowest):
        1. Environment variables
        2. Project TOML config (./foundry-mcp.toml or ./.foundry-mcp.toml)
        3. User TOML config (~/.foundry-mcp.toml)
        4. Default values
        """
        config = cls()

        # Load TOML config if available
        toml_path = config_file or os.environ.get("FOUNDRY_MCP_CONFIG_FILE")
        if toml_path:
            config._load_toml(Path(toml_path))
        else:
            # Layered config loading:
            # 1. Home directory config (user defaults)
            home_config = Path.home() / ".foundry-mcp.toml"
            if home_config.exists():
                config._load_toml(home_config)
                logger.debug(f"Loaded user config from {home_config}")

            # 2. Project directory config (project overrides)
            # Try foundry-mcp.toml first, fall back to .foundry-mcp.toml for compatibility
            project_config = Path("foundry-mcp.toml")
            if project_config.exists():
                config._load_toml(project_config)
                logger.debug(f"Loaded project config from {project_config}")
            else:
                legacy_config = Path(".foundry-mcp.toml")
                if legacy_config.exists():
                    config._load_toml(legacy_config)
                    logger.debug(f"Loaded project config from {legacy_config}")

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
                if "notes_dir" in ws:
                    self.notes_dir = Path(ws["notes_dir"])
                if "research_dir" in ws:
                    self.research_dir = Path(ws["research_dir"])

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
            # Tools configuration (preferred location for disabled_tools)
            if "tools" in data:
                tools_cfg = data["tools"]
                if "disabled_tools" in tools_cfg:
                    self.disabled_tools = tools_cfg["disabled_tools"]

            # Git workflow settings
            if "git" in data:
                git_cfg = data["git"]
                if "enabled" in git_cfg:
                    self.git.enabled = _parse_bool(git_cfg["enabled"])
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

            # Test runner settings
            if "test" in data:
                self.test = TestConfig.from_toml_dict(data["test"])

            # Research workflows settings
            if "research" in data:
                self.research = ResearchConfig.from_toml_dict(data["research"])

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

        # Notes directory (intake queue storage)
        if notes := os.environ.get("FOUNDRY_MCP_NOTES_DIR"):
            self.notes_dir = Path(notes)

        # Research directory (research state storage)
        if research := os.environ.get("FOUNDRY_MCP_RESEARCH_DIR"):
            self.research_dir = Path(research)

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

        # Search provider API keys (direct env vars, no FOUNDRY_MCP_ prefix)
        # These use standard env var names that match provider documentation
        if tavily_key := os.environ.get("TAVILY_API_KEY"):
            self.research.tavily_api_key = tavily_key
        if perplexity_key := os.environ.get("PERPLEXITY_API_KEY"):
            self.research.perplexity_api_key = perplexity_key
        if google_key := os.environ.get("GOOGLE_API_KEY"):
            self.research.google_api_key = google_key
        if google_cse := os.environ.get("GOOGLE_CSE_ID"):
            self.research.google_cse_id = google_cse
        if semantic_scholar_key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
            self.research.semantic_scholar_api_key = semantic_scholar_key

        # Disabled tools (comma-separated list)
        if disabled := os.environ.get("FOUNDRY_MCP_DISABLED_TOOLS"):
            self.disabled_tools = [t.strip() for t in disabled.split(",") if t.strip()]

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

    def get_notes_dir(self, specs_dir: Optional[Path] = None) -> Path:
        """
        Get the resolved notes directory path.

        Priority:
        1. Explicitly configured notes_dir (from TOML or env var)
        2. Default: specs_dir/.notes (where specs_dir is resolved)

        Args:
            specs_dir: Optional specs directory to use for default path.
                      If not provided, uses self.specs_dir or "./specs"

        Returns:
            Path to notes directory
        """
        if self.notes_dir is not None:
            return self.notes_dir.expanduser()

        # Fall back to default: specs/.notes
        base_specs = specs_dir or self.specs_dir or Path("./specs")
        return base_specs / ".notes"

    def get_research_dir(self, specs_dir: Optional[Path] = None) -> Path:
        """
        Get the resolved research directory path.

        Priority:
        1. Explicitly configured research_dir (from TOML or env var)
        2. Default: specs_dir/.research (where specs_dir is resolved)

        Args:
            specs_dir: Optional specs directory to use for default path.
                      If not provided, uses self.specs_dir or "./specs"

        Returns:
            Path to research directory
        """
        if self.research_dir is not None:
            return self.research_dir.expanduser()

        # Fall back to default: specs/.research
        base_specs = specs_dir or self.specs_dir or Path("./specs")
        return base_specs / ".research"

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
