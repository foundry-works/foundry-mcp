"""Health check system for foundry-mcp.

Provides Kubernetes-style health probes (liveness, readiness, health)
with pluggable dependency checkers and configurable thresholds.

Usage:
    from foundry_mcp.core.health import (
        get_health_manager,
        HealthStatus,
        check_liveness,
        check_readiness,
        check_health,
    )

    # Quick checks
    if check_liveness().is_healthy:
        print("Server is alive")

    # Full health check with details
    result = check_health()
    print(f"Status: {result.status.value}")
    print(f"Dependencies: {result.dependencies}")
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status values following Kubernetes conventions."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class DependencyHealth:
    """Health status of a single dependency."""

    name: str
    healthy: bool
    status: HealthStatus
    message: str = ""
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "name": self.name,
            "healthy": self.healthy,
            "status": self.status.value,
        }
        if self.message:
            result["message"] = self.message
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class HealthResult:
    """Result of a health check operation."""

    status: HealthStatus
    is_healthy: bool
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    dependencies: List[DependencyHealth] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "timestamp": self.timestamp,
        }
        if self.message:
            result["message"] = self.message
        if self.dependencies:
            result["dependencies"] = [d.to_dict() for d in self.dependencies]
        if self.details:
            result["details"] = self.details
        return result


class DependencyChecker(Protocol):
    """Protocol for dependency health checkers."""

    @property
    def name(self) -> str:
        """Unique name for this dependency."""
        ...

    def check(self, timeout: float = 5.0) -> DependencyHealth:
        """Check the health of this dependency."""
        ...


# =============================================================================
# Built-in Dependency Checkers
# =============================================================================


class SpecsDirectoryChecker:
    """Check that specs directory exists and is accessible."""

    name = "specs_directory"

    def __init__(self, specs_dir: Optional[Path] = None):
        self.specs_dir = specs_dir

    def check(self, timeout: float = 5.0) -> DependencyHealth:
        start = time.perf_counter()
        try:
            # Try to get specs_dir from config if not provided
            if self.specs_dir is None:
                from foundry_mcp.config import get_config

                config = get_config()
                self.specs_dir = config.specs_dir if config else None

            if self.specs_dir is None:
                return DependencyHealth(
                    name=self.name,
                    healthy=False,
                    status=HealthStatus.UNHEALTHY,
                    message="specs_dir not configured",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            if not self.specs_dir.exists():
                return DependencyHealth(
                    name=self.name,
                    healthy=False,
                    status=HealthStatus.UNHEALTHY,
                    message=f"specs_dir does not exist: {self.specs_dir}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            if not self.specs_dir.is_dir():
                return DependencyHealth(
                    name=self.name,
                    healthy=False,
                    status=HealthStatus.UNHEALTHY,
                    message=f"specs_dir is not a directory: {self.specs_dir}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            # Check if readable
            try:
                list(self.specs_dir.iterdir())
            except PermissionError:
                return DependencyHealth(
                    name=self.name,
                    healthy=False,
                    status=HealthStatus.UNHEALTHY,
                    message=f"specs_dir not readable: {self.specs_dir}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            return DependencyHealth(
                name=self.name,
                healthy=True,
                status=HealthStatus.HEALTHY,
                message="specs_dir accessible",
                latency_ms=(time.perf_counter() - start) * 1000,
                details={"path": str(self.specs_dir)},
            )

        except Exception as e:
            return DependencyHealth(
                name=self.name,
                healthy=False,
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking specs_dir: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )


class DiskSpaceChecker:
    """Check available disk space meets threshold."""

    name = "disk_space"

    def __init__(
        self,
        path: Optional[Path] = None,
        threshold_mb: int = 100,
        warning_mb: int = 500,
    ):
        self.path = path or Path(".")
        self.threshold_mb = threshold_mb
        self.warning_mb = warning_mb

    def check(self, timeout: float = 5.0) -> DependencyHealth:
        start = time.perf_counter()
        try:
            usage = shutil.disk_usage(self.path)
            free_mb = usage.free / (1024 * 1024)

            details = {
                "path": str(self.path),
                "free_mb": round(free_mb, 2),
                "total_mb": round(usage.total / (1024 * 1024), 2),
                "threshold_mb": self.threshold_mb,
            }

            if free_mb < self.threshold_mb:
                return DependencyHealth(
                    name=self.name,
                    healthy=False,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Disk space critically low: {free_mb:.1f}MB free",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    details=details,
                )

            if free_mb < self.warning_mb:
                return DependencyHealth(
                    name=self.name,
                    healthy=True,
                    status=HealthStatus.DEGRADED,
                    message=f"Disk space low: {free_mb:.1f}MB free",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    details=details,
                )

            return DependencyHealth(
                name=self.name,
                healthy=True,
                status=HealthStatus.HEALTHY,
                message=f"Disk space OK: {free_mb:.1f}MB free",
                latency_ms=(time.perf_counter() - start) * 1000,
                details=details,
            )

        except Exception as e:
            return DependencyHealth(
                name=self.name,
                healthy=False,
                status=HealthStatus.UNHEALTHY,
                message=f"Error checking disk space: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )


class OpenTelemetryChecker:
    """Check OpenTelemetry availability."""

    name = "opentelemetry"

    def check(self, timeout: float = 5.0) -> DependencyHealth:
        start = time.perf_counter()
        try:
            from foundry_mcp.core.observability import get_observability_manager

            manager = get_observability_manager()
            is_enabled = manager.is_tracing_enabled()

            # OTel being disabled is not unhealthy, just a different state
            if is_enabled:
                return DependencyHealth(
                    name=self.name,
                    healthy=True,
                    status=HealthStatus.HEALTHY,
                    message="OpenTelemetry tracing enabled",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    details={"enabled": True},
                )
            else:
                return DependencyHealth(
                    name=self.name,
                    healthy=True,  # Disabled is still healthy
                    status=HealthStatus.HEALTHY,
                    message="OpenTelemetry tracing disabled (optional)",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    details={"enabled": False},
                )

        except Exception as e:
            return DependencyHealth(
                name=self.name,
                healthy=True,  # OTel errors shouldn't fail health check
                status=HealthStatus.DEGRADED,
                message=f"OpenTelemetry check failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )


class PrometheusChecker:
    """Check Prometheus metrics availability."""

    name = "prometheus"

    def check(self, timeout: float = 5.0) -> DependencyHealth:
        start = time.perf_counter()
        try:
            from foundry_mcp.core.observability import get_observability_manager

            manager = get_observability_manager()
            is_enabled = manager.is_metrics_enabled()

            # Prometheus being disabled is not unhealthy
            if is_enabled:
                return DependencyHealth(
                    name=self.name,
                    healthy=True,
                    status=HealthStatus.HEALTHY,
                    message="Prometheus metrics enabled",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    details={"enabled": True},
                )
            else:
                return DependencyHealth(
                    name=self.name,
                    healthy=True,
                    status=HealthStatus.HEALTHY,
                    message="Prometheus metrics disabled (optional)",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    details={"enabled": False},
                )

        except Exception as e:
            return DependencyHealth(
                name=self.name,
                healthy=True,  # Prometheus errors shouldn't fail health check
                status=HealthStatus.DEGRADED,
                message=f"Prometheus check failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )


class AIProviderChecker:
    """Check AI provider availability."""

    name = "ai_provider"

    def check(self, timeout: float = 5.0) -> DependencyHealth:
        start = time.perf_counter()
        try:
            from foundry_mcp.core.providers import (
                available_providers,
                get_provider_statuses,
            )

            available = available_providers()
            statuses = get_provider_statuses()

            # AI providers are optional - just report what's available
            if available:
                return DependencyHealth(
                    name=self.name,
                    healthy=True,
                    status=HealthStatus.HEALTHY,
                    message=f"AI providers available: {', '.join(available)}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    details={
                        "available": available,
                        # statuses is Dict[str, bool], not enum values
                        "statuses": statuses,
                    },
                )
            else:
                return DependencyHealth(
                    name=self.name,
                    healthy=True,  # No providers is not unhealthy
                    status=HealthStatus.DEGRADED,
                    message="No AI providers available (optional)",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    details={"available": [], "statuses": {}},
                )

        except ImportError:
            return DependencyHealth(
                name=self.name,
                healthy=True,
                status=HealthStatus.HEALTHY,
                message="AI provider module not available (optional)",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return DependencyHealth(
                name=self.name,
                healthy=True,
                status=HealthStatus.DEGRADED,
                message=f"AI provider check failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )


# =============================================================================
# Health Manager
# =============================================================================


@dataclass
class HealthConfig:
    """Configuration for health checks.

    Attributes:
        enabled: Whether health checks are enabled
        liveness_timeout: Timeout for liveness checks (seconds)
        readiness_timeout: Timeout for readiness checks (seconds)
        health_timeout: Timeout for full health checks (seconds)
        disk_space_threshold_mb: Minimum disk space before unhealthy
        disk_space_warning_mb: Minimum disk space before degraded
    """

    enabled: bool = True
    liveness_timeout: float = 1.0
    readiness_timeout: float = 5.0
    health_timeout: float = 10.0
    disk_space_threshold_mb: int = 100
    disk_space_warning_mb: int = 500

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "HealthConfig":
        """Create config from TOML dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            liveness_timeout=data.get("liveness_timeout", 1.0),
            readiness_timeout=data.get("readiness_timeout", 5.0),
            health_timeout=data.get("health_timeout", 10.0),
            disk_space_threshold_mb=data.get("disk_space_threshold_mb", 100),
            disk_space_warning_mb=data.get("disk_space_warning_mb", 500),
        )


class HealthManager:
    """Manages health checks for the foundry-mcp server.

    Provides three levels of health checks:
    - Liveness: Is the process running? (always true if this code executes)
    - Readiness: Can the server handle requests? (checks critical deps)
    - Health: Full health status (all dependencies)
    """

    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self._liveness_checkers: List[DependencyChecker] = []
        self._readiness_checkers: List[DependencyChecker] = []
        self._health_checkers: List[DependencyChecker] = []
        self._setup_default_checkers()

    def _setup_default_checkers(self) -> None:
        """Set up default dependency checkers."""
        # Readiness checks - critical for serving requests
        specs_checker = SpecsDirectoryChecker()
        disk_checker = DiskSpaceChecker(
            threshold_mb=self.config.disk_space_threshold_mb,
            warning_mb=self.config.disk_space_warning_mb,
        )

        self._readiness_checkers = [specs_checker, disk_checker]

        # Health checks - full system status
        self._health_checkers = [
            specs_checker,
            disk_checker,
            OpenTelemetryChecker(),
            PrometheusChecker(),
            AIProviderChecker(),
        ]

    def register_checker(
        self,
        checker: DependencyChecker,
        *,
        liveness: bool = False,
        readiness: bool = False,
        health: bool = True,
    ) -> None:
        """Register a custom dependency checker.

        Args:
            checker: The dependency checker to register
            liveness: Include in liveness checks
            readiness: Include in readiness checks
            health: Include in full health checks (default True)
        """
        if liveness:
            self._liveness_checkers.append(checker)
        if readiness:
            self._readiness_checkers.append(checker)
        if health:
            self._health_checkers.append(checker)

    def check_liveness(self) -> HealthResult:
        """Check if the server is alive.

        Liveness checks are intentionally minimal - if this code runs,
        we're alive. Custom checkers can be added for process-level health.

        Returns:
            HealthResult indicating liveness status
        """
        if not self.config.enabled:
            return HealthResult(
                status=HealthStatus.HEALTHY,
                is_healthy=True,
                message="Health checks disabled",
            )

        dependencies = []
        for checker in self._liveness_checkers:
            try:
                result = checker.check(timeout=self.config.liveness_timeout)
                dependencies.append(result)
            except Exception as e:
                dependencies.append(
                    DependencyHealth(
                        name=checker.name,
                        healthy=False,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {e}",
                    )
                )

        # If no liveness checkers, we're alive
        if not dependencies:
            return HealthResult(
                status=HealthStatus.HEALTHY,
                is_healthy=True,
                message="Server is alive",
            )

        # Check if any are unhealthy
        unhealthy = [d for d in dependencies if not d.healthy]
        if unhealthy:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                is_healthy=False,
                message=f"Liveness check failed: {unhealthy[0].message}",
                dependencies=dependencies,
            )

        return HealthResult(
            status=HealthStatus.HEALTHY,
            is_healthy=True,
            message="Server is alive",
            dependencies=dependencies,
        )

    def check_readiness(self) -> HealthResult:
        """Check if the server is ready to handle requests.

        Readiness checks verify critical dependencies are available.

        Returns:
            HealthResult indicating readiness status
        """
        if not self.config.enabled:
            return HealthResult(
                status=HealthStatus.HEALTHY,
                is_healthy=True,
                message="Health checks disabled",
            )

        dependencies = []
        for checker in self._readiness_checkers:
            try:
                result = checker.check(timeout=self.config.readiness_timeout)
                dependencies.append(result)
            except Exception as e:
                dependencies.append(
                    DependencyHealth(
                        name=checker.name,
                        healthy=False,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {e}",
                    )
                )

        # Check if any critical dependencies are unhealthy
        unhealthy = [d for d in dependencies if not d.healthy]
        degraded = [d for d in dependencies if d.status == HealthStatus.DEGRADED]

        if unhealthy:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                is_healthy=False,
                message=f"Not ready: {unhealthy[0].message}",
                dependencies=dependencies,
            )

        if degraded:
            return HealthResult(
                status=HealthStatus.DEGRADED,
                is_healthy=True,  # Still ready, but degraded
                message=f"Ready with warnings: {degraded[0].message}",
                dependencies=dependencies,
            )

        return HealthResult(
            status=HealthStatus.HEALTHY,
            is_healthy=True,
            message="Server is ready",
            dependencies=dependencies,
        )

    def check_health(self) -> HealthResult:
        """Perform a full health check of all dependencies.

        Returns:
            HealthResult with complete system health status
        """
        if not self.config.enabled:
            return HealthResult(
                status=HealthStatus.HEALTHY,
                is_healthy=True,
                message="Health checks disabled",
            )

        dependencies = []
        for checker in self._health_checkers:
            try:
                result = checker.check(timeout=self.config.health_timeout)
                dependencies.append(result)
            except Exception as e:
                dependencies.append(
                    DependencyHealth(
                        name=checker.name,
                        healthy=False,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {e}",
                    )
                )

        # Aggregate status
        unhealthy = [d for d in dependencies if not d.healthy]
        degraded = [d for d in dependencies if d.status == HealthStatus.DEGRADED]

        if unhealthy:
            return HealthResult(
                status=HealthStatus.UNHEALTHY,
                is_healthy=False,
                message=f"Unhealthy: {len(unhealthy)} failed check(s)",
                dependencies=dependencies,
                details={
                    "unhealthy_count": len(unhealthy),
                    "degraded_count": len(degraded),
                    "healthy_count": len(dependencies)
                    - len(unhealthy)
                    - len(degraded),
                },
            )

        if degraded:
            return HealthResult(
                status=HealthStatus.DEGRADED,
                is_healthy=True,
                message=f"Degraded: {len(degraded)} warning(s)",
                dependencies=dependencies,
                details={
                    "unhealthy_count": 0,
                    "degraded_count": len(degraded),
                    "healthy_count": len(dependencies) - len(degraded),
                },
            )

        return HealthResult(
            status=HealthStatus.HEALTHY,
            is_healthy=True,
            message="All systems healthy",
            dependencies=dependencies,
            details={
                "unhealthy_count": 0,
                "degraded_count": 0,
                "healthy_count": len(dependencies),
            },
        )


# =============================================================================
# Global Manager Instance
# =============================================================================

_health_manager: Optional[HealthManager] = None
_manager_lock = __import__("threading").Lock()


def get_health_manager(config: Optional[HealthConfig] = None) -> HealthManager:
    """Get or create the global health manager.

    Args:
        config: Optional config (only used on first call)

    Returns:
        Global HealthManager instance
    """
    global _health_manager
    if _health_manager is None:
        with _manager_lock:
            if _health_manager is None:
                _health_manager = HealthManager(config)
    return _health_manager


def reset_health_manager() -> None:
    """Reset the global health manager (for testing)."""
    global _health_manager
    with _manager_lock:
        _health_manager = None


# =============================================================================
# Convenience Functions
# =============================================================================


def check_liveness() -> HealthResult:
    """Quick liveness check.

    Returns:
        HealthResult indicating if server is alive
    """
    return get_health_manager().check_liveness()


def check_readiness() -> HealthResult:
    """Quick readiness check.

    Returns:
        HealthResult indicating if server is ready
    """
    return get_health_manager().check_readiness()


def check_health() -> HealthResult:
    """Full health check.

    Returns:
        HealthResult with complete system status
    """
    return get_health_manager().check_health()
