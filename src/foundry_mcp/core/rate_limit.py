"""
Rate limiting module for foundry-mcp.

Provides per-tool rate limiting with configurable limits and audit logging.
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from foundry_mcp.core.observability import audit_log

logger = logging.getLogger(__name__)


# Schema version
SCHEMA_VERSION = "1.0.0"


@dataclass
class RateLimitConfig:
    """
    Configuration for a rate limit.
    """
    requests_per_minute: int = 60
    burst_limit: int = 10
    enabled: bool = True
    reason: str = ""


@dataclass
class RateLimitState:
    """
    Current state of a rate limiter.
    """
    tokens: float = 0.0
    last_update: float = 0.0
    request_count: int = 0
    throttle_count: int = 0


@dataclass
class RateLimitResult:
    """
    Result of a rate limit check.
    """
    allowed: bool
    remaining: int = 0
    reset_in: float = 0.0
    limit: int = 0
    reason: str = ""


class TokenBucketLimiter:
    """
    Token bucket rate limiter implementation.

    Provides smooth rate limiting with burst support.
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.state = RateLimitState(
            tokens=float(config.burst_limit),
            last_update=time.time()
        )

    def check(self) -> RateLimitResult:
        """
        Check if a request is allowed without consuming tokens.

        Returns:
            RateLimitResult indicating if request would be allowed
        """
        if not self.config.enabled:
            return RateLimitResult(allowed=True, remaining=-1)

        self._refill()
        return RateLimitResult(
            allowed=self.state.tokens >= 1.0,
            remaining=int(self.state.tokens),
            reset_in=self._time_to_next_token(),
            limit=self.config.requests_per_minute,
        )

    def acquire(self) -> RateLimitResult:
        """
        Attempt to acquire a token for a request.

        Returns:
            RateLimitResult indicating if request is allowed
        """
        if not self.config.enabled:
            return RateLimitResult(allowed=True, remaining=-1)

        self._refill()
        self.state.request_count += 1

        if self.state.tokens >= 1.0:
            self.state.tokens -= 1.0
            return RateLimitResult(
                allowed=True,
                remaining=int(self.state.tokens),
                reset_in=self._time_to_next_token(),
                limit=self.config.requests_per_minute,
            )

        self.state.throttle_count += 1
        return RateLimitResult(
            allowed=False,
            remaining=0,
            reset_in=self._time_to_next_token(),
            limit=self.config.requests_per_minute,
            reason=self.config.reason or "Rate limit exceeded",
        )

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.state.last_update
        self.state.last_update = now

        # Calculate tokens to add (tokens per second = rpm / 60)
        tokens_per_second = self.config.requests_per_minute / 60.0
        new_tokens = elapsed * tokens_per_second

        self.state.tokens = min(
            float(self.config.burst_limit),
            self.state.tokens + new_tokens
        )

    def _time_to_next_token(self) -> float:
        """Calculate time until next token is available."""
        if self.state.tokens >= 1.0:
            return 0.0
        tokens_per_second = self.config.requests_per_minute / 60.0
        needed = 1.0 - self.state.tokens
        return needed / tokens_per_second

    def get_stats(self) -> Dict[str, Any]:
        """Get limiter statistics."""
        return {
            "requests": self.state.request_count,
            "throttled": self.state.throttle_count,
            "current_tokens": int(self.state.tokens),
            "limit": self.config.requests_per_minute,
            "burst": self.config.burst_limit,
            "enabled": self.config.enabled,
        }


class RateLimitManager:
    """
    Manages rate limits for multiple tools/operations.

    Loads configuration from manifest and environment,
    enforces limits, and logs throttling events.
    """

    def __init__(self):
        """Initialize rate limit manager."""
        self._limiters: Dict[str, TokenBucketLimiter] = {}
        self._tenant_limiters: Dict[str, Dict[str, TokenBucketLimiter]] = defaultdict(dict)
        self._global_config = RateLimitConfig()
        self._tool_configs: Dict[str, RateLimitConfig] = {}

    def load_from_manifest(self, manifest_path: Optional[Path] = None) -> None:
        """
        Load rate limit configurations from capabilities manifest.

        Args:
            manifest_path: Path to manifest file (auto-detected if not provided)
        """
        if manifest_path is None:
            search_paths = [
                Path.cwd() / "mcp" / "capabilities_manifest.json",
                Path(__file__).parent.parent.parent / "mcp" / "capabilities_manifest.json",
            ]
            for path in search_paths:
                if path.exists():
                    manifest_path = path
                    break

        if not manifest_path or not manifest_path.exists():
            logger.debug("No manifest found for rate limit configuration")
            return

        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            # Load tool-specific rate limits
            for category, tools in manifest.get("tools", {}).items():
                if isinstance(tools, list):
                    for tool in tools:
                        name = tool.get("name")
                        rate_limit = tool.get("rate_limit")
                        if name and rate_limit:
                            self._tool_configs[name] = RateLimitConfig(
                                requests_per_minute=rate_limit.get("requests_per_minute", 60),
                                burst_limit=rate_limit.get("burst_limit", 10),
                                enabled=True,
                                reason=rate_limit.get("reason", ""),
                            )

            logger.info(f"Loaded rate limits for {len(self._tool_configs)} tools from manifest")

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load manifest: {e}")

    def load_from_env(self) -> None:
        """
        Load rate limit overrides from environment variables.

        Supports:
        - FOUNDRY_RATE_LIMIT_DEFAULT: Default requests per minute
        - FOUNDRY_RATE_LIMIT_BURST: Default burst limit
        - FOUNDRY_RATE_LIMIT_{TOOL}: Per-tool override (e.g., FOUNDRY_RATE_LIMIT_RUN_TESTS=5)
        """
        # Global defaults
        default_rpm = os.environ.get("FOUNDRY_RATE_LIMIT_DEFAULT")
        if default_rpm:
            try:
                self._global_config.requests_per_minute = int(default_rpm)
            except ValueError:
                pass

        default_burst = os.environ.get("FOUNDRY_RATE_LIMIT_BURST")
        if default_burst:
            try:
                self._global_config.burst_limit = int(default_burst)
            except ValueError:
                pass

        # Per-tool overrides
        for key, value in os.environ.items():
            if key.startswith("FOUNDRY_RATE_LIMIT_") and key not in ("FOUNDRY_RATE_LIMIT_DEFAULT", "FOUNDRY_RATE_LIMIT_BURST"):
                tool_name = key[19:].lower().replace("_", "_")  # Keep underscores
                try:
                    rpm = int(value)
                    if tool_name not in self._tool_configs:
                        self._tool_configs[tool_name] = RateLimitConfig()
                    self._tool_configs[tool_name].requests_per_minute = rpm
                except ValueError:
                    pass

    def get_limiter(self, tool_name: str, tenant_id: Optional[str] = None) -> TokenBucketLimiter:
        """
        Get or create a rate limiter for a tool.

        Args:
            tool_name: Name of the tool
            tenant_id: Optional tenant ID for per-tenant limiting

        Returns:
            TokenBucketLimiter for the tool
        """
        if tenant_id:
            if tool_name not in self._tenant_limiters[tenant_id]:
                config = self._tool_configs.get(tool_name, self._global_config)
                self._tenant_limiters[tenant_id][tool_name] = TokenBucketLimiter(config)
            return self._tenant_limiters[tenant_id][tool_name]

        if tool_name not in self._limiters:
            config = self._tool_configs.get(tool_name, self._global_config)
            self._limiters[tool_name] = TokenBucketLimiter(config)
        return self._limiters[tool_name]

    def check_limit(
        self,
        tool_name: str,
        tenant_id: Optional[str] = None,
        log_on_throttle: bool = True
    ) -> RateLimitResult:
        """
        Check and enforce rate limit for a tool invocation.

        Args:
            tool_name: Name of the tool being invoked
            tenant_id: Optional tenant ID
            log_on_throttle: Whether to log throttle events

        Returns:
            RateLimitResult indicating if request is allowed
        """
        limiter = self.get_limiter(tool_name, tenant_id)
        result = limiter.acquire()

        if not result.allowed and log_on_throttle:
            self._log_throttle(tool_name, tenant_id, result)

        return result

    def _log_throttle(
        self,
        tool_name: str,
        tenant_id: Optional[str],
        result: RateLimitResult
    ) -> None:
        """Log a throttle event."""
        audit_log(
            "rate_limit_exceeded",
            tool=tool_name,
            tenant_id=tenant_id,
            limit=result.limit,
            reset_in=result.reset_in,
            reason=result.reason,
            success=False,
        )
        logger.warning(
            f"Rate limit exceeded for {tool_name}"
            + (f" (tenant: {tenant_id})" if tenant_id else "")
            + f": {result.reason}"
        )

    def log_auth_failure(
        self,
        tool_name: str,
        tenant_id: Optional[str] = None,
        reason: str = "Authentication failed"
    ) -> None:
        """
        Log an authentication failure.

        Args:
            tool_name: Tool that was accessed
            tenant_id: Optional tenant ID
            reason: Failure reason
        """
        audit_log(
            "auth_failure",
            tool=tool_name,
            tenant_id=tenant_id,
            reason=reason,
            success=False,
        )
        logger.warning(
            f"Auth failure for {tool_name}"
            + (f" (tenant: {tenant_id})" if tenant_id else "")
            + f": {reason}"
        )

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all limiters."""
        stats = {
            "schema_version": SCHEMA_VERSION,
            "global_config": {
                "requests_per_minute": self._global_config.requests_per_minute,
                "burst_limit": self._global_config.burst_limit,
            },
            "tools": {},
            "tenants": {},
        }

        for tool_name, limiter in self._limiters.items():
            stats["tools"][tool_name] = limiter.get_stats()

        for tenant_id, limiters in self._tenant_limiters.items():
            stats["tenants"][tenant_id] = {
                name: limiter.get_stats()
                for name, limiter in limiters.items()
            }

        return stats

    def reset(self, tool_name: Optional[str] = None, tenant_id: Optional[str] = None) -> None:
        """
        Reset rate limit state.

        Args:
            tool_name: Specific tool to reset (all if None)
            tenant_id: Specific tenant to reset (all if None)
        """
        if tool_name and tenant_id:
            if tenant_id in self._tenant_limiters and tool_name in self._tenant_limiters[tenant_id]:
                del self._tenant_limiters[tenant_id][tool_name]
        elif tool_name:
            if tool_name in self._limiters:
                del self._limiters[tool_name]
        elif tenant_id:
            if tenant_id in self._tenant_limiters:
                del self._tenant_limiters[tenant_id]
        else:
            self._limiters.clear()
            self._tenant_limiters.clear()


# Global instance
_manager: Optional[RateLimitManager] = None


def get_rate_limit_manager() -> RateLimitManager:
    """Get the global rate limit manager."""
    global _manager
    if _manager is None:
        _manager = RateLimitManager()
        _manager.load_from_manifest()
        _manager.load_from_env()
    return _manager


def check_rate_limit(
    tool_name: str,
    tenant_id: Optional[str] = None
) -> RateLimitResult:
    """
    Check rate limit for a tool invocation.

    Args:
        tool_name: Name of the tool
        tenant_id: Optional tenant ID

    Returns:
        RateLimitResult indicating if request is allowed
    """
    return get_rate_limit_manager().check_limit(tool_name, tenant_id)
