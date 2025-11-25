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
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, TypeVar

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback


logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ServerConfig:
    """Server configuration with support for env vars and TOML overrides."""

    # Workspace configuration
    workspace_roots: List[Path] = field(default_factory=list)
    specs_dir: Optional[Path] = None
    journals_path: Optional[Path] = None

    # Logging configuration
    log_level: str = "INFO"
    structured_logging: bool = True

    # Authentication configuration
    api_keys: List[str] = field(default_factory=list)
    require_auth: bool = False

    # Server configuration
    server_name: str = "foundry-mcp"
    server_version: str = "0.1.0"

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

        # Log level
        if level := os.environ.get("FOUNDRY_MCP_LOG_LEVEL"):
            self.log_level = level.upper()

        # API keys
        if keys := os.environ.get("FOUNDRY_MCP_API_KEYS"):
            self.api_keys = [k.strip() for k in keys.split(",") if k.strip()]

        # Require auth
        if require := os.environ.get("FOUNDRY_MCP_REQUIRE_AUTH"):
            self.require_auth = require.lower() in ("true", "1", "yes")

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

def log_call(logger_name: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to log function calls with structured data.

    Args:
        logger_name: Optional logger name (defaults to function module)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        log = logging.getLogger(logger_name or func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            log.debug(f"Calling {func.__name__}", extra={
                "function": func.__name__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()),
            })
            try:
                result = func(*args, **kwargs)
                log.debug(f"Completed {func.__name__}", extra={
                    "function": func.__name__,
                    "success": True,
                })
                return result
            except Exception as e:
                log.error(f"Error in {func.__name__}: {e}", extra={
                    "function": func.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                raise

        return wrapper
    return decorator


def timed(metric_name: Optional[str] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
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
                log.info(f"Timer: {name}", extra={
                    "metric": name,
                    "duration_ms": round(elapsed * 1000, 2),
                    "success": True,
                })
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                log.info(f"Timer: {name}", extra={
                    "metric": name,
                    "duration_ms": round(elapsed * 1000, 2),
                    "success": False,
                    "error": str(e),
                })
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
