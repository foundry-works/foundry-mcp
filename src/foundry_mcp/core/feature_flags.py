"""
Feature flags infrastructure for foundry-mcp.

Provides feature flag management with lifecycle states, percentage rollouts,
client-specific overrides, and testing utilities.

See docs/mcp_best_practices/14-feature-flags.md for guidance.

Example:
    from foundry_mcp.core.feature_flags import (
        FlagState, FeatureFlag, FeatureFlagRegistry, feature_flag, flag_override
    )

    # Define flags
    registry = FeatureFlagRegistry()
    registry.register(FeatureFlag(
        name="new_algorithm",
        description="Use improved processing algorithm",
        state=FlagState.BETA,
        default_enabled=False,
    ))

    # Gate feature with decorator
    @feature_flag("new_algorithm")
    def process_data(data: dict) -> dict:
        return improved_process(data)

    # Test with override
    with flag_override("new_algorithm", True):
        result = process_data({"input": "test"})
"""

import hashlib
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Optional,
    ParamSpec,
    Set,
    TypeVar,
)

from foundry_mcp.core.responses import error_response

logger = logging.getLogger(__name__)

# Context variable for current client ID
current_client_id: ContextVar[str] = ContextVar("client_id", default="anonymous")


class FlagState(str, Enum):
    """Lifecycle state for feature flags.

    Feature flags progress through states as they mature:

    EXPERIMENTAL: Early development, opt-in only, may change without notice, no SLA
    BETA: Feature-complete but not fully validated, default off, opt-in available, some SLA
    STABLE: Production-ready, default on, full SLA guarantees
    DEPRECATED: Being phased out, warns on use, will be removed after expiration

    Example:
        >>> flag = FeatureFlag(name="my_feature", state=FlagState.EXPERIMENTAL)
        >>> if flag.state == FlagState.DEPRECATED:
        ...     logger.warning("Feature is deprecated")
    """

    EXPERIMENTAL = "experimental"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"


@dataclass
class FeatureFlag:
    """Definition of a feature flag with lifecycle and rollout configuration.

    Attributes:
        name: Unique identifier for the flag (used in code and config)
        description: Human-readable description of what this flag controls
        state: Current lifecycle state (experimental/beta/stable/deprecated)
        default_enabled: Whether the flag is enabled by default
        created_at: When the flag was created (defaults to now)
        expires_at: Optional expiration date after which flag should be removed
        owner: Team or individual responsible for this flag
        percentage_rollout: Percentage of clients that should have flag enabled (0-100)
        allowed_clients: If non-empty, only these clients can access the flag
        blocked_clients: These clients are explicitly denied access
        dependencies: Other flags that must be enabled for this flag to be enabled
        metadata: Additional key-value metadata for the flag

    Example:
        >>> flag = FeatureFlag(
        ...     name="new_search",
        ...     description="Enable new search algorithm",
        ...     state=FlagState.BETA,
        ...     default_enabled=False,
        ...     percentage_rollout=25.0,
        ...     owner="search-team",
        ... )
        >>> flag.is_expired()
        False
    """

    name: str
    description: str
    state: FlagState
    default_enabled: bool
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    owner: str = ""
    percentage_rollout: float = 100.0
    allowed_clients: Set[str] = field(default_factory=set)
    blocked_clients: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the flag has passed its expiration date."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "default_enabled": self.default_enabled,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "owner": self.owner,
            "percentage_rollout": self.percentage_rollout,
            "allowed_clients": list(self.allowed_clients),
            "blocked_clients": list(self.blocked_clients),
            "dependencies": list(self.dependencies),
            "metadata": self.metadata,
        }


class FeatureFlagRegistry:
    """Registry for managing feature flags with evaluation logic.

    Provides flag registration, client-specific evaluation, percentage rollouts,
    and override support for testing.

    Example:
        >>> registry = FeatureFlagRegistry()
        >>> registry.register(FeatureFlag(
        ...     name="new_feature",
        ...     description="Test feature",
        ...     state=FlagState.BETA,
        ...     default_enabled=False,
        ... ))
        >>> registry.is_enabled("new_feature", client_id="user123")
        False
    """

    def __init__(self) -> None:
        """Initialize an empty flag registry."""
        self._flags: Dict[str, FeatureFlag] = {}
        self._overrides: Dict[
            str, Dict[str, bool]
        ] = {}  # client_id -> flag_name -> value

    def register(self, flag: FeatureFlag) -> None:
        """Register a feature flag.

        Args:
            flag: The feature flag to register

        Raises:
            ValueError: If a flag with the same name already exists
        """
        if flag.name in self._flags:
            raise ValueError(f"Flag '{flag.name}' is already registered")
        self._flags[flag.name] = flag
        logger.debug(f"Registered feature flag: {flag.name} ({flag.state.value})")

    def get(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get a flag by name.

        Args:
            flag_name: Name of the flag to retrieve

        Returns:
            The FeatureFlag if found, None otherwise
        """
        return self._flags.get(flag_name)

    def is_enabled(
        self,
        flag_name: str,
        client_id: Optional[str] = None,
        default: bool = False,
    ) -> bool:
        """Check if a feature flag is enabled for a client.

        Evaluation order:
        1. Check for client-specific override
        2. Check if flag exists
        3. Check if flag is expired
        4. Check client blocklist
        5. Check client allowlist
        6. Check flag dependencies
        7. Evaluate percentage rollout
        8. Return default_enabled value

        Args:
            flag_name: Name of the flag to check
            client_id: Client ID for evaluation (defaults to context variable)
            default: Value to return if flag doesn't exist

        Returns:
            True if the flag is enabled for this client, False otherwise
        """
        client_id = client_id or current_client_id.get()

        # Check for client-specific override first
        if client_id in self._overrides:
            if flag_name in self._overrides[client_id]:
                return self._overrides[client_id][flag_name]

        # Check if flag exists
        flag = self._flags.get(flag_name)
        if not flag:
            logger.warning(f"Unknown feature flag: {flag_name}")
            return default

        # Warn if deprecated
        if flag.state == FlagState.DEPRECATED:
            logger.warning(
                f"Deprecated feature flag '{flag_name}' accessed",
                extra={"client_id": client_id, "flag": flag_name},
            )

        # Check expiration
        if flag.is_expired():
            logger.warning(f"Expired feature flag: {flag_name}")
            return default

        # Check blocklist
        if flag.blocked_clients and client_id in flag.blocked_clients:
            return False

        # Check allowlist (empty means all allowed)
        if flag.allowed_clients and client_id not in flag.allowed_clients:
            return False

        # Check dependencies
        for dep_flag in flag.dependencies:
            if not self.is_enabled(dep_flag, client_id):
                return False

        # Evaluate percentage rollout
        if not self._evaluate_percentage(flag, client_id):
            return False

        return flag.default_enabled

    def _evaluate_percentage(self, flag: FeatureFlag, client_id: str) -> bool:
        """Evaluate percentage-based rollout using deterministic hashing.

        Args:
            flag: The feature flag to evaluate
            client_id: Client ID for bucket assignment

        Returns:
            True if client falls within rollout percentage, False otherwise
        """
        if flag.percentage_rollout >= 100.0:
            return True

        if flag.percentage_rollout <= 0.0:
            return False

        # Use consistent hashing for stable bucket assignment
        hash_input = f"{flag.name}:{client_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 100) + 1

        return bucket <= flag.percentage_rollout

    def set_override(self, client_id: str, flag_name: str, enabled: bool) -> None:
        """Set a client-specific override for a flag.

        Args:
            client_id: The client to override for
            flag_name: Name of the flag to override
            enabled: Override value
        """
        if client_id not in self._overrides:
            self._overrides[client_id] = {}
        self._overrides[client_id][flag_name] = enabled
        logger.debug(f"Set override: {flag_name}={enabled} for client {client_id}")

    def clear_override(self, client_id: str, flag_name: str) -> None:
        """Clear a client-specific override.

        Args:
            client_id: The client to clear override for
            flag_name: Name of the flag to clear
        """
        if client_id in self._overrides:
            self._overrides[client_id].pop(flag_name, None)
            if not self._overrides[client_id]:
                del self._overrides[client_id]

    def clear_all_overrides(self, client_id: Optional[str] = None) -> None:
        """Clear all overrides, optionally for a specific client.

        Args:
            client_id: If provided, only clear overrides for this client
        """
        if client_id:
            self._overrides.pop(client_id, None)
        else:
            self._overrides.clear()

    def get_flags_for_capabilities(
        self,
        client_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Get flag status for capabilities endpoint.

        Returns a dictionary suitable for including in a capabilities response,
        showing each flag's enabled status, state, and description.

        Args:
            client_id: Client ID for evaluation (defaults to context variable)

        Returns:
            Dictionary mapping flag names to their status information
        """
        client_id = client_id or current_client_id.get()
        result = {}

        for name, flag in self._flags.items():
            status = {
                "enabled": self.is_enabled(name, client_id),
                "state": flag.state.value,
                "description": flag.description,
            }

            if flag.state == FlagState.DEPRECATED:
                expires_str = (
                    flag.expires_at.isoformat() if flag.expires_at else "unspecified"
                )
                status["deprecation_notice"] = (
                    f"This feature is deprecated and will be removed after {expires_str}"
                )

            result[name] = status

        return result

    def list_flags(self) -> Dict[str, FeatureFlag]:
        """Get all registered flags.

        Returns:
            Dictionary mapping flag names to FeatureFlag objects
        """
        return dict(self._flags)


# Global registry instance
_default_registry = FeatureFlagRegistry()


def _register_builtin_flags(registry: FeatureFlagRegistry) -> None:
    """Register Foundry MCP built-in flags.

    This keeps commonly-referenced flags discoverable and avoids noisy
    "Unknown feature flag" warnings when running without env overrides.
    """

    builtin_flags = [
        FeatureFlag(
            name="unified_tools_phase2",
            description="Enable unified routers for medium tool families",
            state=FlagState.BETA,
            default_enabled=False,
            percentage_rollout=0.0,
            owner="foundry-mcp core",
        ),
        FeatureFlag(
            name="unified_tools_phase3",
            description="Enable unified routers for large tool families",
            state=FlagState.BETA,
            default_enabled=False,
            percentage_rollout=0.0,
            owner="foundry-mcp core",
        ),
        FeatureFlag(
            name="unified_manifest",
            description="Expose only the 17 unified tools for discovery/token reduction",
            state=FlagState.BETA,
            default_enabled=False,
            percentage_rollout=0.0,
            owner="foundry-mcp core",
        ),
    ]

    for flag in builtin_flags:
        if registry.get(flag.name) is not None:
            continue
        try:
            registry.register(flag)
        except ValueError:
            # Already registered (race or re-import)
            continue


_register_builtin_flags(_default_registry)


def get_registry() -> FeatureFlagRegistry:
    """Get the default feature flag registry."""
    return _default_registry


# Alias for server.py compatibility
get_flag_service = get_registry


P = ParamSpec("P")
R = TypeVar("R")


def feature_flag(
    flag_name: str,
    *,
    fallback: Optional[Callable[P, Any]] = None,
    registry: Optional[FeatureFlagRegistry] = None,
) -> Callable[[Callable[P, R]], Callable[P, Any]]:
    """Decorator to gate function execution behind a feature flag.

    When the feature flag is not enabled for the current client, the decorated
    function will not execute. Instead, it returns a FEATURE_DISABLED error
    response (or invokes the fallback if provided).

    Args:
        flag_name: Name of the feature flag to check
        fallback: Optional fallback function to call when flag is disabled.
            If provided, this function is called with the same arguments.
            If not provided, returns a FEATURE_DISABLED error response.
        registry: Optional registry to use. Defaults to the global registry.

    Returns:
        Decorated function that checks the flag before execution

    Example:
        >>> @feature_flag("new_algorithm")
        ... def process_data(data: dict) -> dict:
        ...     return {"processed": True}
        ...
        >>> # When flag is disabled, returns error response
        >>> result = process_data({"input": "test"})
        >>> result["success"]
        False
        >>> result["data"]["error_code"]
        'FEATURE_DISABLED'

        >>> # With fallback function
        >>> @feature_flag("new_algorithm", fallback=legacy_process)
        ... def process_data(data: dict) -> dict:
        ...     return {"processed": True, "algorithm": "new"}
        ...
        >>> # When flag is disabled, calls legacy_process instead
    """

    def decorator(func: Callable[P, R]) -> Callable[P, Any]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            reg = registry or get_registry()

            if not reg.is_enabled(flag_name):
                logger.debug(
                    f"Feature flag '{flag_name}' is disabled, "
                    f"blocking execution of {func.__name__}"
                )

                if fallback is not None:
                    return fallback(*args, **kwargs)

                # Return FEATURE_DISABLED error response
                response = error_response(
                    message=f"Feature '{flag_name}' is not enabled",
                    error_code="FEATURE_DISABLED",
                    error_type="feature_flag",
                    data={
                        "feature": flag_name,
                    },
                    remediation="Contact support to enable this feature or check feature flag configuration",
                )
                return asdict(response)

            return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def flag_override(
    flag_name: str,
    enabled: bool,
    *,
    client_id: Optional[str] = None,
    registry: Optional[FeatureFlagRegistry] = None,
) -> Generator[None, None, None]:
    """Context manager for temporarily overriding a feature flag value.

    Useful for testing code paths that depend on feature flags without
    modifying the actual flag configuration. The override is automatically
    cleaned up when the context exits, even if an exception occurs.

    Args:
        flag_name: Name of the feature flag to override
        enabled: The override value (True to enable, False to disable)
        client_id: Optional client ID for the override (defaults to current_client_id)
        registry: Optional registry to use (defaults to global registry)

    Yields:
        None

    Example:
        >>> registry = FeatureFlagRegistry()
        >>> registry.register(FeatureFlag(
        ...     name="new_feature",
        ...     description="Test",
        ...     state=FlagState.BETA,
        ...     default_enabled=False,
        ... ))
        >>> # Flag is disabled by default
        >>> registry.is_enabled("new_feature")
        False
        >>> # Override to enable temporarily
        >>> with flag_override("new_feature", True, registry=registry):
        ...     registry.is_enabled("new_feature")
        True
        >>> # Back to original state after context
        >>> registry.is_enabled("new_feature")
        False

        >>> # Use in tests to force flag states
        >>> with flag_override("new_feature", True):
        ...     result = my_flagged_function()  # Runs with flag enabled
    """
    reg = registry or get_registry()
    cid = client_id or current_client_id.get()

    # Check if there's an existing override we need to restore
    had_previous_override = False
    previous_override_value = False

    if cid in reg._overrides and flag_name in reg._overrides[cid]:
        had_previous_override = True
        previous_override_value = reg._overrides[cid][flag_name]

    try:
        reg.set_override(cid, flag_name, enabled)
        yield
    finally:
        # Restore previous state
        if had_previous_override:
            reg.set_override(cid, flag_name, previous_override_value)
        else:
            reg.clear_override(cid, flag_name)


# Export all public symbols
__all__ = [
    "FlagState",
    "FeatureFlag",
    "FeatureFlagRegistry",
    "current_client_id",
    "get_registry",
    "get_flag_service",
    "feature_flag",
    "flag_override",
]
