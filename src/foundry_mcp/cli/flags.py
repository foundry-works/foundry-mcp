"""CLI feature flag bootstrap bridging CLI options and discovery manifest.

Provides CLI-specific flag management that wraps the core feature flag
infrastructure, enabling runtime flag overrides via CLI options and
exposing flag status for tool discovery.

See docs/mcp_best_practices/14-feature-flags.md for guidance.
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar

import click

from foundry_mcp.core.feature_flags import (
    FeatureFlag,
    FeatureFlagRegistry,
    FlagState,
    get_registry,
)

__all__ = [
    "CLIFlagRegistry",
    "get_cli_flags",
    "apply_cli_flag_overrides",
    "flags_for_discovery",
    "with_flag_options",
]

T = TypeVar("T")


class CLIFlagRegistry:
    """Registry of CLI-specific feature flags.

    Wraps the core FeatureFlagRegistry and provides CLI-specific
    functionality like mapping command-line options to flag overrides
    and generating discovery manifests.

    Example:
        >>> registry = CLIFlagRegistry()
        >>> registry.register_cli_flag(
        ...     name="experimental_commands",
        ...     description="Enable experimental CLI commands",
        ...     default_enabled=False,
        ...     state=FlagState.BETA,
        ... )
        >>> registry.is_enabled("experimental_commands")
        False
    """

    def __init__(self, core_registry: Optional[FeatureFlagRegistry] = None):
        """Initialize with optional core registry.

        Args:
            core_registry: Core feature flag registry. Uses global if None.
        """
        self._core = core_registry or get_registry()
        self._cli_flags: Dict[str, FeatureFlag] = {}

    def register_cli_flag(
        self,
        name: str,
        description: str,
        default_enabled: bool = False,
        state: FlagState = FlagState.BETA,
        **kwargs: Any,
    ) -> None:
        """Register a CLI-specific feature flag.

        Creates a flag in both the CLI registry and the core registry
        for unified evaluation.

        Args:
            name: Unique flag identifier (e.g., "experimental_commands").
            description: Human-readable description for discovery.
            default_enabled: Whether enabled by default.
            state: Flag lifecycle state.
            **kwargs: Additional FeatureFlag parameters.
        """
        flag = FeatureFlag(
            name=name,
            description=description,
            default_enabled=default_enabled,
            state=state,
            **kwargs,
        )
        self._cli_flags[name] = flag
        try:
            self._core.register(flag)
        except ValueError:
            # Flag already registered in core, update our local copy
            existing = self._core.get(name)
            if existing:
                self._cli_flags[name] = existing

    def is_enabled(self, flag_name: str, default: bool = False) -> bool:
        """Check if a CLI flag is enabled.

        Args:
            flag_name: Name of the flag to check.
            default: Value if flag doesn't exist.

        Returns:
            True if flag is enabled, False otherwise.
        """
        return self._core.is_enabled(flag_name, client_id="cli", default=default)

    def apply_overrides(self, overrides: Dict[str, bool]) -> None:
        """Apply multiple flag overrides.

        Used to translate CLI options into flag state. Overrides
        persist for the duration of the CLI command execution.

        Args:
            overrides: Mapping of flag names to enabled/disabled state.
        """
        for flag_name, enabled in overrides.items():
            self._core.set_override("cli", flag_name, enabled)

    def clear_overrides(self) -> None:
        """Clear all CLI-applied overrides."""
        self._core.clear_all_overrides("cli")

    def get_discovery_manifest(self) -> Dict[str, Dict[str, Any]]:
        """Generate discovery manifest for CLI flags.

        Returns flag information suitable for tool discovery responses,
        allowing AI coding assistants to understand available features.

        Returns:
            Dictionary with flag names as keys and info dicts as values.
        """
        manifest = {}
        for name, flag in self._cli_flags.items():
            manifest[name] = {
                "enabled": self.is_enabled(name),
                "state": flag.state.value,
                "description": flag.description,
                "default": flag.default_enabled,
            }
            if flag.state == FlagState.DEPRECATED and flag.expires_at:
                manifest[name]["expires"] = flag.expires_at.isoformat()
        return manifest

    def list_flags(self) -> List[str]:
        """List all registered CLI flag names."""
        return list(self._cli_flags.keys())


# Global CLI flag registry
_cli_registry: Optional[CLIFlagRegistry] = None


def get_cli_flags() -> CLIFlagRegistry:
    """Get the global CLI flag registry."""
    global _cli_registry
    if _cli_registry is None:
        _cli_registry = CLIFlagRegistry()
    return _cli_registry


def apply_cli_flag_overrides(
    enable: Optional[List[str]] = None,
    disable: Optional[List[str]] = None,
) -> None:
    """Apply flag overrides from CLI options.

    Translates --enable-feature and --disable-feature CLI options
    into feature flag overrides.

    Args:
        enable: List of flag names to enable.
        disable: List of flag names to disable.
    """
    registry = get_cli_flags()
    overrides: Dict[str, bool] = {}

    if enable:
        for flag_name in enable:
            overrides[flag_name] = True

    if disable:
        for flag_name in disable:
            overrides[flag_name] = False

    if overrides:
        registry.apply_overrides(overrides)


def flags_for_discovery() -> Dict[str, Any]:
    """Get flag status for inclusion in discovery responses.

    Returns:
        Dictionary suitable for JSON serialization in discovery manifest.
    """
    return get_cli_flags().get_discovery_manifest()


def with_flag_options(
    func: Optional[Callable[..., T]] = None,
) -> Callable[..., T]:
    """Click decorator that adds --enable-feature/--disable-feature options.

    Adds common flag override options to a Click command and applies
    them before command execution.

    Example:
        >>> @cli.command()
        ... @with_flag_options
        ... def my_command():
        ...     # flags are already applied
        ...     if get_cli_flags().is_enabled("experimental"):
        ...         do_experimental_thing()

    Args:
        func: The Click command function to wrap.

    Returns:
        Decorated function with flag options.
    """

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        # Add the Click options
        f = click.option(
            "--enable-feature",
            "enable_features",
            multiple=True,
            help="Enable feature flag(s) for this command.",
        )(f)
        f = click.option(
            "--disable-feature",
            "disable_features",
            multiple=True,
            help="Disable feature flag(s) for this command.",
        )(f)

        # Wrap to apply flags before execution
        original = f

        @click.pass_context
        def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> T:
            enable = kwargs.pop("enable_features", ())
            disable = kwargs.pop("disable_features", ())

            apply_cli_flag_overrides(
                enable=list(enable) if enable else None,
                disable=list(disable) if disable else None,
            )

            try:
                # Call original with remaining kwargs
                return ctx.invoke(original, *args, **kwargs)
            finally:
                # Clean up overrides after command
                get_cli_flags().clear_overrides()

        # Preserve function metadata
        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)

    return decorator  # type: ignore[return-value]
