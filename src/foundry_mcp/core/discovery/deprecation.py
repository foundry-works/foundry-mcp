"""
Deprecation utilities for MCP tools.

Provides decorators and introspection functions for marking tools as
deprecated and querying deprecation status/metadata.
"""

from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


def deprecated_tool(
    replacement: str,
    removal_version: str,
) -> Callable[[F], F]:
    """
    Decorator to mark MCP tools as deprecated.

    Modifies the function's docstring to include deprecation notice and
    adds deprecation warning to response meta.warnings.

    Args:
        replacement: Name of the tool that replaces this one
        removal_version: Version in which this tool will be removed

    Returns:
        Decorated function that adds deprecation warnings to responses

    Example:
        >>> @mcp.tool()
        ... @deprecated_tool(replacement="get_user", removal_version="3.0.0")
        ... def fetch_user(user_id: str) -> dict:
        ...     '''Fetch user by ID (deprecated).'''
        ...     return get_user(user_id)
    """

    def decorator(func: F) -> F:
        original_doc = func.__doc__ or ""

        # Update docstring with deprecation notice
        func.__doc__ = f"""[DEPRECATED] {original_doc}

        ⚠️  This tool is deprecated and will be removed in version {removal_version}.
        Use '{replacement}' instead.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Add deprecation warning to response if it has meta
            if isinstance(result, dict):
                if "meta" not in result:
                    result["meta"] = {"version": "response-v2"}

                meta: Dict[str, Any] = result["meta"]
                if "warnings" not in meta:
                    meta["warnings"] = []

                deprecation_warning = (
                    f"DEPRECATED: '{func.__name__}' will be removed in {removal_version}. Use '{replacement}' instead."
                )

                warnings_list: list[str] = meta["warnings"]
                if deprecation_warning not in warnings_list:
                    warnings_list.append(deprecation_warning)

            return result

        # Store deprecation metadata on the wrapper for introspection
        wrapper._deprecated = True  # type: ignore[attr-defined]
        wrapper._replacement = replacement  # type: ignore[attr-defined]
        wrapper._removal_version = removal_version  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def is_deprecated(func: Callable[..., Any]) -> bool:
    """
    Check if a function is marked as deprecated.

    Args:
        func: Function to check

    Returns:
        True if function has @deprecated_tool decorator
    """
    return getattr(func, "_deprecated", False)


def get_deprecation_info(func: Callable[..., Any]) -> Optional[Dict[str, str]]:
    """
    Get deprecation info for a deprecated function.

    Args:
        func: Function to check

    Returns:
        Dict with replacement and removal_version, or None if not deprecated
    """
    if not is_deprecated(func):
        return None

    return {
        "replacement": getattr(func, "_replacement", ""),
        "removal_version": getattr(func, "_removal_version", ""),
    }
