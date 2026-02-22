"""Execution and routing error classes.

Moved from foundry_mcp.core.executor, foundry_mcp.tools.unified.router,
and foundry_mcp.skills.foundry_implement_v2 for centralized error management.
"""

from typing import Any, Optional, Sequence


class ExecutorExhaustedError(Exception):
    """Raised when both dedicated and fallback executors are unavailable."""

    def __init__(self, message: str = "Executor pool exhausted"):
        super().__init__(message)
        self.message = message


class ActionRouterError(ValueError):
    """Raised when an unsupported action is requested."""

    def __init__(self, message: str, *, allowed_actions: Sequence[str]) -> None:
        super().__init__(message)
        self.allowed_actions = tuple(allowed_actions)


class FoundryImplementV2Error(RuntimeError):
    """Structured error for v2 skill preflight/loop orchestration."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        remediation: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        response: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.remediation = remediation
        self.details = details or {}
        self.response = response
