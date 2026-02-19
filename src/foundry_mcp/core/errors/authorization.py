"""Authorization error classes.

Moved from foundry_mcp.core.authorization for centralized error management.
"""

from typing import Optional, Sequence


class PathValidationError(Exception):
    """Raised when a path fails validation for runner isolation."""

    def __init__(self, path: str, reason: str, detail: Optional[str] = None):
        self.path = path
        self.reason = reason
        self.detail = detail
        super().__init__(f"Path validation failed: {reason} (path={path})")


class StdinTimeoutError(Exception):
    """Raised when a subprocess exceeds the stdin timeout cap."""

    def __init__(self, timeout_seconds: float, command: Sequence[str]):
        self.timeout_seconds = timeout_seconds
        self.command = command
        super().__init__(f"Subprocess killed after {timeout_seconds:g}s stdin timeout: {' '.join(command[:3])}...")
