"""Storage and concurrency error classes.

Moved from foundry_mcp.core.pagination, foundry_mcp.core.intake,
foundry_mcp.core.autonomy.state_migrations, foundry_mcp.core.research.state_migrations,
and foundry_mcp.core.autonomy.memory for centralized error management.
"""

from typing import Optional


class CursorError(Exception):
    """Error during cursor encoding or decoding.

    Attributes:
        cursor: The invalid cursor string (if decoding).
        reason: Description of what went wrong.
    """

    def __init__(
        self,
        message: str,
        cursor: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        super().__init__(message)
        self.cursor = cursor
        self.reason = reason


class LockAcquisitionError(Exception):
    """Raised when file lock cannot be acquired within timeout."""
    pass


class MigrationError(Exception):
    """Raised when a state migration fails.

    Consolidated from foundry_mcp.core.autonomy.state_migrations and
    foundry_mcp.core.research.state_migrations (both were identical).
    """

    pass


class VersionConflictError(Exception):
    """Raised when optimistic version check fails during save."""

    def __init__(self, session_id: str, expected: int, actual: int) -> None:
        self.session_id = session_id
        self.expected_version = expected
        self.actual_version = actual
        super().__init__(
            f"Version conflict for session {session_id}: expected {expected}, on-disk {actual}"
        )


class SessionCorrupted(Exception):
    """Raised when a session file exists but cannot be parsed or validated."""

    def __init__(self, session_id: str, reason: str) -> None:
        self.session_id = session_id
        self.reason = reason
        super().__init__(f"Session {session_id} is corrupted: {reason}")
