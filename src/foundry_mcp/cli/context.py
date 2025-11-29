"""Context tracking for SDD CLI sessions.

Provides session markers, consultation limits, and context window tracking
for CLI-driven LLM workflows.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import os
import uuid


@dataclass
class SessionLimits:
    """Configurable limits for a CLI session."""
    max_consultations: int = 50  # Max LLM consultations per session
    max_context_tokens: int = 100000  # Approximate token budget
    warn_at_percentage: float = 0.8  # Warn at 80% usage


@dataclass
class SessionStats:
    """Runtime statistics for a CLI session."""
    consultation_count: int = 0
    estimated_tokens_used: int = 0
    commands_executed: int = 0
    errors_encountered: int = 0
    last_activity: Optional[str] = None


@dataclass
class ContextSession:
    """
    Tracks a CLI session with limits and markers.

    Used for:
    - Session identification across CLI invocations
    - Tracking consultation usage against limits
    - Providing context budget information
    """
    session_id: str
    started_at: str
    limits: SessionLimits = field(default_factory=SessionLimits)
    stats: SessionStats = field(default_factory=SessionStats)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def consultations_remaining(self) -> int:
        """Number of consultations remaining."""
        return max(0, self.limits.max_consultations - self.stats.consultation_count)

    @property
    def tokens_remaining(self) -> int:
        """Estimated tokens remaining in budget."""
        return max(0, self.limits.max_context_tokens - self.stats.estimated_tokens_used)

    @property
    def consultation_usage_percentage(self) -> float:
        """Percentage of consultation limit used."""
        if self.limits.max_consultations == 0:
            return 0.0
        return (self.stats.consultation_count / self.limits.max_consultations) * 100

    @property
    def token_usage_percentage(self) -> float:
        """Percentage of token budget used."""
        if self.limits.max_context_tokens == 0:
            return 0.0
        return (self.stats.estimated_tokens_used / self.limits.max_context_tokens) * 100

    @property
    def should_warn(self) -> bool:
        """Whether to warn about approaching limits."""
        return (
            self.consultation_usage_percentage >= self.limits.warn_at_percentage * 100 or
            self.token_usage_percentage >= self.limits.warn_at_percentage * 100
        )

    @property
    def at_limit(self) -> bool:
        """Whether session has reached its limits."""
        return (
            self.stats.consultation_count >= self.limits.max_consultations or
            self.stats.estimated_tokens_used >= self.limits.max_context_tokens
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "limits": {
                "max_consultations": self.limits.max_consultations,
                "max_context_tokens": self.limits.max_context_tokens,
                "warn_at_percentage": self.limits.warn_at_percentage,
            },
            "stats": {
                "consultation_count": self.stats.consultation_count,
                "estimated_tokens_used": self.stats.estimated_tokens_used,
                "commands_executed": self.stats.commands_executed,
                "errors_encountered": self.stats.errors_encountered,
                "last_activity": self.stats.last_activity,
            },
            "derived": {
                "consultations_remaining": self.consultations_remaining,
                "tokens_remaining": self.tokens_remaining,
                "consultation_usage_percentage": round(self.consultation_usage_percentage, 1),
                "token_usage_percentage": round(self.token_usage_percentage, 1),
                "should_warn": self.should_warn,
                "at_limit": self.at_limit,
            },
            "metadata": self.metadata,
        }


class ContextTracker:
    """
    Tracks CLI session context across command invocations.

    Provides:
    - Session markers for correlation
    - Consultation counting and limits
    - Context budget tracking
    """

    def __init__(self):
        self._session: Optional[ContextSession] = None
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load limits from environment variables."""
        self._default_limits = SessionLimits(
            max_consultations=int(os.environ.get("SDD_MAX_CONSULTATIONS", "50")),
            max_context_tokens=int(os.environ.get("SDD_MAX_CONTEXT_TOKENS", "100000")),
            warn_at_percentage=float(os.environ.get("SDD_WARN_PERCENTAGE", "0.8")),
        )

    def start_session(
        self,
        session_id: Optional[str] = None,
        limits: Optional[SessionLimits] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextSession:
        """
        Start a new tracking session.

        Args:
            session_id: Custom session ID (auto-generated if not provided)
            limits: Custom limits (uses defaults if not provided)
            metadata: Optional session metadata

        Returns:
            The created ContextSession
        """
        self._session = ContextSession(
            session_id=session_id or self._generate_session_id(),
            started_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            limits=limits or SessionLimits(
                max_consultations=self._default_limits.max_consultations,
                max_context_tokens=self._default_limits.max_context_tokens,
                warn_at_percentage=self._default_limits.warn_at_percentage,
            ),
            metadata=metadata or {},
        )
        return self._session

    def get_session(self) -> Optional[ContextSession]:
        """Get the current session, if any."""
        return self._session

    def get_or_create_session(self) -> ContextSession:
        """Get existing session or create a new one."""
        if self._session is None:
            return self.start_session()
        return self._session

    def record_consultation(self, estimated_tokens: int = 0) -> Dict[str, Any]:
        """
        Record an LLM consultation.

        Args:
            estimated_tokens: Estimated tokens used in this consultation

        Returns:
            Status dictionary with current usage info
        """
        session = self.get_or_create_session()
        session.stats.consultation_count += 1
        session.stats.estimated_tokens_used += estimated_tokens
        session.stats.last_activity = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        return {
            "consultation_number": session.stats.consultation_count,
            "consultations_remaining": session.consultations_remaining,
            "tokens_used": estimated_tokens,
            "tokens_remaining": session.tokens_remaining,
            "should_warn": session.should_warn,
            "at_limit": session.at_limit,
        }

    def record_command(self, error: bool = False) -> None:
        """Record a CLI command execution."""
        session = self.get_or_create_session()
        session.stats.commands_executed += 1
        if error:
            session.stats.errors_encountered += 1
        session.stats.last_activity = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def get_status(self) -> Dict[str, Any]:
        """Get current session status."""
        session = self._session
        if session is None:
            return {
                "active": False,
                "message": "No active session",
            }
        return {
            "active": True,
            **session.to_dict(),
        }

    def reset(self) -> None:
        """Reset the current session."""
        self._session = None

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"sdd_{uuid.uuid4().hex[:12]}"


# Global context tracker
_tracker: Optional[ContextTracker] = None


def get_context_tracker() -> ContextTracker:
    """Get the global context tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ContextTracker()
    return _tracker


def start_cli_session(
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ContextSession:
    """Start a new CLI session."""
    return get_context_tracker().start_session(session_id=session_id, metadata=metadata)


def get_session_status() -> Dict[str, Any]:
    """Get current session status."""
    return get_context_tracker().get_status()


def record_consultation(estimated_tokens: int = 0) -> Dict[str, Any]:
    """Record an LLM consultation."""
    return get_context_tracker().record_consultation(estimated_tokens)
