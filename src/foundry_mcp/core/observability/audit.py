"""Audit logging for security events.

Provides structured audit logging with automatic correlation ID and
client ID population from request context.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from foundry_mcp.core.context import get_client_id, get_correlation_id

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events for security logging."""

    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT = "rate_limit"
    RESOURCE_ACCESS = "resource_access"
    TOOL_INVOCATION = "tool_invocation"
    PERMISSION_DENIED = "permission_denied"
    CONFIG_CHANGE = "config_change"
    TASK_TIMEOUT = "task_timeout"
    TASK_STALE = "task_stale"


@dataclass
class AuditEvent:
    """Structured audit event for security logging."""

    event_type: AuditEventType
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None

    def __post_init__(self) -> None:
        """Auto-populate correlation_id and client_id from context if not set."""
        if self.correlation_id is None:
            self.correlation_id = get_correlation_id() or None
        if self.client_id is None:
            ctx_client = get_client_id()
            if ctx_client and ctx_client != "anonymous":
                self.client_id = ctx_client

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "details": self.details,
        }
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.client_id:
            result["client_id"] = self.client_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.ip_address:
            result["ip_address"] = self.ip_address
        return result


class AuditLogger:
    """
    Structured audit logging for security events.

    Audit logs are written to a separate logger for easy filtering
    and compliance requirements.
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.audit")

    def log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        self._logger.info(f"AUDIT: {event.event_type.value}", extra={"audit": event.to_dict()})

    def auth_success(self, client_id: Optional[str] = None, **details: Any) -> None:
        """Log successful authentication."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.AUTH_SUCCESS,
                client_id=client_id,
                details=details,
            )
        )

    def auth_failure(
        self,
        reason: str,
        client_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **details: Any,
    ) -> None:
        """Log failed authentication."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.AUTH_FAILURE,
                client_id=client_id,
                ip_address=ip_address,
                details={"reason": reason, **details},
            )
        )

    def rate_limit(
        self,
        client_id: Optional[str] = None,
        limit: Optional[int] = None,
        **details: Any,
    ) -> None:
        """Log rate limit event."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.RATE_LIMIT,
                client_id=client_id,
                details={"limit": limit, **details},
            )
        )

    def resource_access(self, resource_type: str, resource_id: str, action: str = "read", **details: Any) -> None:
        """Log resource access."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.RESOURCE_ACCESS,
                details={
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "action": action,
                    **details,
                },
            )
        )

    def tool_invocation(
        self,
        tool_name: str,
        success: bool = True,
        duration_ms: Optional[float] = None,
        correlation_id: Optional[str] = None,
        **details: Any,
    ) -> None:
        """Log tool invocation."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.TOOL_INVOCATION,
                correlation_id=correlation_id,
                details={
                    "tool": tool_name,
                    "success": success,
                    "duration_ms": duration_ms,
                    **details,
                },
            )
        )


# Global audit logger
_audit = AuditLogger()


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger."""
    return _audit


def audit_log(event_type: str, **details: Any) -> None:
    """
    Convenience function for audit logging.

    Args:
        event_type: Type of event (auth_success, auth_failure, rate_limit,
                    resource_access, tool_invocation, permission_denied, config_change)
        **details: Additional details to include in the audit log
    """
    try:
        event_enum = AuditEventType(event_type)
    except ValueError:
        event_enum = AuditEventType.TOOL_INVOCATION
        details["original_event_type"] = event_type

    _audit.log(AuditEvent(event_type=event_enum, details=details))
