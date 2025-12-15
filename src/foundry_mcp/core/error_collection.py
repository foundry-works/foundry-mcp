"""Error data collection infrastructure for foundry-mcp.

Provides structured error capture, fingerprinting, and collection for
future introspection and analysis. This module focuses on data collection
only - analysis/insights are handled separately.

Usage:
    from foundry_mcp.core.error_collection import get_error_collector

    collector = get_error_collector()

    # Collect tool errors
    try:
        do_something()
    except Exception as e:
        collector.collect_tool_error(
            tool_name="my-tool",
            error=e,
            input_params={"key": "value"},
            duration_ms=42.5,
        )
        raise

    # Collect AI provider errors
    collector.collect_provider_error(
        provider_id="gemini",
        error=exc,
        request_context={"workflow": "plan_review"},
    )

    # Query errors
    records, cursor = collector.query(tool_name="my-tool", limit=10)
    stats = collector.get_stats(group_by="fingerprint")
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import traceback
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from foundry_mcp.core.context import get_correlation_id
from foundry_mcp.core.observability import redact_sensitive_data

if TYPE_CHECKING:
    from foundry_mcp.core.error_store import ErrorStore

logger = logging.getLogger(__name__)


# =============================================================================
# ErrorRecord Dataclass
# =============================================================================


@dataclass
class ErrorRecord:
    """Rich error record for storage and analysis.

    Captures comprehensive error context including classification,
    timing, provider info (for AI errors), and aggregation metadata.

    Attributes:
        id: Unique error identifier (err_<uuid>)
        fingerprint: Computed signature for deduplication
        error_code: Machine-readable error code (from ErrorCode enum)
        error_type: Error category (from ErrorType enum)
        tool_name: Name of the tool that generated the error
        correlation_id: Request correlation ID for tracing
        message: User-facing error message (sanitized)
        exception_type: Python exception class name
        stack_trace: Sanitized stack trace (server-side only)
        provider_id: AI provider ID (for provider errors)
        provider_model: AI model used (for provider errors)
        provider_status: Provider status code (for provider errors)
        input_summary: Redacted summary of input parameters
        timestamp: ISO 8601 timestamp of error occurrence
        duration_ms: Operation duration before failure
        count: Occurrence count (for aggregated records)
        first_seen: First occurrence timestamp
        last_seen: Last occurrence timestamp
    """

    # Identity
    id: str
    fingerprint: str

    # Classification
    error_code: str
    error_type: str

    # Context
    tool_name: str
    correlation_id: str

    # Error details
    message: str
    exception_type: Optional[str] = None
    stack_trace: Optional[str] = None

    # Provider context (for AI errors)
    provider_id: Optional[str] = None
    provider_model: Optional[str] = None
    provider_status: Optional[str] = None

    # Input context (redacted)
    input_summary: Optional[Dict[str, Any]] = None

    # Timing
    timestamp: str = ""
    duration_ms: Optional[float] = None

    # Aggregation support
    count: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None

    def __post_init__(self) -> None:
        """Set defaults for timestamp fields."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
        if not self.first_seen:
            self.first_seen = self.timestamp
        if not self.last_seen:
            self.last_seen = self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorRecord":
        """Create ErrorRecord from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# ErrorFingerprinter
# =============================================================================


class ErrorFingerprinter:
    """Generate consistent fingerprints for error deduplication.

    Fingerprints capture the "signature" of an error for grouping:
    - Same tool + same error_code + same exception_type = same fingerprint
    - Provider errors include provider_id in fingerprint
    - Message patterns are normalized (remove IDs, timestamps, etc.)
    """

    # Patterns to normalize in error messages
    _NORMALIZE_PATTERNS = [
        # UUIDs
        (r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<UUID>"),
        # Timestamps (ISO 8601)
        (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?", "<TIMESTAMP>"),
        # Unix timestamps
        (r"\b\d{10,13}\b", "<UNIX_TS>"),
        # File paths
        (r"(/[\w\-.]+)+(\.\w+)?", "<PATH>"),
        # Line numbers in stack traces
        (r"line \d+", "line <N>"),
        # Numeric IDs
        (r"\b\d{5,}\b", "<ID>"),
        # Correlation IDs
        (r"(req|tool|task|err)_[a-f0-9]{12}", "<CORR_ID>"),
    ]

    def __init__(self) -> None:
        """Initialize fingerprinter with compiled patterns."""
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self._NORMALIZE_PATTERNS
        ]

    def _normalize_message(self, message: str) -> str:
        """Normalize a message by removing dynamic content."""
        normalized = message
        for pattern, replacement in self._compiled_patterns:
            normalized = pattern.sub(replacement, normalized)
        return normalized.strip().lower()

    def fingerprint(
        self,
        error_code: str,
        error_type: str,
        tool_name: str,
        exception_type: Optional[str] = None,
        message: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> str:
        """Generate a fingerprint based on error characteristics.

        Args:
            error_code: Machine-readable error code
            error_type: Error category
            tool_name: Name of the tool
            exception_type: Python exception class name
            message: Error message (will be normalized)
            provider_id: AI provider ID (for provider errors)

        Returns:
            SHA256 hash of normalized components (first 16 chars)
        """
        components = [
            tool_name,
            error_code,
            error_type,
        ]

        if exception_type:
            components.append(exception_type)

        if provider_id:
            components.append(f"provider:{provider_id}")

        if message:
            normalized_msg = self._normalize_message(message)
            # Only include first 100 chars of normalized message
            components.append(normalized_msg[:100])

        fingerprint_input = "|".join(components)
        hash_obj = hashlib.sha256(fingerprint_input.encode("utf-8"))
        return hash_obj.hexdigest()[:16]


# =============================================================================
# ErrorCollector
# =============================================================================


class ErrorCollector:
    """Central error collection point with enrichment and storage.

    Provides methods to collect tool errors and AI provider errors,
    automatically enriching them with context (correlation ID, timing)
    and storing them for later analysis.
    """

    def __init__(
        self,
        store: Optional["ErrorStore"] = None,
        fingerprinter: Optional[ErrorFingerprinter] = None,
        enabled: bool = True,
        include_stack_traces: bool = True,
        redact_inputs: bool = True,
    ) -> None:
        """Initialize the error collector.

        Args:
            store: Error storage backend (lazy-loaded if None)
            fingerprinter: Fingerprint generator (created if None)
            enabled: Whether error collection is enabled
            include_stack_traces: Whether to capture stack traces
            redact_inputs: Whether to redact input parameters
        """
        self._store = store
        self._fingerprinter = fingerprinter or ErrorFingerprinter()
        self._enabled = enabled
        self._include_stack_traces = include_stack_traces
        self._redact_inputs = redact_inputs
        self._lock = threading.Lock()

    @property
    def store(self) -> "ErrorStore":
        """Get the error store, lazy-loading if necessary."""
        if self._store is None:
            from foundry_mcp.core.error_store import get_error_store

            self._store = get_error_store()
        return self._store

    def is_enabled(self) -> bool:
        """Check if error collection is enabled."""
        return self._enabled

    def initialize(
        self,
        store: "ErrorStore",
        config: Any,
    ) -> None:
        """Initialize the collector with a store and configuration.

        Called by server initialization to set up the error collection
        infrastructure with the configured storage backend.

        Args:
            store: Error storage backend
            config: ErrorCollectionConfig instance
        """
        with self._lock:
            self._store = store
            self._enabled = config.enabled
            self._include_stack_traces = config.include_stack_traces
            self._redact_inputs = config.redact_inputs
            logger.debug(
                f"ErrorCollector initialized: enabled={self._enabled}, "
                f"include_stack_traces={self._include_stack_traces}, "
                f"redact_inputs={self._redact_inputs}"
            )

    def _generate_id(self) -> str:
        """Generate a unique error ID."""
        return f"err_{uuid.uuid4().hex[:12]}"

    def _extract_exception_info(
        self, error: Exception
    ) -> Tuple[str, Optional[str]]:
        """Extract exception type and sanitized stack trace.

        Args:
            error: The exception to extract info from

        Returns:
            Tuple of (exception_type, sanitized_stack_trace)
        """
        exception_type = type(error).__name__

        stack_trace = None
        if self._include_stack_traces:
            try:
                # Get the full traceback
                tb_lines = traceback.format_exception(
                    type(error), error, error.__traceback__
                )
                raw_trace = "".join(tb_lines)
                # Redact sensitive data from stack trace
                stack_trace = redact_sensitive_data(raw_trace)
            except Exception:
                pass

        return exception_type, stack_trace

    # Sensitive key name patterns that should always be redacted
    _SENSITIVE_KEY_PATTERNS = [
        "api_key", "apikey", "api-key",
        "password", "passwd", "pwd",
        "secret", "token", "auth",
        "credential", "private", "key",
        "bearer", "access_token", "refresh_token",
    ]

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a parameter key name indicates sensitive data."""
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in self._SENSITIVE_KEY_PATTERNS)

    def _redact_input_params(
        self, params: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Redact sensitive data from input parameters.

        Args:
            params: Input parameters dictionary

        Returns:
            Redacted parameters or None
        """
        if not params or not self._redact_inputs:
            return None

        try:
            # Create a summary with redacted values
            summary: Dict[str, Any] = {}
            for key, value in params.items():
                if value is None:
                    summary[key] = None
                elif self._is_sensitive_key(key):
                    # Always redact values for sensitive key names
                    summary[key] = "<REDACTED>"
                elif isinstance(value, str):
                    # Redact string values using pattern matching
                    redacted = redact_sensitive_data(value)
                    # Truncate long strings
                    if len(redacted) > 100:
                        redacted = redacted[:100] + "..."
                    summary[key] = redacted
                elif isinstance(value, (int, float, bool)):
                    summary[key] = value
                elif isinstance(value, (list, tuple)):
                    summary[key] = f"<{type(value).__name__}[{len(value)}]>"
                elif isinstance(value, dict):
                    summary[key] = f"<dict[{len(value)}]>"
                else:
                    summary[key] = f"<{type(value).__name__}>"
            return summary
        except Exception:
            return None

    def _map_exception_to_codes(
        self, error: Exception
    ) -> Tuple[str, str]:
        """Map an exception to error_code and error_type.

        Args:
            error: The exception to map

        Returns:
            Tuple of (error_code, error_type)
        """
        # Import here to avoid circular imports
        from foundry_mcp.core.responses import ErrorCode, ErrorType

        exception_name = type(error).__name__

        # Map common exceptions to error codes
        exception_mapping: Dict[str, Tuple[str, str]] = {
            "FileNotFoundError": (ErrorCode.NOT_FOUND.value, ErrorType.NOT_FOUND.value),
            "PermissionError": (ErrorCode.FORBIDDEN.value, ErrorType.AUTHORIZATION.value),
            "ValueError": (ErrorCode.VALIDATION_ERROR.value, ErrorType.VALIDATION.value),
            "TypeError": (ErrorCode.VALIDATION_ERROR.value, ErrorType.VALIDATION.value),
            "KeyError": (ErrorCode.NOT_FOUND.value, ErrorType.NOT_FOUND.value),
            "TimeoutError": (ErrorCode.UNAVAILABLE.value, ErrorType.UNAVAILABLE.value),
            "ConnectionError": (ErrorCode.UNAVAILABLE.value, ErrorType.UNAVAILABLE.value),
            "JSONDecodeError": (ErrorCode.INVALID_FORMAT.value, ErrorType.VALIDATION.value),
        }

        if exception_name in exception_mapping:
            return exception_mapping[exception_name]

        # Check for provider-specific exceptions
        if "Provider" in exception_name or "AI" in exception_name:
            if "Timeout" in exception_name:
                return (ErrorCode.AI_PROVIDER_TIMEOUT.value, ErrorType.AI_PROVIDER.value)
            elif "Unavailable" in exception_name:
                return (ErrorCode.AI_NO_PROVIDER.value, ErrorType.AI_PROVIDER.value)
            else:
                return (ErrorCode.AI_PROVIDER_ERROR.value, ErrorType.AI_PROVIDER.value)

        # Default to internal error
        return (ErrorCode.INTERNAL_ERROR.value, ErrorType.INTERNAL.value)

    def collect_tool_error(
        self,
        tool_name: str,
        error: Exception,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        input_params: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
    ) -> Optional[ErrorRecord]:
        """Collect and store a tool error with full context.

        Args:
            tool_name: Name of the tool that generated the error
            error: The exception that occurred
            error_code: Override error code (auto-detected if None)
            error_type: Override error type (auto-detected if None)
            input_params: Tool input parameters (will be redacted)
            duration_ms: Operation duration before failure

        Returns:
            The stored ErrorRecord, or None if collection failed/disabled
        """
        if not self._enabled:
            return None

        try:
            # Auto-detect error codes if not provided
            if error_code is None or error_type is None:
                detected_code, detected_type = self._map_exception_to_codes(error)
                error_code = error_code or detected_code
                error_type = error_type or detected_type

            # Extract exception info
            exception_type, stack_trace = self._extract_exception_info(error)

            # Get correlation ID from context
            correlation_id = get_correlation_id() or "unknown"

            # Generate fingerprint
            fingerprint = self._fingerprinter.fingerprint(
                error_code=error_code,
                error_type=error_type,
                tool_name=tool_name,
                exception_type=exception_type,
                message=str(error),
            )

            # Create error record
            record = ErrorRecord(
                id=self._generate_id(),
                fingerprint=fingerprint,
                error_code=error_code,
                error_type=error_type,
                tool_name=tool_name,
                correlation_id=correlation_id,
                message=str(error),
                exception_type=exception_type,
                stack_trace=stack_trace,
                input_summary=self._redact_input_params(input_params),
                duration_ms=duration_ms,
            )

            # Store the record
            self.store.append(record)

            logger.debug(
                "Collected tool error",
                extra={
                    "error_id": record.id,
                    "fingerprint": record.fingerprint,
                    "tool_name": tool_name,
                    "error_code": error_code,
                },
            )

            return record

        except Exception as e:
            logger.warning(f"Failed to collect tool error: {e}")
            return None

    def collect_provider_error(
        self,
        provider_id: str,
        error: Optional[Exception] = None,
        provider_result: Optional[Any] = None,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ErrorRecord]:
        """Collect and store an AI provider error.

        Args:
            provider_id: AI provider identifier
            error: The exception that occurred (if any)
            provider_result: ProviderResult object (if available)
            request_context: Request context (workflow, prompt_id, etc.)

        Returns:
            The stored ErrorRecord, or None if collection failed/disabled
        """
        if not self._enabled:
            return None

        try:
            # Import here to avoid circular imports
            from foundry_mcp.core.responses import ErrorCode, ErrorType

            # Determine error details
            if error:
                exception_type, stack_trace = self._extract_exception_info(error)
                message = str(error)
                error_code, error_type = self._map_exception_to_codes(error)
            else:
                exception_type = None
                stack_trace = None
                message = "Provider error"
                error_code = ErrorCode.AI_PROVIDER_ERROR.value
                error_type = ErrorType.AI_PROVIDER.value

            # Extract provider result info if available
            provider_model = None
            provider_status = None
            if provider_result:
                provider_model = getattr(provider_result, "model", None)
                if hasattr(provider_result, "status"):
                    provider_status = str(provider_result.status)
                # Include stderr in message if present
                stderr = getattr(provider_result, "stderr", None)
                if stderr:
                    message = f"{message} - {stderr[:200]}"

            # Get correlation ID
            correlation_id = get_correlation_id() or "unknown"

            # Determine tool name from context
            tool_name = "ai-consultation"
            if request_context and "workflow" in request_context:
                tool_name = f"ai-consultation:{request_context['workflow']}"

            # Generate fingerprint (includes provider_id)
            fingerprint = self._fingerprinter.fingerprint(
                error_code=error_code,
                error_type=error_type,
                tool_name=tool_name,
                exception_type=exception_type,
                message=message,
                provider_id=provider_id,
            )

            # Create error record
            record = ErrorRecord(
                id=self._generate_id(),
                fingerprint=fingerprint,
                error_code=error_code,
                error_type=error_type,
                tool_name=tool_name,
                correlation_id=correlation_id,
                message=message,
                exception_type=exception_type,
                stack_trace=stack_trace,
                provider_id=provider_id,
                provider_model=provider_model,
                provider_status=provider_status,
                input_summary=self._redact_input_params(request_context),
            )

            # Store the record
            self.store.append(record)

            logger.debug(
                "Collected provider error",
                extra={
                    "error_id": record.id,
                    "fingerprint": record.fingerprint,
                    "provider_id": provider_id,
                    "error_code": error_code,
                },
            )

            return record

        except Exception as e:
            logger.warning(f"Failed to collect provider error: {e}")
            return None

    def query(
        self,
        tool_name: Optional[str] = None,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        fingerprint: Optional[str] = None,
        provider_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ErrorRecord]:
        """Query collected errors with filtering.

        Args:
            tool_name: Filter by tool name
            error_code: Filter by error code
            error_type: Filter by error type
            fingerprint: Filter by fingerprint
            provider_id: Filter by provider ID
            since: ISO 8601 timestamp - filter errors after this time
            until: ISO 8601 timestamp - filter errors before this time
            limit: Maximum records to return
            offset: Number of records to skip

        Returns:
            List of matching ErrorRecord objects
        """
        return self.store.query(
            tool_name=tool_name,
            error_code=error_code,
            error_type=error_type,
            fingerprint=fingerprint,
            provider_id=provider_id,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )

    def get(self, error_id: str) -> Optional[ErrorRecord]:
        """Get a specific error record by ID.

        Args:
            error_id: Error record ID

        Returns:
            ErrorRecord or None if not found
        """
        return self.store.get(error_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated error statistics.

        Returns:
            Statistics dictionary with total_errors, unique_patterns,
            by_tool, by_error_code, and top_patterns
        """
        return self.store.get_stats()


# =============================================================================
# Singleton Instance
# =============================================================================

_collector: Optional[ErrorCollector] = None
_collector_lock = threading.Lock()


def get_error_collector() -> ErrorCollector:
    """Get the singleton ErrorCollector instance.

    Returns:
        ErrorCollector singleton instance
    """
    global _collector

    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = ErrorCollector()

    return _collector


def reset_error_collector() -> None:
    """Reset the singleton error collector (mainly for testing)."""
    global _collector
    with _collector_lock:
        _collector = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "ErrorRecord",
    # Fingerprinting
    "ErrorFingerprinter",
    # Collection
    "ErrorCollector",
    "get_error_collector",
    "reset_error_collector",
]
