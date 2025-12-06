"""Structured logging configuration with automatic context injection.

This module provides logging utilities that automatically inject request context
(correlation ID, client ID, etc.) into log records, enabling unified log
correlation across the foundry-mcp codebase.

Usage:
    from foundry_mcp.core.logging_config import (
        get_logger,
        configure_logging,
        ContextFilter,
    )

    # Get a logger with automatic context injection
    logger = get_logger(__name__)

    # Within a request context, logs automatically include correlation_id
    with sync_request_context() as ctx:
        logger.info("Processing request")  # Includes correlation_id in record
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TextIO, Union

from foundry_mcp.core.context import (
    get_client_id,
    get_correlation_id,
    get_start_time,
    get_trace_context,
)

__all__ = [
    "ContextFilter",
    "StructuredFormatter",
    "configure_logging",
    "get_logger",
    "add_context_filter",
]


class ContextFilter(logging.Filter):
    """Logging filter that injects request context into log records.

    This filter adds the following attributes to every log record:
    - correlation_id: Current request correlation ID
    - client_id: Current client identifier
    - elapsed_ms: Milliseconds since request start
    - trace_id: W3C trace ID if available

    Example:
        handler = logging.StreamHandler()
        handler.addFilter(ContextFilter())
        logger.addHandler(handler)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context attributes to the log record.

        Args:
            record: Log record to enrich

        Returns:
            Always True (allows all records through)
        """
        # Always inject context, even if empty
        record.correlation_id = get_correlation_id() or "-"
        record.client_id = get_client_id() or "anonymous"

        # Calculate elapsed time
        start_time = get_start_time()
        if start_time > 0:
            import time

            record.elapsed_ms = round((time.time() - start_time) * 1000, 2)
        else:
            record.elapsed_ms = 0.0

        # Add trace context if available
        trace_ctx = get_trace_context()
        if trace_ctx:
            record.trace_id = trace_ctx.trace_id
            record.parent_id = trace_ctx.parent_id
            record.trace_sampled = trace_ctx.is_sampled
        else:
            record.trace_id = "-"
            record.parent_id = "-"
            record.trace_sampled = False

        return True


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for machine-readable output.

    Produces newline-delimited JSON logs suitable for log aggregation
    systems like Elasticsearch, Splunk, or CloudWatch.

    Example output:
        {"timestamp":"2024-01-15T10:30:45.123Z","level":"INFO",
         "logger":"foundry_mcp.tools.specs","message":"Spec loaded",
         "correlation_id":"req_a1b2c3d4e5f6","client_id":"user123",
         "elapsed_ms":42.5}
    """

    def __init__(
        self,
        *,
        include_extra: bool = True,
        include_exception: bool = True,
        timestamp_format: str = "iso",
    ):
        """Initialize the structured formatter.

        Args:
            include_extra: Include extra record attributes
            include_exception: Include exception info in output
            timestamp_format: "iso" for ISO 8601, "unix" for Unix timestamp
        """
        super().__init__()
        self.include_extra = include_extra
        self.include_exception = include_exception
        self.timestamp_format = timestamp_format

        # Standard attributes to exclude from "extra"
        self._standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "exc_info",
            "exc_text",
            "stack_info",
            # Context attributes (handled separately)
            "correlation_id",
            "client_id",
            "elapsed_ms",
            "trace_id",
            "parent_id",
            "trace_sampled",
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log line
        """
        # Build base log entry
        log_entry: Dict[str, Any] = {}

        # Timestamp
        if self.timestamp_format == "iso":
            log_entry["timestamp"] = datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat()
        else:
            log_entry["timestamp"] = record.created

        # Core fields
        log_entry["level"] = record.levelname
        log_entry["logger"] = record.name
        log_entry["message"] = record.getMessage()

        # Location (optional, useful for debugging)
        log_entry["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Context fields (injected by ContextFilter)
        log_entry["correlation_id"] = getattr(record, "correlation_id", "-")
        log_entry["client_id"] = getattr(record, "client_id", "anonymous")
        log_entry["elapsed_ms"] = getattr(record, "elapsed_ms", 0.0)

        # Trace context
        trace_id = getattr(record, "trace_id", "-")
        if trace_id and trace_id != "-":
            log_entry["trace"] = {
                "trace_id": trace_id,
                "parent_id": getattr(record, "parent_id", "-"),
                "sampled": getattr(record, "trace_sampled", False),
            }

        # Exception info
        if self.include_exception and record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Extra attributes
        if self.include_extra:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in self._standard_attrs:
                    try:
                        # Ensure value is JSON-serializable
                        json.dumps(value)
                        extra[key] = value
                    except (TypeError, ValueError):
                        extra[key] = str(value)
            if extra:
                log_entry["extra"] = extra

        return json.dumps(log_entry, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter with context prefix.

    Produces logs in format:
        [LEVEL] [correlation_id] logger: message

    Example:
        [INFO] [req_a1b2c3] foundry_mcp.tools.specs: Spec loaded successfully
    """

    def __init__(
        self,
        *,
        include_timestamp: bool = True,
        include_location: bool = False,
        color: bool = False,
    ):
        """Initialize the human-readable formatter.

        Args:
            include_timestamp: Include timestamp in output
            include_location: Include file:line in output
            color: Use ANSI colors (disabled by default per project standards)
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_location = include_location
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record for human reading.

        Args:
            record: Log record to format

        Returns:
            Formatted log line
        """
        parts = []

        # Timestamp
        if self.include_timestamp:
            ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
            parts.append(ts)

        # Level
        parts.append(f"[{record.levelname}]")

        # Correlation ID
        corr_id = getattr(record, "correlation_id", "-")
        if corr_id and corr_id != "-":
            parts.append(f"[{corr_id}]")

        # Logger name (shortened)
        logger_name = record.name
        if logger_name.startswith("foundry_mcp."):
            logger_name = logger_name[12:]  # Remove "foundry_mcp."
        parts.append(f"{logger_name}:")

        # Message
        parts.append(record.getMessage())

        # Location
        if self.include_location:
            parts.append(f"({record.filename}:{record.lineno})")

        result = " ".join(parts)

        # Exception
        if record.exc_info:
            result += "\n" + self.formatException(record.exc_info)

        return result


def configure_logging(
    *,
    level: Union[int, str] = logging.INFO,
    format: str = "structured",  # "structured" or "human"
    stream: Optional[TextIO] = None,
    add_context: bool = True,
) -> logging.Logger:
    """Configure the root foundry_mcp logger.

    Args:
        level: Log level (default: INFO)
        format: Output format ("structured" for JSON, "human" for readable)
        stream: Output stream (default: stderr)
        add_context: Add ContextFilter for automatic context injection

    Returns:
        Configured root logger for foundry_mcp
    """
    logger = logging.getLogger("foundry_mcp")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    # Add formatter
    if format == "structured":
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(HumanReadableFormatter())

    # Add context filter
    if add_context:
        handler.addFilter(ContextFilter())

    logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with automatic context injection.

    The logger inherits from the foundry_mcp root logger, which should
    be configured via configure_logging(). Context injection happens
    automatically via the ContextFilter on the root handler.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    # Ensure name is under foundry_mcp namespace
    if not name.startswith("foundry_mcp"):
        name = f"foundry_mcp.{name}"

    return logging.getLogger(name)


def add_context_filter(handler: logging.Handler) -> None:
    """Add ContextFilter to an existing handler.

    Useful for adding context injection to handlers created elsewhere.

    Args:
        handler: Handler to add filter to
    """
    handler.addFilter(ContextFilter())
