"""
Cross-provider validation, observability hooks, and resilience patterns.

Centralizes ProviderRequest validation, input hygiene, observability hooks,
and rate limiting consistent with docs/mcp_best_practices/{04,05,12,15}.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Optional, Set, TypeVar

from .base import (
    ProviderError,
    ProviderExecutionError,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderTimeoutError,
)

# ---------------------------------------------------------------------------
# Logging Configuration (per docs/mcp_best_practices/05-observability-telemetry.md)
# ---------------------------------------------------------------------------

logger = logging.getLogger("foundry_mcp.providers")

# ANSI escape sequence pattern for stripping from inputs
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[mGKHF]")

# Maximum prompt length (characters) to prevent token explosion
MAX_PROMPT_LENGTH = 500_000  # ~125k tokens at 4 chars/token

# Maximum metadata size (bytes) for logging/storage
MAX_METADATA_SIZE = 64 * 1024  # 64KB


# ---------------------------------------------------------------------------
# Input Hygiene (per docs/mcp_best_practices/04-validation-input-hygiene.md)
# ---------------------------------------------------------------------------


class ValidationError(ProviderExecutionError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs: object) -> None:
        self.field = field
        super().__init__(message, **kwargs)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    if not text:
        return text
    return ANSI_ESCAPE_PATTERN.sub("", text)


def ensure_utf8(text: str) -> str:
    """Ensure text is valid UTF-8, replacing invalid sequences."""
    if not text:
        return text
    # Encode to bytes and decode back, replacing invalid chars
    return text.encode("utf-8", errors="replace").decode("utf-8")


def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize prompt text for safe subprocess execution.

    - Ensures valid UTF-8 encoding
    - Strips ANSI escape sequences
    - Validates length limits
    """
    if not prompt:
        raise ValidationError("Prompt cannot be empty", field="prompt")

    # Ensure UTF-8
    clean = ensure_utf8(prompt)

    # Strip ANSI sequences
    clean = strip_ansi(clean)

    # Check length
    if len(clean) > MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"Prompt exceeds maximum length ({len(clean)} > {MAX_PROMPT_LENGTH})",
            field="prompt",
        )

    return clean


def validate_request(request: ProviderRequest) -> ProviderRequest:
    """
    Validate and sanitize a ProviderRequest before execution.

    Performs validation in order per best practices:
    1. Required fields present
    2. Type correctness (implicit via dataclass)
    3. Format validation
    4. Business rules
    """
    # 1. Required fields
    if not request.prompt:
        raise ValidationError("Prompt is required", field="prompt")

    # 3. Format validation - sanitize prompt
    sanitized_prompt = sanitize_prompt(request.prompt)

    # Sanitize system prompt if present
    sanitized_system = None
    if request.system_prompt:
        sanitized_system = sanitize_prompt(request.system_prompt)

    # 4. Business rules - validate numeric parameters
    if request.temperature is not None:
        if not (0.0 <= request.temperature <= 2.0):
            raise ValidationError(
                f"Temperature must be between 0.0 and 2.0, got {request.temperature}",
                field="temperature",
            )

    if request.max_tokens is not None:
        if request.max_tokens < 1:
            raise ValidationError(
                f"max_tokens must be positive, got {request.max_tokens}",
                field="max_tokens",
            )

    if request.timeout is not None:
        if request.timeout < 0:
            raise ValidationError(
                f"timeout must be non-negative, got {request.timeout}",
                field="timeout",
            )

    # Return sanitized request (create new instance with sanitized fields)
    return ProviderRequest(
        prompt=sanitized_prompt,
        system_prompt=sanitized_system,
        stream=request.stream,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        attachments=request.attachments,
        timeout=request.timeout,
        metadata=request.metadata,
    )


# ---------------------------------------------------------------------------
# Command Allowlists (per docs/mcp_best_practices/04-validation-input-hygiene.md)
# ---------------------------------------------------------------------------

# Common read-only commands that are safe across all providers
COMMON_SAFE_COMMANDS: Set[str] = {
    # File viewing
    "cat",
    "head",
    "tail",
    "bat",
    "less",
    "more",
    # Directory listing
    "ls",
    "tree",
    "pwd",
    "which",
    "whereis",
    # Search
    "grep",
    "rg",
    "ag",
    "find",
    "fd",
    "locate",
    # Git read-only
    "git log",
    "git show",
    "git diff",
    "git status",
    "git grep",
    "git blame",
    "git branch",
    "git rev-parse",
    "git describe",
    "git ls-tree",
    # Text processing
    "wc",
    "cut",
    "paste",
    "column",
    "sort",
    "uniq",
    # Data formats
    "jq",
    "yq",
    # File analysis
    "file",
    "stat",
    "du",
    "df",
    # Checksums
    "md5sum",
    "shasum",
    "sha256sum",
    "sha512sum",
}

# Commands that should never be allowed
BLOCKED_COMMANDS: Set[str] = {
    # Destructive operations
    "rm",
    "rmdir",
    "dd",
    "mkfs",
    "fdisk",
    "shred",
    # File modifications
    "touch",
    "mkdir",
    "mv",
    "cp",
    "chmod",
    "chown",
    "chgrp",
    # Text modification
    "sed",
    "awk",
    "ed",
    # Git write operations
    "git add",
    "git commit",
    "git push",
    "git pull",
    "git merge",
    "git rebase",
    "git reset",
    "git checkout",
    "git stash",
    # Package management
    "npm",
    "pip",
    "apt",
    "brew",
    "yum",
    "dnf",
    # System operations
    "sudo",
    "su",
    "halt",
    "reboot",
    "shutdown",
    "kill",
    "pkill",
    # Network (data exfiltration risk)
    "curl",
    "wget",
    "nc",
    "netcat",
    "ssh",
    "scp",
    "rsync",
}


def is_command_allowed(command: str, *, allowlist: Optional[Set[str]] = None) -> bool:
    """
    Check if a command is allowed based on allowlist/blocklist.

    Uses allowlist approach per best practices - only explicitly allowed
    commands pass validation.

    Args:
        command: The command string (may include arguments)
        allowlist: Optional custom allowlist (defaults to COMMON_SAFE_COMMANDS)

    Returns:
        True if command is allowed, False otherwise
    """
    if not command:
        return False

    # Extract base command (first word or "cmd arg" for compound commands)
    parts = command.strip().split()
    if not parts:
        return False

    base_cmd = parts[0]
    effective_allowlist = allowlist or COMMON_SAFE_COMMANDS

    # Check for blocked commands first (deny takes precedence)
    if base_cmd in BLOCKED_COMMANDS:
        return False

    # Check compound commands (e.g., "git log")
    if len(parts) >= 2:
        compound = f"{parts[0]} {parts[1]}"
        if compound in BLOCKED_COMMANDS:
            return False
        if compound in effective_allowlist:
            return True

    # Check base command in allowlist
    return base_cmd in effective_allowlist


# ---------------------------------------------------------------------------
# Observability & Telemetry (per docs/mcp_best_practices/05-observability-telemetry.md)
# ---------------------------------------------------------------------------


@dataclass
class ExecutionSpan:
    """Represents a provider execution span for telemetry."""

    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: Optional[str] = None
    operation: str = "provider_execute"
    provider_id: str = ""
    model: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: Optional[ProviderStatus] = None  # None = pending/in-progress
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(
        self,
        status: ProviderStatus,
        *,
        error: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Mark the span as finished."""
        self.ended_at = datetime.utcnow()
        self.duration_ms = (self.ended_at - self.started_at).total_seconds() * 1000
        self.status = status
        self.error = error
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "operation": self.operation,
            "provider_id": self.provider_id,
            "model": self.model,
            "started_at": self.started_at.isoformat() + "Z",
            "ended_at": self.ended_at.isoformat() + "Z" if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value if self.status else "pending",
            "error": self.error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata,
        }


def create_execution_span(
    provider_id: str,
    model: str = "",
    *,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExecutionSpan:
    """Create a new execution span for telemetry."""
    return ExecutionSpan(
        trace_id=trace_id or str(uuid.uuid4()),
        provider_id=provider_id,
        model=model,
        metadata=metadata or {},
    )


def log_span(span: ExecutionSpan, level: int = logging.INFO) -> None:
    """Log an execution span as structured JSON."""
    log_data = span.to_log_dict()
    logger.log(level, "Provider execution span", extra={"span_data": log_data})


# ---------------------------------------------------------------------------
# Retry Matrix (per docs/mcp_best_practices/12-timeout-resilience.md)
# ---------------------------------------------------------------------------

# Status codes that are safe to retry
RETRYABLE_STATUSES: Set[ProviderStatus] = {
    ProviderStatus.TIMEOUT,
    # Note: RATE_LIMITED not in ProviderStatus enum, would need extension
}


def is_retryable(status: ProviderStatus) -> bool:
    """Check if a provider status indicates a retryable error."""
    return status in RETRYABLE_STATUSES


def is_retryable_error(error: Exception) -> bool:
    """Check if an exception indicates a retryable error."""
    if isinstance(error, ProviderTimeoutError):
        return True
    if isinstance(error, ProviderError):
        # Check if the error has a retryable status
        return False  # Most provider errors are not retryable by default
    return False


# ---------------------------------------------------------------------------
# Circuit Breaker (per docs/mcp_best_practices/12-timeout-resilience.md)
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures exceeded threshold, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for provider resilience.

    Prevents cascade failures by stopping requests to failing providers.
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 1

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
            return self._state

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
            return False
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Success in half-open state, close the circuit
                self._state = CircuitState.CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open state, re-open the circuit
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# Global circuit breakers per provider
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_circuit_breaker_lock = Lock()


def get_circuit_breaker(provider_id: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a provider."""
    with _circuit_breaker_lock:
        if provider_id not in _circuit_breakers:
            _circuit_breakers[provider_id] = CircuitBreaker(name=provider_id)
        return _circuit_breakers[provider_id]


def reset_circuit_breakers() -> None:
    """Reset all circuit breakers (for testing)."""
    with _circuit_breaker_lock:
        _circuit_breakers.clear()


# ---------------------------------------------------------------------------
# Rate Limiting (per docs/mcp_best_practices/15-concurrency-patterns.md)
# ---------------------------------------------------------------------------


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for provider calls.

    Prevents overwhelming providers with too many concurrent requests.
    """

    name: str
    max_tokens: int = 10
    refill_rate: float = 1.0  # tokens per second

    _tokens: float = field(default=0.0, init=False)
    _last_refill: float = field(default_factory=time.time, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.max_tokens)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self.max_tokens, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens for a request.

        Returns:
            True if tokens acquired, False if rate limited
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate time to wait for tokens to become available."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            needed = tokens - self._tokens
            return needed / self.refill_rate


# Global rate limiters per provider
_rate_limiters: Dict[str, RateLimiter] = {}
_rate_limiter_lock = Lock()


def get_rate_limiter(provider_id: str) -> RateLimiter:
    """Get or create a rate limiter for a provider."""
    with _rate_limiter_lock:
        if provider_id not in _rate_limiters:
            _rate_limiters[provider_id] = RateLimiter(name=provider_id)
        return _rate_limiters[provider_id]


def reset_rate_limiters() -> None:
    """Reset all rate limiters (for testing)."""
    with _rate_limiter_lock:
        _rate_limiters.clear()


# ---------------------------------------------------------------------------
# Execution Wrapper (combines validation, observability, and resilience)
# ---------------------------------------------------------------------------

T = TypeVar("T")


def with_validation_and_resilience(
    provider_id: str,
    *,
    validate: bool = True,
    circuit_breaker: bool = True,
    rate_limit: bool = True,
    log_spans: bool = True,
) -> Callable[[Callable[..., ProviderResult]], Callable[..., ProviderResult]]:
    """
    Decorator combining validation, circuit breaking, rate limiting, and observability.

    Args:
        provider_id: The provider identifier
        validate: Whether to validate requests
        circuit_breaker: Whether to apply circuit breaker
        rate_limit: Whether to apply rate limiting
        log_spans: Whether to log execution spans

    Returns:
        Decorated function with resilience patterns applied
    """

    def decorator(func: Callable[..., ProviderResult]) -> Callable[..., ProviderResult]:
        @wraps(func)
        def wrapper(request: ProviderRequest, *args: Any, **kwargs: Any) -> ProviderResult:
            span = create_execution_span(provider_id) if log_spans else None

            try:
                # Input validation
                if validate:
                    request = validate_request(request)

                # Circuit breaker check
                if circuit_breaker:
                    cb = get_circuit_breaker(provider_id)
                    if not cb.can_execute():
                        raise ProviderExecutionError(
                            f"Circuit breaker open for provider {provider_id}",
                            provider=provider_id,
                        )

                # Rate limiting check
                if rate_limit:
                    rl = get_rate_limiter(provider_id)
                    if not rl.acquire():
                        raise ProviderExecutionError(
                            f"Rate limited for provider {provider_id}",
                            provider=provider_id,
                        )

                # Execute the actual function
                result = func(request, *args, **kwargs)

                # Record success
                if circuit_breaker:
                    get_circuit_breaker(provider_id).record_success()

                # Update span
                if span and result.tokens:
                    span.finish(
                        result.status,
                        input_tokens=result.tokens.input_tokens,
                        output_tokens=result.tokens.output_tokens,
                    )
                elif span:
                    span.finish(result.status)

                return result

            except Exception as e:
                # Record failure
                if circuit_breaker:
                    get_circuit_breaker(provider_id).record_failure()

                # Update span
                if span:
                    status = ProviderStatus.TIMEOUT if isinstance(e, ProviderTimeoutError) else ProviderStatus.ERROR
                    span.finish(status, error=str(e))

                raise

            finally:
                # Log span
                if span and log_spans:
                    log_span(span)

        return wrapper

    return decorator


__all__ = [
    # Validation
    "ValidationError",
    "strip_ansi",
    "ensure_utf8",
    "sanitize_prompt",
    "validate_request",
    # Command allowlists
    "COMMON_SAFE_COMMANDS",
    "BLOCKED_COMMANDS",
    "is_command_allowed",
    # Observability
    "ExecutionSpan",
    "create_execution_span",
    "log_span",
    # Retry
    "RETRYABLE_STATUSES",
    "is_retryable",
    "is_retryable_error",
    # Circuit breaker
    "CircuitState",
    "CircuitBreaker",
    "get_circuit_breaker",
    "reset_circuit_breakers",
    # Rate limiting
    "RateLimiter",
    "get_rate_limiter",
    "reset_rate_limiters",
    # Execution wrapper
    "with_validation_and_resilience",
]
