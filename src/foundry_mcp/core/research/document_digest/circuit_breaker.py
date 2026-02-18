"""Circuit breaker mixin for document digest.

Provides failure tracking and circuit breaker pattern to protect
against cascading failures during digest generation.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from foundry_mcp.core.observability import audit_log

logger = logging.getLogger(__name__)


class CircuitBreakerMixin:
    """Mixin providing circuit breaker pattern for DocumentDigestor.

    Tracks digest attempts in a sliding window and opens a circuit breaker
    when the failure ratio exceeds 70% with at least 5 samples.
    Auto-resets after 60 seconds.

    Requires the following instance attributes (set by DocumentDigestor.__init__):
        _attempt_window: list[tuple[float, bool]]
        _window_size: int
        _failure_threshold_ratio: float
        _min_samples: int
        _circuit_breaker_open: bool
        _circuit_breaker_opened_at: Optional[float]
        _circuit_breaker_reset_seconds: float
        _failure_window: list[float]
        _failure_window_size: int
        _failure_threshold: int
        _circuit_breaker_triggered: bool
    """

    # Type hints for attributes set by DocumentDigestor.__init__
    _attempt_window: list[tuple[float, bool]]
    _window_size: int
    _failure_threshold_ratio: float
    _min_samples: int
    _circuit_breaker_open: bool
    _circuit_breaker_opened_at: Optional[float]
    _circuit_breaker_reset_seconds: float
    _failure_window: list[float]
    _failure_window_size: int
    _failure_threshold: int
    _circuit_breaker_triggered: bool

    def _record_attempt(self, success: bool) -> None:
        """Record a digest attempt (success or failure) for circuit breaker.

        Maintains a sliding window of recent attempts. When failure ratio exceeds
        70% with at least 5 samples, the circuit breaker opens.

        Args:
            success: Whether the attempt was successful.
        """
        now = time.time()
        self._attempt_window.append((now, success))

        # Trim window to max size (keep most recent)
        if len(self._attempt_window) > self._window_size:
            self._attempt_window = self._attempt_window[-self._window_size:]

        # Calculate failure ratio
        total_attempts = len(self._attempt_window)
        failures = sum(1 for _, s in self._attempt_window if not s)
        failure_ratio = failures / total_attempts if total_attempts > 0 else 0.0

        # Check if threshold exceeded (only with minimum samples)
        if (
            total_attempts >= self._min_samples
            and failure_ratio >= self._failure_threshold_ratio
            and not self._circuit_breaker_open
        ):
            self._circuit_breaker_open = True
            self._circuit_breaker_opened_at = now
            self._circuit_breaker_triggered = True  # Legacy alias
            audit_log(
                "digest.circuit_breaker_triggered",
                window_failures=failures,
                window_size=total_attempts,
                failure_ratio=round(failure_ratio, 2),
                failure_threshold=self._failure_threshold_ratio,
            )
            logger.warning(
                "Digest circuit breaker opened: %.0f%% failures (%d/%d) in window",
                failure_ratio * 100,
                failures,
                total_attempts,
            )

    def _record_failure(self) -> None:
        """Record a digest failure and check for circuit breaker triggering.

        Maintains a sliding window of attempts. When failure ratio exceeds
        70% with at least 5 samples, emits a digest.circuit_breaker_triggered
        audit event.
        """
        self._record_attempt(success=False)
        # Legacy: also append to old failure_window for backward compatibility
        self._failure_window.append(time.time())
        if len(self._failure_window) > self._failure_window_size:
            self._failure_window = self._failure_window[-self._failure_window_size:]

    def _record_success(self) -> None:
        """Record a successful digest operation.

        Records success in the attempt window. If failure ratio drops below
        threshold, the circuit breaker closes.
        """
        self._record_attempt(success=True)

        # Check if circuit breaker should close (ratio dropped below threshold)
        total_attempts = len(self._attempt_window)
        failures = sum(1 for _, s in self._attempt_window if not s)
        failure_ratio = failures / total_attempts if total_attempts > 0 else 0.0

        if self._circuit_breaker_open and failure_ratio < self._failure_threshold_ratio:
            self._circuit_breaker_open = False
            self._circuit_breaker_opened_at = None
            self._circuit_breaker_triggered = False  # Legacy alias
            logger.info(
                "Digest circuit breaker closed: %.0f%% failures (%d/%d) - below threshold",
                failure_ratio * 100,
                failures,
                total_attempts,
            )

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open (should skip digest attempts).

        The circuit breaker auto-resets after 60 seconds to allow retry.

        Returns:
            True if circuit breaker is open and digest should be skipped.
        """
        if not self._circuit_breaker_open:
            return False

        # Check for auto-reset after timeout
        if self._circuit_breaker_opened_at is not None:
            elapsed = time.time() - self._circuit_breaker_opened_at
            if elapsed >= self._circuit_breaker_reset_seconds:
                logger.info(
                    "Digest circuit breaker auto-reset after %.1f seconds",
                    elapsed,
                )
                self._circuit_breaker_open = False
                self._circuit_breaker_opened_at = None
                self._circuit_breaker_triggered = False
                # Clear attempt window to start fresh
                self._attempt_window.clear()
                return False

        return True

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (e.g., for new iteration).

        Call this at the start of a new research iteration to allow
        retrying digests even if the breaker was previously open.
        """
        self._circuit_breaker_open = False
        self._circuit_breaker_opened_at = None
        self._circuit_breaker_triggered = False
        self._attempt_window.clear()
        self._failure_window.clear()
        logger.debug("Digest circuit breaker manually reset")
