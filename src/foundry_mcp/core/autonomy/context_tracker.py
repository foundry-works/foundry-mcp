"""Robust context usage tracking for autonomous spec execution.

Implements a three-tier approach for determining effective context usage:
  Tier 1 (Sidecar): In sandbox mode, reads a JSON sidecar file written by
      a Claude Code hook for ground-truth context usage.
  Tier 2 (Caller-reported): Validates and hardens caller-reported values
      with monotonicity checks and staleness detection.
  Tier 3 (Estimated): When no fresh data is available, applies pessimistic
      linear estimation from the last known value.

The tracker ensures the orchestrator's context pause guard always has a
usable context_usage_pct, preventing runaway sessions that silently hit
the hard context window limit.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from foundry_mcp.core.autonomy.models import AutonomousSessionState

logger = logging.getLogger(__name__)

# Sidecar file path relative to workspace
SIDECAR_REL_PATH = Path("specs") / ".autonomy" / "context" / "context.json"

# Default maximum age for a sidecar file to be considered fresh
SIDECAR_MAX_AGE = timedelta(minutes=2)

# Env var that indicates sandbox mode (hook writes sidecar)
SANDBOX_ENV_VAR = "FOUNDRY_SANDBOX"

# Default context usage below this threshold is treated as a possible /clear reset
RESET_THRESHOLD_PCT = 10


def is_sandbox_mode() -> bool:
    """Check if running in sandbox mode where sidecar files are available."""
    val = os.environ.get(SANDBOX_ENV_VAR, "").strip().lower()
    return val in ("1", "true", "yes")


class ContextTracker:
    """Tracks and hardens context usage reporting for autonomous sessions.

    Provides get_effective_context_pct() which returns the best available
    context usage percentage along with its source tier.
    """

    def __init__(self, workspace_path: Path) -> None:
        self.workspace_path = workspace_path
        self._sidecar_path = workspace_path / SIDECAR_REL_PATH

    def get_effective_context_pct(
        self,
        session: AutonomousSessionState,
        caller_reported_pct: Optional[int],
        now: Optional[datetime] = None,
    ) -> Tuple[int, str]:
        """Determine the effective context usage percentage.

        Tries tiers in priority order:
          1. Sidecar file (sandbox mode only)
          2. Caller-reported value (with validation/hardening)
          3. Pessimistic estimation from last known value

        Args:
            session: Current session state
            caller_reported_pct: Value reported by the caller (may be None)
            now: Current timestamp (defaults to utcnow)

        Returns:
            Tuple of (effective_pct, source) where source is one of
            "sidecar", "caller", or "estimated".
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Compute sidecar max age from session limits if available
        sidecar_max_age = timedelta(
            seconds=getattr(session.limits, "sidecar_max_age_seconds", int(SIDECAR_MAX_AGE.total_seconds()))
        )

        # Tier 1: Sidecar (sandbox mode)
        if is_sandbox_mode():
            sidecar_pct = self._read_sidecar(now, sidecar_max_age)
            if sidecar_pct is not None:
                self._update_report_tracking(session, sidecar_pct, now)
                return sidecar_pct, "sidecar"

        # Tier 2: Caller-reported (with hardening)
        if caller_reported_pct is not None:
            hardened_pct = self._validate_and_harden(
                session, caller_reported_pct, now
            )
            self._update_report_tracking(session, hardened_pct, now)
            return hardened_pct, "caller"

        # Tier 3: Pessimistic estimation
        estimated_pct = self._estimate_growth(session)
        return estimated_pct, "estimated"

    def update_step_counter(self, session: AutonomousSessionState) -> None:
        """Increment the steps-since-last-report counter."""
        session.context.steps_since_last_report += 1

    # -----------------------------------------------------------------
    # Tier 1: Sidecar file reading
    # -----------------------------------------------------------------

    def _read_sidecar(self, now: datetime, max_age: Optional[timedelta] = None) -> Optional[int]:
        """Read context usage from sidecar file.

        The sidecar is written atomically by the Claude Code hook.
        We validate freshness and clamp to 0-100.

        Args:
            now: Current timestamp for freshness check.
            max_age: Maximum age for sidecar to be considered fresh.
                     Defaults to SIDECAR_MAX_AGE module constant.

        Returns:
            Context percentage if fresh sidecar exists, None otherwise.
        """
        if max_age is None:
            max_age = SIDECAR_MAX_AGE

        try:
            if not self._sidecar_path.exists():
                logger.debug("Sidecar file not found: %s", self._sidecar_path)
                return None

            raw = self._sidecar_path.read_text(encoding="utf-8")
            data = json.loads(raw)

            # Validate required fields
            pct = data.get("context_usage_pct")
            if pct is None or not isinstance(pct, (int, float)):
                logger.warning("Sidecar missing or invalid context_usage_pct")
                return None

            # Check timestamp freshness
            ts_str = data.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if now - ts > max_age:
                        logger.debug(
                            "Sidecar file stale: %s (age > %s)",
                            ts_str,
                            max_age,
                        )
                        return None
                except (ValueError, TypeError):
                    logger.warning("Sidecar has unparseable timestamp: %s", ts_str)
                    return None
            else:
                # No timestamp — fall through to stat-based freshness
                try:
                    mtime = datetime.fromtimestamp(
                        self._sidecar_path.stat().st_mtime, tz=timezone.utc
                    )
                    if now - mtime > max_age:
                        logger.debug("Sidecar file stale by mtime")
                        return None
                except OSError:
                    return None

            return max(0, min(100, int(pct)))

        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Failed to read sidecar: %s", e)
            return None

    # -----------------------------------------------------------------
    # Tier 2: Caller-reported validation and hardening
    # -----------------------------------------------------------------

    def _validate_and_harden(
        self,
        session: AutonomousSessionState,
        reported_pct: int,
        now: datetime,
    ) -> int:
        """Validate and harden a caller-reported context percentage.

        Applies:
        - Clamping to 0-100
        - Monotonicity check (rejects decreases unless likely /clear reset)
        - Staleness penalty for repeated identical values

        Returns:
            Hardened context percentage.
        """
        pct = max(0, min(100, reported_pct))
        last_pct = session.context.last_context_report_pct

        # Monotonicity check: reject decreases unless it looks like a /clear reset
        reset_threshold = getattr(session.limits, "context_reset_threshold_pct", RESET_THRESHOLD_PCT)
        if last_pct is not None and pct < last_pct:
            if pct < reset_threshold:
                # Accept: likely a /clear or new session context
                logger.info(
                    "Context usage dropped from %d%% to %d%% (treating as reset)",
                    last_pct,
                    pct,
                )
                session.context.consecutive_same_reports = 0
            else:
                # Reject decrease: keep the higher value
                logger.debug(
                    "Rejecting context decrease from %d%% to %d%%",
                    last_pct,
                    pct,
                )
                pct = last_pct

        # Staleness detection: consecutive identical reports
        if last_pct is not None and pct == last_pct:
            session.context.consecutive_same_reports += 1
        else:
            session.context.consecutive_same_reports = 0

        # Apply staleness penalty
        if (
            session.context.consecutive_same_reports
            >= session.limits.context_staleness_threshold
        ):
            penalty = session.limits.context_staleness_penalty_pct
            pct = min(100, pct + penalty)
            logger.info(
                "Context staleness penalty applied: +%d%% (now %d%%)",
                penalty,
                pct,
            )

        return pct

    # -----------------------------------------------------------------
    # Tier 3: Pessimistic estimation
    # -----------------------------------------------------------------

    def _estimate_growth(self, session: AutonomousSessionState) -> int:
        """Estimate context usage from last known value + step growth.

        Uses: last_known + (steps_since_report * avg_pct_per_step)
        Capped at 100. Returns 0 if no history available.
        """
        base = session.context.context_usage_pct or 0
        steps = session.context.steps_since_last_report
        growth = steps * session.limits.avg_pct_per_step
        estimated = min(100, base + growth)
        return estimated

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _update_report_tracking(
        self,
        session: AutonomousSessionState,
        pct: int,
        now: datetime,
    ) -> None:
        """Update session tracking fields after receiving a context report."""
        session.context.last_context_report_at = now
        session.context.last_context_report_pct = pct
        session.context.steps_since_last_report = 0
