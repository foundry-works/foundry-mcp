"""Crash recovery and process lifecycle infrastructure.

Manages active research session tracking, crash handlers, SIGTERM
graceful shutdown, and atexit cleanup to ensure research state is
persisted on abnormal exit.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import traceback
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from foundry_mcp.core.research.memory import ResearchMemory
    from foundry_mcp.core.research.models.deep_research import DeepResearchState

logger = logging.getLogger(__name__)

# Track active research sessions for crash recovery
# Protected by _active_sessions_lock to prevent race conditions during iteration
_active_research_sessions: dict[str, DeepResearchState] = {}
_active_sessions_lock = threading.Lock()
_active_research_memory: Optional[ResearchMemory] = None

_crash_handler_installed = False
_crash_handler_lock = threading.Lock()

# Store previous SIGTERM handler to chain calls
_previous_sigterm_handler: Optional[Any] = None


def _persist_active_sessions() -> None:
    """Best-effort persistence for active research sessions.

    Note: Caller should hold _active_sessions_lock or call during shutdown
    when no other threads are modifying the dict.
    """
    memory = _active_research_memory
    if memory is None:
        try:
            from foundry_mcp.core.research.memory import ResearchMemory

            memory = ResearchMemory()
        except Exception as exc:
            print(
                f"Failed to initialize ResearchMemory for persistence: {exc}",
                file=sys.stderr,
            )
            return

    # Copy values while holding lock to avoid iteration issues
    with _active_sessions_lock:
        sessions_snapshot = list(_active_research_sessions.values())

    for state in sessions_snapshot:
        try:
            memory.save_deep_research(state)
        except Exception:
            pass


def _crash_handler(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
    """Handle uncaught exceptions by logging to stderr and writing crash markers.

    This handler catches process-level crashes that escape normal exception handling
    and ensures we have visibility into what went wrong.
    """
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

    # Take a snapshot of sessions under lock to avoid race conditions
    with _active_sessions_lock:
        session_keys = list(_active_research_sessions.keys())
        sessions_snapshot = list(_active_research_sessions.items())

    # Always write to stderr for visibility
    print(
        f"\n{'=' * 60}\n"
        f"DEEP RESEARCH CRASH HANDLER\n"
        f"{'=' * 60}\n"
        f"Exception: {exc_type.__name__}: {exc_value}\n"
        f"Active sessions: {session_keys}\n"
        f"Traceback:\n{tb_str}"
        f"{'=' * 60}\n",
        file=sys.stderr,
        flush=True,
    )

    # Try to save crash markers for active research sessions
    for research_id, state in sessions_snapshot:
        try:
            state.metadata["crash"] = True
            state.metadata["crash_error"] = str(exc_value)
            # Write crash marker file
            crash_path = Path.home() / ".foundry-mcp" / "research" / "deep_research" / f"{research_id}.crash"
            crash_path.parent.mkdir(parents=True, exist_ok=True)
            crash_path.write_text(tb_str)
        except Exception:
            pass  # Best effort - don't fail the crash handler
    _persist_active_sessions()

    # Call original handler
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def _sigterm_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Handle SIGTERM by cancelling all active deep research sessions.

    Sets cancel events on all active background tasks via the task registry
    and marks sessions as INTERRUPTED (distinct from user-initiated CANCELLED).
    Does not call sys.exit — allows the MCP server to handle its own shutdown.
    """
    # Take snapshot under lock
    with _active_sessions_lock:
        session_keys = list(_active_research_sessions.keys())
        sessions_snapshot = list(_active_research_sessions.items())

    if session_keys:
        logger.warning(
            "SIGTERM received — cancelling %d active deep research session(s): %s",
            len(session_keys),
            session_keys,
        )
    else:
        logger.info("SIGTERM received — no active deep research sessions to cancel")

    # Cancel background tasks via the task registry
    cancelled_ids: list[str] = []
    try:
        from foundry_mcp.core import task_registry

        for research_id in session_keys:
            bg_task = task_registry.get(research_id)
            if bg_task is not None:
                # Set the cancel event so workflow checks pick it up
                bg_task._cancel_event.set()
                cancelled_ids.append(research_id)
    except Exception as exc:
        logger.error("Failed to cancel background tasks via registry: %s", exc)

    # Mark each active session as INTERRUPTED and persist
    for research_id, state in sessions_snapshot:
        try:
            if state.completed_at is None:
                state.mark_interrupted(reason="SIGTERM")
                logger.info(
                    "Session %s marked INTERRUPTED at phase=%s, iteration=%d",
                    research_id,
                    state.phase.value,
                    state.iteration,
                )
        except Exception as exc:
            logger.error(
                "Failed to mark session %s as interrupted: %s",
                research_id,
                exc,
            )

    # Best-effort persist all sessions
    _persist_active_sessions()

    logger.info(
        "SIGTERM shutdown complete — cancelled=%s, interrupted=%d sessions",
        cancelled_ids,
        len(sessions_snapshot),
    )

    # Chain to previous handler if it was callable
    if callable(_previous_sigterm_handler):
        _previous_sigterm_handler(signum, frame)


def _cleanup_on_exit() -> None:
    """Mark any active sessions as interrupted on normal exit."""
    # Take snapshot under lock to avoid race conditions
    with _active_sessions_lock:
        sessions_snapshot = list(_active_research_sessions.items())

    for _research_id, state in sessions_snapshot:
        if state.completed_at is None:
            state.mark_interrupted(reason="process_exit")
    _persist_active_sessions()


def install_crash_handler() -> None:
    """Install crash handler, SIGTERM handler, and atexit hook (idempotent)."""
    global _crash_handler_installed, _previous_sigterm_handler
    if _crash_handler_installed:
        return
    with _crash_handler_lock:
        if _crash_handler_installed:
            return
        sys.excepthook = _crash_handler

        # Install SIGTERM handler only on main thread (signal requirement)
        if threading.current_thread() is threading.main_thread():
            _previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _sigterm_handler)

        import atexit

        atexit.register(_cleanup_on_exit)
        _crash_handler_installed = True
