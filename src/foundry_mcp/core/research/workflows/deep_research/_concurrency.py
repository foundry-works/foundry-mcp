"""Shared async concurrency utilities for deep research phases.

Provides helpers for common patterns used across multiple phases,
such as propagating cancellation from ``asyncio.gather`` results.
"""

from __future__ import annotations

import asyncio


def check_gather_cancellation(results: list) -> None:
    """Re-raise ``CancelledError`` or ``KeyboardInterrupt`` from gather results.

    After ``asyncio.gather(*tasks, return_exceptions=True)``, exceptions are
    returned as list elements rather than raised.  This helper scans the
    results and re-raises cancellation/interrupt exceptions so that the
    caller's cancellation semantics are preserved.

    Args:
        results: The list returned by ``asyncio.gather(return_exceptions=True)``.

    Raises:
        asyncio.CancelledError: If any result is a ``CancelledError``.
        KeyboardInterrupt: If any result is a ``KeyboardInterrupt``.
    """
    for r in results:
        if isinstance(r, asyncio.CancelledError):
            raise r
        if isinstance(r, KeyboardInterrupt):
            raise r
