"""Protocol definition for deep research workflow mixins.

Defines ``DeepResearchWorkflowProtocol`` — the structural interface that
all phase mixins expect from the composed ``DeepResearchWorkflow`` class.

Using a Protocol instead of bare ``Any`` annotations catches interface
drift at type-check time: if a mixin references a method that the
concrete workflow no longer provides, ``mypy`` / ``pyright`` will flag it.

Usage in mixins::

    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from ._protocols import DeepResearchWorkflowProtocol

    class SomePhaseMixin:
        # Narrows ``self`` for type-checkers without changing runtime MRO
        if TYPE_CHECKING:
            _proto: DeepResearchWorkflowProtocol
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult


@runtime_checkable
class DeepResearchWorkflowProtocol(Protocol):
    """Structural interface expected by all deep research phase mixins.

    Covers the cross-cutting attributes and methods that every mixin
    may call on ``self`` at runtime.  Phase-specific stubs (e.g.
    ``_execute_topic_research_async``) remain in each mixin's own
    ``TYPE_CHECKING`` block, since not every mixin needs them.
    """

    # --- Instance attributes ---
    config: Any  # ResearchConfig at runtime
    memory: Any  # ResearchMemory at runtime

    # --- Cross-cutting methods ---

    def _write_audit_event(
        self,
        state: DeepResearchState,
        event_name: str,
        *,
        data: dict[str, Any] | None = ...,
        level: str = ...,
    ) -> None:
        """Record an audit event for observability."""
        ...

    def _check_cancellation(self, state: DeepResearchState) -> None:
        """Raise ``asyncio.CancelledError`` if cancellation is requested."""
        ...

    async def _execute_provider_async(self, **kwargs: Any) -> WorkflowResult:
        """Execute an LLM provider call."""
        ...
