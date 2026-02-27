"""Protocol definition for deep research workflow mixins.

Defines ``DeepResearchWorkflowProtocol`` — the structural interface that
all phase mixins expect from the composed ``DeepResearchWorkflow`` class.

Using a Protocol instead of bare ``Any`` annotations catches interface
drift at type-check time: if a mixin references a method that the
concrete workflow no longer provides, ``mypy`` / ``pyright`` will flag it.

Usage in mixins::

    from __future__ import annotations
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from foundry_mcp.config.research import ResearchConfig
        from foundry_mcp.core.research.memory import ResearchMemory

    class SomePhaseMixin:
        # Properly typed — catches config attribute typos at type-check time
        config: ResearchConfig
        memory: ResearchMemory
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig
    from foundry_mcp.core.research.memory import ResearchMemory

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    TopicResearchResult,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult


@runtime_checkable
class DeepResearchWorkflowProtocol(Protocol):
    """Structural interface expected by all deep research phase mixins.

    Covers the cross-cutting attributes and methods that every mixin
    may call on ``self`` at runtime.  This is the **single source of
    truth** for the composed class interface — individual mixins should
    not redeclare these signatures.
    """

    # --- Instance attributes ---
    config: ResearchConfig
    memory: ResearchMemory

    # --- Cross-cutting methods (used by most/all mixins) ---

    def _write_audit_event(
        self,
        state: DeepResearchState | None,
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

    # --- Phase-specific methods (used across mixin boundaries) ---

    async def _execute_topic_research_async(
        self,
        state: DeepResearchState,
        query: Any,
        *,
        provider_id: Optional[str] = ...,
        timeout: float = ...,
        search_providers: Optional[list[str]] = ...,
    ) -> TopicResearchResult:
        """Run a ReAct research loop for a single sub-query."""
        ...

    def _get_search_provider(self, provider_name: str) -> Any:
        """Return a configured search provider instance by name."""
        ...

    def _get_tavily_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]:
        """Build Tavily search kwargs based on config and research mode."""
        ...

    def _get_perplexity_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]:
        """Build Perplexity search kwargs based on config and research mode."""
        ...

    def _get_semantic_scholar_search_kwargs(self, state: DeepResearchState) -> dict[str, Any]:
        """Build Semantic Scholar search kwargs based on config."""
        ...

    async def _compress_single_topic_async(
        self,
        topic_result: TopicResearchResult,
        state: DeepResearchState,
        timeout: float,
    ) -> tuple[int, int, bool]:
        """Compress a single topic's findings. Returns (original, compressed, success)."""
        ...

    def _attach_source_summarizer(self, provider: Any) -> None:
        """Attach the fetch-time summarization callback to a search provider."""
        ...

    # --- Persistence methods (used by action handlers / workflow execution) ---

    def _persist_state_if_needed(self, state: DeepResearchState) -> bool:
        """Conditionally persist state based on throttle rules."""
        ...

    def _flush_state(self, state: DeepResearchState) -> None:
        """Force-persist state, bypassing throttle rules."""
        ...

    # --- Background task methods ---

    def get_background_task(self, research_id: str) -> Any:
        """Get background task by research ID."""
        ...

    def _start_background_task(self, **kwargs: Any) -> WorkflowResult:
        """Start a background research task."""
        ...

    def _cleanup_completed_task(self, research_id: str) -> None:
        """Remove completed task from registry."""
        ...

    async def _execute_workflow_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout_per_operation: float,
        max_concurrent: int,
    ) -> WorkflowResult:
        """Execute the full workflow pipeline."""
        ...
