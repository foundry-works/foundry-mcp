"""Unified research router — split into domain-focused handler modules."""

from __future__ import annotations

import logging
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.tools.unified.common import dispatch_with_standard_errors
from foundry_mcp.tools.unified.research_handlers._helpers import (
    _ACTION_SUMMARY,
    _config,  # noqa: F401
    _get_config,
    _get_memory,  # noqa: F401
    _memory,  # noqa: F401
    _validation_error,  # noqa: F401
)
from foundry_mcp.tools.unified.research_handlers.handlers_deep_research import (
    _handle_deep_research,
    _handle_deep_research_delete,
    _handle_deep_research_evaluate,
    _handle_deep_research_export,
    _handle_deep_research_list,
    _handle_deep_research_provenance,
    _handle_deep_research_report,
    _handle_deep_research_status,
)
from foundry_mcp.tools.unified.research_handlers.handlers_extract import (
    _handle_extract,
)
from foundry_mcp.tools.unified.research_handlers.handlers_spec_nodes import (
    _handle_node_execute,
    _handle_node_findings,
    _handle_node_record,
    _handle_node_status,
    _load_research_node,  # noqa: F401
)
from foundry_mcp.tools.unified.research_handlers.handlers_threads import (
    _handle_thread_delete,
    _handle_thread_get,
    _handle_thread_list,
)
from foundry_mcp.tools.unified.research_handlers.handlers_workflows import (
    _handle_chat,
    _handle_consensus,
    _handle_ideate,
    _handle_thinkdeep,
)
from foundry_mcp.tools.unified.router import ActionDefinition, ActionRouter

logger = logging.getLogger(__name__)

_ACTION_DEFINITIONS = [
    ActionDefinition(
        name="chat",
        handler=_handle_chat,
        summary=_ACTION_SUMMARY["chat"],
    ),
    ActionDefinition(
        name="consensus",
        handler=_handle_consensus,
        summary=_ACTION_SUMMARY["consensus"],
    ),
    ActionDefinition(
        name="thinkdeep",
        handler=_handle_thinkdeep,
        summary=_ACTION_SUMMARY["thinkdeep"],
    ),
    ActionDefinition(
        name="ideate",
        handler=_handle_ideate,
        summary=_ACTION_SUMMARY["ideate"],
    ),
    ActionDefinition(
        name="deep-research",
        handler=_handle_deep_research,
        summary=_ACTION_SUMMARY["deep-research"],
    ),
    ActionDefinition(
        name="deep-research-status",
        handler=_handle_deep_research_status,
        summary=_ACTION_SUMMARY["deep-research-status"],
    ),
    ActionDefinition(
        name="deep-research-report",
        handler=_handle_deep_research_report,
        summary=_ACTION_SUMMARY["deep-research-report"],
    ),
    ActionDefinition(
        name="deep-research-list",
        handler=_handle_deep_research_list,
        summary=_ACTION_SUMMARY["deep-research-list"],
    ),
    ActionDefinition(
        name="deep-research-delete",
        handler=_handle_deep_research_delete,
        summary=_ACTION_SUMMARY["deep-research-delete"],
    ),
    ActionDefinition(
        name="deep-research-evaluate",
        handler=_handle_deep_research_evaluate,
        summary=_ACTION_SUMMARY["deep-research-evaluate"],
    ),
    ActionDefinition(
        name="deep-research-provenance",
        handler=_handle_deep_research_provenance,
        summary=_ACTION_SUMMARY["deep-research-provenance"],
    ),
    ActionDefinition(
        name="deep-research-export",
        handler=_handle_deep_research_export,
        summary=_ACTION_SUMMARY["deep-research-export"],
    ),
    ActionDefinition(
        name="thread-list",
        handler=_handle_thread_list,
        summary=_ACTION_SUMMARY["thread-list"],
    ),
    ActionDefinition(
        name="thread-get",
        handler=_handle_thread_get,
        summary=_ACTION_SUMMARY["thread-get"],
    ),
    ActionDefinition(
        name="thread-delete",
        handler=_handle_thread_delete,
        summary=_ACTION_SUMMARY["thread-delete"],
    ),
    # Spec-integrated research actions
    ActionDefinition(
        name="node-execute",
        handler=_handle_node_execute,
        summary=_ACTION_SUMMARY["node-execute"],
    ),
    ActionDefinition(
        name="node-record",
        handler=_handle_node_record,
        summary=_ACTION_SUMMARY["node-record"],
    ),
    ActionDefinition(
        name="node-status",
        handler=_handle_node_status,
        summary=_ACTION_SUMMARY["node-status"],
    ),
    ActionDefinition(
        name="node-findings",
        handler=_handle_node_findings,
        summary=_ACTION_SUMMARY["node-findings"],
    ),
    # Tavily extract action
    ActionDefinition(
        name="extract",
        handler=_handle_extract,
        summary=_ACTION_SUMMARY["extract"],
    ),
]

_RESEARCH_ROUTER = ActionRouter(tool_name="research", actions=_ACTION_DEFINITIONS)


def _dispatch_research_action(action: str, **kwargs: Any) -> dict:
    """Dispatch action to appropriate handler.

    Catches all exceptions to ensure graceful failure with error response
    instead of crashing the MCP server.
    """
    return dispatch_with_standard_errors(
        _RESEARCH_ROUTER,
        "research",
        action,
        include_details_in_router_error=True,
        config=_get_config(),
        **kwargs,
    )


def register_unified_research_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the unified research tool.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    from foundry_mcp.tools.unified.research_handlers import _helpers

    _helpers._config = config
    _helpers._memory = None  # Reset to use new config

    # Check if research tools are enabled
    if not config.research.enabled:
        logger.info("Research tools disabled in config")
        return

    @canonical_tool(mcp, canonical_name="research")
    def research(
        action: str,
        prompt: Optional[str] = None,
        thread_id: Optional[str] = None,
        investigation_id: Optional[str] = None,
        ideation_id: Optional[str] = None,
        research_id: Optional[str] = None,
        topic: Optional[str] = None,
        query: Optional[str] = None,
        system_prompt: Optional[str] = None,
        provider_id: Optional[str] = None,
        model: Optional[str] = None,
        providers: Optional[list[str]] = None,
        strategy: Optional[str] = None,
        synthesis_provider: Optional[str] = None,
        timeout_per_provider: float = 360.0,
        timeout_per_operation: float = 360.0,
        max_concurrent: int = 3,
        require_all: bool = False,
        min_responses: int = 1,
        max_depth: Optional[int] = None,
        max_iterations: int = 3,
        max_sub_queries: int = 5,
        max_sources_per_query: int = 5,
        follow_links: bool = True,
        deep_research_action: str = "start",
        task_timeout: Optional[float] = None,
        ideate_action: str = "generate",
        perspective: Optional[str] = None,
        perspectives: Optional[list[str]] = None,
        cluster_ids: Optional[list[str]] = None,
        scoring_criteria: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        title: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
        completed_only: bool = False,
        wait: bool = True,
        wait_timeout: float = 90,
        research_profile: Optional[str] = None,
        profile_overrides: Optional[dict] = None,
    ) -> dict:
        """Execute research workflows via the action router.

        Actions:
        - chat: Single-model conversation with thread persistence
        - consensus: Multi-model parallel consultation with synthesis
        - thinkdeep: Hypothesis-driven systematic investigation
        - ideate: Creative brainstorming with idea clustering
        - deep-research: Multi-phase iterative deep research with query decomposition
        - deep-research-status: Get status of deep research session
        - deep-research-report: Get final report from deep research
        - deep-research-list: List deep research sessions
        - deep-research-delete: Delete a deep research session
        - deep-research-evaluate: Evaluate research report quality (LLM-as-judge)
        - thread-list: List conversation threads
        - thread-get: Get thread details including messages
        - thread-delete: Delete a conversation thread

        Args:
            action: The research action to execute
            prompt: User prompt/message (chat, consensus)
            thread_id: Thread ID for continuing conversations (chat)
            investigation_id: Investigation ID to continue (thinkdeep)
            ideation_id: Ideation session ID to continue (ideate)
            research_id: Deep research session ID (deep-research-*)
            topic: Topic for new investigation/ideation
            query: Research query (deep-research) or follow-up (thinkdeep)
            system_prompt: System prompt for workflows
            provider_id: Provider to use for single-model operations
            model: Model override
            providers: Provider list for consensus
            strategy: Consensus strategy (all_responses, synthesize, majority, first_valid)
            synthesis_provider: Provider for synthesis
            timeout_per_provider: Timeout per provider in seconds (consensus)
            timeout_per_operation: Timeout per operation in seconds (deep-research)
            max_concurrent: Max concurrent provider/operation calls
            require_all: Require all providers to succeed
            min_responses: Minimum successful responses needed
            max_depth: Maximum investigation depth (thinkdeep)
            max_iterations: Maximum refinement iterations (deep-research)
            max_sub_queries: Maximum sub-queries to generate (deep-research)
            max_sources_per_query: Maximum sources per sub-query (deep-research)
            follow_links: Whether to follow and extract links (deep-research)
            deep_research_action: Sub-action for deep-research (start, continue, resume)
            task_timeout: Overall timeout for background research task in seconds
            ideate_action: Ideation sub-action (generate, cluster, score, select, elaborate)
            perspective: Specific perspective for idea generation
            perspectives: Custom perspectives list
            cluster_ids: Cluster IDs for selection/elaboration
            scoring_criteria: Custom scoring criteria
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            title: Title for new threads
            status: Filter threads by status
            limit: Maximum items to return
            cursor: Pagination cursor for deep-research-list
            completed_only: Filter to completed sessions only (deep-research-list)
            wait: Block until state changes (long-poll) for deep-research-status
            wait_timeout: Max seconds to wait (clamped to 90) for deep-research-status
            research_profile: Named profile for deep-research (general, academic, systematic-review, bibliometric, technical)
            profile_overrides: Per-request overrides applied on top of the resolved profile

        Returns:
            Response envelope with action results
        """
        return _dispatch_research_action(**locals())

    logger.debug("Registered unified research tool")


__all__ = [
    "register_unified_research_tool",
]
