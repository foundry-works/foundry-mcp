"""Unified research tool with action routing.

Provides multi-model orchestration capabilities through CHAT, CONSENSUS,
THINKDEEP, and IDEATE workflows via a unified MCP tool interface.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.feature_flags import FeatureFlag, FlagState, get_flag_service
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models import ConsensusStrategy, ThreadStatus
from foundry_mcp.core.research.workflows import (
    ChatWorkflow,
    ConsensusWorkflow,
    DeepResearchWorkflow,
    IdeateWorkflow,
    ThinkDeepWorkflow,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Register feature flag for research tools
_flag_service = get_flag_service()
try:
    _flag_service.register(
        FeatureFlag(
            name="research_tools",
            description="Multi-model research workflows (chat, consensus, thinkdeep, ideate)",
            state=FlagState.BETA,
            default_enabled=False,
            owner="foundry-mcp core",
        )
    )
except ValueError:
    # Flag already registered (e.g., via config reload)
    pass


# =============================================================================
# Action Summaries
# =============================================================================

_ACTION_SUMMARY = {
    "chat": "Single-model conversation with thread persistence",
    "consensus": "Multi-model parallel consultation with synthesis",
    "thinkdeep": "Hypothesis-driven systematic investigation",
    "ideate": "Creative brainstorming with idea clustering",
    "deep-research": "Multi-phase iterative deep research with query decomposition",
    "deep-research-status": "Get status of deep research session",
    "deep-research-report": "Get final report from deep research",
    "deep-research-list": "List deep research sessions",
    "deep-research-delete": "Delete a deep research session",
    "thread-list": "List conversation threads",
    "thread-get": "Get full thread details including messages",
    "thread-delete": "Delete a conversation thread",
}


# =============================================================================
# Module State
# =============================================================================

_config: Optional[ServerConfig] = None
_memory: Optional[ResearchMemory] = None


def _get_memory() -> ResearchMemory:
    """Get or create the research memory instance."""
    global _memory, _config
    if _memory is None:
        if _config is not None:
            _memory = ResearchMemory(
                base_path=_config.research.get_storage_path(),
                ttl_hours=_config.research.ttl_hours,
            )
        else:
            _memory = ResearchMemory()
    return _memory


def _get_config() -> ServerConfig:
    """Get the server config, raising if not initialized."""
    global _config
    if _config is None:
        # Create default config if not set
        _config = ServerConfig()
    return _config


# =============================================================================
# Validation Helpers
# =============================================================================

def _validation_error(
    field: str,
    action: str,
    message: str,
    *,
    code: ErrorCode = ErrorCode.VALIDATION_ERROR,
    remediation: Optional[str] = None,
) -> dict:
    """Create a validation error response."""
    return asdict(
        error_response(
            f"Invalid field '{field}' for research.{action}: {message}",
            error_code=code,
            error_type=ErrorType.VALIDATION,
            remediation=remediation or f"Provide a valid '{field}' value",
            details={"field": field, "action": f"research.{action}"},
        )
    )


# =============================================================================
# Action Handlers
# =============================================================================

def _handle_chat(
    *,
    prompt: Optional[str] = None,
    thread_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    provider_id: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    title: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle chat action."""
    if not prompt:
        return _validation_error("prompt", "chat", "Required non-empty string")

    config = _get_config()
    workflow = ChatWorkflow(config.research, _get_memory())

    result = workflow.execute(
        prompt=prompt,
        thread_id=thread_id,
        system_prompt=system_prompt,
        provider_id=provider_id,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        title=title,
    )

    if result.success:
        return asdict(
            success_response(
                data={
                    "content": result.content,
                    "thread_id": result.metadata.get("thread_id"),
                    "message_count": result.metadata.get("message_count"),
                    "provider_id": result.provider_id,
                    "model_used": result.model_used,
                    "tokens_used": result.tokens_used,
                }
            )
        )
    else:
        return asdict(
            error_response(
                result.error or "Chat failed",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check provider availability and retry",
            )
        )


def _handle_consensus(
    *,
    prompt: Optional[str] = None,
    providers: Optional[list[str]] = None,
    strategy: Optional[str] = None,
    synthesis_provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
    timeout_per_provider: float = 30.0,
    max_concurrent: int = 3,
    require_all: bool = False,
    min_responses: int = 1,
    **kwargs: Any,
) -> dict:
    """Handle consensus action."""
    if not prompt:
        return _validation_error("prompt", "consensus", "Required non-empty string")

    # Parse strategy
    consensus_strategy = ConsensusStrategy.SYNTHESIZE
    if strategy:
        try:
            consensus_strategy = ConsensusStrategy(strategy)
        except ValueError:
            valid = [s.value for s in ConsensusStrategy]
            return _validation_error(
                "strategy",
                "consensus",
                f"Invalid value. Valid: {valid}",
                remediation=f"Use one of: {', '.join(valid)}",
            )

    config = _get_config()
    workflow = ConsensusWorkflow(config.research, _get_memory())

    result = workflow.execute(
        prompt=prompt,
        providers=providers,
        strategy=consensus_strategy,
        synthesis_provider=synthesis_provider,
        system_prompt=system_prompt,
        timeout_per_provider=timeout_per_provider,
        max_concurrent=max_concurrent,
        require_all=require_all,
        min_responses=min_responses,
    )

    if result.success:
        return asdict(
            success_response(
                data={
                    "content": result.content,
                    "consensus_id": result.metadata.get("consensus_id"),
                    "providers_consulted": result.metadata.get("providers_consulted"),
                    "strategy": result.metadata.get("strategy"),
                    "response_count": result.metadata.get("response_count"),
                }
            )
        )
    else:
        return asdict(
            error_response(
                result.error or "Consensus failed",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check provider availability and retry",
                details=result.metadata,
            )
        )


def _handle_thinkdeep(
    *,
    topic: Optional[str] = None,
    investigation_id: Optional[str] = None,
    query: Optional[str] = None,
    system_prompt: Optional[str] = None,
    provider_id: Optional[str] = None,
    max_depth: Optional[int] = None,
    **kwargs: Any,
) -> dict:
    """Handle thinkdeep action."""
    if not topic and not investigation_id:
        return _validation_error(
            "topic/investigation_id",
            "thinkdeep",
            "Either 'topic' (new) or 'investigation_id' (continue) required",
        )

    config = _get_config()
    workflow = ThinkDeepWorkflow(config.research, _get_memory())

    result = workflow.execute(
        topic=topic,
        investigation_id=investigation_id,
        query=query,
        system_prompt=system_prompt,
        provider_id=provider_id,
        max_depth=max_depth,
    )

    if result.success:
        return asdict(
            success_response(
                data={
                    "content": result.content,
                    "investigation_id": result.metadata.get("investigation_id"),
                    "current_depth": result.metadata.get("current_depth"),
                    "max_depth": result.metadata.get("max_depth"),
                    "converged": result.metadata.get("converged"),
                    "hypothesis_count": result.metadata.get("hypothesis_count"),
                    "step_count": result.metadata.get("step_count"),
                }
            )
        )
    else:
        return asdict(
            error_response(
                result.error or "ThinkDeep failed",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check investigation ID or topic validity",
            )
        )


def _handle_ideate(
    *,
    topic: Optional[str] = None,
    ideation_id: Optional[str] = None,
    ideate_action: str = "generate",
    perspective: Optional[str] = None,
    cluster_ids: Optional[list[str]] = None,
    system_prompt: Optional[str] = None,
    provider_id: Optional[str] = None,
    perspectives: Optional[list[str]] = None,
    scoring_criteria: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Handle ideate action."""
    if not topic and not ideation_id:
        return _validation_error(
            "topic/ideation_id",
            "ideate",
            "Either 'topic' (new) or 'ideation_id' (continue) required",
        )

    config = _get_config()
    workflow = IdeateWorkflow(config.research, _get_memory())

    result = workflow.execute(
        topic=topic,
        ideation_id=ideation_id,
        action=ideate_action,
        perspective=perspective,
        cluster_ids=cluster_ids,
        system_prompt=system_prompt,
        provider_id=provider_id,
        perspectives=perspectives,
        scoring_criteria=scoring_criteria,
    )

    if result.success:
        return asdict(
            success_response(
                data={
                    "content": result.content,
                    "ideation_id": result.metadata.get("ideation_id"),
                    "phase": result.metadata.get("phase"),
                    "idea_count": result.metadata.get("idea_count"),
                    "cluster_count": result.metadata.get("cluster_count"),
                }
            )
        )
    else:
        return asdict(
            error_response(
                result.error or "Ideate failed",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check ideation ID or topic validity",
            )
        )


def _handle_deep_research(
    *,
    query: Optional[str] = None,
    research_id: Optional[str] = None,
    deep_research_action: str = "start",
    provider_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    max_iterations: int = 3,
    max_sub_queries: int = 5,
    max_sources_per_query: int = 5,
    follow_links: bool = True,
    timeout_per_operation: float = 120.0,
    max_concurrent: int = 3,
    task_timeout: Optional[float] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research action with background execution.

    CRITICAL: This handler uses asyncio.create_task() via the workflow's
    background mode to start research and return immediately with the
    research_id. The workflow runs in the background and can be polled
    via deep-research-status.

    Supports:
    - start: Begin new research, returns immediately with research_id
    - continue: Resume paused research in background
    - resume: Alias for continue (for backward compatibility)
    """
    # Normalize 'resume' to 'continue' for workflow compatibility
    if deep_research_action == "resume":
        deep_research_action = "continue"

    # Validate based on action
    if deep_research_action == "start" and not query:
        return _validation_error(
            "query",
            "deep-research",
            "Query is required to start deep research",
            remediation="Provide a research query to investigate",
        )

    if deep_research_action in ("continue",) and not research_id:
        return _validation_error(
            "research_id",
            "deep-research",
            f"research_id is required for '{deep_research_action}' action",
            remediation="Use deep-research-list to find existing research sessions",
        )

    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    # Execute with background=True for non-blocking execution
    # This uses asyncio.create_task() internally and returns immediately
    result = workflow.execute(
        query=query,
        research_id=research_id,
        action=deep_research_action,
        provider_id=provider_id,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        max_sub_queries=max_sub_queries,
        max_sources_per_query=max_sources_per_query,
        follow_links=follow_links,
        timeout_per_operation=timeout_per_operation,
        max_concurrent=max_concurrent,
        background=True,  # CRITICAL: Run in background, return immediately
        task_timeout=task_timeout,
    )

    if result.success:
        # For background execution, return started status with research_id
        response_data = {
            "research_id": result.metadata.get("research_id"),
            "status": "started",
            "message": "Deep research started in background. Use deep-research-status to poll progress.",
        }

        # Include additional metadata if available (for continue/resume)
        if result.metadata.get("phase"):
            response_data["phase"] = result.metadata.get("phase")
        if result.metadata.get("iteration") is not None:
            response_data["iteration"] = result.metadata.get("iteration")

        return asdict(success_response(data=response_data))
    else:
        return asdict(
            error_response(
                result.error or "Deep research failed to start",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check query or research_id validity and provider availability",
                details={"action": deep_research_action},
            )
        )


def _handle_deep_research_status(
    *,
    research_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-status action."""
    if not research_id:
        return _validation_error("research_id", "deep-research-status", "Required")

    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    result = workflow.execute(
        research_id=research_id,
        action="status",
    )

    if result.success:
        return asdict(success_response(data=result.metadata))
    else:
        return asdict(
            error_response(
                result.error or "Failed to get status",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use deep-research-list to find valid research IDs",
            )
        )


def _handle_deep_research_report(
    *,
    research_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-report action."""
    if not research_id:
        return _validation_error("research_id", "deep-research-report", "Required")

    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    result = workflow.execute(
        research_id=research_id,
        action="report",
    )

    if result.success:
        return asdict(
            success_response(
                data={
                    "report": result.content,
                    **result.metadata,
                }
            )
        )
    else:
        return asdict(
            error_response(
                result.error or "Failed to get report",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure research is complete or use deep-research-status to check",
            )
        )


def _handle_deep_research_list(
    *,
    limit: int = 50,
    cursor: Optional[str] = None,
    completed_only: bool = False,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-list action."""
    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    sessions = workflow.list_sessions(
        limit=limit,
        cursor=cursor,
        completed_only=completed_only,
    )

    # Build response with pagination support
    response_data: dict[str, Any] = {
        "sessions": sessions,
        "count": len(sessions),
    }

    # Include next cursor if there are more results
    if sessions and len(sessions) == limit:
        # Use last session's ID as cursor for next page
        response_data["next_cursor"] = sessions[-1].get("id")

    return asdict(success_response(data=response_data))


def _handle_deep_research_delete(
    *,
    research_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle deep-research-delete action."""
    if not research_id:
        return _validation_error("research_id", "deep-research-delete", "Required")

    config = _get_config()
    workflow = DeepResearchWorkflow(config.research, _get_memory())

    deleted = workflow.delete_session(research_id)

    if not deleted:
        return asdict(
            error_response(
                f"Research session '{research_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use deep-research-list to find valid research IDs",
            )
        )

    return asdict(
        success_response(
            data={
                "deleted": True,
                "research_id": research_id,
            }
        )
    )


def _handle_thread_list(
    *,
    status: Optional[str] = None,
    limit: int = 50,
    **kwargs: Any,
) -> dict:
    """Handle thread-list action."""
    thread_status = None
    if status:
        try:
            thread_status = ThreadStatus(status)
        except ValueError:
            valid = [s.value for s in ThreadStatus]
            return _validation_error(
                "status",
                "thread-list",
                f"Invalid value. Valid: {valid}",
            )

    config = _get_config()
    workflow = ChatWorkflow(config.research, _get_memory())
    threads = workflow.list_threads(status=thread_status, limit=limit)

    return asdict(
        success_response(
            data={
                "threads": threads,
                "count": len(threads),
            }
        )
    )


def _handle_thread_get(
    *,
    thread_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle thread-get action."""
    if not thread_id:
        return _validation_error("thread_id", "thread-get", "Required")

    config = _get_config()
    workflow = ChatWorkflow(config.research, _get_memory())
    thread = workflow.get_thread(thread_id)

    if not thread:
        return asdict(
            error_response(
                f"Thread '{thread_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use thread-list to find valid thread IDs",
            )
        )

    return asdict(success_response(data=thread))


def _handle_thread_delete(
    *,
    thread_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle thread-delete action."""
    if not thread_id:
        return _validation_error("thread_id", "thread-delete", "Required")

    config = _get_config()
    workflow = ChatWorkflow(config.research, _get_memory())
    deleted = workflow.delete_thread(thread_id)

    if not deleted:
        return asdict(
            error_response(
                f"Thread '{thread_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use thread-list to find valid thread IDs",
            )
        )

    return asdict(
        success_response(
            data={
                "deleted": True,
                "thread_id": thread_id,
            }
        )
    )


# =============================================================================
# Router Setup
# =============================================================================

def _build_router() -> ActionRouter:
    """Build the action router for research tool."""
    definitions = [
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
    ]
    return ActionRouter(tool_name="research", actions=definitions)


_RESEARCH_ROUTER = _build_router()


def _dispatch_research_action(action: str, **kwargs: Any) -> dict:
    """Dispatch action to appropriate handler."""
    try:
        return _RESEARCH_ROUTER.dispatch(action=action, **kwargs)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported research action '{action}'. Allowed: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                details={"action": action, "allowed_actions": exc.allowed_actions},
            )
        )


# =============================================================================
# Tool Registration
# =============================================================================

def register_unified_research_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the unified research tool.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    global _config, _memory
    _config = config
    _memory = None  # Reset to use new config

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
        timeout_per_provider: float = 30.0,
        timeout_per_operation: float = 120.0,
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

        Returns:
            Response envelope with action results
        """
        return _dispatch_research_action(
            action=action,
            prompt=prompt,
            thread_id=thread_id,
            investigation_id=investigation_id,
            ideation_id=ideation_id,
            research_id=research_id,
            topic=topic,
            query=query,
            system_prompt=system_prompt,
            provider_id=provider_id,
            model=model,
            providers=providers,
            strategy=strategy,
            synthesis_provider=synthesis_provider,
            timeout_per_provider=timeout_per_provider,
            timeout_per_operation=timeout_per_operation,
            max_concurrent=max_concurrent,
            require_all=require_all,
            min_responses=min_responses,
            max_depth=max_depth,
            max_iterations=max_iterations,
            max_sub_queries=max_sub_queries,
            max_sources_per_query=max_sources_per_query,
            follow_links=follow_links,
            deep_research_action=deep_research_action,
            task_timeout=task_timeout,
            ideate_action=ideate_action,
            perspective=perspective,
            perspectives=perspectives,
            cluster_ids=cluster_ids,
            scoring_criteria=scoring_criteria,
            temperature=temperature,
            max_tokens=max_tokens,
            title=title,
            status=status,
            limit=limit,
            cursor=cursor,
            completed_only=completed_only,
        )

    logger.debug("Registered unified research tool")


__all__ = [
    "register_unified_research_tool",
]
