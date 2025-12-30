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
from foundry_mcp.core.feature_flags import get_flag_service
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models import ConsensusStrategy, ThreadStatus
from foundry_mcp.core.research.workflows import (
    ChatWorkflow,
    ConsensusWorkflow,
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


# =============================================================================
# Action Summaries
# =============================================================================

_ACTION_SUMMARY = {
    "chat": "Single-model conversation with thread persistence",
    "consensus": "Multi-model parallel consultation with synthesis",
    "thinkdeep": "Hypothesis-driven systematic investigation",
    "ideate": "Creative brainstorming with idea clustering",
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
        topic: Optional[str] = None,
        query: Optional[str] = None,
        system_prompt: Optional[str] = None,
        provider_id: Optional[str] = None,
        model: Optional[str] = None,
        providers: Optional[list[str]] = None,
        strategy: Optional[str] = None,
        synthesis_provider: Optional[str] = None,
        timeout_per_provider: float = 30.0,
        max_concurrent: int = 3,
        require_all: bool = False,
        min_responses: int = 1,
        max_depth: Optional[int] = None,
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
    ) -> dict:
        """Execute research workflows via the action router.

        Actions:
        - chat: Single-model conversation with thread persistence
        - consensus: Multi-model parallel consultation with synthesis
        - thinkdeep: Hypothesis-driven systematic investigation
        - ideate: Creative brainstorming with idea clustering
        - route: Intelligent workflow selection based on prompt
        - thread-list: List conversation threads
        - thread-get: Get thread details including messages
        - thread-delete: Delete a conversation thread

        Args:
            action: The research action to execute
            prompt: User prompt/message (chat, consensus, route)
            thread_id: Thread ID for continuing conversations (chat)
            investigation_id: Investigation ID to continue (thinkdeep)
            ideation_id: Ideation session ID to continue (ideate)
            topic: Topic for new investigation/ideation
            query: Follow-up query (thinkdeep)
            system_prompt: System prompt for workflows
            provider_id: Provider to use for single-model operations
            model: Model override
            providers: Provider list for consensus
            strategy: Consensus strategy (all_responses, synthesize, majority, first_valid)
            synthesis_provider: Provider for synthesis
            timeout_per_provider: Timeout per provider in seconds
            max_concurrent: Max concurrent provider calls
            require_all: Require all providers to succeed
            min_responses: Minimum successful responses needed
            max_depth: Maximum investigation depth (thinkdeep)
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

        Returns:
            Response envelope with action results
        """
        # Check feature flag
        flag_service = get_flag_service()
        if not flag_service.is_enabled("research_tools"):
            return asdict(
                error_response(
                    "Research tools are not enabled",
                    error_code=ErrorCode.FEATURE_DISABLED,
                    error_type=ErrorType.UNAVAILABLE,
                    remediation="Enable 'research_tools' feature flag in configuration",
                )
            )

        return _dispatch_research_action(
            action=action,
            prompt=prompt,
            thread_id=thread_id,
            investigation_id=investigation_id,
            ideation_id=ideation_id,
            topic=topic,
            query=query,
            system_prompt=system_prompt,
            provider_id=provider_id,
            model=model,
            providers=providers,
            strategy=strategy,
            synthesis_provider=synthesis_provider,
            timeout_per_provider=timeout_per_provider,
            max_concurrent=max_concurrent,
            require_all=require_all,
            min_responses=min_responses,
            max_depth=max_depth,
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
        )

    logger.debug("Registered unified research tool")


__all__ = [
    "register_unified_research_tool",
]
