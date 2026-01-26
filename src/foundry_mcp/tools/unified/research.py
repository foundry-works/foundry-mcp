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
    # Spec-integrated research actions
    "node-execute": "Execute research workflow linked to spec node",
    "node-record": "Record research findings to spec node",
    "node-status": "Get research node status and linked session info",
    "node-findings": "Retrieve recorded findings from spec node",
    # Tavily extract action
    "extract": "Extract content from URLs using Tavily Extract API",
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
                base_path=_config.get_research_dir(),
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
    timeout_per_provider: float = 360.0,
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

    # Apply config default for task_timeout if not explicitly set
    # Precedence: explicit param > config > hardcoded fallback (600s)
    effective_timeout = task_timeout
    if effective_timeout is None:
        effective_timeout = config.research.deep_research_timeout

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
        task_timeout=effective_timeout,
    )

    if result.success:
        # For background execution, return started status with research_id
        response_data = {
            "research_id": result.metadata.get("research_id"),
            "status": "started",
            "effective_timeout": effective_timeout,
            "message": (
                "Deep research started. This typically takes 3-5 minutes. "
                "IMPORTANT: Communicate progress to user before each status check. "
                "Maximum 5 status checks allowed. "
                "Do NOT use WebSearch/WebFetch while this research is running."
            ),
            "polling_guidance": {
                "max_checks": 5,
                "typical_duration_minutes": 5,
                "require_user_communication": True,
                "no_independent_research": True,
            },
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
        # Add next_action guidance based on check count
        status_data = dict(result.metadata) if result.metadata else {}
        check_count = status_data.get("status_check_count", 1)
        checks_remaining = max(0, 5 - check_count)

        if checks_remaining > 0:
            status_data["next_action"] = (
                f"BEFORE next check: Tell user about progress. {checks_remaining} checks remaining."
            )
        else:
            status_data["next_action"] = (
                "Max checks reached. Offer user options: wait, background, or cancel."
            )

        return asdict(success_response(data=status_data))
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
        # Extract warnings from metadata for routing to meta.warnings
        metadata = result.metadata or {}
        warnings = metadata.pop("warnings", None)

        # Build response data with all fields
        response_data = {
            "report": result.content,
            **metadata,
        }

        return asdict(
            success_response(
                data=response_data,
                warnings=warnings,  # Route warnings to meta.warnings
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
# Spec-Integrated Research Actions
# =============================================================================


def _load_research_node(
    spec_id: str,
    research_node_id: str,
    workspace: Optional[str] = None,
) -> tuple[Optional[dict], Optional[dict], Optional[str]]:
    """Load spec and validate research node exists.

    Returns:
        (spec_data, node_data, error_message)
    """
    from foundry_mcp.core.spec import load_spec, find_specs_directory

    specs_dir = find_specs_directory(workspace)
    if specs_dir is None:
        return None, None, "No specs directory found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, None, f"Specification '{spec_id}' not found"

    hierarchy = spec_data.get("hierarchy", {})
    node = hierarchy.get(research_node_id)
    if node is None:
        return None, None, f"Node '{research_node_id}' not found"

    if node.get("type") != "research":
        return None, None, f"Node '{research_node_id}' is not a research node (type: {node.get('type')})"

    return spec_data, node, None


def _handle_node_execute(
    *,
    spec_id: Optional[str] = None,
    research_node_id: Optional[str] = None,
    workspace: Optional[str] = None,
    prompt: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Execute research workflow linked to spec node.

    Starts the research workflow configured in the node's metadata,
    and stores the session_id back in the node for tracking.
    """
    from datetime import datetime, timezone
    from foundry_mcp.core.spec import save_spec, find_specs_directory

    if not spec_id:
        return _validation_error("spec_id", "node-execute", "Required")
    if not research_node_id:
        return _validation_error("research_node_id", "node-execute", "Required")

    spec_data, node, error = _load_research_node(spec_id, research_node_id, workspace)
    if error:
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.NOT_FOUND if "not found" in error.lower() else ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.NOT_FOUND if "not found" in error.lower() else ErrorType.VALIDATION,
            )
        )

    metadata = node.get("metadata", {})
    research_type = metadata.get("research_type", "consensus")
    query = prompt or metadata.get("query", "")

    if not query:
        return _validation_error("query", "node-execute", "No query found in node or prompt parameter")

    # Execute the appropriate research workflow
    config = _get_config()
    session_id = None
    result_data: dict[str, Any] = {
        "spec_id": spec_id,
        "research_node_id": research_node_id,
        "research_type": research_type,
    }

    if research_type == "chat":
        workflow = ChatWorkflow(config.research, _get_memory())
        result = workflow.chat(prompt=query)
        session_id = result.thread_id
        result_data["thread_id"] = session_id
    elif research_type == "consensus":
        workflow = ConsensusWorkflow(config.research, _get_memory())
        result = workflow.run(prompt=query)
        session_id = result.session_id
        result_data["consensus_id"] = session_id
        result_data["strategy"] = result.strategy.value if result.strategy else None
    elif research_type == "thinkdeep":
        workflow = ThinkDeepWorkflow(config.research, _get_memory())
        result = workflow.run(topic=query)
        session_id = result.investigation_id
        result_data["investigation_id"] = session_id
    elif research_type == "ideate":
        workflow = IdeateWorkflow(config.research, _get_memory())
        result = workflow.run(topic=query)
        session_id = result.ideation_id
        result_data["ideation_id"] = session_id
    elif research_type == "deep-research":
        workflow = DeepResearchWorkflow(config.research, _get_memory())
        result = workflow.start(query=query)
        session_id = result.research_id
        result_data["research_id"] = session_id
    else:
        return _validation_error("research_type", "node-execute", f"Unsupported: {research_type}")

    # Update node metadata with session info
    metadata["session_id"] = session_id
    history = metadata.setdefault("research_history", [])
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "started",
        "workflow": research_type,
        "session_id": session_id,
    })
    node["metadata"] = metadata
    node["status"] = "in_progress"

    # Save spec
    specs_dir = find_specs_directory(workspace)
    if specs_dir and not save_spec(spec_id, spec_data, specs_dir):
        return asdict(
            error_response(
                "Failed to save specification",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
            )
        )

    result_data["session_id"] = session_id
    result_data["status"] = "started"
    return asdict(success_response(data=result_data))


def _handle_node_record(
    *,
    spec_id: Optional[str] = None,
    research_node_id: Optional[str] = None,
    workspace: Optional[str] = None,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    key_insights: Optional[list[str]] = None,
    recommendations: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
    confidence: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Record research findings to spec node."""
    from datetime import datetime, timezone
    from foundry_mcp.core.spec import save_spec, find_specs_directory
    from foundry_mcp.core.validation import VALID_RESEARCH_RESULTS

    if not spec_id:
        return _validation_error("spec_id", "node-record", "Required")
    if not research_node_id:
        return _validation_error("research_node_id", "node-record", "Required")
    if not result:
        return _validation_error("result", "node-record", "Required (completed, inconclusive, blocked, cancelled)")
    if result not in VALID_RESEARCH_RESULTS:
        return _validation_error("result", "node-record", f"Must be one of: {', '.join(sorted(VALID_RESEARCH_RESULTS))}")

    spec_data, node, error = _load_research_node(spec_id, research_node_id, workspace)
    if error:
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.NOT_FOUND if "not found" in error.lower() else ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.NOT_FOUND if "not found" in error.lower() else ErrorType.VALIDATION,
            )
        )

    metadata = node.get("metadata", {})

    # Store findings
    metadata["findings"] = {
        "summary": summary or "",
        "key_insights": key_insights or [],
        "recommendations": recommendations or [],
        "sources": sources or [],
        "confidence": confidence or "medium",
    }

    # Update session link if provided
    if session_id:
        metadata["session_id"] = session_id

    # Add to history
    history = metadata.setdefault("research_history", [])
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "completed",
        "result": result,
        "session_id": session_id or metadata.get("session_id"),
    })

    node["metadata"] = metadata

    # Update node status based on result
    if result == "completed":
        node["status"] = "completed"
    elif result == "blocked":
        node["status"] = "blocked"
    else:
        node["status"] = "pending"  # inconclusive or cancelled

    # Save spec
    specs_dir = find_specs_directory(workspace)
    if specs_dir and not save_spec(spec_id, spec_data, specs_dir):
        return asdict(
            error_response(
                "Failed to save specification",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
            )
        )

    return asdict(
        success_response(
            data={
                "spec_id": spec_id,
                "research_node_id": research_node_id,
                "result": result,
                "status": node["status"],
                "findings_recorded": True,
            }
        )
    )


def _handle_node_status(
    *,
    spec_id: Optional[str] = None,
    research_node_id: Optional[str] = None,
    workspace: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Get research node status and linked session info."""
    if not spec_id:
        return _validation_error("spec_id", "node-status", "Required")
    if not research_node_id:
        return _validation_error("research_node_id", "node-status", "Required")

    spec_data, node, error = _load_research_node(spec_id, research_node_id, workspace)
    if error:
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.NOT_FOUND if "not found" in error.lower() else ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.NOT_FOUND if "not found" in error.lower() else ErrorType.VALIDATION,
            )
        )

    metadata = node.get("metadata", {})

    return asdict(
        success_response(
            data={
                "spec_id": spec_id,
                "research_node_id": research_node_id,
                "title": node.get("title"),
                "status": node.get("status"),
                "research_type": metadata.get("research_type"),
                "blocking_mode": metadata.get("blocking_mode"),
                "session_id": metadata.get("session_id"),
                "query": metadata.get("query"),
                "has_findings": bool(metadata.get("findings", {}).get("summary")),
                "history_count": len(metadata.get("research_history", [])),
            }
        )
    )


def _handle_node_findings(
    *,
    spec_id: Optional[str] = None,
    research_node_id: Optional[str] = None,
    workspace: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Retrieve recorded findings from spec node."""
    if not spec_id:
        return _validation_error("spec_id", "node-findings", "Required")
    if not research_node_id:
        return _validation_error("research_node_id", "node-findings", "Required")

    spec_data, node, error = _load_research_node(spec_id, research_node_id, workspace)
    if error:
        return asdict(
            error_response(
                error,
                error_code=ErrorCode.NOT_FOUND if "not found" in error.lower() else ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.NOT_FOUND if "not found" in error.lower() else ErrorType.VALIDATION,
            )
        )

    metadata = node.get("metadata", {})
    findings = metadata.get("findings", {})

    return asdict(
        success_response(
            data={
                "spec_id": spec_id,
                "research_node_id": research_node_id,
                "title": node.get("title"),
                "status": node.get("status"),
                "findings": findings,
                "research_history": metadata.get("research_history", []),
            }
        )
    )


# =============================================================================
# Extract Handler
# =============================================================================


def _handle_extract(
    *,
    urls: Optional[list[str]] = None,
    extract_depth: str = "basic",
    include_images: bool = False,
    format: str = "markdown",
    query: Optional[str] = None,
    chunks_per_source: Optional[int] = None,
    **kwargs: Any,
) -> dict:
    """Extract content from URLs using Tavily Extract API.

    Response envelope patterns (per MCP best practices):
    - Full success: success=True, data contains sources and stats, error=None
    - Partial success: success=True, data.failed_urls populated, meta.warnings contains summary
    - Total failure: success=False, data contains error_code/error_type/remediation/details

    Error codes:
    - VALIDATION_ERROR: Invalid parameters or URL format
    - INVALID_URL: URL parsing or scheme validation failed
    - BLOCKED_HOST: SSRF protection blocked the URL
    - RATE_LIMIT_EXCEEDED: API rate limit hit
    - TIMEOUT: Request timeout
    - EXTRACT_FAILED: General extraction failure

    Args:
        urls: List of URLs to extract content from (required, max 10).
        extract_depth: "basic" or "advanced" (default: "basic").
        include_images: Include images in results (default: False).
        format: Output format, "markdown" or "text" (default: "markdown").
        query: Optional query for relevance-based chunk reranking.
        chunks_per_source: Chunks per URL, 1-5 (default: 3).

    Returns:
        MCP response envelope with extracted content as ResearchSource objects.
    """
    import asyncio
    import os
    from concurrent.futures import ThreadPoolExecutor

    from foundry_mcp.core.research.providers.tavily_extract import (
        TavilyExtractProvider,
        UrlValidationError,
        validate_extract_url,
    )
    from foundry_mcp.core.research.providers.base import (
        RateLimitError,
        AuthenticationError,
    )

    # Validate required parameter
    if not urls:
        return asdict(
            error_response(
                "urls parameter is required",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Provide a list of URLs to extract content from",
            )
        )

    if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
        return asdict(
            error_response(
                "urls must be a list of strings",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
            )
        )

    # Get API key from config or environment
    config = _get_config()
    api_key = config.research.tavily_api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return asdict(
            error_response(
                "Tavily API key not configured",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Set TAVILY_API_KEY environment variable or tavily_api_key in config",
            )
        )

    # Pre-validate URLs and track validation failures
    valid_urls: list[str] = []
    failed_urls: list[str] = []
    error_details: list[dict[str, Any]] = []

    for url in urls:
        try:
            validate_extract_url(url, resolve_dns=False)  # Skip DNS in validation
            valid_urls.append(url)
        except UrlValidationError as e:
            failed_urls.append(url)
            error_details.append({
                "url": url,
                "error": e.reason,
                "error_code": e.error_code,
            })

    # If all URLs failed validation, return total failure
    if not valid_urls:
        return asdict(
            error_response(
                f"All {len(urls)} URLs failed validation",
                error_code="INVALID_URL",
                error_type=ErrorType.VALIDATION,
                remediation="Check URL formats and ensure they are publicly accessible HTTP/HTTPS URLs",
                details={
                    "failed_urls": failed_urls,
                    "error_details": error_details,
                },
            )
        )

    try:
        provider = TavilyExtractProvider(api_key=api_key)

        # Build extract kwargs
        extract_kwargs: dict[str, Any] = {
            "extract_depth": extract_depth,
            "include_images": include_images,
            "format": format,
        }
        if query is not None:
            extract_kwargs["query"] = query
        if chunks_per_source is not None:
            extract_kwargs["chunks_per_source"] = chunks_per_source

        def _run_async(coro: Any) -> Any:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            # Avoid blocking a running loop by executing in a worker thread.
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()

        # Execute extraction for valid URLs only
        sources = _run_async(provider.extract(valid_urls, **extract_kwargs))

        # Convert ResearchSource objects to dicts
        source_dicts = []
        succeeded_urls = set()
        for src in sources:
            src_dict = {
                "url": src.url,
                "title": src.title,
                "source_type": src.source_type.value if src.source_type else "web",
                "snippet": src.snippet,
                "content": src.content,
                "metadata": src.metadata,
            }
            source_dicts.append(src_dict)
            if src.url:
                succeeded_urls.add(src.url)

        # Check for URLs that were valid but failed extraction
        for url in valid_urls:
            if url not in succeeded_urls:
                failed_urls.append(url)
                error_details.append({
                    "url": url,
                    "error": "Extraction returned no content",
                    "error_code": "EXTRACT_FAILED",
                })

        # Build response based on success/failure pattern
        stats = {
            "requested": len(urls),
            "succeeded": len(sources),
            "failed": len(failed_urls),
        }

        # Determine response type
        if len(sources) == 0:
            # Total failure: no sources extracted
            return asdict(
                error_response(
                    f"Extract failed: no content extracted from {len(urls)} URLs",
                    error_code="EXTRACT_FAILED",
                    error_type=ErrorType.INTERNAL,
                    remediation="Check that URLs are publicly accessible and contain extractable content",
                    details={
                        "failed_urls": failed_urls,
                        "error_details": error_details,
                    },
                )
            )
        elif failed_urls:
            # Partial success: some URLs succeeded, some failed
            warnings = [
                f"{len(failed_urls)} of {len(urls)} URLs failed extraction"
            ]
            return asdict(
                success_response(
                    data={
                        "action": "extract",
                        "sources": source_dicts,
                        "stats": stats,
                        "failed_urls": failed_urls,
                        "error_details": error_details,
                    },
                    warnings=warnings,
                )
            )
        else:
            # Full success: all URLs extracted
            return asdict(
                success_response(
                    data={
                        "action": "extract",
                        "sources": source_dicts,
                        "stats": stats,
                    }
                )
            )

    except AuthenticationError as e:
        return asdict(
            error_response(
                f"Authentication failed: {e}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check that TAVILY_API_KEY is valid",
                details={
                    "failed_urls": urls,
                    "error_details": [{
                        "url": None,
                        "error": str(e),
                        "error_code": "AUTHENTICATION_ERROR",
                    }],
                },
            )
        )
    except RateLimitError as e:
        return asdict(
            error_response(
                f"Rate limit exceeded: {e}",
                error_code="RATE_LIMIT_EXCEEDED",
                error_type=ErrorType.RATE_LIMIT,
                remediation=f"Wait {e.retry_after or 60} seconds before retrying" if hasattr(e, 'retry_after') else "Wait before retrying",
                details={
                    "failed_urls": urls,
                    "error_details": [{
                        "url": None,
                        "error": str(e),
                        "error_code": "RATE_LIMIT_EXCEEDED",
                    }],
                },
            )
        )
    except UrlValidationError as e:
        return asdict(
            error_response(
                f"URL validation failed: {e.reason}",
                error_code=e.error_code,
                error_type=ErrorType.VALIDATION,
                details={
                    "failed_urls": [e.url],
                    "error_details": [{
                        "url": e.url,
                        "error": e.reason,
                        "error_code": e.error_code,
                    }],
                },
            )
        )
    except ValueError as e:
        return asdict(
            error_response(
                str(e),
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                details={
                    "failed_urls": urls if urls else [],
                    "error_details": [{
                        "url": None,
                        "error": str(e),
                        "error_code": ErrorCode.VALIDATION_ERROR,
                    }],
                },
            )
        )
    except asyncio.TimeoutError:
        return asdict(
            error_response(
                "Extract request timed out",
                error_code="TIMEOUT",
                error_type=ErrorType.UNAVAILABLE,
                remediation="Try with fewer URLs or increase timeout",
                details={
                    "failed_urls": urls,
                    "error_details": [{
                        "url": None,
                        "error": "Request timed out",
                        "error_code": "TIMEOUT",
                    }],
                },
            )
        )
    except Exception as e:
        logger.exception("Extract failed: %s", e)
        return asdict(
            error_response(
                f"Extract failed: {e}",
                error_code="EXTRACT_FAILED",
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details or try with different URLs",
                details={
                    "failed_urls": urls if urls else [],
                    "error_details": [{
                        "url": None,
                        "error": str(e),
                        "error_code": "EXTRACT_FAILED",
                    }],
                },
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
    return ActionRouter(tool_name="research", actions=definitions)


_RESEARCH_ROUTER = _build_router()


def _dispatch_research_action(action: str, **kwargs: Any) -> dict:
    """Dispatch action to appropriate handler.

    Catches all exceptions to ensure graceful failure with error response
    instead of crashing the MCP server.
    """
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
    except Exception as exc:
        # Catch all other exceptions to prevent MCP server crash
        logger.exception("Research action '%s' failed with unexpected error: %s", action, exc)
        error_msg = str(exc) if str(exc) else exc.__class__.__name__
        return asdict(
            error_response(
                f"Research action '{action}' failed: {error_msg}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check provider availability and configuration. Review logs for details.",
                details={
                    "action": action,
                    "error_type": exc.__class__.__name__,
                },
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
