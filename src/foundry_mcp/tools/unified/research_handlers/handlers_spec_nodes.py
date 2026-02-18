"""Spec-integrated research handlers: node-execute, node-record, node-status, node-findings."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from foundry_mcp.core.research.workflows import (
    ChatWorkflow,
    ConsensusWorkflow,
    DeepResearchWorkflow,
    IdeateWorkflow,
    ThinkDeepWorkflow,
)
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.validation.constants import VALID_RESEARCH_RESULTS
from foundry_mcp.tools.unified.param_schema import Str, validate_payload

from ._helpers import _get_config, _get_memory, _validation_error

# ---------------------------------------------------------------------------
# Declarative validation schemas
# ---------------------------------------------------------------------------

_NODE_EXECUTE_SCHEMA = {
    "spec_id": Str(required=True),
    "research_node_id": Str(required=True),
}

_NODE_RECORD_SCHEMA = {
    "spec_id": Str(required=True),
    "research_node_id": Str(required=True),
    "result": Str(required=True, choices=frozenset(VALID_RESEARCH_RESULTS)),
}

_NODE_STATUS_SCHEMA = {
    "spec_id": Str(required=True),
    "research_node_id": Str(required=True),
}

_NODE_FINDINGS_SCHEMA = {
    "spec_id": Str(required=True),
    "research_node_id": Str(required=True),
}


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

    payload = {"spec_id": spec_id, "research_node_id": research_node_id}
    err = validate_payload(payload, _NODE_EXECUTE_SCHEMA, tool_name="research", action="node-execute")
    if err:
        return err

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

    # Imperative: query depends on runtime metadata fallback
    if not query:
        return _validation_error(field="query", action="node-execute", message="No query found in node or prompt parameter")

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
        return _validation_error(field="research_type", action="node-execute", message=f"Unsupported: {research_type}")

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

    payload = {"spec_id": spec_id, "research_node_id": research_node_id, "result": result}
    err = validate_payload(payload, _NODE_RECORD_SCHEMA, tool_name="research", action="node-record")
    if err:
        return err

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
    payload = {"spec_id": spec_id, "research_node_id": research_node_id}
    err = validate_payload(payload, _NODE_STATUS_SCHEMA, tool_name="research", action="node-status")
    if err:
        return err

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
    payload = {"spec_id": spec_id, "research_node_id": research_node_id}
    err = validate_payload(payload, _NODE_FINDINGS_SCHEMA, tool_name="research", action="node-findings")
    if err:
        return err

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
