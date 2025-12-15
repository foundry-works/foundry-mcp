"""Unified server discovery tool with action routing.

Consolidates discovery/context helpers into a single `server(action=...)` tool.
Only the unified tool surface is exposed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.context import generate_correlation_id, get_correlation_id
from foundry_mcp.core.discovery import get_capabilities, get_tool_registry
from foundry_mcp.core.feature_flags import get_flag_service
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import (
    get_metrics,
    get_observability_manager,
    mcp_tool,
)
from foundry_mcp.core.pagination import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    CursorError,
    decode_cursor,
    encode_cursor,
    paginated_response,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.tools.unified.context_helpers import (
    build_llm_status_response,
    build_server_context_response,
)
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()


def _request_id() -> str:
    return get_correlation_id() or generate_correlation_id(prefix="server")


def _metric(action: str) -> str:
    return f"unified_tools.server.{action.replace('-', '_')}"


MANIFEST_TOKEN_BUDGET = 16_000
MANIFEST_TOKEN_BUDGET_MAX = 18_000


@lru_cache(maxsize=1)
def _get_tokenizer() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _estimate_tokens(text: str) -> int:
    """Estimate token usage for manifest budget dashboards.

    Uses `tiktoken` when available, otherwise falls back to a conservative
    ~4-chars-per-token heuristic.
    """

    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        return len(tokenizer.encode(text))

    return max(1, len(text) // 4)


def _validation_error(
    *, message: str, request_id: str, remediation: Optional[str] = None
) -> dict:
    return asdict(
        error_response(
            message,
            error_code=ErrorCode.VALIDATION_ERROR,
            error_type=ErrorType.VALIDATION,
            remediation=remediation,
            request_id=request_id,
        )
    )


def _build_unified_manifest_tools() -> list[Dict[str, Any]]:
    """Return compact tool entries for the 17-tool unified manifest."""

    from foundry_mcp.tools.unified.authoring import _AUTHORING_ROUTER
    from foundry_mcp.tools.unified.code import _CODE_ROUTER
    from foundry_mcp.tools.unified.environment import _ENVIRONMENT_ROUTER
    from foundry_mcp.tools.unified.error import _ERROR_ROUTER
    from foundry_mcp.tools.unified.health import _HEALTH_ROUTER
    from foundry_mcp.tools.unified.journal import _JOURNAL_ROUTER
    from foundry_mcp.tools.unified.lifecycle import _LIFECYCLE_ROUTER
    from foundry_mcp.tools.unified.metrics import _METRICS_ROUTER
    from foundry_mcp.tools.unified.plan import _PLAN_ROUTER
    from foundry_mcp.tools.unified.pr import _PR_ROUTER
    from foundry_mcp.tools.unified.provider import _PROVIDER_ROUTER
    from foundry_mcp.tools.unified.review import _REVIEW_ROUTER
    from foundry_mcp.tools.unified.spec import _SPEC_ROUTER
    from foundry_mcp.tools.unified.task import _TASK_ROUTER
    from foundry_mcp.tools.unified.test import _TEST_ROUTER
    from foundry_mcp.tools.unified.verification import _VERIFICATION_ROUTER

    routers = {
        "health": _HEALTH_ROUTER,
        "plan": _PLAN_ROUTER,
        "pr": _PR_ROUTER,
        "error": _ERROR_ROUTER,
        "metrics": _METRICS_ROUTER,
        "journal": _JOURNAL_ROUTER,
        "authoring": _AUTHORING_ROUTER,
        "provider": _PROVIDER_ROUTER,
        "environment": _ENVIRONMENT_ROUTER,
        "lifecycle": _LIFECYCLE_ROUTER,
        "verification": _VERIFICATION_ROUTER,
        "task": _TASK_ROUTER,
        "spec": _SPEC_ROUTER,
        "review": _REVIEW_ROUTER,
        "code": _CODE_ROUTER,
        "server": _SERVER_ROUTER,
        "test": _TEST_ROUTER,
    }

    categories = {
        "health": "health",
        "plan": "planning",
        "pr": "workflow",
        "error": "observability",
        "metrics": "observability",
        "journal": "journal",
        "authoring": "specs",
        "provider": "providers",
        "environment": "environment",
        "lifecycle": "lifecycle",
        "verification": "verification",
        "task": "tasks",
        "spec": "specs",
        "review": "review",
        "code": "code",
        "server": "server",
        "test": "testing",
    }

    descriptions = {
        "health": "Health checks and diagnostics.",
        "plan": "Planning helpers (create/list/review plans).",
        "pr": "PR workflows with spec context.",
        "error": "Error collection query and cleanup.",
        "metrics": "Metrics query, stats, and cleanup.",
        "journal": "Journaling add/list helpers.",
        "authoring": "Spec authoring mutations (phases, assumptions, revisions).",
        "provider": "LLM provider discovery and execution.",
        "environment": "Workspace init + environment verification.",
        "lifecycle": "Spec lifecycle transitions.",
        "verification": "Verification definition + execution.",
        "task": "Task preparation, mutation, and listing.",
        "spec": "Spec discovery, validation, and analysis.",
        "review": "LLM-assisted review workflows.",
        "code": "Repo-local code navigation helpers.",
        "server": "Tool discovery, schemas, context, and capabilities.",
        "test": "Pytest discovery and execution.",
    }

    tools: list[Dict[str, Any]] = []
    for name, router in routers.items():
        summaries = router.describe()
        actions = [
            {"name": action, "summary": summaries.get(action)}
            for action in router.allowed_actions()
        ]
        tools.append(
            {
                "name": name,
                "description": descriptions.get(name, ""),
                "category": categories.get(name, "general"),
                "version": "1.0.0",
                "deprecated": False,
                "tags": ["unified"],
                "actions": actions,
            }
        )

    return tools


def _handle_tools(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()

    category = payload.get("category")
    if category is not None and not isinstance(category, str):
        return _validation_error(
            message="category must be a string",
            request_id=request_id,
            remediation="Provide a category name like 'specs'",
        )

    tag = payload.get("tag")
    if tag is not None and not isinstance(tag, str):
        return _validation_error(
            message="tag must be a string",
            request_id=request_id,
            remediation="Provide a tag name like 'read'",
        )

    include_deprecated_value = payload.get("include_deprecated", False)
    if include_deprecated_value is not None and not isinstance(
        include_deprecated_value, bool
    ):
        return _validation_error(
            message="include_deprecated must be a boolean",
            request_id=request_id,
            remediation="Provide include_deprecated=true|false",
        )
    include_deprecated = (
        include_deprecated_value
        if isinstance(include_deprecated_value, bool)
        else False
    )

    cursor = payload.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        return _validation_error(
            message="cursor must be a string",
            request_id=request_id,
            remediation="Use the cursor provided in meta.pagination",
        )

    limit = payload.get("limit", DEFAULT_PAGE_SIZE)
    try:
        limit_int = int(limit)
    except (TypeError, ValueError):
        return _validation_error(
            message="limit must be an integer",
            request_id=request_id,
            remediation=f"Provide an integer between 1 and {MAX_PAGE_SIZE}",
        )
    limit_int = min(max(1, limit_int), MAX_PAGE_SIZE)

    start_time = time.perf_counter()

    categories_list: list[str]
    flag_service = get_flag_service()
    if flag_service.is_enabled("unified_manifest"):
        all_tools = _build_unified_manifest_tools()
        if category:
            all_tools = [tool for tool in all_tools if tool.get("category") == category]
        if tag:
            all_tools = [tool for tool in all_tools if tag in (tool.get("tags") or [])]
        categories_list = sorted(
            {tool.get("category", "general") for tool in all_tools}
        )
    else:
        registry = get_tool_registry()
        all_tools = registry.list_tools(
            category=category,
            tag=tag,
            include_deprecated=include_deprecated,
        )
        categories = registry.list_categories()
        categories_list = [c["name"] for c in categories]

    start_idx = 0
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
            start_idx = int(cursor_data.get("offset", 0))
        except (CursorError, ValueError, TypeError) as exc:
            return asdict(
                error_response(
                    f"Invalid cursor: {exc}",
                    error_code=ErrorCode.INVALID_FORMAT,
                    error_type=ErrorType.VALIDATION,
                    remediation="Use the cursor returned by server(action=tools)",
                    request_id=request_id,
                )
            )

    end_idx = start_idx + limit_int
    paginated_tools = all_tools[start_idx:end_idx]
    has_more = end_idx < len(all_tools)
    next_cursor = encode_cursor({"offset": end_idx}) if has_more else None

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _metrics.timer(_metric("tools") + ".duration_ms", elapsed_ms)

    response = paginated_response(
        data={
            "tools": paginated_tools,
            "categories": categories_list,
            "filters_applied": {
                "category": category,
                "tag": tag,
                "include_deprecated": include_deprecated,
            },
        },
        cursor=next_cursor,
        has_more=has_more,
        page_size=limit_int,
        total_count=len(all_tools),
    )
    telemetry = response.setdefault("meta", {}).setdefault("telemetry", {})
    telemetry["duration_ms"] = round(elapsed_ms, 2)

    manifest_label = "unified"
    manifest_tokens = _estimate_tokens(
        json.dumps(all_tools, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )
    telemetry["manifest_tokens"] = manifest_tokens
    telemetry["manifest_tool_count"] = len(all_tools)
    telemetry["manifest_token_budget"] = MANIFEST_TOKEN_BUDGET
    telemetry["manifest_token_budget_max"] = MANIFEST_TOKEN_BUDGET_MAX

    warning_message: str | None = None
    if manifest_tokens > MANIFEST_TOKEN_BUDGET_MAX:
        warning_message = (
            "Manifest token estimate "
            f"{manifest_tokens} exceeds maximum budget {MANIFEST_TOKEN_BUDGET_MAX}; "
            "clients may fail to load the manifest."
        )
    elif manifest_tokens > MANIFEST_TOKEN_BUDGET:
        warning_message = (
            "Manifest token estimate "
            f"{manifest_tokens} exceeds budget {MANIFEST_TOKEN_BUDGET}; "
            "trim tool/action metadata to reduce token load."
        )

    if warning_message:
        meta = response.setdefault("meta", {})
        warnings = meta.get("warnings")
        if warnings is None:
            warnings = []
        elif not isinstance(warnings, list):
            warnings = [str(warnings)]
        warnings.append(warning_message)
        meta["warnings"] = warnings

    manager = get_observability_manager()
    if manager.is_metrics_enabled():
        exporter = manager.get_prometheus_exporter()
        exporter.record_manifest_snapshot(
            manifest=manifest_label,
            tokens=manifest_tokens,
            tool_count=len(all_tools),
        )
        exporter.record_feature_flag_state(
            "unified_manifest", flag_service.is_enabled("unified_manifest")
        )

    response["meta"]["request_id"] = request_id
    return response


def _handle_schema(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return _validation_error(
            message="tool_name is required",
            request_id=request_id,
            remediation="Provide a tool name like 'spec'",
        )

    registry = get_tool_registry()
    schema = registry.get_tool_schema(tool_name.strip())
    if schema is None:
        return asdict(
            error_response(
                f"Tool '{tool_name}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use server(action=tools) to list available tools",
                request_id=request_id,
            )
        )

    return asdict(success_response(data=schema, request_id=request_id))


def _handle_capabilities(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    try:
        caps = get_capabilities()
        return asdict(success_response(data=caps, request_id=request_id))
    except Exception as exc:
        logger.exception("Error getting capabilities")
        return asdict(
            error_response(
                f"Failed to get capabilities: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check server logs",
                request_id=request_id,
            )
        )


def _handle_context(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()

    include_llm_value = payload.get("include_llm", True)
    if include_llm_value is not None and not isinstance(include_llm_value, bool):
        return _validation_error(
            message="include_llm must be a boolean",
            request_id=request_id,
            remediation="Provide include_llm=true|false",
        )
    include_llm = include_llm_value if isinstance(include_llm_value, bool) else True

    include_workflow_value = payload.get("include_workflow", True)
    if include_workflow_value is not None and not isinstance(
        include_workflow_value, bool
    ):
        return _validation_error(
            message="include_workflow must be a boolean",
            request_id=request_id,
            remediation="Provide include_workflow=true|false",
        )
    include_workflow = (
        include_workflow_value if isinstance(include_workflow_value, bool) else True
    )

    include_workspace_value = payload.get("include_workspace", True)
    if include_workspace_value is not None and not isinstance(
        include_workspace_value, bool
    ):
        return _validation_error(
            message="include_workspace must be a boolean",
            request_id=request_id,
            remediation="Provide include_workspace=true|false",
        )
    include_workspace = (
        include_workspace_value if isinstance(include_workspace_value, bool) else True
    )

    include_capabilities_value = payload.get("include_capabilities", True)
    if include_capabilities_value is not None and not isinstance(
        include_capabilities_value, bool
    ):
        return _validation_error(
            message="include_capabilities must be a boolean",
            request_id=request_id,
            remediation="Provide include_capabilities=true|false",
        )
    include_capabilities = (
        include_capabilities_value
        if isinstance(include_capabilities_value, bool)
        else True
    )

    return build_server_context_response(
        config,
        include_llm=include_llm,
        include_workflow=include_workflow,
        include_workspace=include_workspace,
        include_capabilities=include_capabilities,
        request_id=_request_id(),
    )


def _handle_llm_status(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    return build_llm_status_response(request_id=_request_id())


_ACTION_SUMMARY = {
    "tools": "List available tools with filters and pagination.",
    "schema": "Return schema metadata for a tool.",
    "capabilities": "Return server capability negotiation metadata.",
    "context": "Return server context (paths, config, capabilities).",
    "llm-status": "Return downstream LLM provider configuration health.",
}


def _build_router() -> ActionRouter:
    return ActionRouter(
        tool_name="server",
        actions=[
            ActionDefinition(
                name="tools", handler=_handle_tools, summary=_ACTION_SUMMARY["tools"]
            ),
            ActionDefinition(
                name="schema", handler=_handle_schema, summary=_ACTION_SUMMARY["schema"]
            ),
            ActionDefinition(
                name="capabilities",
                handler=_handle_capabilities,
                summary=_ACTION_SUMMARY["capabilities"],
            ),
            ActionDefinition(
                name="context",
                handler=_handle_context,
                summary=_ACTION_SUMMARY["context"],
            ),
            ActionDefinition(
                name="llm-status",
                handler=_handle_llm_status,
                summary=_ACTION_SUMMARY["llm-status"],
                aliases=("llm_status",),
            ),
        ],
    )


_SERVER_ROUTER = _build_router()


def _dispatch_server_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _SERVER_ROUTER.dispatch(action, config=config, payload=payload)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        request_id = _request_id()
        return asdict(
            error_response(
                f"Unsupported server action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                request_id=request_id,
            )
        )


def register_unified_server_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated server tool."""

    @canonical_tool(mcp, canonical_name="server")
    @mcp_tool(tool_name="server", emit_metrics=True, audit=False)
    def server(
        action: str,
        tool_name: Optional[str] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        include_deprecated: bool = False,
        cursor: Optional[str] = None,
        limit: int = DEFAULT_PAGE_SIZE,
        include_llm: bool = True,
        include_workflow: bool = True,
        include_workspace: bool = True,
        include_capabilities: bool = True,
    ) -> dict:
        payload: Dict[str, Any] = {
            "tool_name": tool_name,
            "category": category,
            "tag": tag,
            "include_deprecated": include_deprecated,
            "cursor": cursor,
            "limit": limit,
            "include_llm": include_llm,
            "include_workflow": include_workflow,
            "include_workspace": include_workspace,
            "include_capabilities": include_capabilities,
        }
        return _dispatch_server_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified server tool")


__all__ = [
    "register_unified_server_tool",
]
