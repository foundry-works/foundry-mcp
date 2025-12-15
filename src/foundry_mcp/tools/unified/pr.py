"""Unified PR workflow tool with action routing."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.journal import get_journal_entries
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
from foundry_mcp.core.progress import get_progress_summary
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
    sanitize_error_message,
)
from foundry_mcp.core.spec import find_spec_file, find_specs_directory, load_spec
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()

_ACTION_SUMMARY = {
    "create": "Delegate PR creation to external CLI integration",
    "get-context": "Generate PR context from spec progress",
}


def perform_pr_create_with_spec(
    *,
    spec_id: str,
    title: Optional[str],
    base_branch: str,
    include_journals: bool,
    include_diffs: bool,
    model: Optional[str],
    path: Optional[str],
    dry_run: bool,
) -> dict:
    """Return the not-implemented response for PR creation."""

    return asdict(
        error_response(
            "PR creation requires GitHub CLI integration and LLM-powered description generation. "
            "Use the sdd-toolkit:sdd-pr skill for AI-powered PR creation.",
            error_code=ErrorCode.UNAVAILABLE,
            error_type=ErrorType.UNAVAILABLE,
            data={
                "spec_id": spec_id,
                "title": title,
                "base_branch": base_branch,
                "dry_run": dry_run,
                "include_journals": include_journals,
                "include_diffs": include_diffs,
                "path": path,
                "model": model,
                "alternative": "sdd-toolkit:sdd-pr skill",
                "feature_status": "requires_external_integration",
            },
            remediation="Use the sdd-toolkit:sdd-pr skill which provides GitHub CLI integration and LLM-powered PR description generation.",
        )
    )


def perform_pr_get_context(
    *,
    spec_id: str,
    include_tasks: bool,
    include_journals: bool,
    include_progress: bool,
    path: Optional[str],
) -> dict:
    """Gather context for manual or automated PR preparation."""

    start_time = time.perf_counter()

    try:
        ws_path = Path(path) if path else Path.cwd()

        specs_dir = find_specs_directory(str(ws_path))
        if not specs_dir:
            return asdict(
                error_response(
                    f"Specs directory not found in {ws_path}",
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    data={"spec_id": spec_id, "workspace": str(ws_path)},
                    remediation="Ensure you're in a project with a specs/ directory or pass a valid path.",
                )
            )

        spec_file = find_spec_file(spec_id, specs_dir)
        if not spec_file:
            return asdict(
                error_response(
                    f"Spec '{spec_id}' not found",
                    error_code=ErrorCode.SPEC_NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    data={"spec_id": spec_id, "specs_dir": str(specs_dir)},
                    remediation='Verify the spec ID exists using spec(action="list").',
                )
            )

        spec_data = load_spec(str(spec_file), specs_dir)
        if not spec_data:
            return asdict(
                error_response(
                    f"Failed to load spec '{spec_id}'",
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_type=ErrorType.INTERNAL,
                    data={"spec_id": spec_id, "spec_file": str(spec_file)},
                    remediation="Check spec JSON validity and retry.",
                )
            )

        context: Dict[str, Any] = {
            "spec_id": spec_id,
            "title": spec_data.get("metadata", {}).get("title", ""),
        }

        if include_progress:
            progress_data = get_progress_summary(spec_data)
            context["progress"] = {
                "total_tasks": progress_data.get("total_tasks", 0),
                "completed_tasks": progress_data.get("completed_tasks", 0),
                "percentage": progress_data.get("percentage", 0),
                "current_phase": progress_data.get("current_phase"),
            }

        if include_tasks:
            hierarchy = spec_data.get("hierarchy", {})
            completed_tasks = []
            for node_id, node in hierarchy.items():
                if (
                    node.get("type") in ("task", "subtask")
                    and node.get("status") == "completed"
                ):
                    completed_tasks.append(
                        {
                            "task_id": node_id,
                            "title": node.get("title", ""),
                            "completed_at": node.get("metadata", {}).get(
                                "completed_at", ""
                            ),
                        }
                    )
            context["tasks"] = completed_tasks

        if include_journals:
            journal_entries = get_journal_entries(spec_data, limit=5)
            context["journals"] = [
                {
                    "timestamp": entry.timestamp,
                    "entry_type": entry.entry_type,
                    "title": entry.title,
                    "task_id": entry.task_id,
                }
                for entry in journal_entries
            ]

        duration_ms = (time.perf_counter() - start_time) * 1000
        _metrics.timer("pr_workflow.pr_get_spec_context.duration_ms", duration_ms)

        return asdict(
            success_response(
                **context,
                telemetry={"duration_ms": round(duration_ms, 2)},
            )
        )

    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception(f"Error getting spec context for {spec_id}")
        return asdict(
            error_response(
                sanitize_error_message(exc, context="PR workflow"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                data={"spec_id": spec_id},
                remediation="Check logs for details and retry.",
            )
        )


def _handle_pr_create(**payload: Any) -> dict:
    return perform_pr_create_with_spec(
        spec_id=payload["spec_id"],
        title=payload.get("title"),
        base_branch=payload.get("base_branch", "main"),
        include_journals=payload.get("include_journals", True),
        include_diffs=payload.get("include_diffs", True),
        model=payload.get("model"),
        path=payload.get("path"),
        dry_run=payload.get("dry_run", False),
    )


def _handle_pr_get_context(**payload: Any) -> dict:
    return perform_pr_get_context(
        spec_id=payload["spec_id"],
        include_tasks=payload.get("include_tasks", True),
        include_journals=payload.get("include_journals", True),
        include_progress=payload.get("include_progress", True),
        path=payload.get("path"),
    )


_PR_ROUTER = ActionRouter(
    tool_name="pr",
    actions=[
        ActionDefinition(
            name="create",
            handler=_handle_pr_create,
            summary=_ACTION_SUMMARY["create"],
        ),
        ActionDefinition(
            name="get-context",
            handler=_handle_pr_get_context,
            summary=_ACTION_SUMMARY["get-context"],
            aliases=("context", "get_context"),
        ),
    ],
)


def _dispatch_pr_action(action: str, payload: Dict[str, Any]) -> dict:
    try:
        return _PR_ROUTER.dispatch(action=action, **payload)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported pr action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
            )
        )


def register_unified_pr_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated PR tool."""

    @canonical_tool(
        mcp,
        canonical_name="pr",
    )
    @mcp_tool(tool_name="pr", emit_metrics=True, audit=False)
    def pr(
        action: str,
        spec_id: str,
        title: Optional[str] = None,
        base_branch: str = "main",
        include_journals: bool = True,
        include_diffs: bool = True,
        model: Optional[str] = None,
        path: Optional[str] = None,
        dry_run: bool = False,
        include_tasks: bool = True,
        include_progress: bool = True,
    ) -> dict:
        """Execute PR workflows via `action` parameter."""

        payload = {
            "spec_id": spec_id,
            "title": title,
            "base_branch": base_branch,
            "include_journals": include_journals,
            "include_diffs": include_diffs,
            "model": model,
            "path": path,
            "dry_run": dry_run,
            "include_tasks": include_tasks,
            "include_progress": include_progress,
        }
        return _dispatch_pr_action(action=action, payload=payload)

    logger.debug("Registered unified pr tool")


__all__ = [
    "register_unified_pr_tool",
]
