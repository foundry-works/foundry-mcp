"""
PR workflow tools for foundry-mcp.

Provides MCP tools for GitHub PR creation with SDD spec context.
Uses direct Python API calls to core modules for progress and journal retrieval.
PR creation requires external GitHub CLI integration and is not directly supported.
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
from foundry_mcp.core.spec import find_specs_directory, load_spec, find_spec_file
from foundry_mcp.core.progress import get_progress_summary
from foundry_mcp.core.journal import get_journal_entries, JournalEntry

logger = logging.getLogger(__name__)

# Metrics singleton for PR workflow tools
_metrics = get_metrics()


def register_pr_workflow_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """Register PR workflow tools with the FastMCP server."""

    @canonical_tool(
        mcp,
        canonical_name="pr-create-with-spec",
    )
    @mcp_tool(tool_name="pr-create-with-spec", emit_metrics=True, audit=True)
    def pr_create_with_spec(
        spec_id: str,
        title: Optional[str] = None,
        base_branch: str = "main",
        include_journals: bool = True,
        include_diffs: bool = True,
        model: Optional[str] = None,
        path: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Create a GitHub PR with AI-enhanced description from SDD spec context.

        Wraps the SDD create-pr command to scaffold PRs with rich context
        from the specification including task completions, journal entries,
        and AI-generated summaries.

        WHEN TO USE:
        - After completing a phase or set of tasks
        - When ready to submit work for review
        - To generate comprehensive PR descriptions automatically

        Args:
            spec_id: Specification ID to create PR for
            title: PR title (default: auto-generated from spec)
            base_branch: Base branch for PR (default: main)
            include_journals: Include journal entries in PR description
            include_diffs: Include git diffs in LLM context
            model: LLM model for description generation
            path: Project root path
            dry_run: Preview PR content without creating

        Returns:
            JSON object with:
            - spec_id: The specification used
            - pr_url: URL of created PR (if not dry_run)
            - title: PR title
            - description_preview: Preview of generated description
            - llm_status: LLM configuration status

        LIMITATIONS:
        - Requires GitHub CLI (gh) to be installed and authenticated
        - Requires LLM for enhanced descriptions (falls back to basic if unavailable)
        - Git working tree must be clean or changes staged
        """
        # PR creation requires GitHub CLI integration and LLM-powered description generation.
        # This functionality is not available as a direct core API.
        # Use the sdd-toolkit:sdd-pr skill for AI-powered PR creation.
        return asdict(
            error_response(
                "PR creation requires GitHub CLI integration and LLM-powered description generation. "
                "Use the sdd-toolkit:sdd-pr skill for AI-powered PR creation.",
                error_code="NOT_IMPLEMENTED",
                error_type="unavailable",
                data={
                    "spec_id": spec_id,
                    "title": title,
                    "base_branch": base_branch,
                    "dry_run": dry_run,
                    "alternative": "sdd-toolkit:sdd-pr skill",
                    "feature_status": "requires_external_integration",
                },
                remediation="Use the sdd-toolkit:sdd-pr skill which provides "
                "GitHub CLI integration and LLM-powered PR description generation.",
            )
        )

    @canonical_tool(
        mcp,
        canonical_name="pr-get-spec-context",
    )
    @mcp_tool(tool_name="pr-get-spec-context", emit_metrics=True, audit=False)
    def pr_get_spec_context(
        spec_id: str,
        include_tasks: bool = True,
        include_journals: bool = True,
        include_progress: bool = True,
        path: Optional[str] = None,
    ) -> dict:
        """
        Get specification context for PR description generation.

        Retrieves comprehensive information about a spec that can be used
        to craft meaningful PR descriptions, including completed tasks,
        journal entries, and overall progress.

        WHEN TO USE:
        - Before creating a PR to understand what to include
        - To gather context for manual PR creation
        - For debugging PR description generation

        Args:
            spec_id: Specification ID
            include_tasks: Include completed task summaries
            include_journals: Include recent journal entries
            include_progress: Include phase/task progress stats
            path: Project root path

        Returns:
            JSON object with:
            - spec_id: The specification ID
            - title: Spec title
            - tasks: Completed tasks (if requested)
            - journals: Recent journal entries (if requested)
            - progress: Progress statistics (if requested)
        """
        start_time = time.perf_counter()

        try:
            # Resolve workspace path
            ws_path = Path(path) if path else Path.cwd()

            # Find and load spec
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                return asdict(
                    error_response(
                        f"Specs directory not found in {ws_path}",
                        data={"spec_id": spec_id, "workspace": str(ws_path)},
                    )
                )

            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                return asdict(
                    error_response(
                        f"Spec '{spec_id}' not found",
                        data={"spec_id": spec_id, "specs_dir": str(specs_dir)},
                    )
                )

            spec_data = load_spec(spec_file)
            if not spec_data:
                return asdict(
                    error_response(
                        f"Failed to load spec '{spec_id}'",
                        data={"spec_id": spec_id, "spec_file": str(spec_file)},
                    )
                )

            # Build context response
            context: Dict[str, Any] = {
                "spec_id": spec_id,
                "title": spec_data.get("metadata", {}).get("title", ""),
            }

            # Get progress information
            if include_progress:
                progress_data = get_progress_summary(spec_data)
                context["progress"] = {
                    "total_tasks": progress_data.get("total_tasks", 0),
                    "completed_tasks": progress_data.get("completed_tasks", 0),
                    "percentage": progress_data.get("percentage", 0),
                    "current_phase": progress_data.get("current_phase"),
                }

            # Get completed tasks if requested
            if include_tasks:
                hierarchy = spec_data.get("hierarchy", {})
                completed_tasks = []
                for node_id, node in hierarchy.items():
                    if node.get("type") in ("task", "subtask") and node.get("status") == "completed":
                        completed_tasks.append({
                            "task_id": node_id,
                            "title": node.get("title", ""),
                            "completed_at": node.get("metadata", {}).get("completed_at", ""),
                        })
                context["tasks"] = completed_tasks

            # Get journal entries if requested
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
                    duration_ms=round(duration_ms, 2),
                    **context,
                )
            )

        except Exception as e:
            logger.exception(f"Error getting spec context for {spec_id}")
            return asdict(
                error_response(
                    sanitize_error_message(e, context="PR workflow"),
                    data={"spec_id": spec_id},
                )
            )
