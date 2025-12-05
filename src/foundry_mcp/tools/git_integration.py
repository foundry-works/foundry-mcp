"""
Git integration tools for foundry-mcp.

Provides MCP tools for git-related SDD operations including task commits
and bulk journaling. Uses direct Python API calls to core modules.
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.spec import find_specs_directory, find_spec_file, load_spec, save_spec
from foundry_mcp.core.journal import bulk_journal, find_unjournaled_tasks

logger = logging.getLogger(__name__)

# Metrics singleton for git integration tools
_metrics = get_metrics()


def register_git_integration_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register git integration tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="task-create-commit",
    )
    def task_create_commit(
        spec_id: str,
        task_id: str,
        skip_status_check: bool = False,
        force: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Generate a git commit for task-scoped changes.

        Creates a git commit with proper task context including spec ID,
        task ID, and task metadata in the commit message.

        WHEN TO USE:
        - Creating commits for completed task work
        - Generating structured commit messages with task context
        - Maintaining traceability between commits and spec tasks
        - Automating commit creation in CI/CD pipelines

        Args:
            spec_id: Specification ID containing the task
            task_id: Task ID to create commit for
            skip_status_check: Skip checking if task is completed
            force: Force commit even if task is not completed
            path: Project root path (default: current directory)

        Returns:
            JSON object with commit results:
            - spec_id: The specification ID
            - task_id: The task ID
            - commit_hash: The created commit hash
            - commit_message: The generated commit message
            - files_committed: List of files included in commit
        """
        # Git commit creation requires git CLI integration and subprocess calls.
        # This functionality is not available as a direct core API.
        # Use git commands directly or the Claude Code commit workflow.
        return asdict(
            error_response(
                "Task commit creation requires git CLI integration. "
                "Use git commands directly to create commits with task context.",
                error_code="NOT_IMPLEMENTED",
                error_type="unavailable",
                data={
                    "spec_id": spec_id,
                    "task_id": task_id,
                    "force": force,
                    "alternative": "git commit with task-id in message",
                    "feature_status": "requires_git_cli",
                },
                remediation="Use 'git commit -m \"task-id: description\"' to create "
                "commits with task context, or use the Claude Code commit workflow.",
            )
        )

    @canonical_tool(
        mcp,
        canonical_name="journal-bulk-add",
    )
    def journal_bulk_add(
        spec_id: str,
        tasks: Optional[str] = None,
        template: Optional[str] = None,
        template_author: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Add multiple journal entries in one shot.

        Adds journal entries to multiple tasks at once. Can use templates
        for consistent journal entry formatting.

        WHEN TO USE:
        - Journaling multiple completed tasks at once
        - Applying consistent journal templates across tasks
        - Bulk updating unjournaled tasks
        - Automating journal entry creation

        Args:
            spec_id: Specification ID to journal
            tasks: Comma-separated list of task IDs (if omitted, journals all unjournaled tasks)
            template: Journal template to apply (completion, decision, blocker)
            template_author: Override author for templated entries
            dry_run: Preview journal entries without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with journaling results:
            - spec_id: The specification ID
            - tasks_journaled: Number of tasks that received journal entries
            - task_ids: List of task IDs that were journaled
            - template_used: Template that was applied (if any)
            - dry_run: Whether this was a dry run
        """
        tool_name = "journal_bulk_add"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            # Validate template if provided
            valid_templates = ("completion", "decision", "blocker")
            if template and template not in valid_templates:
                return asdict(error_response(
                    f"Invalid template '{template}'. Must be one of: {', '.join(valid_templates)}",
                    error_code="VALIDATION_ERROR",
                    error_type="validation",
                    remediation=f"Use one of: {', '.join(valid_templates)}",
                ))

            # Resolve workspace path
            ws_path = Path(path) if path else Path.cwd()

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="journal-bulk-add",
                action="bulk_journal",
                spec_id=spec_id,
                tasks=tasks,
                template=template,
                dry_run=dry_run,
            )

            # Find and load spec
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                return asdict(error_response(
                    f"Specs directory not found in {ws_path}",
                    data={"spec_id": spec_id, "workspace": str(ws_path)},
                ))

            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                return asdict(error_response(
                    f"Specification '{spec_id}' not found",
                    error_code="SPEC_NOT_FOUND",
                    error_type="not_found",
                    remediation="Verify the spec ID exists using spec-list",
                ))

            spec_data = load_spec(spec_file)
            if not spec_data:
                return asdict(error_response(
                    f"Failed to load spec '{spec_id}'",
                    data={"spec_id": spec_id, "spec_file": str(spec_file)},
                ))

            # Determine which tasks to journal
            task_ids: List[str] = []
            if tasks:
                # Parse comma-separated task IDs
                task_ids = [t.strip() for t in tasks.split(",") if t.strip()]
            else:
                # Find all unjournaled tasks
                unjournaled = find_unjournaled_tasks(spec_data)
                task_ids = [t["task_id"] for t in unjournaled]

            if not task_ids:
                return asdict(success_response(
                    spec_id=spec_id,
                    tasks_journaled=0,
                    task_ids=[],
                    dry_run=dry_run,
                    message="No unjournaled tasks found",
                ))

            # Build journal entries based on template
            author = template_author or "claude-code"
            hierarchy = spec_data.get("hierarchy", {})

            entries: List[Dict[str, Any]] = []
            for task_id in task_ids:
                task = hierarchy.get(task_id, {})
                task_title = task.get("title", task_id)

                # Build entry based on template
                if template == "completion":
                    entry = {
                        "title": f"Completed: {task_title}",
                        "content": f"Task {task_id} has been completed.",
                        "entry_type": "status_change",
                        "task_id": task_id,
                        "author": author,
                    }
                elif template == "decision":
                    entry = {
                        "title": f"Decision: {task_title}",
                        "content": f"Decision made for task {task_id}.",
                        "entry_type": "decision",
                        "task_id": task_id,
                        "author": author,
                    }
                elif template == "blocker":
                    entry = {
                        "title": f"Blocker: {task_title}",
                        "content": f"Blocker identified for task {task_id}.",
                        "entry_type": "blocker",
                        "task_id": task_id,
                        "author": author,
                    }
                else:
                    # Default note entry
                    entry = {
                        "title": f"Journal: {task_title}",
                        "content": f"Journal entry for task {task_id}.",
                        "entry_type": "note",
                        "task_id": task_id,
                        "author": author,
                    }
                entries.append(entry)

            # If dry run, just return what would be journaled
            if dry_run:
                duration_ms = (time.perf_counter() - start_time) * 1000
                _metrics.counter(f"git_integration.{tool_name}", labels={
                    "status": "success",
                    "dry_run": "True",
                })

                return asdict(success_response(
                    spec_id=spec_id,
                    tasks_journaled=len(entries),
                    task_ids=task_ids,
                    template_used=template,
                    dry_run=True,
                    preview=entries,
                    duration_ms=round(duration_ms, 2),
                ))

            # Apply bulk journal entries
            created_entries = bulk_journal(spec_data, entries)

            # Save the spec
            if not save_spec(spec_data, spec_file):
                return asdict(error_response(
                    "Failed to save spec after journaling",
                    data={"spec_id": spec_id},
                ))

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.counter(f"git_integration.{tool_name}", labels={
                "status": "success",
                "dry_run": "False",
                "has_template": str(bool(template)),
            })
            _metrics.timer(f"git_integration.{tool_name}.duration_ms", duration_ms)

            return asdict(success_response(
                spec_id=spec_id,
                tasks_journaled=len(created_entries),
                task_ids=[e.task_id for e in created_entries if e.task_id],
                template_used=template,
                dry_run=False,
                duration_ms=round(duration_ms, 2),
            ))

        except Exception as e:
            logger.exception("Unexpected error in journal-bulk-add")
            _metrics.counter(f"git_integration.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="git integration"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))
