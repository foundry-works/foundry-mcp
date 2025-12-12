"""
PR workflow tools for foundry-mcp.

Provides MCP tools for GitHub PR creation with SDD spec context.
PR creation requires external GitHub CLI integration and is not directly supported.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.tools.unified.pr import legacy_pr_action


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
            JSON object describing either the error or delegated action outcome.
        """

        return legacy_pr_action(
            action="create",
            spec_id=spec_id,
            title=title,
            base_branch=base_branch,
            include_journals=include_journals,
            include_diffs=include_diffs,
            model=model,
            path=path,
            dry_run=dry_run,
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
            JSON object with spec context for PR authoring.
        """

        return legacy_pr_action(
            action="get-context",
            spec_id=spec_id,
            include_tasks=include_tasks,
            include_journals=include_journals,
            include_progress=include_progress,
            path=path,
        )
