"""
Validation tools for foundry-mcp.

Provides MCP tools for spec validation, auto-fix, and statistics.
"""

import logging
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.core.spec import (
    find_specs_directory,
    find_spec_file,
    load_spec,
    save_spec,
)
from foundry_mcp.core.validation import (
    validate_spec,
    get_fix_actions,
    apply_fixes,
    calculate_stats,
)

logger = logging.getLogger(__name__)


def register_validation_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register validation tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @mcp.tool()
    @mcp_tool(tool_name="foundry_validate_spec")
    def foundry_validate_spec(
        spec_id: str,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Validate a specification file and return diagnostics.

        Checks structure, hierarchy, nodes, task counts, dependencies,
        and metadata for compliance with SDD spec requirements.

        Args:
            spec_id: Specification ID
            workspace: Optional workspace path

        Returns:
            JSON object with validation result including:
            - is_valid: Whether spec passed validation
            - error_count: Number of errors
            - warning_count: Number of warnings
            - diagnostics: List of diagnostic objects with code, message, severity, etc.
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return {
                    "success": False,
                    "data": {},
                    "error": "No specs directory found"
                }

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Spec not found: {spec_id}"
                }

            result = validate_spec(spec_data)

            # Convert diagnostics to dicts
            diagnostics = []
            for diag in result.diagnostics:
                diagnostics.append({
                    "code": diag.code,
                    "message": diag.message,
                    "severity": diag.severity,
                    "category": diag.category,
                    "location": diag.location,
                    "suggested_fix": diag.suggested_fix,
                    "auto_fixable": diag.auto_fixable,
                })

            return {
                "success": True,
                "data": {
                    "spec_id": result.spec_id,
                    "is_valid": result.is_valid,
                    "error_count": result.error_count,
                    "warning_count": result.warning_count,
                    "info_count": result.info_count,
                    "diagnostics": diagnostics,
                },
                "error": None
            }

        except Exception as e:
            logger.error(f"Error validating spec: {e}")
            return {
                "success": False,
                "data": {},
                "error": str(e)
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_fix_spec")
    def foundry_fix_spec(
        spec_id: str,
        dry_run: bool = False,
        create_backup: bool = True,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Apply auto-fixes to a specification file.

        Validates the spec and applies fixes for auto-fixable issues.
        Creates a backup before modifying unless disabled.

        Args:
            spec_id: Specification ID
            dry_run: If True, show fixes without applying
            create_backup: If True, create backup before modifying
            workspace: Optional workspace path

        Returns:
            JSON object with fix results including:
            - applied_count: Number of fixes applied
            - skipped_count: Number of fixes skipped
            - applied_actions: List of applied fix descriptions
            - backup_path: Path to backup file (if created)
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return {
                    "success": False,
                    "data": {},
                    "error": "No specs directory found"
                }

            spec_path = find_spec_file(spec_id, specs_dir)
            if not spec_path:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Spec not found: {spec_id}"
                }

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Failed to load spec: {spec_id}"
                }

            # Validate to get diagnostics
            result = validate_spec(spec_data)

            # Get fix actions
            actions = get_fix_actions(result, spec_data)

            if not actions:
                return {
                    "success": True,
                    "data": {
                        "spec_id": spec_id,
                        "applied_count": 0,
                        "skipped_count": 0,
                        "message": "No auto-fixable issues found"
                    },
                    "error": None
                }

            # Apply fixes
            report = apply_fixes(
                actions,
                str(spec_path),
                dry_run=dry_run,
                create_backup=create_backup,
            )

            applied_actions = [
                {
                    "id": a.id,
                    "description": a.description,
                    "category": a.category,
                }
                for a in report.applied_actions
            ]

            skipped_actions = [
                {
                    "id": a.id,
                    "description": a.description,
                    "category": a.category,
                }
                for a in report.skipped_actions
            ]

            return {
                "success": True,
                "data": {
                    "spec_id": spec_id,
                    "dry_run": dry_run,
                    "applied_count": len(report.applied_actions),
                    "skipped_count": len(report.skipped_actions),
                    "applied_actions": applied_actions,
                    "skipped_actions": skipped_actions,
                    "backup_path": report.backup_path,
                },
                "error": None
            }

        except Exception as e:
            logger.error(f"Error fixing spec: {e}")
            return {
                "success": False,
                "data": {},
                "error": str(e)
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_spec_stats")
    def foundry_spec_stats(
        spec_id: str,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Get statistics for a specification file.

        Calculates comprehensive statistics including:
        - Task counts by status
        - Phase breakdown
        - Progress percentage
        - Verification coverage
        - File metrics

        Args:
            spec_id: Specification ID
            workspace: Optional workspace path

        Returns:
            JSON object with spec statistics
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return {
                    "success": False,
                    "data": {},
                    "error": "No specs directory found"
                }

            spec_path = find_spec_file(spec_id, specs_dir)
            if not spec_path:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Spec not found: {spec_id}"
                }

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Failed to load spec: {spec_id}"
                }

            stats = calculate_stats(spec_data, str(spec_path))

            return {
                "success": True,
                "data": {
                    "spec_id": stats.spec_id,
                    "title": stats.title,
                    "version": stats.version,
                    "status": stats.status,
                    "totals": stats.totals,
                    "status_counts": stats.status_counts,
                    "max_depth": stats.max_depth,
                    "avg_tasks_per_phase": stats.avg_tasks_per_phase,
                    "verification_coverage": stats.verification_coverage,
                    "progress": stats.progress,
                    "file_size_kb": stats.file_size_kb,
                },
                "error": None
            }

        except Exception as e:
            logger.error(f"Error getting spec stats: {e}")
            return {
                "success": False,
                "data": {},
                "error": str(e)
            }

    @mcp.tool()
    @mcp_tool(tool_name="foundry_validate_and_fix")
    def foundry_validate_and_fix(
        spec_id: str,
        auto_fix: bool = True,
        workspace: Optional[str] = None
    ) -> dict:
        """
        Validate a spec and optionally apply auto-fixes in one operation.

        Combines validation and fixing for convenience.

        Args:
            spec_id: Specification ID
            auto_fix: If True, apply auto-fixes for fixable issues
            workspace: Optional workspace path

        Returns:
            JSON object with validation result and fix summary
        """
        try:
            if workspace:
                specs_dir = find_specs_directory(workspace)
            else:
                specs_dir = config.specs_dir or find_specs_directory()

            if not specs_dir:
                return {
                    "success": False,
                    "data": {},
                    "error": "No specs directory found"
                }

            spec_path = find_spec_file(spec_id, specs_dir)
            if not spec_path:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Spec not found: {spec_id}"
                }

            spec_data = load_spec(spec_id, specs_dir)
            if not spec_data:
                return {
                    "success": False,
                    "data": {},
                    "error": f"Failed to load spec: {spec_id}"
                }

            # Initial validation
            result = validate_spec(spec_data)

            response_data = {
                "spec_id": result.spec_id,
                "is_valid": result.is_valid,
                "error_count": result.error_count,
                "warning_count": result.warning_count,
            }

            # Apply fixes if requested and there are fixable issues
            if auto_fix and not result.is_valid:
                actions = get_fix_actions(result, spec_data)

                if actions:
                    report = apply_fixes(
                        actions,
                        str(spec_path),
                        dry_run=False,
                        create_backup=True,
                    )

                    response_data["fixes_applied"] = len(report.applied_actions)
                    response_data["backup_path"] = report.backup_path

                    # Re-validate after fixes
                    spec_data = load_spec(spec_id, specs_dir)
                    if spec_data:
                        post_result = validate_spec(spec_data)
                        response_data["post_fix_is_valid"] = post_result.is_valid
                        response_data["post_fix_error_count"] = post_result.error_count
                else:
                    response_data["fixes_applied"] = 0
                    response_data["message"] = "No auto-fixable issues found"
            else:
                response_data["fixes_applied"] = 0

            # Convert diagnostics
            diagnostics = []
            for diag in result.diagnostics:
                diagnostics.append({
                    "code": diag.code,
                    "message": diag.message,
                    "severity": diag.severity,
                    "category": diag.category,
                    "location": diag.location,
                    "auto_fixable": diag.auto_fixable,
                })

            response_data["diagnostics"] = diagnostics

            return {
                "success": True,
                "data": response_data,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error in validate_and_fix: {e}")
            return {
                "success": False,
                "data": {},
                "error": str(e)
            }

    logger.debug("Registered validation tools: foundry_validate_spec, foundry_fix_spec, "
                 "foundry_spec_stats, foundry_validate_and_fix")
