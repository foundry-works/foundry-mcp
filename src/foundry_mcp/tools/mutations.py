"""
Mutation tools for foundry-mcp.

Provides MCP tools for batch modifications to SDD specifications.
These tools use direct Python API calls to core modules for applying plans,
verification, and metadata mutations.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import (
    success_response,
    error_response,
    sanitize_error_message,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.modifications import apply_modifications, load_modifications_file
from foundry_mcp.core.spec import (
    find_specs_directory,
    load_spec,
    save_spec,
    find_spec_file,
)
from foundry_mcp.core.task import update_estimate, update_task_metadata
from foundry_mcp.core.validation import (
    format_verification_summary,
    _recalculate_counts,
)
from foundry_mcp.tools.unified.verification import legacy_verification_action
from foundry_mcp.tools.unified.task import legacy_task_action

logger = logging.getLogger(__name__)

# Metrics singleton for mutation tools
_metrics = get_metrics()


def register_mutation_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register mutation tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-apply-plan",
    )
    def spec_apply_plan(
        spec_id: str,
        modifications_file: str,
        dry_run: bool = False,
        output_file: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Apply bulk structural edits from a diff/plan JSON file.

        Wraps the SDD CLI apply-modifications command to batch apply changes
        to a specification from a JSON file. The modifications file contains
        structured changes like task additions, updates, removals, and
        metadata modifications.

        WHEN TO USE:
        - Applying review feedback to a specification
        - Batch updating multiple tasks from parsed review output
        - Migrating specs with automated transformations
        - Applying AI-generated spec modifications
        - Implementing bulk changes from sdd parse-review output

        Args:
            spec_id: Specification ID to modify
            modifications_file: Path to JSON file containing modifications
            dry_run: Preview changes without applying them
            output_file: Output path for modified spec (default: overwrite original)
            path: Project root path (default: current directory)

        Returns:
            JSON object with modification results:
            - spec_id: The specification ID
            - modifications_applied: Number of modifications applied
            - modifications_skipped: Number of modifications skipped
            - changes: Array of change summaries
            - dry_run: Whether this was a dry run
            - output_path: Path where modified spec was written
        """
        tool_name = "spec_apply_plan"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            if not modifications_file:
                return asdict(
                    error_response(
                        "modifications_file is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a path to the modifications JSON file",
                    )
                )

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-apply-plan",
                action="apply_modifications",
                spec_id=spec_id,
                modifications_file=modifications_file,
                dry_run=dry_run,
            )

            # Load modifications from file
            try:
                modifications = load_modifications_file(modifications_file)
            except FileNotFoundError:
                _metrics.counter(
                    f"mutations.{tool_name}", labels={"status": "file_not_found"}
                )
                return asdict(
                    error_response(
                        f"Modifications file not found: {modifications_file}",
                        error_code="FILE_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the modifications file path exists",
                    )
                )
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error in modifications file: {e}")
                _metrics.counter(
                    f"mutations.{tool_name}", labels={"status": "invalid_json"}
                )
                return asdict(
                    error_response(
                        "Invalid modifications JSON format",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Ensure the modifications file contains valid JSON",
                    )
                )

            # Find specs directory
            specs_dir = find_specs_directory(path)
            if not specs_dir:
                _metrics.counter(
                    f"mutations.{tool_name}", labels={"status": "specs_not_found"}
                )
                return asdict(
                    error_response(
                        "Could not find specs directory",
                        error_code="SPECS_NOT_FOUND",
                        error_type="not_found",
                        remediation="Ensure you are in a project with a specs/ directory",
                    )
                )

            # Apply modifications using direct Python API
            try:
                applied, skipped, changes = apply_modifications(
                    spec_id=spec_id,
                    modifications=modifications,
                    specs_dir=specs_dir,
                    dry_run=dry_run,
                )
            except ValueError as e:
                logger.debug(f"ValueError in spec-apply-plan: {e}")
                _metrics.counter(
                    f"mutations.{tool_name}", labels={"status": "spec_not_found"}
                )
                return asdict(
                    error_response(
                        "Specification not found or invalid",
                        error_code="SPEC_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the spec ID exists using spec-list",
                    )
                )

            # Build response data
            data: Dict[str, Any] = {
                "spec_id": spec_id,
                "modifications_applied": applied,
                "modifications_skipped": skipped,
                "changes": changes,
                "dry_run": dry_run,
            }

            # Include output path if specified
            if output_file:
                data["output_path"] = output_file

            # Include warnings if any modifications were skipped
            warnings = []
            if skipped > 0:
                warnings.append(f"{skipped} modifications were skipped")

            # Track metrics
            _metrics.counter(
                f"mutations.{tool_name}",
                labels={"status": "success", "dry_run": str(dry_run)},
            )

            if warnings:
                return asdict(success_response(data, warnings=warnings))
            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in spec-apply-plan")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="mutations"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="verification-add",
    )
    def verification_add_tool(
        spec_id: str,
        verify_id: str,
        result: str,
        command: Optional[str] = None,
        output: Optional[str] = None,
        issues: Optional[str] = None,
        notes: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """Add verification result to a task or spec verification node."""

        return legacy_verification_action(
            "add",
            config=config,
            spec_id=spec_id,
            verify_id=verify_id,
            result=result,
            command=command,
            output=output,
            issues=issues,
            notes=notes,
            dry_run=dry_run,
            path=path,
        )

    @canonical_tool(
        mcp,
        canonical_name="verification-execute",
    )
    def verification_execute_tool(
        spec_id: str,
        verify_id: str,
        record: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """Execute verification steps defined in a verification node."""

        return legacy_verification_action(
            "execute",
            config=config,
            spec_id=spec_id,
            verify_id=verify_id,
            record=record,
            path=path,
        )

    @canonical_tool(
        mcp,
        canonical_name="verification-format-summary",
    )
    def verification_format_summary_tool(
        json_file: Optional[str] = None,
        json_input: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Format verification results into a human-readable summary.

        Wraps the SDD CLI format-verification-summary command to produce
        formatted summaries of verification results from JSON data.

        WHEN TO USE:
        - Generating human-readable test reports
        - Formatting verification results for display
        - Creating summary reports from raw verification data
        - Preparing verification output for documentation

        Args:
            json_file: Path to JSON file with verification results
            json_input: JSON string with verification results (alternative to file)
            path: Project root path (default: current directory)

        Returns:
            JSON object with formatted summary:
            - summary: Human-readable summary text
            - total_verifications: Total number of verifications
            - passed: Number of passed verifications
            - failed: Number of failed verifications
            - partial: Number of partial verifications
        """
        tool_name = "verification_format_summary"
        try:
            # Validate that exactly one input source is provided
            if not json_file and not json_input:
                return asdict(
                    error_response(
                        "Either json_file or json_input is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide either json_file (path) or json_input (JSON string)",
                    )
                )

            if json_file and json_input:
                return asdict(
                    error_response(
                        "Only one of json_file or json_input should be provided",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Provide either json_file or json_input, not both",
                    )
                )

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="verification-format-summary",
                action="format_summary",
                has_file=bool(json_file),
                has_input=bool(json_input),
            )

            # Load verification data from file or parse from input
            verification_data = None
            if json_file:
                file_path = Path(json_file)
                if not file_path.exists():
                    _metrics.counter(
                        f"mutations.{tool_name}", labels={"status": "file_not_found"}
                    )
                    return asdict(
                        error_response(
                            f"JSON file not found: {json_file}",
                            error_code="FILE_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the JSON file path exists",
                        )
                    )
                try:
                    with open(file_path, "r") as f:
                        verification_data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error in verification file: {e}")
                    _metrics.counter(
                        f"mutations.{tool_name}", labels={"status": "invalid_json"}
                    )
                    return asdict(
                        error_response(
                            "Invalid JSON in verification file",
                            error_code="VALIDATION_ERROR",
                            error_type="validation",
                            remediation="Ensure the file contains valid JSON",
                        )
                    )
            elif json_input:
                try:
                    verification_data = json.loads(json_input)
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error in verification input: {e}")
                    _metrics.counter(
                        f"mutations.{tool_name}", labels={"status": "invalid_json"}
                    )
                    return asdict(
                        error_response(
                            "Invalid JSON input",
                            error_code="VALIDATION_ERROR",
                            error_type="validation",
                            remediation="Ensure the input contains valid JSON",
                        )
                    )

            if verification_data is None:  # pragma: no cover - defensive guard
                return asdict(
                    error_response(
                        "No verification data loaded",
                        error_code="INTERNAL_ERROR",
                        error_type="internal",
                        remediation="Provide json_file or json_input containing verification results",
                    )
                )

            # Format using core function
            summary_data = format_verification_summary(
                cast(Dict[str, Any] | list[Dict[str, Any]], verification_data)
            )

            # Track metrics
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "success"})

            return asdict(success_response(summary_data))

        except Exception as e:
            logger.exception("Unexpected error in verification-format-summary")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="mutations"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="task-update-estimate",
    )
    def task_update_estimate_tool(
        spec_id: str,
        task_id: str,
        hours: Optional[float] = None,
        complexity: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Update effort/time estimates for a task.

        Wraps the SDD CLI update-estimate command to modify estimated_hours
        and complexity metadata for a task.

        WHEN TO USE:
        - Adjusting task estimates based on new information
        - Recording actual vs estimated time
        - Updating complexity assessments mid-implementation
        - Re-estimating work after scope changes

        Args:
            spec_id: Specification ID containing the task
            task_id: Task ID to update
            hours: Estimated hours (float)
            complexity: Complexity level (low, medium, high)
            dry_run: Preview changes without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with update results:
            - spec_id: The specification ID
            - task_id: The task ID
            - hours: Updated hours estimate (if provided)
            - complexity: Updated complexity (if provided)
            - previous_hours: Previous hours estimate
            - previous_complexity: Previous complexity
            - dry_run: Whether this was a dry run
        """
        return legacy_task_action(
            "update-estimate",
            config=config,
            spec_id=spec_id,
            task_id=task_id,
            estimated_hours=hours,
            complexity=complexity,
            dry_run=dry_run,
            workspace=path,
        )

    @canonical_tool(
        mcp,
        canonical_name="task-update-metadata",
    )
    def task_update_metadata_tool(
        spec_id: str,
        task_id: str,
        file_path: Optional[str] = None,
        description: Optional[str] = None,
        task_category: Optional[str] = None,
        actual_hours: Optional[float] = None,
        status_note: Optional[str] = None,
        verification_type: Optional[str] = None,
        command: Optional[str] = None,
        metadata_json: Optional[str] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Update arbitrary metadata fields for a task.

        Wraps the SDD CLI update-task-metadata command to mutate per-task
        metadata payloads. Supports both named fields and custom JSON metadata.

        WHEN TO USE:
        - Updating task file paths after implementation
        - Recording actual hours spent
        - Adding status notes or completion notes
        - Setting verification type and commands
        - Storing custom metadata fields

        Args:
            spec_id: Specification ID containing the task
            task_id: Task ID to update
            file_path: File path for this task
            description: Task description
            task_category: Task category (implementation, testing, etc.)
            actual_hours: Actual hours spent on task
            status_note: Status note or completion note
            verification_type: Verification type (auto, manual, none)
            command: Command executed
            metadata_json: JSON string with custom metadata fields
            dry_run: Preview changes without saving
            path: Project root path (default: current directory)

        Returns:
            JSON object with update results:
            - spec_id: The specification ID
            - task_id: The task ID
            - fields_updated: List of fields that were updated
            - dry_run: Whether this was a dry run
        """
        custom_metadata: Optional[Dict[str, Any]] = None
        if metadata_json:
            try:
                parsed = json.loads(metadata_json)
            except json.JSONDecodeError:
                return asdict(
                    error_response(
                        "Invalid JSON in metadata_json",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Ensure metadata_json contains valid JSON",
                    )
                )
            if not isinstance(parsed, dict):
                return asdict(
                    error_response(
                        "metadata_json must be a JSON object",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation='Provide a JSON object like {"key": "value"}',
                    )
                )
            custom_metadata = parsed

        return legacy_task_action(
            "update-metadata",
            config=config,
            spec_id=spec_id,
            task_id=task_id,
            file_path=file_path,
            description=description,
            task_category=task_category,
            actual_hours=actual_hours,
            status_note=status_note,
            verification_type=verification_type,
            command=command,
            custom_metadata=custom_metadata,
            dry_run=dry_run,
            workspace=path,
        )

        tool_name = "task_update_metadata"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            if not task_id:
                return asdict(
                    error_response(
                        "task_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a task_id parameter",
                    )
                )

            # Validate at least one update field is provided
            has_update = any(
                [
                    file_path,
                    description,
                    task_category,
                    actual_hours is not None,
                    status_note,
                    verification_type,
                    command,
                    metadata_json,
                ]
            )
            if not has_update:
                return asdict(
                    error_response(
                        "At least one metadata field must be provided",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide at least one field to update",
                    )
                )

            # Validate verification_type if provided
            if verification_type and verification_type.lower() not in (
                "auto",
                "manual",
                "none",
            ):
                return asdict(
                    error_response(
                        f"Invalid verification_type '{verification_type}'. Must be one of: auto, manual, none",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use one of: auto, manual, none",
                    )
                )

            # Parse custom metadata JSON if provided
            custom_metadata = None
            if metadata_json:
                try:
                    custom_metadata = json.loads(metadata_json)
                    if not isinstance(custom_metadata, dict):
                        return asdict(
                            error_response(
                                "metadata_json must be a JSON object",
                                error_code="VALIDATION_ERROR",
                                error_type="validation",
                                remediation='Provide a JSON object like {"key": "value"}',
                            )
                        )
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error in metadata_json: {e}")
                    return asdict(
                        error_response(
                            "Invalid JSON in metadata_json",
                            error_code="VALIDATION_ERROR",
                            error_type="validation",
                            remediation="Ensure metadata_json contains valid JSON",
                        )
                    )

            # Build list of fields being updated
            fields_updated = []
            if file_path:
                fields_updated.append("file_path")
            if description:
                fields_updated.append("description")
            if task_category:
                fields_updated.append("task_category")
            if actual_hours is not None:
                fields_updated.append("actual_hours")
            if status_note:
                fields_updated.append("status_note")
            if verification_type:
                fields_updated.append("verification_type")
            if command:
                fields_updated.append("command")
            if custom_metadata:
                fields_updated.extend(list(custom_metadata.keys()))

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="task-update-metadata",
                action="update_metadata",
                spec_id=spec_id,
                task_id=task_id,
                fields_updated=fields_updated,
                dry_run=dry_run,
            )

            # Find specs directory
            specs_dir = find_specs_directory(path)
            if not specs_dir:
                _metrics.counter(
                    f"mutations.{tool_name}", labels={"status": "specs_not_found"}
                )
                return asdict(
                    error_response(
                        "Could not find specs directory",
                        error_code="SPECS_NOT_FOUND",
                        error_type="not_found",
                        remediation="Ensure you are in a project with a specs/ directory",
                    )
                )

            # For dry_run, load spec and validate task exists
            if dry_run:
                spec_data = load_spec(spec_id, specs_dir)
                if not spec_data:
                    _metrics.counter(
                        f"mutations.{tool_name}", labels={"status": "spec_not_found"}
                    )
                    return asdict(
                        error_response(
                            f"Specification '{spec_id}' not found",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list",
                        )
                    )

                hierarchy = spec_data.get("hierarchy", {})
                task = hierarchy.get(task_id)
                if not task:
                    _metrics.counter(
                        f"mutations.{tool_name}", labels={"status": "task_not_found"}
                    )
                    return asdict(
                        error_response(
                            f"Task '{task_id}' not found in spec",
                            error_code="TASK_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the task ID exists in the specification",
                        )
                    )

                data: Dict[str, Any] = {
                    "spec_id": spec_id,
                    "task_id": task_id,
                    "fields_updated": fields_updated,
                    "dry_run": True,
                }
                _metrics.counter(
                    f"mutations.{tool_name}",
                    labels={"status": "success", "dry_run": "true"},
                )
                return asdict(success_response(data))

            # Use core function to update metadata
            result_data, error_msg = update_task_metadata(
                spec_id=spec_id,
                task_id=task_id,
                file_path=file_path,
                description=description,
                task_category=task_category,
                actual_hours=actual_hours,
                status_note=status_note,
                verification_type=verification_type,
                command=command,
                custom_metadata=custom_metadata,
                specs_dir=specs_dir,
            )

            if error_msg:
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
                if "not found" in error_msg.lower():
                    if "spec" in error_msg.lower():
                        return asdict(
                            error_response(
                                error_msg,
                                error_code="SPEC_NOT_FOUND",
                                error_type="not_found",
                                remediation="Verify the spec ID exists using spec-list",
                            )
                        )
                    else:
                        return asdict(
                            error_response(
                                error_msg,
                                error_code="TASK_NOT_FOUND",
                                error_type="not_found",
                                remediation="Verify the task ID exists in the specification",
                            )
                        )
                return asdict(
                    error_response(
                        error_msg,
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Check input parameters",
                    )
                )

            # Build response data
            data = {
                "spec_id": spec_id,
                "task_id": task_id,
                "fields_updated": result_data.get("fields_updated", fields_updated),
                "dry_run": False,
            }

            # Track metrics
            _metrics.counter(
                f"mutations.{tool_name}",
                labels={
                    "status": "success",
                    "field_count": str(len(fields_updated)),
                },
            )

            return asdict(success_response(data))

        except Exception as e:
            logger.exception("Unexpected error in task-update-metadata")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="mutations"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="spec-sync-metadata",
    )
    def spec_sync_metadata(
        spec_id: str,
        dry_run: bool = False,
        path: Optional[str] = None,
    ) -> dict:
        """
        Synchronize spec metadata across stores.

        Recalculates task counts and ensures consistency between spec file
        and derived artifacts.

        WHEN TO USE:
        - Syncing spec metadata after bulk edits
        - Refreshing documentation cache with latest spec data
        - Ensuring consistency between spec file and derived artifacts
        - Propagating metadata changes to downstream consumers

        Args:
            spec_id: Specification ID to sync
            dry_run: Preview changes without applying them
            path: Project root path (default: current directory)

        Returns:
            JSON object with sync results:
            - spec_id: The specification ID
            - synced: Whether sync was successful
            - changes: List of metadata changes synced
            - dry_run: Whether this was a dry run
        """
        tool_name = "spec_sync_metadata"
        try:
            # Validate required parameters
            if not spec_id:
                return asdict(
                    error_response(
                        "spec_id is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a spec_id parameter",
                    )
                )

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-sync-metadata",
                action="sync_metadata",
                spec_id=spec_id,
                dry_run=dry_run,
            )

            # Resolve workspace
            workspace = Path(path) if path else Path.cwd()
            specs_dir = find_specs_directory(str(workspace))
            if not specs_dir:
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
                return asdict(
                    error_response(
                        "Could not find specs directory",
                        error_code="SPECS_DIR_NOT_FOUND",
                        error_type="not_found",
                        remediation="Ensure you're in a project with a specs/ directory",
                    )
                )

            # Find the spec file
            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
                return asdict(
                    error_response(
                        f"Specification '{spec_id}' not found",
                        error_code="SPEC_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the spec ID exists using spec-list",
                    )
                )

            # Load the spec
            spec_data = load_spec(str(spec_file))
            if not spec_data:
                _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
                return asdict(
                    error_response(
                        f"Failed to load specification '{spec_id}'",
                        error_code="SPEC_NOT_FOUND",
                        error_type="not_found",
                        remediation="Verify the spec ID exists and the JSON is valid",
                    )
                )

            # Capture original counts for change tracking
            hierarchy = spec_data.get("hierarchy", {})
            if not isinstance(hierarchy, dict):
                hierarchy = {}
            original_counts: Dict[str, Any] = {}
            for node_id, node in hierarchy.items():
                if "task_count" in node or "completed_count" in node:
                    original_counts[node_id] = {
                        "task_count": node.get("task_count", 0),
                        "completed_count": node.get("completed_count", 0),
                    }

            # Recalculate counts
            _recalculate_counts(spec_data)

            # Track changes
            changes = []
            updated_hierarchy = spec_data.get("hierarchy", {})
            if not isinstance(updated_hierarchy, dict):
                updated_hierarchy = {}
            for node_id, node in updated_hierarchy.items():
                if node_id in original_counts:
                    orig = original_counts[node_id]
                    new_task = node.get("task_count", 0)
                    new_completed = node.get("completed_count", 0)
                    if (
                        orig["task_count"] != new_task
                        or orig["completed_count"] != new_completed
                    ):
                        changes.append(
                            {
                                "node_id": node_id,
                                "field": "counts",
                                "old": f"{orig['completed_count']}/{orig['task_count']}",
                                "new": f"{new_completed}/{new_task}",
                            }
                        )

            # Save unless dry_run
            if not dry_run:
                save_spec(str(spec_file), spec_data)

            # Build response data
            data: Dict[str, Any] = {
                "spec_id": spec_id,
                "synced": True,
                "dry_run": dry_run,
                "changes": changes,
            }

            if not changes:
                data["message"] = "No metadata changes to sync"

            # Track metrics
            _metrics.counter(
                f"mutations.{tool_name}",
                labels={
                    "status": "success",
                    "dry_run": str(dry_run),
                },
            )

            return asdict(success_response(data))

        except PermissionError as e:
            logger.debug(f"Permission error in spec-sync-metadata: {e}")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    "Permission denied while syncing metadata",
                    error_code="PERMISSION_DENIED",
                    error_type="validation",
                    remediation="Check file permissions and access rights",
                )
            )
        except Exception as e:
            logger.exception("Unexpected error in spec-sync-metadata")
            _metrics.counter(f"mutations.{tool_name}", labels={"status": "error"})
            return asdict(
                error_response(
                    sanitize_error_message(e, context="mutations"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )
