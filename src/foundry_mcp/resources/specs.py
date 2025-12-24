"""
Spec resources for foundry-mcp.

Provides MCP resources for accessing specs, journals, and templates.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.spec import (
    load_spec,
    list_specs,
    find_specs_directory,
    find_spec_file,
)
from foundry_mcp.core.journal import get_journal_entries

logger = logging.getLogger(__name__)


# Schema version for resource responses
SCHEMA_VERSION = "1.0.0"


def register_spec_resources(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register spec resources with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    def _get_specs_dir(workspace: Optional[str] = None) -> Optional[Path]:
        """Get the specs directory for the given workspace."""
        if workspace:
            ws_path = Path(workspace)
            if ws_path.is_dir():
                specs_path = ws_path / "specs"
                if specs_path.is_dir():
                    return specs_path
                return find_specs_directory(workspace)
        return config.specs_dir or find_specs_directory()

    def _validate_sandbox(path: Path, workspace: Optional[Path] = None) -> bool:
        """
        Validate that a path is within the workspace sandbox.

        Args:
            path: Path to validate
            workspace: Workspace root (defaults to specs_dir parent)

        Returns:
            True if path is within sandbox, False otherwise
        """
        if workspace is None:
            specs_dir = _get_specs_dir()
            if specs_dir:
                workspace = specs_dir.parent
            else:
                return False

        try:
            path.resolve().relative_to(workspace.resolve())
            return True
        except ValueError:
            return False

    # Resource: foundry://specs/ - List all specs
    @mcp.resource("foundry://specs/")
    def resource_specs_list() -> str:
        """
        List all specifications.

        Returns JSON with all specs across all status folders.
        """
        specs_dir = _get_specs_dir()
        if not specs_dir:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": "No specs directory found",
            }, separators=(",", ":"))

        specs = list_specs(specs_dir=specs_dir)

        return json.dumps({
            "success": True,
            "schema_version": SCHEMA_VERSION,
            "specs": specs,
            "count": len(specs),
        }, separators=(",", ":"))

    # Resource: foundry://specs/{status}/ - List specs by status
    @mcp.resource("foundry://specs/{status}/")
    def resource_specs_by_status(status: str) -> str:
        """
        List specifications filtered by status.

        Args:
            status: Status folder (active, pending, completed, archived)

        Returns JSON with specs in the specified status folder.
        """
        valid_statuses = {"active", "pending", "completed", "archived"}
        if status not in valid_statuses:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": f"Invalid status: {status}. Must be one of: {', '.join(sorted(valid_statuses))}",
            }, separators=(",", ":"))

        specs_dir = _get_specs_dir()
        if not specs_dir:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": "No specs directory found",
            }, separators=(",", ":"))

        specs = list_specs(specs_dir=specs_dir, status=status)

        return json.dumps({
            "success": True,
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "specs": specs,
            "count": len(specs),
        }, separators=(",", ":"))

    # Resource: foundry://specs/{status}/{spec_id} - Get specific spec
    @mcp.resource("foundry://specs/{status}/{spec_id}")
    def resource_spec_by_status(status: str, spec_id: str) -> str:
        """
        Get a specification by status and ID.

        Args:
            status: Status folder (active, pending, completed, archived)
            spec_id: Specification ID

        Returns JSON with full spec data.
        """
        valid_statuses = {"active", "pending", "completed", "archived"}
        if status not in valid_statuses:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": f"Invalid status: {status}. Must be one of: {', '.join(sorted(valid_statuses))}",
            }, separators=(",", ":"))

        specs_dir = _get_specs_dir()
        if not specs_dir:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": "No specs directory found",
            }, separators=(",", ":"))

        # Verify spec is in the specified status folder
        spec_file = specs_dir / status / f"{spec_id}.json"
        if not spec_file.exists():
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": f"Spec not found in {status}: {spec_id}",
            }, separators=(",", ":"))

        # Validate sandbox
        if not _validate_sandbox(spec_file):
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": "Access denied: path outside workspace sandbox",
            }, separators=(",", ":"))

        spec_data = load_spec(spec_id, specs_dir)
        if spec_data is None:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": f"Failed to load spec: {spec_id}",
            }, separators=(",", ":"))

        # Calculate progress
        hierarchy = spec_data.get("hierarchy", {})
        total_tasks = len(hierarchy)
        completed_tasks = sum(
            1 for task in hierarchy.values()
            if task.get("status") == "completed"
        )

        return json.dumps({
            "success": True,
            "schema_version": SCHEMA_VERSION,
            "spec_id": spec_id,
            "status": status,
            "title": spec_data.get("metadata", {}).get("title", spec_data.get("title", "Untitled")),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "progress_percentage": int((completed_tasks / total_tasks * 100)) if total_tasks > 0 else 0,
            "hierarchy": hierarchy,
            "metadata": spec_data.get("metadata", {}),
            "journal": spec_data.get("journal", []),
        }, separators=(",", ":"))

    # Resource: foundry://specs/{spec_id}/journal - Get spec journal
    @mcp.resource("foundry://specs/{spec_id}/journal")
    def resource_spec_journal(spec_id: str) -> str:
        """
        Get journal entries for a specification.

        Args:
            spec_id: Specification ID

        Returns JSON with journal entries.
        """
        specs_dir = _get_specs_dir()
        if not specs_dir:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": "No specs directory found",
            }, separators=(",", ":"))

        # Find spec file (in any status folder)
        spec_file = find_spec_file(spec_id, specs_dir)
        if not spec_file:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": f"Spec not found: {spec_id}",
            }, separators=(",", ":"))

        # Validate sandbox
        if not _validate_sandbox(spec_file):
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": "Access denied: path outside workspace sandbox",
            }, separators=(",", ":"))

        spec_data = load_spec(spec_id, specs_dir)
        if spec_data is None:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": f"Failed to load spec: {spec_id}",
            }, separators=(",", ":"))

        # Get journal entries
        entries = get_journal_entries(spec_data)

        # Convert to serializable format
        journal_data = [
            {
                "timestamp": entry.timestamp,
                "entry_type": entry.entry_type,
                "title": entry.title,
                "content": entry.content,
                "author": entry.author,
                "task_id": entry.task_id,
                "metadata": entry.metadata,
            }
            for entry in entries
        ]

        return json.dumps({
            "success": True,
            "schema_version": SCHEMA_VERSION,
            "spec_id": spec_id,
            "journal": journal_data,
            "count": len(journal_data),
        }, separators=(",", ":"))

    # Resource: foundry://templates/ - List available templates
    @mcp.resource("foundry://templates/")
    def resource_templates_list() -> str:
        """
        List available spec templates.

        Returns JSON with template information.
        """
        specs_dir = _get_specs_dir()
        if not specs_dir:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": "No specs directory found",
            }, separators=(",", ":"))

        # Look for templates in specs/templates/ directory
        templates_dir = specs_dir / "templates"
        templates = []

        if templates_dir.is_dir():
            for template_file in sorted(templates_dir.glob("*.json")):
                try:
                    with open(template_file, "r") as f:
                        template_data = json.load(f)

                    templates.append({
                        "template_id": template_file.stem,
                        "title": template_data.get("metadata", {}).get("title", template_data.get("title", template_file.stem)),
                        "description": template_data.get("metadata", {}).get("description", ""),
                        "file": str(template_file.name),
                    })
                except (json.JSONDecodeError, IOError):
                    # Skip invalid templates
                    continue

        # Add built-in templates info
        builtin_templates = [
            {
                "template_id": "basic",
                "title": "Basic Spec",
                "description": "Minimal spec with a single phase",
                "builtin": True,
            },
            {
                "template_id": "feature",
                "title": "Feature Development",
                "description": "Standard feature with design, implementation, and verification phases",
                "builtin": True,
            },
            {
                "template_id": "bugfix",
                "title": "Bug Fix",
                "description": "Bug investigation, fix, and verification",
                "builtin": True,
            },
        ]

        return json.dumps({
            "success": True,
            "schema_version": SCHEMA_VERSION,
            "templates": templates,
            "builtin_templates": builtin_templates,
            "count": len(templates),
            "builtin_count": len(builtin_templates),
        }, separators=(",", ":"))

    # Resource: foundry://templates/{template_id} - Get specific template
    @mcp.resource("foundry://templates/{template_id}")
    def resource_template(template_id: str) -> str:
        """
        Get a specific template by ID.

        Args:
            template_id: Template ID

        Returns JSON with template data.
        """
        specs_dir = _get_specs_dir()
        if not specs_dir:
            return json.dumps({
                "success": False,
                "schema_version": SCHEMA_VERSION,
                "error": "No specs directory found",
            }, separators=(",", ":"))

        # Check for custom template
        templates_dir = specs_dir / "templates"
        template_file = templates_dir / f"{template_id}.json"

        if template_file.exists():
            # Validate sandbox
            if not _validate_sandbox(template_file):
                return json.dumps({
                    "success": False,
                    "schema_version": SCHEMA_VERSION,
                    "error": "Access denied: path outside workspace sandbox",
                }, separators=(",", ":"))

            try:
                with open(template_file, "r") as f:
                    template_data = json.load(f)

                return json.dumps({
                    "success": True,
                    "schema_version": SCHEMA_VERSION,
                    "template_id": template_id,
                    "template": template_data,
                    "builtin": False,
                }, separators=(",", ":"))
            except (json.JSONDecodeError, IOError) as e:
                return json.dumps({
                    "success": False,
                    "schema_version": SCHEMA_VERSION,
                    "error": f"Failed to load template: {e}",
                }, separators=(",", ":"))

        # Check for builtin template
        builtin_templates = {
            "basic": _get_basic_template(),
            "feature": _get_feature_template(),
            "bugfix": _get_bugfix_template(),
        }

        if template_id in builtin_templates:
            return json.dumps({
                "success": True,
                "schema_version": SCHEMA_VERSION,
                "template_id": template_id,
                "template": builtin_templates[template_id],
                "builtin": True,
            }, separators=(",", ":"))

        return json.dumps({
            "success": False,
            "schema_version": SCHEMA_VERSION,
            "error": f"Template not found: {template_id}",
        }, separators=(",", ":"))

    logger.debug("Registered spec resources: foundry://specs/, foundry://specs/{status}/, "
                 "foundry://specs/{status}/{spec_id}, foundry://specs/{spec_id}/journal, "
                 "foundry://templates/, foundry://templates/{template_id}")


def _get_basic_template() -> dict:
    """Get the basic builtin template."""
    return {
        "spec_id": "{{spec_id}}",
        "title": "{{title}}",
        "metadata": {
            "title": "{{title}}",
            "description": "{{description}}",
            "created_at": "{{timestamp}}",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "{{title}}",
                "status": "pending",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Implementation",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-1-1"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Initial task",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
            },
        },
        "journal": [],
    }


def _get_feature_template() -> dict:
    """Get the feature development builtin template."""
    return {
        "spec_id": "{{spec_id}}",
        "title": "{{title}}",
        "metadata": {
            "title": "{{title}}",
            "description": "{{description}}",
            "created_at": "{{timestamp}}",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "{{title}}",
                "status": "pending",
                "children": ["phase-1", "phase-2", "phase-3"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Design",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-1-1"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Design document",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
            },
            "phase-2": {
                "type": "phase",
                "title": "Implementation",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-2-1"],
            },
            "task-2-1": {
                "type": "task",
                "title": "Core implementation",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
            },
            "phase-3": {
                "type": "phase",
                "title": "Verification",
                "status": "pending",
                "parent": "spec-root",
                "children": ["verify-3-1"],
            },
            "verify-3-1": {
                "type": "verify",
                "title": "All tests pass",
                "status": "pending",
                "parent": "phase-3",
                "children": [],
                "metadata": {
                    "verification_type": "run-tests",
                },
            },
        },
        "journal": [],
    }


def _get_bugfix_template() -> dict:
    """Get the bugfix builtin template."""
    return {
        "spec_id": "{{spec_id}}",
        "title": "{{title}}",
        "metadata": {
            "title": "{{title}}",
            "description": "{{description}}",
            "created_at": "{{timestamp}}",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "{{title}}",
                "status": "pending",
                "children": ["phase-1", "phase-2"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Investigation",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Reproduce bug",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
            },
            "task-1-2": {
                "type": "task",
                "title": "Root cause analysis",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
            },
            "phase-2": {
                "type": "phase",
                "title": "Fix & Verify",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-2-1", "verify-2-1"],
            },
            "task-2-1": {
                "type": "task",
                "title": "Implement fix",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
            },
            "verify-2-1": {
                "type": "verify",
                "title": "Bug no longer reproduces",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
                "metadata": {
                    "verification_type": "manual",
                },
            },
        },
        "journal": [],
    }
