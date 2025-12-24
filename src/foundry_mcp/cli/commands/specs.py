"""Spec management commands for SDD CLI.

Provides commands for creating, listing, and managing specifications.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    handle_keyboard_interrupt,
    MEDIUM_TIMEOUT,
    with_sync_timeout,
)
from foundry_mcp.core.journal import list_blocked_tasks
from foundry_mcp.core.progress import list_phases as core_list_phases
from foundry_mcp.core.spec import list_specs as core_list_specs, load_spec

logger = get_cli_logger()

# Valid templates and categories
TEMPLATES = ("simple", "medium", "complex", "security")
CATEGORIES = ("investigation", "implementation", "refactoring", "decision", "research")


def generate_spec_id(name: str) -> str:
    """Generate a spec ID from a name.

    Args:
        name: Human-readable spec name.

    Returns:
        URL-safe spec ID with date suffix.
    """
    # Normalize: lowercase, replace spaces/special chars with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    # Add date suffix
    date_suffix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Add sequence number (001 for new specs)
    return f"{slug}-{date_suffix}-001"


def get_template_structure(template: str, category: str) -> Dict[str, Any]:
    """Get the hierarchical structure for a spec template.

    Args:
        template: Template type (simple, medium, complex, security).
        category: Default task category.

    Returns:
        Hierarchy dict for the spec.
    """
    base_hierarchy = {
        "spec-root": {
            "type": "spec",
            "title": "",  # Filled in later
            "status": "pending",
            "parent": None,
            "children": ["phase-1"],
            "total_tasks": 0,
            "completed_tasks": 0,
            "metadata": {
                "purpose": "",
                "category": category,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        },
        "phase-1": {
            "type": "phase",
            "title": "Planning & Discovery",
            "status": "pending",
            "parent": "spec-root",
            "children": ["task-1-1"],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "purpose": "Initial planning and requirements gathering",
                "estimated_hours": 2,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        },
        "task-1-1": {
            "type": "task",
            "title": "Define requirements",
            "status": "pending",
            "parent": "phase-1",
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "details": "Document the requirements and acceptance criteria",
                "category": category,
                "estimated_hours": 1,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        },
    }

    if template == "simple":
        return base_hierarchy

    # Medium template adds implementation phase
    if template in ("medium", "complex", "security"):
        base_hierarchy["spec-root"]["children"].append("phase-2")
        base_hierarchy["phase-1"]["dependencies"]["blocks"].append("phase-2")
        base_hierarchy["phase-2"] = {
            "type": "phase",
            "title": "Implementation",
            "status": "pending",
            "parent": "spec-root",
            "children": ["task-2-1"],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "purpose": "Core implementation work",
                "estimated_hours": 8,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": ["phase-1"],
                "depends": [],
            },
        }
        base_hierarchy["task-2-1"] = {
            "type": "task",
            "title": "Implement core functionality",
            "status": "pending",
            "parent": "phase-2",
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "details": "Implement the main features",
                "category": category,
                "estimated_hours": 4,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        }
        base_hierarchy["spec-root"]["total_tasks"] = 2
        base_hierarchy["phase-1"]["total_tasks"] = 1

    # Complex template adds verification phase
    if template in ("complex", "security"):
        base_hierarchy["spec-root"]["children"].append("phase-3")
        base_hierarchy["phase-2"]["dependencies"]["blocks"].append("phase-3")
        base_hierarchy["phase-3"] = {
            "type": "phase",
            "title": "Verification & Testing",
            "status": "pending",
            "parent": "spec-root",
            "children": ["verify-3-1"],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "purpose": "Verify implementation meets requirements",
                "estimated_hours": 4,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": ["phase-2"],
                "depends": [],
            },
        }
        base_hierarchy["verify-3-1"] = {
            "type": "verify",
            "title": "Run test suite",
            "status": "pending",
            "parent": "phase-3",
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "verification_type": "run-tests",
                "command": "pytest",
                "expected": "All tests pass",
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        }
        base_hierarchy["spec-root"]["total_tasks"] = 3

    # Security template adds security review phase
    if template == "security":
        base_hierarchy["spec-root"]["children"].append("phase-4")
        base_hierarchy["phase-3"]["dependencies"]["blocks"].append("phase-4")
        base_hierarchy["phase-4"] = {
            "type": "phase",
            "title": "Security Review",
            "status": "pending",
            "parent": "spec-root",
            "children": ["task-4-1"],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "purpose": "Security audit and hardening",
                "estimated_hours": 4,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": ["phase-3"],
                "depends": [],
            },
        }
        base_hierarchy["task-4-1"] = {
            "type": "task",
            "title": "Security audit",
            "status": "pending",
            "parent": "phase-4",
            "children": [],
            "total_tasks": 1,
            "completed_tasks": 0,
            "metadata": {
                "details": "Review for security vulnerabilities",
                "category": "investigation",
                "estimated_hours": 2,
            },
            "dependencies": {
                "blocks": [],
                "blocked_by": [],
                "depends": [],
            },
        }
        base_hierarchy["spec-root"]["total_tasks"] = 4

    return base_hierarchy


@click.group("specs")
def specs() -> None:
    """Specification management commands."""
    pass


# Template definitions for listing/showing
TEMPLATE_INFO = {
    "simple": {
        "name": "simple",
        "description": "Minimal spec with single planning phase",
        "phases": 1,
        "tasks": 1,
        "use_cases": ["Quick fixes", "Small features", "Simple investigations"],
    },
    "medium": {
        "name": "medium",
        "description": "Standard spec with planning and implementation phases",
        "phases": 2,
        "tasks": 2,
        "use_cases": ["New features", "Moderate refactoring", "Standard development"],
    },
    "complex": {
        "name": "complex",
        "description": "Full spec with planning, implementation, and verification phases",
        "phases": 3,
        "tasks": 3,
        "use_cases": ["Large features", "Major refactoring", "Critical systems"],
    },
    "security": {
        "name": "security",
        "description": "Complete spec with security review phase",
        "phases": 4,
        "tasks": 4,
        "use_cases": ["Security-sensitive features", "Authentication", "Data handling"],
    },
}


@specs.command("template")
@click.argument("action", type=click.Choice(["list", "show"]))
@click.argument("template_name", required=False)
@click.pass_context
@cli_command("template")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Template lookup timed out")
def template(
    ctx: click.Context,
    action: str,
    template_name: Optional[str] = None,
) -> None:
    """List or show spec templates.

    ACTION is either 'list' (show all templates) or 'show' (show template details).
    TEMPLATE_NAME is required for 'show' action.
    """
    if action == "list":
        templates = [
            {
                "name": info["name"],
                "description": info["description"],
                "phases": info["phases"],
                "tasks": info["tasks"],
            }
            for info in TEMPLATE_INFO.values()
        ]
        emit_success(
            {
                "templates": templates,
                "count": len(templates),
            }
        )

    elif action == "show":
        if not template_name:
            emit_error(
                "Template name required for 'show' action",
                code="MISSING_REQUIRED",
                error_type="validation",
                remediation="Provide a template name: sdd specs template show <template_name>",
                details={"required": "template_name"},
            )

        if template_name not in TEMPLATE_INFO:
            emit_error(
                f"Unknown template: {template_name}",
                code="NOT_FOUND",
                error_type="not_found",
                remediation=f"Use one of the available templates: {', '.join(TEMPLATE_INFO.keys())}",
                details={
                    "template": template_name,
                    "available": list(TEMPLATE_INFO.keys()),
                },
            )

        info = TEMPLATE_INFO[template_name]
        # Get the actual structure
        structure = get_template_structure(template_name, "implementation")

        emit_success(
            {
                "template": info,
                "structure": {
                    "nodes": list(structure.keys()),
                    "hierarchy": {
                        node_id: {
                            "type": node["type"],
                            "title": node["title"],
                            "children": node.get("children", []),
                        }
                        for node_id, node in structure.items()
                        if isinstance(node, dict)
                    },
                },
            }
        )


@specs.command("analyze")
@click.argument("directory", required=False)
@click.pass_context
@cli_command("analyze")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Spec analysis timed out")
def analyze(ctx: click.Context, directory: Optional[str] = None) -> None:
    """Analyze specs directory structure and health.

    DIRECTORY is the path to analyze (defaults to current directory).
    """
    cli_ctx = get_context(ctx)
    target_dir = Path(directory) if directory else Path.cwd()

    # Check for specs directory
    specs_dir = cli_ctx.specs_dir
    if specs_dir is None:
        # Try to find specs in the target directory
        for subdir in ("specs", "."):
            candidate = target_dir / subdir
            if candidate.is_dir():
                for folder in ("pending", "active", "completed", "archived"):
                    if (candidate / folder).is_dir():
                        specs_dir = candidate
                        break
            if specs_dir:
                break

    # Gather analysis data
    analysis: Dict[str, Any] = {
        "directory": str(target_dir.resolve()),
        "has_specs": specs_dir is not None,
        "specs_dir": str(specs_dir) if specs_dir else None,
    }

    if specs_dir and specs_dir.is_dir():
        # Count specs by folder
        folder_counts = {}
        total_specs = 0
        for folder in ("pending", "active", "completed", "archived"):
            folder_path = specs_dir / folder
            if folder_path.is_dir():
                count = len(list(folder_path.glob("*.json")))
                folder_counts[folder] = count
                total_specs += count
            else:
                folder_counts[folder] = 0

        analysis["spec_counts"] = folder_counts
        analysis["total_specs"] = total_specs

        # Check for documentation
        docs_dir = specs_dir / ".human-readable"
        analysis["documentation_available"] = docs_dir.is_dir() and any(
            docs_dir.glob("*.md")
        )

        # Check for codebase docs
        codebase_json = target_dir / "docs" / "codebase.json"
        analysis["codebase_docs_available"] = codebase_json.is_file()

        # Workspace health indicators
        analysis["health"] = {
            "has_active_specs": folder_counts.get("active", 0) > 0,
            "has_pending_specs": folder_counts.get("pending", 0) > 0,
            "completion_rate": (
                round(folder_counts.get("completed", 0) / total_specs * 100, 1)
                if total_specs > 0
                else 0
            ),
        }
    else:
        analysis["spec_counts"] = None
        analysis["total_specs"] = 0
        analysis["documentation_available"] = False
        analysis["codebase_docs_available"] = False
        analysis["health"] = None

    emit_success(analysis)


@specs.command("create")
@click.argument("name")
@click.option(
    "--template",
    type=click.Choice(TEMPLATES),
    default="medium",
    help="Spec template: simple, medium, complex, or security.",
)
@click.option(
    "--category",
    type=click.Choice(CATEGORIES),
    default="implementation",
    help="Default task category.",
)
@click.option(
    "--mission",
    type=str,
    default="",
    help="Mission statement (required for medium/complex templates).",
)
@click.pass_context
@cli_command("create")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Spec creation timed out")
def create(
    ctx: click.Context,
    name: str,
    template: str,
    category: str,
    mission: str,
) -> None:
    """Create a new specification.

    NAME is the human-readable name for the specification.
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    if template in ("medium", "complex") and not mission.strip():
        emit_error(
            "Mission is required for medium/complex specs",
            code="MISSING_REQUIRED",
            error_type="validation",
            remediation="Provide --mission with a concise goal statement",
            details={"field": "mission"},
        )

    # Ensure pending directory exists
    pending_dir = specs_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Generate spec ID
    spec_id = generate_spec_id(name)

    # Check if spec already exists
    spec_path = pending_dir / f"{spec_id}.json"
    if spec_path.exists():
        emit_error(
            f"Specification already exists: {spec_id}",
            code="DUPLICATE_ENTRY",
            error_type="conflict",
            remediation="Use a different name or delete the existing specification",
            details={"spec_id": spec_id, "path": str(spec_path)},
        )

    # Generate spec structure
    now = datetime.now(timezone.utc).isoformat()
    hierarchy = get_template_structure(template, category)

    # Fill in the title
    hierarchy["spec-root"]["title"] = name

    spec_data = {
        "spec_id": spec_id,
        "title": name,
        "generated": now,
        "last_updated": now,
        "metadata": {
            "description": "",
            "mission": mission.strip(),
            "objectives": [],
            "complexity": "medium" if template in ("medium", "complex") else "low",
            "estimated_hours": sum(
                node.get("metadata", {}).get("estimated_hours", 0)
                for node in hierarchy.values()
                if isinstance(node, dict)
            ),
            "assumptions": [],
            "status": "pending",
            "owner": "",
            "progress_percentage": 0,
            "current_phase": "phase-1",
            "category": category,
            "template": template,
        },
        "progress_percentage": 0,
        "status": "pending",
        "current_phase": "phase-1",
        "hierarchy": hierarchy,
        "journal": [],
    }

    # Write the spec file
    with open(spec_path, "w") as f:
        json.dump(spec_data, f, indent=2)

    # Count tasks
    task_count = sum(
        1
        for node in hierarchy.values()
        if isinstance(node, dict) and node.get("type") in ("task", "subtask", "verify")
    )

    emit_success(
        {
            "spec_id": spec_id,
            "spec_path": str(spec_path),
            "template": template,
            "category": category,
            "name": name,
            "structure": {
                "phases": len(
                    [
                        n
                        for n in hierarchy.values()
                        if isinstance(n, dict) and n.get("type") == "phase"
                    ]
                ),
                "tasks": task_count,
            },
        }
    )


@specs.command("schema")
@click.pass_context
@cli_command("schema")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Schema export timed out")
def schema_cmd(ctx: click.Context) -> None:
    """Export the SDD spec JSON schema.

    Returns the complete JSON schema for SDD specification files,
    useful for validation, IDE integration, and agent understanding.
    """
    from foundry_mcp.schemas import get_spec_schema

    schema, error = get_spec_schema()

    if schema is None:
        emit_error(
            "Failed to load schema",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="This may indicate a corrupted installation. Try reinstalling the package.",
            details={"error": error},
        )

    emit_success(
        {
            "schema": schema,
            "version": "1.0.0",
            "source": "bundled",
        }
    )


@specs.command("find")
@click.option(
    "--status",
    "-s",
    type=click.Choice(["active", "pending", "completed", "archived", "all"]),
    default="all",
    help="Filter by status folder.",
)
@click.pass_context
@cli_command("find")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Spec discovery timed out")
def find_specs_cmd(ctx: click.Context, status: str) -> None:
    """Find all specifications with progress information.

    Lists specs sorted by status (active first) and completion percentage.

    Examples:
        sdd specs find
        sdd specs find --status active
        sdd specs find
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )
        return

    # Use core function to list specs
    status_filter = None if status == "all" else status
    specs_list = core_list_specs(specs_dir, status=status_filter)

    emit_success(
        {
            "count": len(specs_list),
            "status_filter": status if status != "all" else None,
            "specs": specs_list,
        }
    )


@specs.command("list-phases")
@click.argument("spec_id")
@click.pass_context
@cli_command("list-phases")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "List phases timed out")
def list_phases_cmd(ctx: click.Context, spec_id: str) -> None:
    """List all phases in a specification with progress.

    SPEC_ID is the specification identifier.

    Examples:
        sdd specs list-phases my-spec
        sdd list-phases my-spec
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs find",
            details={"spec_id": spec_id, "specs_dir": str(specs_dir)},
        )
        return

    phases = core_list_phases(spec_data)

    emit_success(
        {
            "spec_id": spec_id,
            "phase_count": len(phases),
            "phases": phases,
        }
    )


@specs.command("query-tasks")
@click.argument("spec_id")
@click.option(
    "--status",
    "-s",
    help="Filter by status (pending, in_progress, completed, blocked).",
)
@click.option("--parent", "-p", help="Filter by parent node ID (e.g., phase-1).")
@click.pass_context
@cli_command("query-tasks")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Query tasks timed out")
def query_tasks_cmd(
    ctx: click.Context,
    spec_id: str,
    status: Optional[str],
    parent: Optional[str],
) -> None:
    """Query tasks in a specification with filters.

    SPEC_ID is the specification identifier.

    Examples:
        sdd specs query-tasks my-spec
        sdd specs query-tasks my-spec --status pending
        sdd specs query-tasks my-spec --parent phase-2
        sdd query-tasks my-spec --status in_progress
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs find",
            details={"spec_id": spec_id, "specs_dir": str(specs_dir)},
        )
        return

    hierarchy = spec_data.get("hierarchy", {})
    tasks = []

    for node_id, node in hierarchy.items():
        node_type = node.get("type", "")
        if node_type not in ("task", "subtask", "verify"):
            continue

        # Apply filters
        if status and node.get("status") != status:
            continue
        if parent and node.get("parent") != parent:
            continue

        tasks.append(
            {
                "task_id": node_id,
                "title": node.get("title", ""),
                "type": node_type,
                "status": node.get("status", "pending"),
                "parent": node.get("parent"),
                "children": node.get("children", []),
            }
        )

    # Sort by task_id
    tasks.sort(key=lambda t: t["task_id"])

    emit_success(
        {
            "spec_id": spec_id,
            "filters": {
                "status": status,
                "parent": parent,
            },
            "task_count": len(tasks),
            "tasks": tasks,
        }
    )


@specs.command("list-blockers")
@click.argument("spec_id")
@click.pass_context
@cli_command("list-blockers")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "List blockers timed out")
def list_blockers_cmd(ctx: click.Context, spec_id: str) -> None:
    """List all blocked tasks in a specification.

    SPEC_ID is the specification identifier.

    Returns tasks with status='blocked' and their blocker information.

    Examples:
        sdd specs list-blockers my-spec
        sdd list-blockers my-spec
    """
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )
        return

    # Load spec
    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        emit_error(
            f"Specification not found: {spec_id}",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID exists using: sdd specs find",
            details={"spec_id": spec_id, "specs_dir": str(specs_dir)},
        )
        return

    blocked = list_blocked_tasks(spec_data)

    emit_success(
        {
            "spec_id": spec_id,
            "blocker_count": len(blocked),
            "blocked_tasks": blocked,
        }
    )
