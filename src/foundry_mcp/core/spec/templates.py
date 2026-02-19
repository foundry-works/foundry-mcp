"""
Spec creation, templates, assumptions, revisions, and frontmatter operations.

Functions for creating specs from templates, applying phase templates,
managing assumptions/revisions, and updating frontmatter metadata.

Imports ``io`` (for find/load/save), ``hierarchy`` (for add_phase_bulk),
and ``_constants`` only.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from foundry_mcp.core.spec._constants import (
    CATEGORIES,
    FRONTMATTER_KEYS,
    PHASE_TEMPLATES,
    TEMPLATES,
    TEMPLATE_DESCRIPTIONS,
)
from foundry_mcp.core.spec.hierarchy import add_phase_bulk
from foundry_mcp.core.spec.io import (
    find_spec_file,
    find_specs_directory,
    generate_spec_id,
    load_spec,
    save_spec,
)


def get_template_structure(template: str, category: str) -> Dict[str, Any]:
    """
    Get the hierarchical structure for a spec template.

    Only the 'empty' template is supported. Use phase templates to add structure.

    Args:
        template: Template type (only 'empty' is valid).
        category: Default task category.

    Returns:
        Hierarchy dict for the spec.

    Raises:
        ValueError: If template is not 'empty'.
    """
    if template != "empty":
        raise ValueError(
            f"Invalid template '{template}'. Only 'empty' template is supported. "
            f"Use phase templates (phase-add-bulk or phase-template apply) to add structure."
        )

    return {
        "spec-root": {
            "type": "spec",
            "title": "",  # Filled in later
            "status": "pending",
            "parent": None,
            "children": [],
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
    }


def get_phase_template_structure(
    template: str, category: str = "implementation"
) -> Dict[str, Any]:
    """
    Get the structure definition for a phase template.

    Phase templates define reusable phase structures with pre-configured tasks.
    Each template includes automatic verification scaffolding (run-tests + fidelity).

    Args:
        template: Phase template type (planning, implementation, testing, security, documentation).
        category: Default task category for tasks in this phase.

    Returns:
        Dict with phase structure including:
        - title: Phase title
        - description: Phase description
        - purpose: Phase purpose for metadata
        - estimated_hours: Total estimated hours
        - tasks: List of task definitions (title, description, category, estimated_hours)
        - includes_verification: Always True (verification auto-added)
    """
    templates: Dict[str, Dict[str, Any]] = {
        "planning": {
            "title": "Planning & Discovery",
            "description": "Requirements gathering, analysis, and initial planning",
            "purpose": "Define scope, requirements, and acceptance criteria",
            "estimated_hours": 4,
            "tasks": [
                {
                    "title": "Define requirements",
                    "description": "Document functional and non-functional requirements",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Requirements are documented and reviewed",
                    ],
                    "estimated_hours": 2,
                },
                {
                    "title": "Design solution approach",
                    "description": "Outline the technical approach and architecture decisions",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Solution approach and key decisions are documented",
                    ],
                    "estimated_hours": 2,
                },
            ],
        },
        "implementation": {
            "title": "Implementation",
            "description": "Core development and feature implementation",
            "purpose": "Build the primary functionality",
            "estimated_hours": 8,
            "tasks": [
                {
                    "title": "Implement core functionality",
                    "description": "Build the main features and business logic",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Core functionality is implemented and verified",
                    ],
                    "estimated_hours": 6,
                },
                {
                    "title": "Add error handling",
                    "description": "Implement error handling and edge cases",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Error handling covers expected edge cases",
                    ],
                    "estimated_hours": 2,
                },
            ],
        },
        "testing": {
            "title": "Testing & Validation",
            "description": "Comprehensive testing and quality assurance",
            "purpose": "Ensure code quality and correctness",
            "estimated_hours": 6,
            "tasks": [
                {
                    "title": "Write unit tests",
                    "description": "Create unit tests for individual components",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Unit tests cover primary logic paths",
                    ],
                    "estimated_hours": 3,
                },
                {
                    "title": "Write integration tests",
                    "description": "Create integration tests for component interactions",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Integration tests cover critical workflows",
                    ],
                    "estimated_hours": 3,
                },
            ],
        },
        "security": {
            "title": "Security Review",
            "description": "Security audit, vulnerability assessment, and hardening",
            "purpose": "Identify and remediate security vulnerabilities",
            "estimated_hours": 6,
            "tasks": [
                {
                    "title": "Security audit",
                    "description": "Review code for security vulnerabilities (OWASP Top 10)",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Security findings are documented with severity",
                    ],
                    "estimated_hours": 3,
                },
                {
                    "title": "Security remediation",
                    "description": "Fix identified vulnerabilities and harden implementation",
                    "task_category": "investigation",
                    "acceptance_criteria": [
                        "Security findings are addressed or tracked",
                    ],
                    "estimated_hours": 3,
                },
            ],
        },
        "documentation": {
            "title": "Documentation",
            "description": "Technical documentation and knowledge capture",
            "purpose": "Document the implementation for maintainability",
            "estimated_hours": 4,
            "tasks": [
                {
                    "title": "Write API documentation",
                    "description": "Document public APIs, parameters, and return values",
                    "task_category": "research",
                    "acceptance_criteria": [
                        "API documentation is updated with current behavior",
                    ],
                    "estimated_hours": 2,
                },
                {
                    "title": "Write user guide",
                    "description": "Create usage examples and integration guide",
                    "task_category": "research",
                    "acceptance_criteria": [
                        "User guide includes usage examples",
                    ],
                    "estimated_hours": 2,
                },
            ],
        },
    }

    if template not in templates:
        raise ValueError(
            f"Invalid phase template '{template}'. Must be one of: {', '.join(PHASE_TEMPLATES)}"
        )

    result = templates[template].copy()
    result["includes_verification"] = True
    result["template_name"] = template
    return result


def apply_phase_template(
    spec_id: str,
    template: str,
    specs_dir=None,
    category: str = "implementation",
    position: Optional[int] = None,
    link_previous: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Apply a phase template to an existing spec.

    Creates a new phase with pre-configured tasks based on the template.
    Automatically includes verification scaffolding (run-tests + fidelity).

    Args:
        spec_id: ID of the spec to add the phase to.
        template: Phase template name (planning, implementation, testing, security, documentation).
        specs_dir: Path to specs directory (auto-detected if not provided).
        category: Default task category for tasks (can be overridden by template).
        position: Position to insert phase (None = append at end).
        link_previous: Whether to link this phase to the previous one with dependencies.

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"phase_id": ..., "tasks_created": [...], ...}, None)
        On failure: (None, "error message")
    """
    # Validate template
    if template not in PHASE_TEMPLATES:
        return (
            None,
            f"Invalid phase template '{template}'. Must be one of: {', '.join(PHASE_TEMPLATES)}",
        )

    # Get template structure
    template_struct = get_phase_template_structure(template, category)

    # Build tasks list for add_phase_bulk
    tasks = []
    for task_def in template_struct["tasks"]:
        tasks.append({
            "type": "task",
            "title": task_def["title"],
            "description": task_def.get("description", ""),
            "task_category": task_def.get("task_category", task_def.get("category", category)),
            "acceptance_criteria": task_def.get("acceptance_criteria"),
            "estimated_hours": task_def.get("estimated_hours", 1),
        })

    # Append verification scaffolding (run-tests + fidelity-review)
    tasks.append({
        "type": "verify",
        "title": "Run tests",
        "verification_type": "run-tests",
    })
    tasks.append({
        "type": "verify",
        "title": "Fidelity review",
        "verification_type": "fidelity",
    })

    # Use add_phase_bulk to create the phase atomically
    result, error = add_phase_bulk(
        spec_id=spec_id,
        phase_title=template_struct["title"],
        tasks=tasks,
        specs_dir=specs_dir,
        phase_description=template_struct.get("description"),
        phase_purpose=template_struct.get("purpose"),
        phase_estimated_hours=template_struct.get("estimated_hours"),
        position=position,
        link_previous=link_previous,
    )

    if error:
        return None, error

    # Enhance result with template info
    if result:
        result["template_applied"] = template
        result["template_title"] = template_struct["title"]

    return result, None


def generate_spec_data(
    name: str,
    template: str = "empty",
    category: str = "implementation",
    mission: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate spec data structure without writing to disk.

    Used for preflight validation (dry_run) and by create_spec.

    Args:
        name: Human-readable name for the specification.
        template: Template type (only 'empty' is valid).
        category: Default task category.
        mission: Optional mission statement for the spec.

    Returns:
        Tuple of (spec_data, error_message).
        On success: (dict, None)
        On failure: (None, "error message")
    """
    # Validate template - only 'empty' is supported
    if template not in TEMPLATES:
        return (
            None,
            f"Invalid template '{template}'. Only 'empty' template is supported. "
            f"Use phase templates to add structure.",
        )

    # Validate category
    if category not in CATEGORIES:
        return (
            None,
            f"Invalid category '{category}'. Must be one of: {', '.join(CATEGORIES)}",
        )

    # Generate spec ID
    spec_id = generate_spec_id(name)

    # Generate spec structure
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    hierarchy = get_template_structure(template, category)

    # Fill in the title
    hierarchy["spec-root"]["title"] = name

    # Calculate estimated hours from hierarchy
    estimated_hours = sum(
        node.get("metadata", {}).get("estimated_hours", 0)
        for node in hierarchy.values()
        if isinstance(node, dict)
    )

    spec_data = {
        "spec_id": spec_id,
        "title": name,
        "generated": now,
        "last_updated": now,
        "metadata": {
            "description": "",
            "mission": mission.strip() if isinstance(mission, str) else "",
            "objectives": [],
            "complexity": "low",  # Complexity set via explicit metadata, not template
            "estimated_hours": estimated_hours,
            "assumptions": [],
            "owner": "",
            "category": category,
            "template": template,
        },
        "progress_percentage": 0,
        "status": "pending",
        "current_phase": None,  # Empty template has no phases
        "hierarchy": hierarchy,
        "journal": [],
    }

    return spec_data, None


def create_spec(
    name: str,
    template: str = "empty",
    category: str = "implementation",
    mission: Optional[str] = None,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Create a new specification file from a template.

    Args:
        name: Human-readable name for the specification.
        template: Template type (only 'empty' is valid). Use phase templates to add structure.
        category: Default task category. Default: implementation.
        mission: Optional mission statement for the spec.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "spec_path": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Generate spec data (handles validation)
    spec_data, error = generate_spec_data(
        name=name,
        template=template,
        category=category,
        mission=mission,
    )
    if error or spec_data is None:
        return None, error or "Failed to generate spec data"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR.",
        )

    # Ensure pending directory exists
    pending_dir = specs_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Check if spec already exists
    spec_id = spec_data["spec_id"]
    spec_path = pending_dir / f"{spec_id}.json"
    if spec_path.exists():
        return None, f"Specification already exists: {spec_id}"

    # Write the spec file
    try:
        with open(spec_path, "w") as f:
            json.dump(spec_data, f, indent=2)
    except (IOError, OSError) as e:
        return None, f"Failed to write spec file: {e}"

    # Count tasks and phases
    hierarchy = spec_data["hierarchy"]
    task_count = sum(
        1
        for node in hierarchy.values()
        if isinstance(node, dict) and node.get("type") in ("task", "subtask", "verify")
    )
    phase_count = sum(
        1
        for node in hierarchy.values()
        if isinstance(node, dict) and node.get("type") == "phase"
    )

    return {
        "spec_id": spec_id,
        "spec_path": str(spec_path),
        "template": template,
        "category": category,
        "name": name,
        "structure": {
            "phases": phase_count,
            "tasks": task_count,
        },
    }, None


def add_assumption(
    spec_id: str,
    text: str,
    assumption_type: Optional[str] = None,
    author: Optional[str] = None,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add an assumption to a specification's assumptions array.

    The schema expects assumptions to be stored as strings. The assumption_type
    and author are included in the returned result for API compatibility but
    are not stored in the spec (the text itself should be descriptive).

    Args:
        spec_id: Specification ID to add assumption to.
        text: Assumption text/description.
        assumption_type: Optional type/category (any string accepted, e.g. "constraint",
            "architectural", "security"). For API compatibility only.
        author: Optional author. For API compatibility.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "text": ..., ...}, None)
        On failure: (None, "error message")
    """

    # Validate text
    if not text or not text.strip():
        return None, "Assumption text is required"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    # Ensure metadata.assumptions exists
    if "metadata" not in spec_data:
        spec_data["metadata"] = {}
    if "assumptions" not in spec_data["metadata"]:
        spec_data["metadata"]["assumptions"] = []

    assumptions = spec_data["metadata"]["assumptions"]

    # Schema expects strings, so store text directly
    assumption_text = text.strip()

    # Check for duplicates
    if assumption_text in assumptions:
        return None, f"Assumption already exists: {assumption_text[:50]}..."

    # Add to assumptions array (as string per schema)
    assumptions.append(assumption_text)

    # Update last_updated
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    spec_data["last_updated"] = now

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    # Return index as "ID" for API compatibility
    assumption_index = len(assumptions)

    return {
        "spec_id": spec_id,
        "assumption_id": f"a-{assumption_index}",
        "text": assumption_text,
        "type": assumption_type,
        "author": author,
        "index": assumption_index,
    }, None


def add_revision(
    spec_id: str,
    version: str,
    changelog: str,
    author: Optional[str] = None,
    modified_by: Optional[str] = None,
    review_triggered_by: Optional[str] = None,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a revision entry to a specification's revision_history array.

    Args:
        spec_id: Specification ID to add revision to.
        version: Version number (e.g., "1.0", "1.1", "2.0").
        changelog: Description of changes made in this revision.
        author: Optional author who made the revision.
        modified_by: Optional tool or command that made the modification.
        review_triggered_by: Optional path to review report that triggered this revision.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "version": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate version
    if not version or not version.strip():
        return None, "Version is required"

    # Validate changelog
    if not changelog or not changelog.strip():
        return None, "Changelog is required"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    # Ensure metadata.revision_history exists
    if "metadata" not in spec_data:
        spec_data["metadata"] = {}
    if "revision_history" not in spec_data["metadata"]:
        spec_data["metadata"]["revision_history"] = []

    revision_history = spec_data["metadata"]["revision_history"]

    # Create revision entry per schema
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    revision_entry = {
        "version": version.strip(),
        "date": now,
        "changelog": changelog.strip(),
    }

    # Add optional fields if provided
    if author:
        revision_entry["author"] = author.strip()
    if modified_by:
        revision_entry["modified_by"] = modified_by.strip()
    if review_triggered_by:
        revision_entry["review_triggered_by"] = review_triggered_by.strip()

    # Append to revision history
    revision_history.append(revision_entry)

    # Update last_updated
    spec_data["last_updated"] = now

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "version": revision_entry["version"],
        "date": revision_entry["date"],
        "changelog": revision_entry["changelog"],
        "author": author,
        "modified_by": modified_by,
        "review_triggered_by": review_triggered_by,
        "revision_index": len(revision_history),
    }, None


def list_assumptions(
    spec_id: str,
    assumption_type: Optional[str] = None,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    List assumptions from a specification.

    Args:
        spec_id: Specification ID to list assumptions from.
        assumption_type: Optional filter parameter (any string accepted).
            Note: Since assumptions are stored as strings, this filter is
            provided for API compatibility but has no effect.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "assumptions": [...], ...}, None)
        On failure: (None, "error message")
    """
    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    # Get assumptions from metadata
    assumptions = spec_data.get("metadata", {}).get("assumptions", [])

    # Build assumption list with indices
    assumption_list = []
    for i, assumption in enumerate(assumptions, 1):
        if isinstance(assumption, str):
            assumption_list.append(
                {
                    "id": f"a-{i}",
                    "text": assumption,
                    "index": i,
                }
            )

    return {
        "spec_id": spec_id,
        "assumptions": assumption_list,
        "total_count": len(assumption_list),
        "filter_type": assumption_type,
    }, None


def update_frontmatter(
    spec_id: str,
    key: str,
    value: Any,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Update a top-level metadata field in a specification.

    Updates fields in the spec's metadata block. For arrays like assumptions
    or revision_history, use the dedicated add_assumption() and add_revision()
    functions instead.

    Args:
        spec_id: Specification ID to update.
        key: Metadata key to update (e.g., "title", "status", "description").
        value: New value for the key.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
        On success: ({"spec_id": ..., "key": ..., "value": ..., ...}, None)
        On failure: (None, "error message")
    """
    # Validate key
    if not key or not key.strip():
        return None, "Key is required"

    key = key.strip()

    # Block array fields that have dedicated functions
    if key in ("assumptions", "revision_history"):
        return (
            None,
            f"Use dedicated function for '{key}' (add_assumption or add_revision)",
        )

    # Validate value is not None (but allow empty string, 0, False, etc.)
    if value is None:
        return None, "Value cannot be None"

    # Find specs directory
    if specs_dir is None:
        specs_dir = find_specs_directory()

    if specs_dir is None:
        return (
            None,
            "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR.",
        )

    # Find and load the spec
    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    # Ensure metadata exists
    if "metadata" not in spec_data:
        spec_data["metadata"] = {}

    # Get previous value for result (check appropriate location)
    if key in ("status", "progress_percentage", "current_phase"):
        previous_value = spec_data.get(key)
    else:
        previous_value = spec_data["metadata"].get(key)

    # Process value based on type
    if isinstance(value, str):
        value = value.strip() if value else value

    # Computed fields (status, progress_percentage, current_phase) are now
    # stored only at top-level. Title is kept in metadata for descriptive purposes.
    if key in ("status", "progress_percentage", "current_phase"):
        # Update top-level only (canonical location for computed fields)
        spec_data[key] = value
    else:
        # Regular metadata field
        spec_data["metadata"][key] = value
        # Also sync title to top-level if updating it
        if key == "title":
            spec_data[key] = value

    # Update last_updated
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    spec_data["last_updated"] = now

    # Save the spec
    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "key": key,
        "value": value,
        "previous_value": previous_value,
    }, None
