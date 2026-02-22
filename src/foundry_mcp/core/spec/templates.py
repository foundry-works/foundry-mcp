"""
Spec creation, templates, assumptions, revisions, and frontmatter operations.

Functions for creating specs from templates, managing assumptions/revisions,
and updating frontmatter metadata.

Imports ``io`` (for find/load/save) and ``_constants`` only.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from foundry_mcp.core.spec._constants import (
    CATEGORIES,
    TEMPLATES,
)
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

    Only the 'empty' template is supported. Use phase-add-bulk to add structure.

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
            f"Invalid template '{template}'. Only 'empty' template is supported. Use phase-add-bulk to add structure."
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


def generate_spec_data(
    name: str,
    template: str = "empty",
    category: str = "implementation",
    mission: Optional[str] = None,
    plan_path: Optional[str] = None,
    plan_review_path: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generate spec data structure without writing to disk.

    Used for preflight validation (dry_run) and by create_spec.

    Args:
        name: Human-readable name for the specification.
        template: Template type (only 'empty' is valid).
        category: Default task category.
        mission: Optional mission statement for the spec.
        plan_path: Optional path to the markdown plan file (relative to specs dir).
        plan_review_path: Optional path to the synthesized plan review file.

    Returns:
        Tuple of (spec_data, error_message).
        On success: (dict, None)
        On failure: (None, "error message")
    """
    # Validate template - only 'empty' is supported
    if template not in TEMPLATES:
        return (
            None,
            f"Invalid template '{template}'. Only 'empty' template is supported. Use phase-add-bulk to add structure.",
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

    metadata: Dict[str, Any] = {
        "description": "",
        "mission": mission.strip() if isinstance(mission, str) else "",
        "objectives": [],
        "assumptions": [],
        "success_criteria": [],
        "constraints": [],
        "risks": [],
        "open_questions": [],
    }
    if plan_path is not None:
        metadata["plan_path"] = plan_path.strip()
    if plan_review_path is not None:
        metadata["plan_review_path"] = plan_review_path.strip()

    spec_data = {
        "spec_id": spec_id,
        "title": name,
        "generated": now,
        "last_updated": now,
        "metadata": metadata,
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
    plan_path: Optional[str] = None,
    plan_review_path: Optional[str] = None,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Create a new specification file from a template.

    Args:
        name: Human-readable name for the specification.
        template: Template type (only 'empty' is valid). Use phase-add-bulk to add structure.
        category: Default task category. Default: implementation.
        mission: Optional mission statement for the spec.
        plan_path: Optional path to the markdown plan file (relative to specs dir).
        plan_review_path: Optional path to the synthesized plan review file.
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
        plan_path=plan_path,
        plan_review_path=plan_review_path,
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

    # Validate plan file existence at creation time.
    # Callers may pass paths in several forms:
    #   - Relative from specs_dir:  .plans/foo.md  (canonical)
    #   - Prefixed with specs/:     specs/.plans/foo.md
    #   - Absolute:                 /home/.../specs/.plans/foo.md
    # We normalise to a relative-from-specs_dir form and store that.
    specs_dir_path = Path(specs_dir).resolve()

    def _resolve_plan_file(raw: str) -> Tuple[Optional[Path], Optional[str]]:
        """Return (resolved_absolute_path, normalised_relative_str) or raise."""
        p = Path(raw.strip())
        # Absolute path — use directly, then relativise for storage
        if p.is_absolute():
            if p.exists():
                try:
                    return p, str(p.relative_to(specs_dir_path))
                except ValueError:
                    return p, raw.strip()
            return None, None
        # Try as-is (relative to specs_dir)
        candidate = specs_dir_path / p
        if candidate.exists():
            return candidate, str(p)
        # Strip leading "specs/" prefix (common LLM mistake)
        parts = p.parts
        if parts and parts[0] == specs_dir_path.name:
            stripped = Path(*parts[1:]) if len(parts) > 1 else p
            candidate = specs_dir_path / stripped
            if candidate.exists():
                return candidate, str(stripped)
        return None, None

    if plan_path is not None:
        resolved, normed = _resolve_plan_file(plan_path)
        if resolved is None:
            return None, f"Plan file not found: {specs_dir_path / plan_path.strip()}"
        plan_path = normed  # store normalised relative path
        spec_data["metadata"]["plan_path"] = normed

    if plan_review_path is not None:
        resolved, normed = _resolve_plan_file(plan_review_path)
        if resolved is None:
            return None, f"Plan review file not found: {specs_dir_path / plan_review_path.strip()}"
        plan_review_path = normed
        spec_data["metadata"]["plan_review_path"] = normed

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
        1 for node in hierarchy.values() if isinstance(node, dict) and node.get("type") in ("task", "subtask", "verify")
    )
    phase_count = sum(1 for node in hierarchy.values() if isinstance(node, dict) and node.get("type") == "phase")

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


def add_constraint(
    spec_id: str,
    text: str,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a constraint to a specification's constraints array.

    Args:
        spec_id: Specification ID to add constraint to.
        text: Constraint text/description.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
    """
    if not text or not text.strip():
        return None, "Constraint text is required"

    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR."

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    if "metadata" not in spec_data:
        spec_data["metadata"] = {}
    if "constraints" not in spec_data["metadata"]:
        spec_data["metadata"]["constraints"] = []

    constraints = spec_data["metadata"]["constraints"]
    constraint_text = text.strip()

    if constraint_text in constraints:
        return None, f"Constraint already exists: {constraint_text[:50]}..."

    constraints.append(constraint_text)

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    spec_data["last_updated"] = now

    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "text": constraint_text,
        "index": len(constraints),
    }, None


def add_risk(
    spec_id: str,
    description: str,
    likelihood: Optional[str] = None,
    impact: Optional[str] = None,
    mitigation: Optional[str] = None,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a risk to a specification's risks array.

    Args:
        spec_id: Specification ID to add risk to.
        description: Risk description (required).
        likelihood: Risk likelihood (low, medium, high). Optional.
        impact: Risk impact (low, medium, high). Optional.
        mitigation: Mitigation strategy. Optional.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
    """
    if not description or not description.strip():
        return None, "Risk description is required"

    valid_levels = ("low", "medium", "high")
    if likelihood and likelihood.strip().lower() not in valid_levels:
        return None, f"Invalid likelihood '{likelihood}'. Must be one of: {', '.join(valid_levels)}"
    if impact and impact.strip().lower() not in valid_levels:
        return None, f"Invalid impact '{impact}'. Must be one of: {', '.join(valid_levels)}"

    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR."

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    if "metadata" not in spec_data:
        spec_data["metadata"] = {}
    if "risks" not in spec_data["metadata"]:
        spec_data["metadata"]["risks"] = []

    risk_description = description.strip()

    # Check for duplicate by description
    for existing in spec_data["metadata"]["risks"]:
        if isinstance(existing, dict) and existing.get("description", "").strip() == risk_description:
            return None, f"Risk already exists: {risk_description[:50]}..."

    risk_entry: Dict[str, Any] = {"description": risk_description}
    if likelihood:
        risk_entry["likelihood"] = likelihood.strip().lower()
    if impact:
        risk_entry["impact"] = impact.strip().lower()
    if mitigation:
        risk_entry["mitigation"] = mitigation.strip()

    spec_data["metadata"]["risks"].append(risk_entry)

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    spec_data["last_updated"] = now

    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "risk": risk_entry,
        "index": len(spec_data["metadata"]["risks"]),
    }, None


def add_question(
    spec_id: str,
    text: str,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add an open question to a specification's open_questions array.

    Args:
        spec_id: Specification ID to add question to.
        text: Question text.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
    """
    if not text or not text.strip():
        return None, "Question text is required"

    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR."

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    if "metadata" not in spec_data:
        spec_data["metadata"] = {}
    if "open_questions" not in spec_data["metadata"]:
        spec_data["metadata"]["open_questions"] = []

    questions = spec_data["metadata"]["open_questions"]
    question_text = text.strip()

    if question_text in questions:
        return None, f"Question already exists: {question_text[:50]}..."

    questions.append(question_text)

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    spec_data["last_updated"] = now

    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "text": question_text,
        "index": len(questions),
    }, None


def add_success_criterion(
    spec_id: str,
    text: str,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Add a success criterion to a specification's success_criteria array.

    Args:
        spec_id: Specification ID to add success criterion to.
        text: Success criterion text.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
    """
    if not text or not text.strip():
        return None, "Success criterion text is required"

    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR."

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    if "metadata" not in spec_data:
        spec_data["metadata"] = {}
    if "success_criteria" not in spec_data["metadata"]:
        spec_data["metadata"]["success_criteria"] = []

    criteria = spec_data["metadata"]["success_criteria"]
    criterion_text = text.strip()

    if criterion_text in criteria:
        return None, f"Success criterion already exists: {criterion_text[:50]}..."

    criteria.append(criterion_text)

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    spec_data["last_updated"] = now

    success = save_spec(spec_id, spec_data, specs_dir)
    if not success:
        return None, "Failed to save specification"

    return {
        "spec_id": spec_id,
        "text": criterion_text,
        "index": len(criteria),
    }, None


def list_constraints(
    spec_id: str,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    List constraints from a specification.

    Args:
        spec_id: Specification ID to list constraints from.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
    """
    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR."

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    constraints = spec_data.get("metadata", {}).get("constraints", [])
    constraint_list = [
        {"id": f"c-{i}", "text": c, "index": i} for i, c in enumerate(constraints, 1) if isinstance(c, str)
    ]

    return {
        "spec_id": spec_id,
        "constraints": constraint_list,
        "total_count": len(constraint_list),
    }, None


def list_risks(
    spec_id: str,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    List risks from a specification.

    Args:
        spec_id: Specification ID to list risks from.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
    """
    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR."

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    risks = spec_data.get("metadata", {}).get("risks", [])
    risk_list = [
        {"id": f"r-{i}", "index": i, **r}
        for i, r in enumerate(risks, 1)
        if isinstance(r, dict) and r.get("description")
    ]

    return {
        "spec_id": spec_id,
        "risks": risk_list,
        "total_count": len(risk_list),
    }, None


def list_questions(
    spec_id: str,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    List open questions from a specification.

    Args:
        spec_id: Specification ID to list questions from.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
    """
    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR."

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    questions = spec_data.get("metadata", {}).get("open_questions", [])
    question_list = [{"id": f"q-{i}", "text": q, "index": i} for i, q in enumerate(questions, 1) if isinstance(q, str)]

    return {
        "spec_id": spec_id,
        "questions": question_list,
        "total_count": len(question_list),
    }, None


def list_success_criteria(
    spec_id: str,
    specs_dir=None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    List success criteria from a specification.

    Args:
        spec_id: Specification ID to list success criteria from.
        specs_dir: Path to specs directory (auto-detected if not provided).

    Returns:
        Tuple of (result_dict, error_message).
    """
    if specs_dir is None:
        specs_dir = find_specs_directory()
    if specs_dir is None:
        return None, "No specs directory found. Use specs_dir parameter or set FOUNDRY_SPECS_DIR."

    spec_path = find_spec_file(spec_id, specs_dir)
    if spec_path is None:
        return None, f"Specification '{spec_id}' not found"

    spec_data = load_spec(spec_id, specs_dir)
    if spec_data is None:
        return None, f"Failed to load specification '{spec_id}'"

    criteria = spec_data.get("metadata", {}).get("success_criteria", [])
    criteria_list = [{"id": f"sc-{i}", "text": c, "index": i} for i, c in enumerate(criteria, 1) if isinstance(c, str)]

    return {
        "spec_id": spec_id,
        "success_criteria": criteria_list,
        "total_count": len(criteria_list),
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
    blocked_array_fields = {
        "assumptions": "add_assumption",
        "revision_history": "add_revision",
        "success_criteria": "add_success_criterion",
        "constraints": "add_constraint",
        "risks": "add_risk",
        "open_questions": "add_question",
    }
    if key in blocked_array_fields:
        return (
            None,
            f"Use dedicated function for '{key}' ({blocked_array_fields[key]})",
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
