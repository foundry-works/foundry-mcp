"""
Rendering operations for SDD spec files.
Provides markdown generation with basic and enhanced modes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Data structures

@dataclass
class RenderOptions:
    """
    Options for spec rendering.
    """
    mode: str = "basic"  # basic, enhanced
    include_metadata: bool = True
    include_progress: bool = True
    include_dependencies: bool = True
    include_journal: bool = False
    max_depth: int = 0  # 0 = unlimited
    phase_filter: Optional[List[str]] = None


@dataclass
class RenderResult:
    """
    Result of rendering a spec.
    """
    markdown: str
    spec_id: str
    title: str
    total_sections: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0


# Status icons
STATUS_ICONS = {
    "pending": "⏳",
    "in_progress": "🔄",
    "completed": "✅",
    "blocked": "🚫",
    "failed": "❌",
}


# Main rendering functions

def render_spec_to_markdown(
    spec_data: Dict[str, Any],
    options: Optional[RenderOptions] = None,
) -> RenderResult:
    """
    Render a spec to human-readable markdown.

    Args:
        spec_data: Parsed spec JSON data
        options: Optional rendering options

    Returns:
        RenderResult with markdown content and metadata
    """
    if options is None:
        options = RenderOptions()

    hierarchy = spec_data.get("hierarchy", {})
    metadata = spec_data.get("metadata", {})
    spec_id = spec_data.get("spec_id", "unknown")

    sections: List[str] = []

    # Render header
    sections.append(_render_header(spec_data, options))

    # Render objectives if present
    if metadata.get("objectives") and options.include_metadata:
        sections.append(_render_objectives(metadata))

    # Render phases
    root = hierarchy.get("spec-root", {})
    phase_ids = root.get("children", [])

    total_sections = 0
    for phase_id in phase_ids:
        if options.phase_filter and phase_id not in options.phase_filter:
            continue
        sections.append(_render_phase(hierarchy, phase_id, options))
        total_sections += 1

    # Render journal if requested
    if options.include_journal:
        journal = spec_data.get("journal", [])
        if journal:
            sections.append(_render_journal_summary(journal))

    markdown = "\n\n".join(sections)

    return RenderResult(
        markdown=markdown,
        spec_id=spec_id,
        title=root.get("title", metadata.get("title", "Untitled")),
        total_sections=total_sections,
        total_tasks=root.get("total_tasks", 0),
        completed_tasks=root.get("completed_tasks", 0),
    )


def render_progress_bar(
    completed: int,
    total: int,
    width: int = 20,
) -> str:
    """
    Generate a text-based progress bar.

    Args:
        completed: Number of completed items
        total: Total number of items
        width: Width of the bar in characters

    Returns:
        Progress bar string like [████████░░░░░░░░░░░░] 40%
    """
    if total == 0:
        return "[" + "░" * width + "] 0%"

    progress = completed / total
    filled = int(width * progress)
    empty = width - filled

    percentage = int(progress * 100)
    return f"[{'█' * filled}{'░' * empty}] {percentage}%"


def get_status_icon(status: str) -> str:
    """
    Get icon for a task status.

    Args:
        status: Task status string

    Returns:
        Status icon character
    """
    return STATUS_ICONS.get(status, "❓")


def render_task_list(
    spec_data: Dict[str, Any],
    status_filter: Optional[str] = None,
    include_completed: bool = True,
) -> str:
    """
    Render a flat list of all tasks.

    Args:
        spec_data: Parsed spec JSON data
        status_filter: Optional filter by status
        include_completed: Whether to include completed tasks

    Returns:
        Markdown list of tasks
    """
    hierarchy = spec_data.get("hierarchy", {})
    tasks = []

    for node_id, node in hierarchy.items():
        node_type = node.get("type", "")
        if node_type not in ("task", "subtask", "verify"):
            continue

        status = node.get("status", "pending")

        if status_filter and status != status_filter:
            continue

        if not include_completed and status == "completed":
            continue

        tasks.append({
            "id": node_id,
            "title": node.get("title", "Untitled"),
            "status": status,
            "type": node_type,
            "file_path": node.get("metadata", {}).get("file_path"),
        })

    lines = ["## Task List", ""]

    for task in tasks:
        icon = get_status_icon(task["status"])
        line = f"- {icon} **{task['id']}**: {task['title']}"
        if task["file_path"]:
            line += f" (`{task['file_path']}`)"
        lines.append(line)

    if not tasks:
        lines.append("_No tasks found_")

    return "\n".join(lines)


# Internal rendering functions

def _render_header(spec_data: Dict[str, Any], options: RenderOptions) -> str:
    """Render the spec header section."""
    hierarchy = spec_data.get("hierarchy", {})
    metadata = spec_data.get("metadata", {})
    spec_id = spec_data.get("spec_id", "unknown")

    root = hierarchy.get("spec-root", {})
    title = metadata.get("title") or root.get("title", "Untitled Specification")
    status = root.get("status", "pending")
    total_tasks = root.get("total_tasks", 0)
    completed_tasks = root.get("completed_tasks", 0)

    lines = [f"# {title}", ""]

    if options.include_metadata:
        lines.append(f"**Spec ID:** `{spec_id}`  ")
        lines.append(f"**Status:** {status}  ")

    if options.include_progress and total_tasks > 0:
        progress_bar = render_progress_bar(completed_tasks, total_tasks)
        lines.append(f"**Progress:** {progress_bar} ({completed_tasks}/{total_tasks} tasks)  ")

    if options.include_metadata:
        if metadata.get("estimated_hours"):
            lines.append(f"**Estimated Effort:** {metadata['estimated_hours']} hours  ")
        if metadata.get("complexity"):
            lines.append(f"**Complexity:** {metadata['complexity']}  ")

    if metadata.get("description"):
        lines.extend(["", metadata["description"]])

    return "\n".join(lines)


def _render_objectives(metadata: Dict[str, Any]) -> str:
    """Render the objectives section."""
    objectives = metadata.get("objectives", [])
    lines = ["## Objectives", ""]

    for obj in objectives:
        lines.append(f"- {obj}")

    return "\n".join(lines)


def _render_phase(
    hierarchy: Dict[str, Any],
    phase_id: str,
    options: RenderOptions,
    depth: int = 0,
) -> str:
    """Render a phase with its groups and tasks."""
    if options.max_depth > 0 and depth >= options.max_depth:
        return ""

    phase = hierarchy.get(phase_id, {})
    title = phase.get("title", "Untitled Phase")
    total_tasks = phase.get("total_tasks", 0)
    completed_tasks = phase.get("completed_tasks", 0)
    status = phase.get("status", "pending")

    icon = get_status_icon(status)

    lines = []

    if options.include_progress:
        progress_bar = render_progress_bar(completed_tasks, total_tasks)
        lines.append(f"## {icon} {title}")
        lines.append("")
        lines.append(f"**Progress:** {progress_bar} ({completed_tasks}/{total_tasks} tasks)  ")
    else:
        lines.append(f"## {icon} {title}")
        lines.append("")

    # Phase metadata
    phase_metadata = phase.get("metadata", {})
    if options.include_metadata:
        if phase_metadata.get("purpose"):
            lines.append(f"**Purpose:** {phase_metadata['purpose']}  ")
        if phase_metadata.get("risk_level"):
            lines.append(f"**Risk Level:** {phase_metadata['risk_level']}  ")
        if phase_metadata.get("estimated_hours"):
            lines.append(f"**Estimated:** {phase_metadata['estimated_hours']} hours  ")

    # Dependencies
    if options.include_dependencies:
        deps = phase.get("dependencies", {})
        if deps.get("blocked_by"):
            lines.append(f"**Blocked by:** {', '.join(deps['blocked_by'])}  ")
        if deps.get("depends"):
            lines.append(f"**Depends on:** {', '.join(deps['depends'])}  ")

    # Render groups
    group_ids = phase.get("children", [])
    for group_id in group_ids:
        group_md = _render_group(hierarchy, group_id, options, depth + 1)
        if group_md:
            lines.append("")
            lines.append(group_md)

    return "\n".join(lines)


def _render_group(
    hierarchy: Dict[str, Any],
    group_id: str,
    options: RenderOptions,
    depth: int = 0,
) -> str:
    """Render a task group."""
    if options.max_depth > 0 and depth >= options.max_depth:
        return ""

    group = hierarchy.get(group_id, {})
    title = group.get("title", "Tasks")
    total_tasks = group.get("total_tasks", 0)
    completed_tasks = group.get("completed_tasks", 0)
    status = group.get("status", "pending")

    icon = get_status_icon(status)

    lines = []

    if options.include_progress:
        lines.append(f"### {icon} {title} ({completed_tasks}/{total_tasks})")
    else:
        lines.append(f"### {icon} {title}")

    lines.append("")

    # Group dependencies
    if options.include_dependencies:
        deps = group.get("dependencies", {})
        if deps.get("blocked_by"):
            lines.append(f"**Blocked by:** {', '.join(deps['blocked_by'])}  ")
            lines.append("")

    # Render tasks
    task_ids = group.get("children", [])
    for task_id in task_ids:
        task = hierarchy.get(task_id, {})
        task_type = task.get("type", "task")

        if task_type == "verify":
            lines.append(_render_verification(hierarchy, task_id, options))
        else:
            lines.append(_render_task(hierarchy, task_id, options, depth + 1))

    return "\n".join(lines)


def _render_task(
    hierarchy: Dict[str, Any],
    task_id: str,
    options: RenderOptions,
    depth: int = 0,
    level: int = 4,
) -> str:
    """Render a task or subtask."""
    task = hierarchy.get(task_id, {})
    title = task.get("title", "Untitled Task")
    status = task.get("status", "pending")
    task_metadata = task.get("metadata", {})

    heading = "#" * level
    icon = get_status_icon(status)

    lines = [f"{heading} {icon} {title}", ""]

    if options.include_metadata:
        if task_metadata.get("file_path"):
            lines.append(f"**File:** `{task_metadata['file_path']}`  ")

        lines.append(f"**Status:** {status}  ")

        if task_metadata.get("estimated_hours"):
            lines.append(f"**Estimated:** {task_metadata['estimated_hours']} hours  ")

        if task_metadata.get("changes"):
            lines.append(f"**Changes:** {task_metadata['changes']}  ")

        if task_metadata.get("reasoning"):
            lines.append(f"**Reasoning:** {task_metadata['reasoning']}  ")

        if task_metadata.get("details"):
            details = task_metadata["details"]
            lines.append("")
            if isinstance(details, list):
                lines.append("**Details:**")
                for detail in details:
                    lines.append(f"- {detail}")
            else:
                lines.append(f"**Details:** {details}")

    # Dependencies
    if options.include_dependencies:
        deps = task.get("dependencies", {})
        if deps.get("depends"):
            lines.extend(["", f"**Depends on:** {', '.join(deps['depends'])}"])
        if deps.get("blocked_by"):
            lines.extend(["", f"**Blocked by:** {', '.join(deps['blocked_by'])}"])

    lines.append("")

    # Render subtasks
    if options.max_depth == 0 or depth < options.max_depth:
        subtask_ids = task.get("children", [])
        for subtask_id in subtask_ids:
            lines.append(_render_task(hierarchy, subtask_id, options, depth + 1, level + 1))

    return "\n".join(lines)


def _render_verification(
    hierarchy: Dict[str, Any],
    verify_id: str,
    options: RenderOptions,
) -> str:
    """Render a verification step."""
    verify = hierarchy.get(verify_id, {})
    title = verify.get("title", "Untitled Verification")
    status = verify.get("status", "pending")
    verify_metadata = verify.get("metadata", {})

    icon = get_status_icon(status)

    lines = [f"#### {icon} {title}", ""]

    if options.include_metadata:
        lines.append(f"**Status:** {status}  ")

        verification_type = verify_metadata.get("verification_type", "manual")
        lines.append(f"**Type:** {verification_type}  ")

        if verify_metadata.get("command"):
            lines.extend([
                "",
                "**Command:**",
                "```bash",
                verify_metadata["command"],
                "```",
            ])

        if verify_metadata.get("expected"):
            lines.extend(["", f"**Expected:** {verify_metadata['expected']}"])

    lines.append("")

    return "\n".join(lines)


def _render_journal_summary(journal: List[Dict[str, Any]], limit: int = 5) -> str:
    """Render a summary of recent journal entries."""
    lines = ["## Recent Journal Entries", ""]

    # Sort by timestamp descending
    sorted_entries = sorted(
        journal,
        key=lambda e: e.get("timestamp", ""),
        reverse=True,
    )[:limit]

    for entry in sorted_entries:
        timestamp = entry.get("timestamp", "Unknown")[:10]  # Just the date
        entry_type = entry.get("entry_type", "note")
        title = entry.get("title", "Untitled")
        task_id = entry.get("task_id", "")

        icon = {
            "status_change": "📝",
            "deviation": "⚠️",
            "blocker": "🚫",
            "decision": "💡",
            "note": "📌",
        }.get(entry_type, "📌")

        line = f"- {icon} **{timestamp}** [{entry_type}] {title}"
        if task_id:
            line += f" (task: {task_id})"
        lines.append(line)

    if not sorted_entries:
        lines.append("_No journal entries_")

    return "\n".join(lines)
