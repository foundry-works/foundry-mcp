"""Helpers for building review context sections (implementation artifacts, requirements, etc)."""

from pathlib import Path
from typing import Any, Dict, List, Optional


def _is_fidelity_verify_node(node: Dict[str, Any]) -> bool:
    """Return True if the node is a verify node with verification_type 'fidelity'."""
    return node.get("type") == "verify" and node.get("metadata", {}).get("verification_type") == "fidelity"


def _build_spec_requirements(
    spec_data: Dict[str, Any],
    task_id: Optional[str],
    phase_id: Optional[str],
    exclude_fidelity_verify: bool = False,
) -> str:
    lines: list[str] = []
    if task_id:
        task = _find_task(spec_data, task_id)
        if task:
            if exclude_fidelity_verify and _is_fidelity_verify_node(task):
                return "*Task excluded from review context (fidelity-verify node)*"
            lines.append(f"### Task: {task.get('title', task_id)}")
            lines.append(f"- **Status:** {task.get('status', 'unknown')}")
            if task.get("metadata", {}).get("task_category"):
                lines.append(f"- **Category:** {task['metadata']['task_category']}")
            if task.get("metadata", {}).get("description"):
                lines.append(f"- **Description:** {task['metadata']['description']}")
            if task.get("metadata", {}).get("details"):
                lines.append("- **Details:**")
                for detail in task["metadata"]["details"]:
                    lines.append(f"  - {detail}")
            if task.get("metadata", {}).get("file_path"):
                lines.append(f"- **Expected file:** {task['metadata']['file_path']}")
            ac = task.get("metadata", {}).get("acceptance_criteria")
            if ac and isinstance(ac, list):
                lines.append("- **Acceptance Criteria:**")
                for criterion in ac:
                    if isinstance(criterion, str) and criterion.strip():
                        lines.append(f"  - {criterion}")
    elif phase_id:
        phase = _find_phase(spec_data, phase_id)
        if phase:
            lines.append(f"### Phase: {phase.get('title', phase_id)}")
            lines.append(f"- **Status:** {phase.get('status', 'unknown')}")
            if phase.get("metadata", {}).get("description"):
                lines.append(f"- **Description:** {phase['metadata']['description']}")
            if phase.get("metadata", {}).get("purpose"):
                lines.append(f"- **Purpose:** {phase['metadata']['purpose']}")
            child_nodes = _get_child_nodes(spec_data, phase)
            if child_nodes:
                lines.append("- **Tasks:**")
                for child in child_nodes:
                    if exclude_fidelity_verify and _is_fidelity_verify_node(child):
                        continue
                    lines.append(f"  - {child.get('id', 'unknown')}: {child.get('title', 'Unknown task')}")
                    child_cat = child.get("metadata", {}).get("task_category")
                    if child_cat:
                        lines.append(f"    - Category: {child_cat}")
                    child_desc = child.get("metadata", {}).get("description")
                    if child_desc:
                        lines.append(f"    - Description: {child_desc}")
                    child_details = child.get("metadata", {}).get("details")
                    if child_details and isinstance(child_details, list):
                        for detail in child_details:
                            lines.append(f"    - Detail: {detail}")
                    child_fp = child.get("metadata", {}).get("file_path")
                    if child_fp:
                        lines.append(f"    - File: {child_fp}")
                    ac = child.get("metadata", {}).get("acceptance_criteria")
                    if ac and isinstance(ac, list):
                        for criterion in ac:
                            if isinstance(criterion, str) and criterion.strip():
                                lines.append(f"    - AC: {criterion}")
    else:
        lines.append(f"### Specification: {spec_data.get('title', 'Unknown')}")
        if spec_data.get("description"):
            lines.append(f"- **Description:** {spec_data['description']}")
        if spec_data.get("assumptions"):
            lines.append("- **Assumptions:**")
            for assumption in spec_data["assumptions"]:
                if isinstance(assumption, dict):
                    lines.append(f"  - {assumption.get('text', str(assumption))}")
                else:
                    lines.append(f"  - {assumption}")
    return "\n".join(lines) if lines else "*No requirements available*"


def _split_file_paths(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            parts.extend(_split_file_paths(item))
        return parts
    if isinstance(value, str):
        segments = [part.strip() for part in value.split(",")]
        return [segment for segment in segments if segment]
    return [str(value)]


def _normalize_for_comparison(path_value: str, workspace_root: Optional[Path]) -> str:
    raw_path = Path(path_value)
    if raw_path.is_absolute() and workspace_root:
        try:
            raw_path = raw_path.relative_to(workspace_root)
        except ValueError:
            pass
    if workspace_root and raw_path.parts and raw_path.parts[0] == workspace_root.name:
        raw_path = Path(*raw_path.parts[1:])
    return raw_path.as_posix()


def _resolve_path(path_value: str, workspace_root: Optional[Path]) -> Path:
    raw_path = Path(path_value)
    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(raw_path)
        if workspace_root:
            candidates.append(workspace_root / raw_path)
            if raw_path.parts and raw_path.parts[0] == workspace_root.name:
                candidates.append(workspace_root / Path(*raw_path.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else raw_path


def _build_implementation_artifacts(
    spec_data: Dict[str, Any],
    task_id: Optional[str],
    phase_id: Optional[str],
    files: Optional[List[str]],
    incremental: bool,
    base_branch: str,
    workspace_root: Optional[Path] = None,
    exclude_fidelity_verify: bool = False,
) -> str:
    lines: list[str] = []
    file_paths: list[str] = []
    if workspace_root is not None and not isinstance(workspace_root, Path):
        workspace_root = Path(str(workspace_root))
    if files:
        file_paths = _split_file_paths(files)
    elif task_id:
        task = _find_task(spec_data, task_id)
        if task and task.get("metadata", {}).get("file_path"):
            file_paths = _split_file_paths(task["metadata"]["file_path"])
    elif phase_id:
        phase = _find_phase(spec_data, phase_id)
        if phase:
            for child in _get_child_nodes(spec_data, phase):
                if exclude_fidelity_verify and _is_fidelity_verify_node(child):
                    continue
                if child.get("metadata", {}).get("file_path"):
                    file_paths.extend(_split_file_paths(child["metadata"]["file_path"]))
    else:
        # Full spec review - collect file_path from all tasks/subtasks/verify nodes
        hierarchy_nodes = _get_hierarchy_nodes(spec_data)
        for node in hierarchy_nodes.values():
            if node.get("type") in ("task", "subtask", "verify"):
                if exclude_fidelity_verify and _is_fidelity_verify_node(node):
                    continue
                if node.get("metadata", {}).get("file_path"):
                    file_paths.extend(_split_file_paths(node["metadata"]["file_path"]))
    if file_paths:
        deduped: List[str] = []
        seen = set()
        for file_path in file_paths:
            if file_path not in seen:
                seen.add(file_path)
                deduped.append(file_path)
        file_paths = deduped
    if incremental:
        try:
            import subprocess

            result = subprocess.run(
                ["git", "diff", "--name-only", base_branch],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                changed_files = result.stdout.strip().split("\n") if result.stdout else []
                if file_paths:
                    changed_set = {_normalize_for_comparison(path, workspace_root) for path in changed_files if path}
                    file_paths = [
                        path for path in file_paths if _normalize_for_comparison(path, workspace_root) in changed_set
                    ]
                else:
                    file_paths = [path for path in changed_files if path]
                lines.append(f"*Incremental review: {len(file_paths)} changed files since {base_branch}*\n")
        except Exception:
            lines.append(f"*Warning: Could not get git diff from {base_branch}*\n")
    for file_path in file_paths:
        path = _resolve_path(file_path, workspace_root)
        exists = path.exists()
        marker = "+" if exists else "-"
        lines.append(f"- [{marker}] `{file_path}`")
    if not lines:
        lines.append("*No implementation artifacts available*")
    return "\n".join(lines)


def _build_test_results(
    spec_data: Dict[str, Any],
    task_id: Optional[str],
    phase_id: Optional[str],
    exclude_fidelity_verify: bool = False,
) -> str:
    journal = spec_data.get("journal", [])
    # Scope to task first
    if task_id:
        if exclude_fidelity_verify:
            task = _find_task(spec_data, task_id)
            if task and _is_fidelity_verify_node(task):
                return "*No test results available*"
        journal = [e for e in journal if e.get("task_id") == task_id]
    elif phase_id:
        phase = _find_phase(spec_data, phase_id)
        if phase:
            children = _get_child_nodes(spec_data, phase)
            if exclude_fidelity_verify:
                children = [c for c in children if not _is_fidelity_verify_node(c)]
            phase_task_ids = {c["id"] for c in children if "id" in c}
            journal = [
                entry
                for entry in journal
                if entry.get("task_id") in phase_task_ids
                or (not entry.get("task_id") and entry.get("metadata", {}).get("phase_id") == phase_id)
            ]
    elif exclude_fidelity_verify:
        # Full spec scope — exclude entries from fidelity-verify nodes
        hierarchy_nodes = _get_hierarchy_nodes(spec_data)
        excluded_ids = {nid for nid, n in hierarchy_nodes.items() if _is_fidelity_verify_node(n)}
        journal = [e for e in journal if e.get("task_id") not in excluded_ids]
    # Then apply keyword filter
    test_entries = [
        entry
        for entry in journal
        if "test" in entry.get("title", "").lower()
        or "verify" in entry.get("title", "").lower()
        or "verification" in entry.get("title", "").lower()
    ]
    if test_entries:
        lines = [f"*{len(test_entries)} test-related journal entries:*"]
        for entry in test_entries:
            lines.append(f"- **{entry.get('title', 'Unknown')}** ({entry.get('timestamp', 'unknown')})")
            if entry.get("content"):
                lines.append(f"  {entry['content']}")
        return "\n".join(lines)
    return "*No test results available*"


def _build_journal_entries(
    spec_data: Dict[str, Any],
    task_id: Optional[str],
    phase_id: Optional[str],
    exclude_fidelity_verify: bool = False,
) -> str:
    journal = spec_data.get("journal", [])
    if task_id:
        if exclude_fidelity_verify:
            task = _find_task(spec_data, task_id)
            if task and _is_fidelity_verify_node(task):
                return "*No journal entries found*"
        journal = [entry for entry in journal if entry.get("task_id") == task_id]
    elif phase_id:
        # Filter to entries from tasks belonging to this phase
        phase = _find_phase(spec_data, phase_id)
        if phase:
            children = _get_child_nodes(spec_data, phase)
            if exclude_fidelity_verify:
                children = [c for c in children if not _is_fidelity_verify_node(c)]
            phase_task_ids = {c["id"] for c in children if "id" in c}
            # Include entries that belong to phase tasks or have no task_id (phase-level)
            journal = [
                entry
                for entry in journal
                if entry.get("task_id") in phase_task_ids
                or (not entry.get("task_id") and entry.get("metadata", {}).get("phase_id") == phase_id)
            ]
    if journal:
        lines = [f"*{len(journal)} journal entries found:*"]
        for entry in journal:
            entry_type = entry.get("entry_type", "note")
            timestamp = entry.get("timestamp", "unknown")[:10] if entry.get("timestamp") else "unknown"
            lines.append(f"- **[{entry_type}]** {entry.get('title', 'Untitled')} ({timestamp})")
            if entry.get("content"):
                lines.append(f"  {entry['content']}")
        return "\n".join(lines)
    return "*No journal entries found*"


def _build_spec_overview(spec_data: Dict[str, Any]) -> str:
    """Build a spec-level overview section for fidelity review context."""
    lines: list[str] = []
    metadata = spec_data.get("metadata", {})
    title = spec_data.get("title") or metadata.get("title", "Unknown")
    lines.append(f"### Specification Overview: {title}")

    description = spec_data.get("description") or metadata.get("description")
    if description:
        lines.append(f"- **Description:** {description}")

    mission = spec_data.get("mission") or metadata.get("mission")
    if mission:
        lines.append(f"- **Mission:** {mission}")

    category = spec_data.get("category") or metadata.get("category")
    if category:
        lines.append(f"- **Category:** {category}")

    complexity = spec_data.get("complexity") or metadata.get("complexity")
    if complexity:
        lines.append(f"- **Complexity:** {complexity}")

    status = spec_data.get("status") or metadata.get("status")
    if status:
        lines.append(f"- **Status:** {status}")

    progress = spec_data.get("progress") or metadata.get("progress")
    if progress:
        if isinstance(progress, dict):
            pct = progress.get("percentage", progress.get("percent"))
            if pct is not None:
                lines.append(f"- **Progress:** {pct}%")
        else:
            lines.append(f"- **Progress:** {progress}")

    objectives = spec_data.get("objectives") or metadata.get("objectives")
    if objectives and isinstance(objectives, list):
        lines.append("- **Objectives:**")
        for obj in objectives:
            if isinstance(obj, dict):
                lines.append(f"  - {obj.get('text', obj.get('title', str(obj)))}")
            else:
                lines.append(f"  - {obj}")

    assumptions = spec_data.get("assumptions") or metadata.get("assumptions")
    if assumptions and isinstance(assumptions, list):
        lines.append("- **Assumptions:**")
        for assumption in assumptions:
            if isinstance(assumption, dict):
                lines.append(f"  - {assumption.get('text', str(assumption))}")
            else:
                lines.append(f"  - {assumption}")

    return "\n".join(lines) if lines else "*No spec overview available*"


def _build_subsequent_phases(
    spec_data: Dict[str, Any],
    phase_id: Optional[str],
) -> str:
    """Build a section listing phases after the current one.

    This gives reviewers visibility into upcoming work so they don't
    penalize the implementation for features planned in later phases.
    """
    if not phase_id:
        return ""

    hierarchy = spec_data.get("hierarchy", {})
    spec_root = hierarchy.get("spec-root", {})
    phase_order = spec_root.get("children", [])

    if not phase_order:
        return ""

    try:
        current_idx = phase_order.index(phase_id)
    except ValueError:
        return ""

    subsequent_ids = phase_order[current_idx + 1 :]
    if not subsequent_ids:
        return ""

    hierarchy_nodes = _get_hierarchy_nodes(spec_data)
    lines: list[str] = [
        "*The following phases are planned but not yet expected to be implemented:*",
        "",
    ]

    for sid in subsequent_ids:
        phase_node = hierarchy_nodes.get(sid)
        if not phase_node:
            continue
        title = phase_node.get("title", sid)
        status = phase_node.get("status", "unknown")
        lines.append(f"### {title} (`{sid}`)")
        lines.append(f"- **Status:** {status}")

        desc = phase_node.get("description") or phase_node.get("purpose")
        if desc:
            lines.append(f"- **Description:** {desc}")

        children = phase_node.get("children", [])
        if children:
            lines.append(f"- **Tasks ({len(children)}):**")
            for child_id in children:
                child_node = hierarchy_nodes.get(child_id, {})
                child_title = child_node.get("title", child_id)
                lines.append(f"  - {child_title}")
        lines.append("")

    return "\n".join(lines) if len(lines) > 2 else ""


def _find_task(spec_data: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    hierarchy_nodes = _get_hierarchy_nodes(spec_data)
    if task_id in hierarchy_nodes:
        return hierarchy_nodes[task_id]
    return None


def _find_phase(spec_data: Dict[str, Any], phase_id: str) -> Optional[Dict[str, Any]]:
    hierarchy_nodes = _get_hierarchy_nodes(spec_data)
    if phase_id in hierarchy_nodes:
        return hierarchy_nodes[phase_id]
    return None


def _get_hierarchy_nodes(spec_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    hierarchy = spec_data.get("hierarchy", {})
    nodes: Dict[str, Dict[str, Any]] = {}
    if isinstance(hierarchy, dict):
        if all(isinstance(value, dict) for value in hierarchy.values()):
            for node_id, node in hierarchy.items():
                node_copy = dict(node)
                node_copy.setdefault("id", node_id)
                nodes[node_id] = node_copy
    return nodes


def _get_child_nodes(spec_data: Dict[str, Any], node: Dict[str, Any]) -> List[Dict[str, Any]]:
    hierarchy_nodes = _get_hierarchy_nodes(spec_data)
    children = node.get("children", [])
    return [hierarchy_nodes[child_id] for child_id in children if child_id in hierarchy_nodes]
