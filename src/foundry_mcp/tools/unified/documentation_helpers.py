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
            if task.get("metadata", {}).get("details"):
                lines.append("- **Details:**")
                for detail in task["metadata"]["details"]:
                    lines.append(f"  - {detail}")
            if task.get("metadata", {}).get("file_path"):
                lines.append(f"- **Expected file:** {task['metadata']['file_path']}")
    elif phase_id:
        phase = _find_phase(spec_data, phase_id)
        if phase:
            lines.append(f"### Phase: {phase.get('title', phase_id)}")
            lines.append(f"- **Status:** {phase.get('status', 'unknown')}")
            child_nodes = _get_child_nodes(spec_data, phase)
            if child_nodes:
                lines.append("- **Tasks:**")
                for child in child_nodes:
                    if exclude_fidelity_verify and _is_fidelity_verify_node(child):
                        continue
                    lines.append(f"  - {child.get('id', 'unknown')}: {child.get('title', 'Unknown task')}")
    else:
        lines.append(f"### Specification: {spec_data.get('title', 'Unknown')}")
        if spec_data.get("description"):
            lines.append(f"- **Description:** {spec_data['description']}")
        if spec_data.get("assumptions"):
            lines.append("- **Assumptions:**")
            for assumption in spec_data["assumptions"][:5]:
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
    for file_path in file_paths[:5]:
        path = _resolve_path(file_path, workspace_root)
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                if len(content) > 10_000:
                    content = content[:10_000] + "\n... [truncated] ..."
                file_type = path.suffix.lstrip(".") or "text"
                lines.append(f"### File: `{file_path}`")
                lines.append(f"```{file_type}")
                lines.append(content)
                lines.append("```\n")
            except Exception as exc:
                lines.append(f"### File: `{file_path}`")
                lines.append(f"*Error reading file: {exc}*\n")
        else:
            lines.append(f"### File: `{file_path}`")
            lines.append("*File not found*\n")
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
    if exclude_fidelity_verify and phase_id:
        phase = _find_phase(spec_data, phase_id)
        if phase:
            children = _get_child_nodes(spec_data, phase)
            excluded_ids = {c["id"] for c in children if "id" in c and _is_fidelity_verify_node(c)}
            journal = [e for e in journal if e.get("task_id") not in excluded_ids]
    test_entries = [
        entry
        for entry in journal
        if "test" in entry.get("title", "").lower()
        or "verify" in entry.get("title", "").lower()
        or "verification" in entry.get("title", "").lower()
    ]
    if test_entries:
        lines = ["*Recent test-related journal entries:*"]
        for entry in test_entries[-3:]:
            lines.append(f"- **{entry.get('title', 'Unknown')}** ({entry.get('timestamp', 'unknown')})")
            if entry.get("content"):
                content = entry["content"][:500]
                if len(entry["content"]) > 500:
                    content += "..."
                lines.append(f"  {content}")
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
        for entry in journal[-5:]:
            entry_type = entry.get("entry_type", "note")
            timestamp = entry.get("timestamp", "unknown")[:10] if entry.get("timestamp") else "unknown"
            lines.append(f"- **[{entry_type}]** {entry.get('title', 'Untitled')} ({timestamp})")
            if entry.get("content"):
                content = entry["content"][:500]
                if len(entry["content"]) > 500:
                    content += "..."
                lines.append(f"  {content}")
        return "\n".join(lines)
    return "*No journal entries found*"


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
