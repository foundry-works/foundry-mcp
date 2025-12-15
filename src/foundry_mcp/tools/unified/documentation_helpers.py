"""Helpers for building review context sections (implementation artifacts, requirements, etc)."""

from typing import Any, Dict, List, Optional
from pathlib import Path


def _build_spec_requirements(
    spec_data: Dict[str, Any], task_id: Optional[str], phase_id: Optional[str]
) -> str:
    lines: list[str] = []
    if task_id:
        task = _find_task(spec_data, task_id)
        if task:
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
                    lines.append(
                        f"  - {child.get('id', 'unknown')}: {child.get('title', 'Unknown task')}"
                    )
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


def _build_implementation_artifacts(
    spec_data: Dict[str, Any],
    task_id: Optional[str],
    phase_id: Optional[str],
    files: Optional[List[str]],
    incremental: bool,
    base_branch: str,
) -> str:
    lines: list[str] = []
    file_paths: list[str] = []
    if files:
        file_paths = list(files)
    elif task_id:
        task = _find_task(spec_data, task_id)
        if task and task.get("metadata", {}).get("file_path"):
            file_paths = [task["metadata"]["file_path"]]
    elif phase_id:
        phase = _find_phase(spec_data, phase_id)
        if phase:
            for child in _get_child_nodes(spec_data, phase):
                if child.get("metadata", {}).get("file_path"):
                    file_paths.append(child["metadata"]["file_path"])
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
                changed_files = (
                    result.stdout.strip().split("\n") if result.stdout else []
                )
                if file_paths:
                    file_paths = [path for path in file_paths if path in changed_files]
                else:
                    file_paths = changed_files
                lines.append(
                    f"*Incremental review: {len(file_paths)} changed files since {base_branch}*\n"
                )
        except Exception:
            lines.append(f"*Warning: Could not get git diff from {base_branch}*\n")
    for file_path in file_paths[:5]:
        path = Path(file_path)
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
    spec_data: Dict[str, Any], task_id: Optional[str], phase_id: Optional[str]
) -> str:
    journal = spec_data.get("journal", [])
    test_entries = [
        entry
        for entry in journal
        if "test" in entry.get("title", "").lower()
        or "verify" in entry.get("title", "").lower()
    ]
    if test_entries:
        lines = ["*Recent test-related journal entries:*"]
        for entry in test_entries[-3:]:
            lines.append(
                f"- **{entry.get('title', 'Unknown')}** ({entry.get('timestamp', 'unknown')})"
            )
            if entry.get("content"):
                content = entry["content"][:500]
                if len(entry["content"]) > 500:
                    content += "..."
                lines.append(f"  {content}")
        return "\n".join(lines)
    return "*No test results available*"


def _build_journal_entries(
    spec_data: Dict[str, Any], task_id: Optional[str], phase_id: Optional[str]
) -> str:
    journal = spec_data.get("journal", [])
    if task_id:
        journal = [entry for entry in journal if entry.get("task_id") == task_id]
    if journal:
        lines = [f"*{len(journal)} journal entries found:*"]
        for entry in journal[-5:]:
            entry_type = entry.get("entry_type", "note")
            timestamp = (
                entry.get("timestamp", "unknown")[:10]
                if entry.get("timestamp")
                else "unknown"
            )
            lines.append(
                f"- **[{entry_type}]** {entry.get('title', 'Untitled')} ({timestamp})"
            )
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


def _get_child_nodes(
    spec_data: Dict[str, Any], node: Dict[str, Any]
) -> List[Dict[str, Any]]:
    hierarchy_nodes = _get_hierarchy_nodes(spec_data)
    children = node.get("children", [])
    return [
        hierarchy_nodes[child_id]
        for child_id in children
        if child_id in hierarchy_nodes
    ]
