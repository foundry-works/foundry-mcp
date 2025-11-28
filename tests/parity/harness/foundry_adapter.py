"""
Foundry-MCP adapter for parity testing.

Calls foundry-mcp core functions directly for comparison with sdd-toolkit CLI.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .base import SpecToolAdapter

# Import foundry-mcp core modules
from foundry_mcp.core.spec import (
    find_specs_directory,
    find_spec_file,
    load_spec,
    save_spec,
    list_specs as core_list_specs,
    get_node,
    update_node,
)
from foundry_mcp.core.task import (
    get_next_task,
    check_dependencies as core_check_deps,
    prepare_task as core_prepare_task,
)
from foundry_mcp.core.progress import (
    get_progress_summary,
    update_parent_status,
)
from foundry_mcp.core.journal import (
    add_journal_entry,
    get_journal_entries,
    mark_blocked as mark_task_blocked,
    unblock as unblock_task,
    list_blocked_tasks as get_blocked_tasks,
)
from foundry_mcp.core.validation import (
    validate_spec as core_validate,
    apply_fixes,
    calculate_stats,
    get_fix_actions,
)
from foundry_mcp.core.lifecycle import (
    move_spec as core_move_spec,
    activate_spec as core_activate,
    complete_spec as core_complete,
    archive_spec as core_archive,
)
from foundry_mcp.core.rendering import (
    RenderOptions,
    render_spec_to_markdown,
    render_progress_bar,
)


class FoundryMcpAdapter(SpecToolAdapter):
    """
    Adapter for foundry-mcp.

    Calls core Python functions directly rather than going through MCP protocol.
    Includes spec caching for performance optimization in tests.
    """

    def __init__(self, specs_dir: Path):
        """
        Initialize adapter.

        Args:
            specs_dir: Path to the specs directory
        """
        super().__init__(specs_dir)
        # Resolve symlinks (for macOS temp dirs) - cached at init
        self.specs_dir = Path(specs_dir).resolve()
        # Spec cache for read-heavy test operations
        self._spec_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def _load_spec_cached(self, spec_id: str) -> Optional[Dict[str, Any]]:
        """Load spec with caching for read operations."""
        if spec_id not in self._spec_cache:
            self._spec_cache[spec_id] = load_spec(spec_id, self.specs_dir)
        return self._spec_cache[spec_id]

    def invalidate_cache(self, spec_id: Optional[str] = None) -> None:
        """Invalidate cache after mutations."""
        if spec_id:
            self._spec_cache.pop(spec_id, None)
        else:
            self._spec_cache.clear()

    def _find_spec_status(self, spec_id: str) -> Optional[str]:
        """Find which status folder contains a spec."""
        for status in ["active", "pending", "completed", "archived"]:
            spec_path = self.specs_dir / status / f"{spec_id}.json"
            if spec_path.exists():
                return status
        return None

    def _diagnostic_to_dict(self, diagnostic: Any) -> Dict[str, Any]:
        """Convert Validation Diagnostic dataclasses to plain dicts."""
        if hasattr(diagnostic, "__dataclass_fields__"):
            return asdict(diagnostic)
        return diagnostic

    def _journal_entry_to_dict(self, entry: Any) -> Any:
        """Convert JournalEntry dataclasses to dicts for serialization."""
        if hasattr(entry, "__dataclass_fields__"):
            return asdict(entry)
        return entry

    def _fix_action_summary(self, action: Any) -> Dict[str, Any]:
        """Return a simplified view of FixAction dataclasses."""
        if hasattr(action, "__dataclass_fields__"):
            data = asdict(action)
            return {
                "id": data.get("id"),
                "description": data.get("description"),
                "category": data.get("category"),
            }
        return action

    # =========================================================================
    # Spec Operations
    # =========================================================================

    def list_specs(self, status: str = "all") -> Dict[str, Any]:
        """List specifications by status."""
        try:
            specs = core_list_specs(self.specs_dir, status)
            return {
                "success": True,
                "specs": specs,
                "count": len(specs),
                "status_filter": status,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_spec(self, spec_id: str) -> Dict[str, Any]:
        """Get specification details."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}
            return {
                "success": True,
                "spec": spec_data,
                "spec_id": spec_id,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Get task details."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            task = get_node(spec_data, task_id)
            if task is None:
                return {"success": False, "error": f"Task not found: {task_id}"}

            return {
                "success": True,
                "task": task,
                "task_id": task_id,
                "spec_id": spec_id,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Task Operations
    # =========================================================================

    def next_task(self, spec_id: str) -> Dict[str, Any]:
        """Find the next actionable task."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            result = get_next_task(spec_data)
            if result is None:
                return {
                    "success": True,
                    "task_id": None,
                    "message": "No actionable tasks found",
                }

            # get_next_task returns (task_id, task_data) tuple
            task_id, task_data = result
            return {
                "success": True,
                "task_id": task_id,
                "task": task_data,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def prepare_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Prepare task context and dependencies."""
        try:
            result = core_prepare_task(spec_id, self.specs_dir, task_id)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_dependencies(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Check if task dependencies are satisfied."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            result = core_check_deps(spec_data, task_id)
            return {
                "success": True,
                "can_start": result.get("can_start", True),
                "unmet_hard": result.get("unmet_hard", []),
                "unmet_soft": result.get("unmet_soft", []),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_status(
        self, spec_id: str, task_id: str, status: str
    ) -> Dict[str, Any]:
        """Update task status."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            # Update the task
            task = get_node(spec_data, task_id)
            if task is None:
                return {"success": False, "error": f"Task not found: {task_id}"}

            old_status = task.get("status")
            update_node(spec_data, task_id, {"status": status})

            # Update parent phase status if needed
            update_parent_status(spec_data, task_id)

            # Find the spec file and save
            spec_status = self._find_spec_status(spec_id)
            if spec_status:
                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            return {
                "success": True,
                "task_id": task_id,
                "old_status": old_status,
                "new_status": status,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def complete_task(
        self, spec_id: str, task_id: str, journal_entry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mark task as completed."""
        result = self.update_status(spec_id, task_id, "completed")

        if result.get("success") and journal_entry:
            self.add_journal(
                spec_id=spec_id,
                title=f"Completed {task_id}",
                content=journal_entry,
                entry_type="completion",
                task_id=task_id,
            )

        return result

    def start_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Mark task as in_progress."""
        return self.update_status(spec_id, task_id, "in_progress")

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def progress(self, spec_id: str) -> Dict[str, Any]:
        """Get progress summary for a spec."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            summary = get_progress_summary(spec_data)
            return {
                "success": True,
                "spec_id": spec_id,
                **summary,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_journal(
        self,
        spec_id: str,
        title: str,
        content: str,
        entry_type: str = "note",
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a journal entry."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            entry = add_journal_entry(
                spec_data,
                title=title,
                content=content,
                entry_type=entry_type,
                task_id=task_id,
            )

            # Save the spec
            spec_status = self._find_spec_status(spec_id)
            if spec_status:
                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            return {
                "success": True,
                "entry": self._journal_entry_to_dict(entry),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_journal(
        self, spec_id: str, entry_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get journal entries."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            entries = get_journal_entries(spec_data, entry_type=entry_type)
            entry_dicts = [self._journal_entry_to_dict(entry) for entry in entries]
            return {
                "success": True,
                "entries": entry_dicts,
                "count": len(entry_dicts),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def mark_blocked(
        self, spec_id: str, task_id: str, reason: str
    ) -> Dict[str, Any]:
        """Mark a task as blocked."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            result = mark_task_blocked(spec_data, task_id, reason)

            # Save the spec
            spec_status = self._find_spec_status(spec_id)
            if spec_status:
                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            return {
                "success": True,
                "task_id": task_id,
                "blocked": True,
                "reason": reason,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def unblock(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Remove blocked status from a task."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            result = unblock_task(spec_data, task_id)

            # Save the spec
            spec_status = self._find_spec_status(spec_id)
            if spec_status:
                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            return {
                "success": True,
                "task_id": task_id,
                "unblocked": True,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_blocked(self, spec_id: str) -> Dict[str, Any]:
        """List all blocked tasks."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            blocked = get_blocked_tasks(spec_data)
            return {
                "success": True,
                "blocked_tasks": blocked,
                "count": len(blocked),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_spec(self, spec_id: str) -> Dict[str, Any]:
        """Validate specification structure."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            result = core_validate(spec_data)
            errors = [
                self._diagnostic_to_dict(d)
                for d in result.diagnostics
                if d.severity == "error"
            ]
            warnings = [
                self._diagnostic_to_dict(d)
                for d in result.diagnostics
                if d.severity == "warning"
            ]
            return {
                "success": True,
                "is_valid": result.is_valid,
                "errors": errors,
                "warnings": warnings,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fix_spec(self, spec_id: str) -> Dict[str, Any]:
        """Auto-fix spec issues."""
        try:
            spec_path = find_spec_file(spec_id, self.specs_dir)
            if not spec_path:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            validation = core_validate(spec_data)
            actions = get_fix_actions(validation, spec_data)

            if not actions:
                return {
                    "success": True,
                    "spec_id": spec_id,
                    "applied_count": 0,
                    "skipped_count": 0,
                    "message": "No auto-fixable issues found",
                }

            report = apply_fixes(
                actions,
                str(spec_path),
                dry_run=False,
                create_backup=False,
            )

            return {
                "success": True,
                "spec_id": spec_id,
                "applied_count": len(report.applied_actions),
                "skipped_count": len(report.skipped_actions),
                "applied_actions": [
                    self._fix_action_summary(action)
                    for action in report.applied_actions
                ],
                "skipped_actions": [
                    self._fix_action_summary(action)
                    for action in report.skipped_actions
                ],
                "backup_path": report.backup_path,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def spec_stats(self, spec_id: str) -> Dict[str, Any]:
        """Get specification statistics."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            stats = calculate_stats(spec_data)
            # SpecStats is a dataclass, convert to dict
            return {
                "success": True,
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
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def activate_spec(self, spec_id: str) -> Dict[str, Any]:
        """Move spec from pending to active."""
        try:
            result = core_activate(spec_id, self.specs_dir)
            response = {
                "success": result.success,
                "spec_id": spec_id,
                "new_status": "active" if result.success else None,
                "from_folder": result.from_folder,
                "to_folder": result.to_folder,
                "old_path": result.old_path,
                "new_path": result.new_path,
            }
            if result.error:
                response["error"] = result.error
            return response
        except Exception as e:
            return {"success": False, "error": str(e)}

    def complete_spec(self, spec_id: str) -> Dict[str, Any]:
        """Move spec from active to completed."""
        try:
            result = core_complete(spec_id, self.specs_dir, force=True)
            response = {
                "success": result.success,
                "spec_id": spec_id,
                "new_status": "completed" if result.success else None,
                "from_folder": result.from_folder,
                "to_folder": result.to_folder,
                "old_path": result.old_path,
                "new_path": result.new_path,
            }
            if result.error:
                response["error"] = result.error
            return response
        except Exception as e:
            return {"success": False, "error": str(e)}

    def archive_spec(self, spec_id: str) -> Dict[str, Any]:
        """Move spec to archived."""
        try:
            result = core_archive(spec_id, self.specs_dir)
            response = {
                "success": result.success,
                "spec_id": spec_id,
                "new_status": "archived" if result.success else None,
                "from_folder": result.from_folder,
                "to_folder": result.to_folder,
                "old_path": result.old_path,
                "new_path": result.new_path,
            }
            if result.error:
                response["error"] = result.error
            return response
        except Exception as e:
            return {"success": False, "error": str(e)}

    def move_spec(self, spec_id: str, target_status: str) -> Dict[str, Any]:
        """Move spec to a specific status folder."""
        try:
            result = core_move_spec(spec_id, target_status, self.specs_dir)
            response = {
                "success": result.success,
                "spec_id": spec_id,
                "new_status": target_status if result.success else None,
                "from_folder": result.from_folder,
                "to_folder": result.to_folder,
                "old_path": result.old_path,
                "new_path": result.new_path,
            }
            if result.error:
                response["error"] = result.error
            return response
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Authoring Operations
    # =========================================================================

    def add_task(
        self,
        spec_id: str,
        parent: str,
        title: str,
        description: Optional[str] = None,
        task_type: str = "task",
        hours: Optional[float] = None,
        position: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Add a new task to a specification."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})
            parent_node = hierarchy.get(parent)
            if parent_node is None:
                return {"success": False, "error": f"Parent not found: {parent}"}

            # Generate task ID based on parent
            children = parent_node.get("children", [])
            if parent.startswith("phase-"):
                phase_num = parent.split("-")[1]
                task_num = len(children) + 1
                task_id = f"task-{phase_num}-{task_num}"
            elif parent.startswith("task-"):
                task_id = f"subtask-{parent.split('-', 1)[1]}-{len(children) + 1}"
            else:
                task_id = f"task-{len(children) + 1}"

            # Create new task node
            new_task = {
                "type": task_type,
                "title": title,
                "status": "pending",
                "parent": parent,
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
            }

            if description:
                new_task["description"] = description
            if hours is not None:
                new_task["metadata"]["estimated_hours"] = hours

            if not dry_run:
                # Add to hierarchy
                hierarchy[task_id] = new_task

                # Update parent's children
                if position is not None and 0 <= position <= len(children):
                    children.insert(position, task_id)
                else:
                    children.append(task_id)
                parent_node["children"] = children
                parent_node["total_tasks"] = parent_node.get("total_tasks", 0) + 1

                # Save spec
                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            return {
                "success": True,
                "task_id": task_id,
                "parent": parent,
                "title": title,
                "type": task_type,
                "dry_run": dry_run,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def remove_task(
        self,
        spec_id: str,
        task_id: str,
        cascade: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Remove a task from a specification."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})
            task = hierarchy.get(task_id)
            if task is None:
                return {"success": False, "error": f"Task not found: {task_id}"}

            children = task.get("children", [])
            if children and not cascade:
                return {
                    "success": False,
                    "error": f"Task '{task_id}' has children. Use cascade=True to remove recursively",
                }

            def count_descendants(node_id: str) -> int:
                node = hierarchy.get(node_id, {})
                count = 0
                for child_id in node.get("children", []):
                    count += 1 + count_descendants(child_id)
                return count

            children_removed = count_descendants(task_id) if cascade else 0

            if not dry_run:
                # Remove children recursively if cascade
                def remove_recursively(node_id: str):
                    node = hierarchy.get(node_id, {})
                    for child_id in node.get("children", []):
                        remove_recursively(child_id)
                    if node_id in hierarchy:
                        del hierarchy[node_id]

                if cascade:
                    remove_recursively(task_id)
                else:
                    del hierarchy[task_id]

                # Update parent's children list
                parent_id = task.get("parent")
                if parent_id and parent_id in hierarchy:
                    parent = hierarchy[parent_id]
                    if task_id in parent.get("children", []):
                        parent["children"].remove(task_id)
                    parent["total_tasks"] = max(0, parent.get("total_tasks", 1) - 1 - children_removed)

                # Save spec
                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            return {
                "success": True,
                "task_id": task_id,
                "spec_id": spec_id,
                "cascade": cascade,
                "children_removed": children_removed,
                "dry_run": dry_run,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_assumption(
        self,
        spec_id: str,
        text: str,
        assumption_type: str = "constraint",
        author: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Add an assumption to a specification."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            assumptions = spec_data.get("assumptions", [])
            assumption_id = f"assumption-{len(assumptions) + 1}"

            from datetime import datetime
            new_assumption = {
                "id": assumption_id,
                "type": assumption_type,
                "text": text,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            if author:
                new_assumption["author"] = author

            if not dry_run:
                assumptions.append(new_assumption)
                spec_data["assumptions"] = assumptions
                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            return {
                "success": True,
                "spec_id": spec_id,
                "assumption_id": assumption_id,
                "text": text,
                "type": assumption_type,
                "dry_run": dry_run,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_assumptions(
        self,
        spec_id: str,
        assumption_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List assumptions from a specification."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            assumptions = spec_data.get("assumptions", [])

            if assumption_type:
                assumptions = [a for a in assumptions if a.get("type") == assumption_type]

            return {
                "success": True,
                "spec_id": spec_id,
                "assumptions": assumptions,
                "total_count": len(assumptions),
                "filter_type": assumption_type,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_revision(
        self,
        spec_id: str,
        version: str,
        changes: str,
        author: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Add a revision history entry."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            from datetime import datetime
            revision_history = spec_data.get("revision_history", [])
            new_revision = {
                "version": version,
                "date": datetime.utcnow().isoformat() + "Z",
                "changes": changes,
            }
            if author:
                new_revision["author"] = author

            if not dry_run:
                revision_history.append(new_revision)
                spec_data["revision_history"] = revision_history
                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            return {
                "success": True,
                "spec_id": spec_id,
                "version": version,
                "changes": changes,
                "dry_run": dry_run,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_estimate(
        self,
        spec_id: str,
        task_id: str,
        hours: Optional[float] = None,
        complexity: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Update task estimate."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            task = get_node(spec_data, task_id)
            if task is None:
                return {"success": False, "error": f"Task not found: {task_id}"}

            metadata = task.get("metadata", {})
            previous_hours = metadata.get("estimated_hours")
            previous_complexity = metadata.get("complexity")

            if not dry_run:
                if hours is not None:
                    metadata["estimated_hours"] = hours
                if complexity:
                    metadata["complexity"] = complexity
                task["metadata"] = metadata
                update_node(spec_data, task_id, task)
                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            result = {
                "success": True,
                "spec_id": spec_id,
                "task_id": task_id,
                "dry_run": dry_run,
            }
            if hours is not None:
                result["hours"] = hours
            if complexity:
                result["complexity"] = complexity
            if previous_hours is not None:
                result["previous_hours"] = previous_hours
            if previous_complexity is not None:
                result["previous_complexity"] = previous_complexity

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_task_metadata(
        self,
        spec_id: str,
        task_id: str,
        file_path: Optional[str] = None,
        description: Optional[str] = None,
        actual_hours: Optional[float] = None,
        status_note: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Update task metadata fields."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            task = get_node(spec_data, task_id)
            if task is None:
                return {"success": False, "error": f"Task not found: {task_id}"}

            fields_updated = []
            metadata = task.get("metadata", {})

            if not dry_run:
                if file_path:
                    metadata["file_path"] = file_path
                    fields_updated.append("file_path")
                if description:
                    task["description"] = description
                    fields_updated.append("description")
                if actual_hours is not None:
                    metadata["actual_hours"] = actual_hours
                    fields_updated.append("actual_hours")
                if status_note:
                    metadata["status_note"] = status_note
                    fields_updated.append("status_note")

                task["metadata"] = metadata
                update_node(spec_data, task_id, task)
                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            return {
                "success": True,
                "spec_id": spec_id,
                "task_id": task_id,
                "fields_updated": fields_updated,
                "dry_run": dry_run,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def task_list(
        self,
        spec_id: str,
        status_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List all tasks in a specification."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})
            tasks = []

            for node_id, node in hierarchy.items():
                if node.get("type") in ("task", "subtask", "verify"):
                    if status_filter is None or node.get("status") == status_filter:
                        tasks.append({
                            "task_id": node_id,
                            "title": node.get("title"),
                            "status": node.get("status"),
                            "type": node.get("type"),
                            "parent": node.get("parent"),
                        })

            return {
                "success": True,
                "spec_id": spec_id,
                "tasks": tasks,
                "count": len(tasks),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Analysis Operations
    # =========================================================================

    def analyze_deps(
        self,
        spec_id: str,
        bottleneck_threshold: int = 3,
    ) -> Dict[str, Any]:
        """Analyze dependency graph for bottlenecks and critical path."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})

            # Build dependency graph
            dependents: Dict[str, list] = {}  # task_id -> list of tasks depending on it
            all_deps: list = []

            for node_id, node in hierarchy.items():
                deps = node.get("dependencies", {})
                hard_deps = deps.get("hard", []) if isinstance(deps, dict) else []

                for dep_id in hard_deps:
                    all_deps.append({"from": node_id, "to": dep_id})
                    if dep_id not in dependents:
                        dependents[dep_id] = []
                    dependents[dep_id].append(node_id)

            # Find bottlenecks (tasks blocking many others)
            bottlenecks = []
            for task_id, blocked_tasks in dependents.items():
                if len(blocked_tasks) >= bottleneck_threshold:
                    bottlenecks.append({
                        "task_id": task_id,
                        "blocking_count": len(blocked_tasks),
                        "blocked_tasks": blocked_tasks,
                    })

            # Sort bottlenecks by impact
            bottlenecks.sort(key=lambda x: x["blocking_count"], reverse=True)

            # Find critical path (longest dependency chain)
            def get_chain_length(task_id: str, visited: set) -> int:
                if task_id in visited:
                    return 0
                visited.add(task_id)
                node = hierarchy.get(task_id, {})
                deps = node.get("dependencies", {})
                hard_deps = deps.get("hard", []) if isinstance(deps, dict) else []
                if not hard_deps:
                    return 1
                return 1 + max(get_chain_length(d, visited.copy()) for d in hard_deps)

            critical_path = []
            max_depth = 0
            for node_id in hierarchy:
                depth = get_chain_length(node_id, set())
                if depth > max_depth:
                    max_depth = depth

            return {
                "success": True,
                "spec_id": spec_id,
                "dependency_count": len(all_deps),
                "bottlenecks": bottlenecks,
                "critical_path_length": max_depth,
                "has_bottlenecks": len(bottlenecks) > 0,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def detect_cycles(
        self,
        spec_id: str,
    ) -> Dict[str, Any]:
        """Detect circular dependencies in task graph."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})

            # Build adjacency list from dependencies
            graph: Dict[str, list] = {}
            for node_id, node in hierarchy.items():
                deps = node.get("dependencies", {})
                hard_deps = deps.get("hard", []) if isinstance(deps, dict) else []
                graph[node_id] = hard_deps

            # Find cycles using DFS
            cycles = []
            visited = set()
            rec_stack = set()
            path = []

            def dfs(node: str) -> bool:
                visited.add(node)
                rec_stack.add(node)
                path.append(node)

                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        if dfs(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        cycles.append(cycle)
                        return True

                path.pop()
                rec_stack.remove(node)
                return False

            for node in graph:
                if node not in visited:
                    dfs(node)

            # Get affected tasks
            affected_tasks = set()
            for cycle in cycles:
                affected_tasks.update(cycle)

            return {
                "success": True,
                "spec_id": spec_id,
                "has_cycles": len(cycles) > 0,
                "cycles": cycles,
                "cycle_count": len(cycles),
                "affected_tasks": list(affected_tasks),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def find_patterns(
        self,
        spec_id: str,
        pattern: str,
    ) -> Dict[str, Any]:
        """Find tasks with file_path matching pattern."""
        try:
            import fnmatch

            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})
            matches = []

            for node_id, node in hierarchy.items():
                metadata = node.get("metadata", {})
                file_path = metadata.get("file_path", "")
                if file_path and fnmatch.fnmatch(file_path, pattern):
                    matches.append({
                        "task_id": node_id,
                        "file_path": file_path,
                        "title": node.get("title"),
                    })

            return {
                "success": True,
                "spec_id": spec_id,
                "pattern": pattern,
                "matches": matches,
                "total_count": len(matches),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Verification Operations
    # =========================================================================

    def add_verification(
        self,
        spec_id: str,
        verify_id: str,
        result: str,
        command: Optional[str] = None,
        output: Optional[str] = None,
        issues: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add verification result to verify node."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})
            if verify_id not in hierarchy:
                return {"success": False, "error": f"Verify node not found: {verify_id}"}

            node = hierarchy[verify_id]
            if node.get("type") != "verify":
                return {"success": False, "error": f"Node {verify_id} is not a verify node"}

            # Validate result
            valid_results = ("PASSED", "FAILED", "PARTIAL")
            if result not in valid_results:
                return {
                    "success": False,
                    "error": f"Invalid result '{result}'. Must be one of: {', '.join(valid_results)}",
                }

            # Update node metadata
            metadata = node.get("metadata", {})
            metadata["result"] = result
            metadata["executed_at"] = "2025-01-02T12:00:00Z"  # Static for tests
            if command:
                metadata["command"] = command
            if output:
                metadata["output"] = output
            if issues:
                metadata["issues"] = issues.split(",") if "," in issues else [issues]
            if notes:
                metadata["notes"] = notes
            node["metadata"] = metadata

            # Update status based on result
            if result == "PASSED":
                node["status"] = "completed"
            elif result == "FAILED":
                node["status"] = "blocked"
            else:  # PARTIAL
                node["status"] = "in_progress"

            # Save spec
            save_spec(spec_id, spec_data, self.specs_dir)
            self.invalidate_cache(spec_id)

            return {
                "success": True,
                "spec_id": spec_id,
                "verify_id": verify_id,
                "result": result,
                "command": command,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_verification_results(
        self,
        spec_id: str,
    ) -> Dict[str, Any]:
        """Get all verification results from spec."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})
            results = []

            for node_id, node in hierarchy.items():
                if node.get("type") == "verify":
                    metadata = node.get("metadata", {})
                    if "result" in metadata:
                        results.append({
                            "verify_id": node_id,
                            "title": node.get("title"),
                            "result": metadata.get("result"),
                            "command": metadata.get("command"),
                            "output": metadata.get("output"),
                            "issues": metadata.get("issues", []),
                            "executed_at": metadata.get("executed_at"),
                        })

            return {
                "success": True,
                "spec_id": spec_id,
                "results": results,
                "count": len(results),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def format_verification_summary(
        self,
        spec_id: str,
    ) -> Dict[str, Any]:
        """Format summary of verification results."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})
            passed = 0
            failed = 0
            partial = 0
            total = 0

            for node_id, node in hierarchy.items():
                if node.get("type") == "verify":
                    metadata = node.get("metadata", {})
                    result = metadata.get("result")
                    if result:
                        total += 1
                        if result == "PASSED":
                            passed += 1
                        elif result == "FAILED":
                            failed += 1
                        elif result == "PARTIAL":
                            partial += 1

            summary_lines = [
                f"Verification Summary for {spec_id}:",
                f"  Total: {total}",
                f"  Passed: {passed}",
                f"  Failed: {failed}",
                f"  Partial: {partial}",
            ]

            return {
                "success": True,
                "spec_id": spec_id,
                "summary": "\n".join(summary_lines),
                "total_verifications": total,
                "passed": passed,
                "failed": failed,
                "partial": partial,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Rendering & Reporting Operations
    # =========================================================================

    def render_spec(
        self,
        spec_id: str,
        mode: str = "basic",
        include_journal: bool = False,
        max_depth: int = 0,
    ) -> Dict[str, Any]:
        """Render spec to markdown."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            options = RenderOptions(
                mode=mode,
                include_metadata=True,
                include_progress=True,
                include_dependencies=True,
                include_journal=include_journal,
                max_depth=max_depth,
            )

            result = render_spec_to_markdown(spec_data, options)

            return {
                "success": True,
                "spec_id": result.spec_id,
                "title": result.title,
                "markdown": result.markdown,
                "total_sections": result.total_sections,
                "total_tasks": result.total_tasks,
                "completed_tasks": result.completed_tasks,
                "progress_percentage": (
                    result.completed_tasks / result.total_tasks * 100
                    if result.total_tasks > 0
                    else 0
                ),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def render_progress(
        self,
        spec_id: str,
        bar_width: int = 20,
    ) -> Dict[str, Any]:
        """Get visual progress summary for a spec."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})
            root = hierarchy.get("spec-root", {})
            metadata = spec_data.get("metadata", {})

            total_tasks = root.get("total_tasks", 0)
            completed_tasks = root.get("completed_tasks", 0)

            # Overall progress
            overall_bar = render_progress_bar(completed_tasks, total_tasks, bar_width)

            # Phase progress
            phases = []
            for phase_id in root.get("children", []):
                phase = hierarchy.get(phase_id, {})
                phase_total = phase.get("total_tasks", 0)
                phase_completed = phase.get("completed_tasks", 0)
                phase_status = phase.get("status", "pending")
                phase_bar = render_progress_bar(phase_completed, phase_total, bar_width)

                phases.append({
                    "id": phase_id,
                    "title": phase.get("title", "Untitled"),
                    "status": phase_status,
                    "progress_bar": phase_bar,
                    "completed": phase_completed,
                    "total": phase_total,
                })

            return {
                "success": True,
                "spec_id": spec_id,
                "title": metadata.get("title") or root.get("title", "Untitled"),
                "overall": {
                    "status": root.get("status", "pending"),
                    "progress_bar": overall_bar,
                    "completed": completed_tasks,
                    "total": total_tasks,
                },
                "phases": phases,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def spec_report(
        self,
        spec_id: str,
        format: str = "markdown",
        sections: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive report for a spec."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            spec_path = find_spec_file(spec_id, self.specs_dir)

            # Parse sections
            requested_sections = set()
            if sections is None or sections.lower() == "all":
                requested_sections = {"validation", "stats", "health"}
            else:
                for s in sections.lower().split(","):
                    s = s.strip()
                    if s in ("validation", "stats", "health"):
                        requested_sections.add(s)

            # Validation
            validation_result = {}
            if "validation" in requested_sections or "health" in requested_sections:
                result = core_validate(spec_data)
                validation_result = {
                    "is_valid": result.is_valid,
                    "error_count": result.error_count,
                    "warning_count": result.warning_count,
                    "info_count": result.info_count,
                }

            # Stats
            stats_result = {}
            if "stats" in requested_sections or "health" in requested_sections:
                stats = calculate_stats(spec_data, str(spec_path) if spec_path else None)
                # Get completed/total from spec-root node
                root = spec_data.get("hierarchy", {}).get("spec-root", {})
                total_tasks = root.get("total_tasks", 0)
                completed_tasks = root.get("completed_tasks", 0)
                stats_result = {
                    "title": stats.title,
                    "status": stats.status,
                    "progress": stats.progress,  # This is a float (0-1)
                    "progress_pct": int(stats.progress * 100),
                    "completed": completed_tasks,
                    "total": total_tasks,
                    "totals": stats.totals,
                }

            # Health score
            health_score = 100
            if validation_result.get("is_valid") is False:
                health_score -= min(30, validation_result.get("error_count", 0) * 10)
            warning_count = validation_result.get("warning_count", 0)
            if warning_count > 5:
                health_score -= min(20, warning_count * 2)
            health_score = max(0, health_score)

            # Build report
            report_lines = [f"# Report: {spec_id}", ""]
            if "validation" in requested_sections:
                report_lines.extend([
                    "## Validation",
                    f"Valid: {validation_result.get('is_valid', True)}",
                    f"Errors: {validation_result.get('error_count', 0)}",
                    f"Warnings: {validation_result.get('warning_count', 0)}",
                    "",
                ])
            if "stats" in requested_sections:
                report_lines.extend([
                    "## Statistics",
                    f"Status: {stats_result.get('status', 'unknown')}",
                    f"Progress: {stats_result.get('completed', 0)}/{stats_result.get('total', 0)} ({stats_result.get('progress_pct', 0)}%)",
                    "",
                ])
            if "health" in requested_sections:
                report_lines.extend([
                    "## Health",
                    f"Score: {health_score}",
                    f"Status: {'healthy' if health_score >= 80 else 'needs_attention' if health_score >= 50 else 'critical'}",
                    "",
                ])

            return {
                "success": True,
                "spec_id": spec_id,
                "report": "\n".join(report_lines),
                "format": format,
                "sections": list(requested_sections),
                "summary": {
                    "validation": validation_result,
                    "stats": stats_result,
                    "health": {"score": health_score},
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def spec_report_summary(
        self,
        spec_id: str,
    ) -> Dict[str, Any]:
        """Generate quick summary report for a spec."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            spec_path = find_spec_file(spec_id, self.specs_dir)

            # Validate
            result = core_validate(spec_data)

            # Stats
            stats = calculate_stats(spec_data, str(spec_path) if spec_path else None)

            # Get completed/total from spec-root node
            root = spec_data.get("hierarchy", {}).get("spec-root", {})
            total_tasks = root.get("total_tasks", 0)
            completed_tasks = root.get("completed_tasks", 0)

            # Health score
            health_score = 100
            if not result.is_valid:
                health_score -= min(30, result.error_count * 10)
            if result.warning_count > 5:
                health_score -= min(20, result.warning_count * 2)
            health_score = max(0, health_score)

            return {
                "success": True,
                "spec_id": spec_id,
                "title": stats.title,
                "status": stats.status,
                "validation": {
                    "is_valid": result.is_valid,
                    "errors": result.error_count,
                    "warnings": result.warning_count,
                },
                "progress": {
                    "completed": completed_tasks,
                    "total": total_tasks,
                    "percentage": int(stats.progress * 100),
                },
                "health": {
                    "score": health_score,
                    "status": "healthy" if health_score >= 80 else (
                        "needs_attention" if health_score >= 50 else "critical"
                    ),
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Authoring Extensions
    # =========================================================================

    def update_frontmatter(
        self,
        spec_id: str,
        key: str,
        value: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Update spec frontmatter/metadata."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            metadata = spec_data.get("metadata", {})
            previous_value = metadata.get(key)

            if not dry_run:
                metadata[key] = value
                spec_data["metadata"] = metadata

                # Also update top-level fields if they exist
                if key in spec_data:
                    spec_data[key] = value

                save_spec(spec_id, spec_data, self.specs_dir)
                self.invalidate_cache(spec_id)

            return {
                "success": True,
                "spec_id": spec_id,
                "key": key,
                "value": value,
                "previous_value": previous_value,
                "dry_run": dry_run,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # File Operations
    # =========================================================================

    def find_related_files(
        self,
        file_path: str,
        spec_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find files related to a source file."""
        try:
            related_files = []
            spec_references = []

            # Search specs for file references
            if spec_id:
                spec_ids = [spec_id]
            else:
                # List all specs
                specs = core_list_specs(self.specs_dir)
                spec_ids = [s.get("spec_id") for s in specs]

            for sid in spec_ids:
                spec_data = load_spec(sid, self.specs_dir)
                if spec_data is None:
                    continue

                hierarchy = spec_data.get("hierarchy", {})
                for node_id, node in hierarchy.items():
                    node_file = node.get("metadata", {}).get("file_path", "")
                    if node_file == file_path:
                        spec_references.append({
                            "spec_id": sid,
                            "node_id": node_id,
                            "title": node.get("title"),
                        })
                    # Check for related files (same directory, tests, etc.)
                    elif node_file and self._files_related(file_path, node_file):
                        related_files.append({
                            "file_path": node_file,
                            "relationship": "same_directory" if self._same_dir(file_path, node_file) else "related",
                            "spec_id": sid,
                            "node_id": node_id,
                        })

            return {
                "success": True,
                "file_path": file_path,
                "related_files": related_files,
                "spec_references": spec_references,
                "total_count": len(related_files),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _files_related(self, file1: str, file2: str) -> bool:
        """Check if two files are related."""
        from pathlib import Path
        p1, p2 = Path(file1), Path(file2)
        # Same directory
        if p1.parent == p2.parent:
            return True
        # Test file relationship
        if "test" in str(p1) or "test" in str(p2):
            base1 = p1.stem.replace("test_", "").replace("_test", "")
            base2 = p2.stem.replace("test_", "").replace("_test", "")
            if base1 == base2 or base1 in base2 or base2 in base1:
                return True
        return False

    def _same_dir(self, file1: str, file2: str) -> bool:
        """Check if files are in the same directory."""
        from pathlib import Path
        return Path(file1).parent == Path(file2).parent

    def validate_paths(
        self,
        paths: list,
        base_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate that file paths exist on disk."""
        try:
            from pathlib import Path
            base = Path(base_directory) if base_directory else Path.cwd()

            valid_paths = []
            invalid_paths = []

            for p in paths:
                full_path = base / p if not Path(p).is_absolute() else Path(p)
                if full_path.exists():
                    valid_paths.append(p)
                else:
                    invalid_paths.append(p)

            return {
                "success": True,
                "paths_checked": len(paths),
                "valid_paths": valid_paths,
                "invalid_paths": invalid_paths,
                "all_valid": len(invalid_paths) == 0,
                "valid_count": len(valid_paths),
                "invalid_count": len(invalid_paths),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Pagination Operations
    # =========================================================================

    def task_query(
        self,
        spec_id: str,
        status: Optional[str] = None,
        parent: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query tasks with pagination."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            hierarchy = spec_data.get("hierarchy", {})
            tasks = []

            for node_id, node in hierarchy.items():
                if node.get("type") not in ("task", "subtask", "verify"):
                    continue
                if status and node.get("status") != status:
                    continue
                if parent and node.get("parent") != parent:
                    continue
                tasks.append({
                    "task_id": node_id,
                    "title": node.get("title"),
                    "status": node.get("status"),
                    "type": node.get("type"),
                    "parent": node.get("parent"),
                })

            # Sort by task_id for consistent pagination
            tasks.sort(key=lambda t: t["task_id"])

            # Apply cursor (offset-based for simplicity)
            start_idx = 0
            if cursor:
                try:
                    start_idx = int(cursor)
                except ValueError:
                    start_idx = 0

            end_idx = start_idx + limit
            paginated_tasks = tasks[start_idx:end_idx]

            # Generate next cursor
            next_cursor = None
            if end_idx < len(tasks):
                next_cursor = str(end_idx)

            return {
                "success": True,
                "spec_id": spec_id,
                "tasks": paginated_tasks,
                "count": len(paginated_tasks),
                "total": len(tasks),
                "cursor": next_cursor,
                "has_more": next_cursor is not None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def journal_list_paginated(
        self,
        spec_id: str,
        entry_type: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List journal entries with pagination."""
        try:
            spec_data = self._load_spec_cached(spec_id)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            entries = get_journal_entries(spec_data, task_id=task_id, entry_type=entry_type)
            entry_dicts = [self._journal_entry_to_dict(e) for e in entries]

            # Apply cursor
            start_idx = 0
            if cursor:
                try:
                    start_idx = int(cursor)
                except ValueError:
                    start_idx = 0

            end_idx = start_idx + limit
            paginated = entry_dicts[start_idx:end_idx]

            next_cursor = None
            if end_idx < len(entry_dicts):
                next_cursor = str(end_idx)

            return {
                "success": True,
                "spec_id": spec_id,
                "entries": paginated,
                "count": len(paginated),
                "total": len(entry_dicts),
                "cursor": next_cursor,
                "has_more": next_cursor is not None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
