"""
Foundry-MCP adapter for parity testing.

Calls foundry-mcp core functions directly for comparison with sdd-toolkit CLI.
"""

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
)
from foundry_mcp.core.lifecycle import (
    move_spec as core_move_spec,
    activate_spec as core_activate,
    complete_spec as core_complete,
    archive_spec as core_archive,
)


class FoundryMcpAdapter(SpecToolAdapter):
    """
    Adapter for foundry-mcp.

    Calls core Python functions directly rather than going through MCP protocol.
    """

    def __init__(self, specs_dir: Path):
        """
        Initialize adapter.

        Args:
            specs_dir: Path to the specs directory
        """
        super().__init__(specs_dir)
        # Resolve symlinks (for macOS temp dirs)
        self.specs_dir = Path(specs_dir).resolve()

    def _find_spec_status(self, spec_id: str) -> Optional[str]:
        """Find which status folder contains a spec."""
        for status in ["active", "pending", "completed", "archived"]:
            spec_path = self.specs_dir / status / f"{spec_id}.json"
            if spec_path.exists():
                return status
        return None

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
            spec_data = load_spec(spec_id, self.specs_dir)
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
            spec_data = load_spec(spec_id, self.specs_dir)
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
            spec_data = load_spec(spec_id, self.specs_dir)
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
            spec_data = load_spec(spec_id, self.specs_dir)
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
            spec_data = load_spec(spec_id, self.specs_dir)
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
            spec_data = load_spec(spec_id, self.specs_dir)
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
            spec_data = load_spec(spec_id, self.specs_dir)
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

            return {
                "success": True,
                "entry": entry,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_journal(
        self, spec_id: str, entry_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get journal entries."""
        try:
            spec_data = load_spec(spec_id, self.specs_dir)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            entries = get_journal_entries(spec_data, entry_type=entry_type)
            return {
                "success": True,
                "entries": entries,
                "count": len(entries),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def mark_blocked(
        self, spec_id: str, task_id: str, reason: str
    ) -> Dict[str, Any]:
        """Mark a task as blocked."""
        try:
            spec_data = load_spec(spec_id, self.specs_dir)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            result = mark_task_blocked(spec_data, task_id, reason)

            # Save the spec
            spec_status = self._find_spec_status(spec_id)
            if spec_status:
                save_spec(spec_id, spec_data, self.specs_dir)

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
            spec_data = load_spec(spec_id, self.specs_dir)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            result = unblock_task(spec_data, task_id)

            # Save the spec
            spec_status = self._find_spec_status(spec_id)
            if spec_status:
                save_spec(spec_id, spec_data, self.specs_dir)

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
            spec_data = load_spec(spec_id, self.specs_dir)
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
            spec_data = load_spec(spec_id, self.specs_dir)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            result = core_validate(spec_data)
            # ValidationResult is a dataclass, access attributes directly
            errors = [d for d in result.diagnostics if d.severity == "error"]
            warnings = [d for d in result.diagnostics if d.severity == "warning"]
            return {
                "success": True,
                "is_valid": result.is_valid,
                "errors": [{"message": e.message, "path": e.path} for e in errors],
                "warnings": [{"message": w.message, "path": w.path} for w in warnings],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fix_spec(self, spec_id: str) -> Dict[str, Any]:
        """Auto-fix spec issues."""
        try:
            spec_data = load_spec(spec_id, self.specs_dir)
            if spec_data is None:
                return {"success": False, "error": f"Spec not found: {spec_id}"}

            result = apply_fixes(spec_data)

            # Save the fixed spec
            spec_status = self._find_spec_status(spec_id)
            if spec_status:
                save_spec(spec_data, self.specs_dir / spec_status / f"{spec_id}.json")

            return {
                "success": True,
                "fixes_applied": result.get("fixes_applied", []),
                "remaining_issues": result.get("remaining_issues", []),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def spec_stats(self, spec_id: str) -> Dict[str, Any]:
        """Get specification statistics."""
        try:
            spec_data = load_spec(spec_id, self.specs_dir)
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
