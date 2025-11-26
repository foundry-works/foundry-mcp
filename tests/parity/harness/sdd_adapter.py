"""
SDD-Toolkit CLI adapter for parity testing.

Invokes sdd CLI commands via subprocess and parses JSON output.
"""

import json
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import SpecToolAdapter


class SddToolkitAdapter(SpecToolAdapter):
    """
    Adapter for sdd-toolkit CLI.

    Invokes CLI commands via subprocess with --json output flag.
    """

    SPEC_ID_PATTERN = re.compile(r"^[\w-]+-\d{4}-\d{2}-\d{2}-\d{3}$")

    def __init__(self, specs_dir: Path):
        """
        Initialize adapter.

        Args:
            specs_dir: Path to the specs directory
        """
        super().__init__(specs_dir)
        self.specs_dir = Path(specs_dir)
        # Project root is parent of specs/
        self.project_root = self.specs_dir.parent

    def _wrap_list_result(

        self,
        result: Any,
        *,
        list_key: str,
        count_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize CLI list responses into success dictionaries."""
        if isinstance(result, dict):
            if result.get("success") is False or "error" in result:
                return result
            if list_key in result and isinstance(result[list_key], list):
                if count_key:
                    result.setdefault(count_key, len(result[list_key]))
                result.setdefault("success", True)
                return result
            data = result.get("result")
            if isinstance(data, list):
                payload = {
                    **{k: v for k, v in result.items() if k != "result"},
                    list_key: data,
                }
                payload.setdefault("success", True)
                if count_key:
                    payload[count_key] = len(data)
                return payload
            return result
        if isinstance(result, list):
            payload = {"success": True, list_key: result}
            if count_key:
                payload[count_key] = len(result)
            return payload
        return result

    def _entry_matches(self, entry: Any, entry_type: str) -> bool:
        """Return True if journal entry matches requested type."""
        if isinstance(entry, dict):
            candidate = entry.get("entry_type") or entry.get("type")
            return candidate == entry_type
        return False

    def _ensure_task_identifier(
        self,
        result: Any,
        *,
        fallback_task_id: Optional[str] = None,
    ) -> Any:
        """Ensure sdd responses include task_id metadata like foundry."""
        if not isinstance(result, dict):
            return result
        if result.get("success") is False or "error" in result:
            return result
        if result.get("task_id"):
            return result
        task_payload = result.get("task")
        candidate: Optional[str] = None
        if isinstance(task_payload, dict):
            candidate = (
                task_payload.get("task_id")
                or task_payload.get("id")
                or task_payload.get("node_id")
            )
        if candidate:
            result["task_id"] = candidate
        elif fallback_task_id:
            result["task_id"] = fallback_task_id
        return result

    def _find_spec_path(self, spec_id: str) -> Optional[Path]:
        """Locate the spec file across status folders."""
        for status in ["pending", "active", "completed", "archived"]:
            candidate = self.specs_dir / status / f"{spec_id}.json"
            if candidate.exists():
                return candidate
        return None

    def _canonical_spec_id(self, spec_id: Optional[str]) -> Optional[str]:
        """Generate a schema-compliant spec_id for CLI validation."""
        if not spec_id:
            return spec_id
        if self.SPEC_ID_PATTERN.match(spec_id):
            return spec_id
        return f"{spec_id}-2000-01-01-000"

    @contextmanager
    def _normalized_spec_id(self, spec_path: Path):
        """Temporarily normalize spec_id for CLI commands that require it."""
        normalized_id = None
        original_id = None
        try:
            with open(spec_path) as source:
                data = json.load(source)
            original_id = data.get("spec_id")
            normalized_id = self._canonical_spec_id(original_id)
            if normalized_id == original_id or normalized_id is None:
                yield
                return
            data["spec_id"] = normalized_id
            with open(spec_path, "w") as target:
                json.dump(data, target, indent=2)
            yield
        finally:
            if (
                original_id is not None
                and normalized_id is not None
                and normalized_id != original_id
            ):
                with open(spec_path) as source:
                    data = json.load(source)
                data["spec_id"] = original_id
                with open(spec_path, "w") as target:
                    json.dump(data, target, indent=2)

    def _run_sdd(
        self,
        *args: str,
        timeout: int = 30,
        check: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute sdd CLI command and return parsed JSON output.

        Args:
            *args: Command arguments (e.g., "list-specs", "--status", "active")
            timeout: Command timeout in seconds
            check: If True, raise on non-zero exit

        Returns:
            Parsed JSON response or error dict
        """
        cmd = ["sdd", "--json", "--path", str(self.project_root)]
        cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root),
            )

            # Try to parse JSON from stdout
            if result.stdout.strip():
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return raw output
                    pass

            # Check for error
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr or f"Command failed with exit code {result.returncode}",
                    "_exit_code": result.returncode,
                    "_stdout": result.stdout,
                }

            # No JSON output but command succeeded
            return {
                "success": True,
                "_stdout": result.stdout,
                "_stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout}s",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "sdd command not found. Is sdd-toolkit installed?",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # Spec Operations
    # =========================================================================

    def list_specs(self, status: str = "all") -> Dict[str, Any]:
        """List specifications by status."""
        args = ["list-specs"]
        if status != "all":
            args.extend(["--status", status])
        result = self._run_sdd(*args)
        result = self._wrap_list_result(result, list_key="specs", count_key="count")
        if isinstance(result, dict) and result.get("success", True):
            result.setdefault("status_filter", status)
        return result

    def get_spec(self, spec_id: str) -> Dict[str, Any]:
        """Get specification details."""
        # sdd-toolkit uses 'progress' command for spec overview
        return self._run_sdd("progress", spec_id)

    def get_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Get task details."""
        result = self._run_sdd("task-info", spec_id, task_id)
        return self._ensure_task_identifier(result, fallback_task_id=task_id)

    # =========================================================================
    # Task Operations
    # =========================================================================

    def next_task(self, spec_id: str) -> Dict[str, Any]:
        """Find the next actionable task."""
        result = self._run_sdd("next-task", spec_id)
        if (
            isinstance(result, dict)
            and result.get("success") is False
            and "No actionable tasks" in result.get("_stdout", "")
        ):
            return {
                "success": True,
                "task_id": None,
                "message": "No actionable tasks found",
            }
        return result

    def prepare_task(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Prepare task context and dependencies."""
        return self._run_sdd("prepare-task", spec_id, task_id)

    def check_dependencies(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Check if task dependencies are satisfied."""
        return self._run_sdd("check-deps", spec_id, task_id)

    def update_status(
        self, spec_id: str, task_id: str, status: str
    ) -> Dict[str, Any]:
        """Update task status."""
        return self._run_sdd("update-status", spec_id, task_id, status)

    def complete_task(
        self, spec_id: str, task_id: str, journal_entry: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mark task as completed."""
        result = self._run_sdd("complete-task", spec_id, task_id)
        if (
            journal_entry
            and isinstance(result, dict)
            and result.get("success", True)
        ):
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
        return self._run_sdd("progress", spec_id)

    def add_journal(
        self,
        spec_id: str,
        title: str,
        content: str,
        entry_type: str = "note",
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a journal entry."""
        args = ["add-journal", spec_id, "--title", title, "--content", content]
        if entry_type != "note":
            args.extend(["--type", entry_type])
        if task_id:
            args.extend(["--task", task_id])
        return self._run_sdd(*args)

    def get_journal(
        self, spec_id: str, entry_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get journal entries."""
        result = self._run_sdd("get-journal", spec_id)
        result = self._wrap_list_result(result, list_key="entries", count_key="count")
        if (
            entry_type
            and isinstance(result, dict)
            and result.get("success", True)
            and isinstance(result.get("entries"), list)
        ):
            filtered = [
                entry
                for entry in result["entries"]
                if self._entry_matches(entry, entry_type)
            ]
            result["entries"] = filtered
            result["count"] = len(filtered)
        return result

    def mark_blocked(
        self, spec_id: str, task_id: str, reason: str
    ) -> Dict[str, Any]:
        """Mark a task as blocked."""
        return self._run_sdd("mark-blocked", spec_id, task_id, "--reason", reason)

    def unblock(self, spec_id: str, task_id: str) -> Dict[str, Any]:
        """Remove blocked status from a task."""
        return self._run_sdd("unblock-task", spec_id, task_id)

    def list_blocked(self, spec_id: str) -> Dict[str, Any]:
        """List all blocked tasks."""
        result = self._run_sdd("list-blockers", spec_id)
        return self._wrap_list_result(result, list_key="blocked_tasks", count_key="count")

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_spec(self, spec_id: str) -> Dict[str, Any]:
        """Validate specification structure."""
        spec_path = self._find_spec_path(spec_id)
        if not spec_path:
            return {"success": False, "error": f"Spec not found: {spec_id}"}

        with self._normalized_spec_id(spec_path):
            result = self._run_sdd("validate", str(spec_path))

        if not isinstance(result, dict):
            return result
        if result.get("success") is False or "error" in result:
            return result

        is_valid = result.get("errors", 0) == 0 and result.get("status") != "errors"
        return {
            "success": True,
            "spec_id": spec_id,
            "is_valid": is_valid,
            "errors": result.get("errors", 0),
            "warnings": result.get("warnings", 0),
        }

    def fix_spec(self, spec_id: str) -> Dict[str, Any]:
        """Auto-fix spec issues."""
        spec_path = self._find_spec_path(spec_id)
        if not spec_path:
            return {"success": False, "error": f"Spec not found: {spec_id}"}

        with self._normalized_spec_id(spec_path):
            result = self._run_sdd("fix", str(spec_path))

        if isinstance(result, dict):
            result.setdefault("success", True)
        return result

    def spec_stats(self, spec_id: str) -> Dict[str, Any]:
        """Get specification statistics."""
        # sdd-toolkit has 'stats' or 'status-report' command
        return self._run_sdd("stats", spec_id)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def activate_spec(self, spec_id: str) -> Dict[str, Any]:
        """Move spec from pending to active."""
        return self._run_sdd("activate-spec", spec_id)

    def complete_spec(self, spec_id: str) -> Dict[str, Any]:
        """Move spec from active to completed."""
        return self._run_sdd("complete-spec", spec_id)

    def archive_spec(self, spec_id: str) -> Dict[str, Any]:
        """Move spec to archived."""
        return self._run_sdd("move-spec", spec_id, "archived")

    def move_spec(self, spec_id: str, target_status: str) -> Dict[str, Any]:
        """Move spec to a specific status folder."""
        return self._run_sdd("move-spec", spec_id, target_status)
