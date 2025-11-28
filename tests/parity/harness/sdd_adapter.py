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
        args = ["add-task", spec_id, "--parent", parent, "--title", title]

        if description:
            args.extend(["--description", description])
        if task_type != "task":
            args.extend(["--type", task_type])
        if hours is not None:
            args.extend(["--hours", str(hours)])
        if position is not None:
            args.extend(["--position", str(position)])
        if dry_run:
            args.append("--dry-run")

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "task_id": result.get("task_id"),
                "parent": parent,
                "title": title,
                "type": task_type,
                "dry_run": dry_run,
            }
        return result

    def remove_task(
        self,
        spec_id: str,
        task_id: str,
        cascade: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Remove a task from a specification."""
        args = ["remove-task", spec_id, task_id]

        if cascade:
            args.append("--cascade")
        if dry_run:
            args.append("--dry-run")

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "task_id": task_id,
                "spec_id": spec_id,
                "cascade": cascade,
                "children_removed": result.get("children_removed", 0),
                "dry_run": dry_run,
            }
        return result

    def add_assumption(
        self,
        spec_id: str,
        text: str,
        assumption_type: str = "constraint",
        author: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Add an assumption to a specification."""
        args = ["add-assumption", spec_id, "--text", text]

        if assumption_type != "constraint":
            args.extend(["--type", assumption_type])
        if author:
            args.extend(["--author", author])
        if dry_run:
            args.append("--dry-run")

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "assumption_id": result.get("assumption_id"),
                "text": text,
                "type": assumption_type,
                "dry_run": dry_run,
            }
        return result

    def list_assumptions(
        self,
        spec_id: str,
        assumption_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List assumptions from a specification."""
        args = ["list-assumptions", spec_id]

        if assumption_type:
            args.extend(["--type", assumption_type])

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            assumptions = result.get("assumptions", [])
            return {
                "success": True,
                "spec_id": spec_id,
                "assumptions": assumptions,
                "total_count": result.get("total_count", len(assumptions)),
                "filter_type": assumption_type,
            }
        return result

    def add_revision(
        self,
        spec_id: str,
        version: str,
        changes: str,
        author: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Add a revision history entry."""
        args = ["add-revision", spec_id, "--version", version, "--changes", changes]

        if author:
            args.extend(["--author", author])
        if dry_run:
            args.append("--dry-run")

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "version": version,
                "changes": changes,
                "dry_run": dry_run,
            }
        return result

    def update_estimate(
        self,
        spec_id: str,
        task_id: str,
        hours: Optional[float] = None,
        complexity: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Update task estimate."""
        args = ["update-estimate", spec_id, task_id]

        if hours is not None:
            args.extend(["--hours", str(hours)])
        if complexity:
            args.extend(["--complexity", complexity])
        if dry_run:
            args.append("--dry-run")

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            response = {
                "success": True,
                "spec_id": spec_id,
                "task_id": task_id,
                "dry_run": dry_run,
            }
            if hours is not None:
                response["hours"] = hours
            if complexity:
                response["complexity"] = complexity
            if result.get("previous_hours") is not None:
                response["previous_hours"] = result["previous_hours"]
            if result.get("previous_complexity") is not None:
                response["previous_complexity"] = result["previous_complexity"]
            return response
        return result

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
        args = ["update-task-metadata", spec_id, task_id]

        fields_updated = []
        if file_path:
            args.extend(["--file-path", file_path])
            fields_updated.append("file_path")
        if description:
            args.extend(["--description", description])
            fields_updated.append("description")
        if actual_hours is not None:
            args.extend(["--actual-hours", str(actual_hours)])
            fields_updated.append("actual_hours")
        if status_note:
            args.extend(["--status-note", status_note])
            fields_updated.append("status_note")
        if dry_run:
            args.append("--dry-run")

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "task_id": task_id,
                "fields_updated": fields_updated,
                "dry_run": dry_run,
            }
        return result

    def task_list(
        self,
        spec_id: str,
        status_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List all tasks in a specification."""
        args = ["task-list", spec_id]

        if status_filter:
            args.extend(["--status", status_filter])

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            tasks = result.get("tasks", [])
            return {
                "success": True,
                "spec_id": spec_id,
                "tasks": tasks,
                "count": len(tasks),
            }
        return result

    # =========================================================================
    # Analysis Operations
    # =========================================================================

    def analyze_deps(
        self,
        spec_id: str,
        bottleneck_threshold: int = 3,
    ) -> Dict[str, Any]:
        """Analyze dependency graph for bottlenecks and critical path."""
        args = ["analyze-deps", spec_id, "--bottleneck-threshold", str(bottleneck_threshold)]

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "dependency_count": result.get("dependency_count", 0),
                "bottlenecks": result.get("bottlenecks", []),
                "critical_path_length": result.get("critical_path_length", 0),
                "has_bottlenecks": result.get("has_bottlenecks", False),
            }
        return result

    def detect_cycles(
        self,
        spec_id: str,
    ) -> Dict[str, Any]:
        """Detect circular dependencies in task graph."""
        args = ["find-circular-deps", spec_id]

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            cycles = result.get("cycles", [])
            affected_tasks = result.get("affected_tasks", [])
            return {
                "success": True,
                "spec_id": spec_id,
                "has_cycles": len(cycles) > 0,
                "cycles": cycles,
                "cycle_count": len(cycles),
                "affected_tasks": affected_tasks,
            }
        return result

    def find_patterns(
        self,
        spec_id: str,
        pattern: str,
    ) -> Dict[str, Any]:
        """Find tasks with file_path matching pattern."""
        args = ["find-pattern", spec_id, pattern]

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            matches = result.get("matches", [])
            return {
                "success": True,
                "spec_id": spec_id,
                "pattern": pattern,
                "matches": matches,
                "total_count": len(matches),
            }
        return result

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
        args = ["add-verification", spec_id, verify_id, result]

        if command:
            args.extend(["--command", command])
        if output:
            args.extend(["--output", output])
        if issues:
            args.extend(["--issues", issues])
        if notes:
            args.extend(["--notes", notes])

        cli_result = self._run_sdd(*args)

        if isinstance(cli_result, dict) and cli_result.get("success", True) and "error" not in cli_result:
            return {
                "success": True,
                "spec_id": spec_id,
                "verify_id": verify_id,
                "result": result,
                "command": command,
            }
        return cli_result

    def get_verification_results(
        self,
        spec_id: str,
    ) -> Dict[str, Any]:
        """Get all verification results from spec."""
        # Use task-list and filter verify nodes
        args = ["task-list", spec_id]

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            tasks = result.get("tasks", [])
            results = []
            for task in tasks:
                if task.get("type") == "verify" and task.get("result"):
                    results.append({
                        "verify_id": task.get("task_id"),
                        "title": task.get("title"),
                        "result": task.get("result"),
                        "command": task.get("command"),
                        "output": task.get("output"),
                        "issues": task.get("issues", []),
                        "executed_at": task.get("executed_at"),
                    })
            return {
                "success": True,
                "spec_id": spec_id,
                "results": results,
                "count": len(results),
            }
        return result

    def format_verification_summary(
        self,
        spec_id: str,
    ) -> Dict[str, Any]:
        """Format summary of verification results."""
        args = ["format-verification-summary", spec_id]

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "summary": result.get("summary", ""),
                "total_verifications": result.get("total_verifications", 0),
                "passed": result.get("passed", 0),
                "failed": result.get("failed", 0),
                "partial": result.get("partial", 0),
            }
        return result

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
        args = ["render", spec_id, "--mode", mode]

        if include_journal:
            args.append("--include-journal")
        if max_depth > 0:
            args.extend(["--max-depth", str(max_depth)])

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "title": result.get("title", ""),
                "markdown": result.get("markdown", ""),
                "total_sections": result.get("total_sections", 0),
                "total_tasks": result.get("total_tasks", 0),
                "completed_tasks": result.get("completed_tasks", 0),
                "progress_percentage": result.get("progress_percentage", 0),
            }
        return result

    def render_progress(
        self,
        spec_id: str,
        bar_width: int = 20,
    ) -> Dict[str, Any]:
        """Get visual progress summary for a spec."""
        args = ["render-progress", spec_id, "--bar-width", str(bar_width)]

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "title": result.get("title", ""),
                "overall": result.get("overall", {}),
                "phases": result.get("phases", []),
            }
        return result

    def spec_report(
        self,
        spec_id: str,
        format: str = "markdown",
        sections: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive report for a spec."""
        args = ["report", spec_id, "--format", format]

        if sections:
            args.extend(["--sections", sections])

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "report": result.get("report", ""),
                "format": format,
                "sections": result.get("sections", []),
                "summary": result.get("summary", {}),
            }
        return result

    def spec_report_summary(
        self,
        spec_id: str,
    ) -> Dict[str, Any]:
        """Generate quick summary report for a spec."""
        args = ["report-summary", spec_id]

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "title": result.get("title", ""),
                "status": result.get("status", ""),
                "validation": result.get("validation", {}),
                "progress": result.get("progress", {}),
                "health": result.get("health", {}),
            }
        return result

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
        args = ["update-frontmatter", spec_id, key, value]

        if dry_run:
            args.append("--dry-run")

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "key": key,
                "value": value,
                "previous_value": result.get("previous_value"),
                "dry_run": dry_run,
            }
        return result

    # =========================================================================
    # File Operations
    # =========================================================================

    def find_related_files(
        self,
        file_path: str,
        spec_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find files related to a source file."""
        args = ["find-related-files", file_path]

        if spec_id:
            args.extend(["--spec-id", spec_id])

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "file_path": file_path,
                "related_files": result.get("related_files", []),
                "spec_references": result.get("spec_references", []),
                "total_count": result.get("total_count", 0),
            }
        return result

    def validate_paths(
        self,
        paths: list,
        base_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate that file paths exist on disk."""
        args = ["validate-paths"] + paths

        if base_directory:
            args.extend(["--base-directory", base_directory])

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "paths_checked": len(paths),
                "valid_paths": result.get("valid_paths", []),
                "invalid_paths": result.get("invalid_paths", []),
                "all_valid": result.get("all_valid", False),
                "valid_count": result.get("valid_count", 0),
                "invalid_count": result.get("invalid_count", 0),
            }
        return result

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
        args = ["task-query", spec_id, "--limit", str(limit)]

        if status:
            args.extend(["--status", status])
        if parent:
            args.extend(["--parent", parent])
        if cursor:
            args.extend(["--cursor", cursor])

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "tasks": result.get("tasks", []),
                "count": result.get("count", 0),
                "total": result.get("total", 0),
                "cursor": result.get("cursor"),
                "has_more": result.get("has_more", False),
            }
        return result

    def journal_list_paginated(
        self,
        spec_id: str,
        entry_type: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List journal entries with pagination."""
        args = ["journal-list", spec_id, "--limit", str(limit)]

        if entry_type:
            args.extend(["--entry-type", entry_type])
        if task_id:
            args.extend(["--task-id", task_id])
        if cursor:
            args.extend(["--cursor", cursor])

        result = self._run_sdd(*args)

        if isinstance(result, dict) and result.get("success", True) and "error" not in result:
            return {
                "success": True,
                "spec_id": spec_id,
                "entries": result.get("entries", []),
                "count": result.get("count", 0),
                "total": result.get("total", 0),
                "cursor": result.get("cursor"),
                "has_more": result.get("has_more", False),
            }
        return result
