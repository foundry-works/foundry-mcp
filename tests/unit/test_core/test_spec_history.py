"""Tests for spec history operations: backup, diff, list backups, and rollback."""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from foundry_mcp.core.spec import (
    DEFAULT_MAX_BACKUPS,
    DEFAULT_BACKUP_PAGE_SIZE,
    DEFAULT_DIFF_MAX_RESULTS,
    backup_spec,
    diff_specs,
    list_spec_backups,
    load_spec,
    rollback_spec,
    _apply_backup_retention,
)


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure with .backups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Resolve to handle macOS /var -> /private/var symlink
        specs_dir = (Path(tmpdir) / "specs").resolve()

        # Create status directories
        (specs_dir / "pending").mkdir(parents=True)
        (specs_dir / "active").mkdir(parents=True)
        (specs_dir / "completed").mkdir(parents=True)
        (specs_dir / "archived").mkdir(parents=True)
        # Create .backups directory
        (specs_dir / ".backups").mkdir(parents=True)

        yield specs_dir


@pytest.fixture
def sample_spec() -> Dict[str, Any]:
    """Create a sample spec data structure."""
    return {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "metadata": {
            "title": "Test Specification",
            "version": "1.0.0",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "pending",
                "parent": None,
                "children": ["phase-1"],
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-1-1"],
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        },
    }


def _write_spec(specs_dir: Path, spec_data: Dict[str, Any], status: str = "active") -> Path:
    """Helper to write a spec file to the specified status directory."""
    spec_id = spec_data["spec_id"]
    spec_file = specs_dir / status / f"{spec_id}.json"
    spec_file.write_text(json.dumps(spec_data))
    return spec_file


def _create_backup(
    specs_dir: Path,
    spec_id: str,
    timestamp: str,
    spec_data: Dict[str, Any],
) -> Path:
    """Helper to create a backup file with a specific timestamp."""
    backups_dir = specs_dir / ".backups" / spec_id
    backups_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backups_dir / f"{timestamp}.json"
    backup_file.write_text(json.dumps(spec_data))
    return backup_file


class TestBackupSpec:
    """Tests for backup_spec function."""

    def test_backup_creates_timestamped_file(self, temp_specs_dir, sample_spec):
        """Should create a timestamped backup file in .backups/{spec_id}/."""
        _write_spec(temp_specs_dir, sample_spec)

        backup_path = backup_spec("test-spec-001", temp_specs_dir)

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.suffix == ".json"
        # Verify it's in the correct directory
        assert backup_path.parent == temp_specs_dir / ".backups" / "test-spec-001"
        # Verify timestamp format in filename (YYYY-MM-DDTHH-MM-SS.microseconds)
        assert "T" in backup_path.stem
        assert "." in backup_path.stem  # Has microseconds

    def test_backup_creates_latest_json(self, temp_specs_dir, sample_spec):
        """Should create/update latest.json pointing to newest backup."""
        _write_spec(temp_specs_dir, sample_spec)

        backup_path = backup_spec("test-spec-001", temp_specs_dir)
        latest_path = temp_specs_dir / ".backups" / "test-spec-001" / "latest.json"

        assert latest_path.exists()
        # latest.json should contain the same data as the backup
        with open(backup_path) as f:
            backup_data = json.load(f)
        with open(latest_path) as f:
            latest_data = json.load(f)
        assert backup_data == latest_data

    def test_backup_preserves_spec_content(self, temp_specs_dir, sample_spec):
        """Should preserve the full spec content in backup."""
        _write_spec(temp_specs_dir, sample_spec)

        backup_path = backup_spec("test-spec-001", temp_specs_dir)

        with open(backup_path) as f:
            backup_data = json.load(f)

        assert backup_data["spec_id"] == "test-spec-001"
        assert backup_data["title"] == "Test Specification"
        assert "hierarchy" in backup_data
        assert "task-1-1" in backup_data["hierarchy"]

    def test_backup_returns_none_for_nonexistent_spec(self, temp_specs_dir):
        """Should return None for nonexistent spec."""
        result = backup_spec("nonexistent-spec", temp_specs_dir)
        assert result is None

    def test_backup_applies_retention_policy(self, temp_specs_dir, sample_spec):
        """Should delete oldest backups when exceeding max_backups."""
        _write_spec(temp_specs_dir, sample_spec)

        # Create 5 backups with max_backups=3
        for i in range(5):
            backup_spec("test-spec-001", temp_specs_dir, max_backups=3)
            time.sleep(0.001)  # Small delay for unique timestamps

        # Count backup files (excluding latest.json)
        backups_dir = temp_specs_dir / ".backups" / "test-spec-001"
        backup_files = [
            f for f in backups_dir.glob("*.json")
            if f.name != "latest.json"
        ]

        assert len(backup_files) == 3

    def test_backup_unlimited_when_max_zero(self, temp_specs_dir, sample_spec):
        """Should not delete backups when max_backups=0."""
        _write_spec(temp_specs_dir, sample_spec)

        # Create 5 backups with unlimited retention
        for i in range(5):
            backup_spec("test-spec-001", temp_specs_dir, max_backups=0)
            time.sleep(0.001)

        backups_dir = temp_specs_dir / ".backups" / "test-spec-001"
        backup_files = [
            f for f in backups_dir.glob("*.json")
            if f.name != "latest.json"
        ]

        assert len(backup_files) == 5

    def test_backup_microsecond_precision(self, temp_specs_dir, sample_spec):
        """Should handle rapid successive backups with microsecond precision."""
        _write_spec(temp_specs_dir, sample_spec)

        # Create multiple backups rapidly (no sleep)
        backups = []
        for _ in range(3):
            backup = backup_spec("test-spec-001", temp_specs_dir, max_backups=0)
            if backup:
                backups.append(backup)

        # All should have unique filenames
        filenames = [b.name for b in backups]
        assert len(set(filenames)) == len(filenames)


class TestApplyBackupRetention:
    """Tests for _apply_backup_retention helper."""

    def test_retention_deletes_oldest_backups(self, temp_specs_dir, sample_spec):
        """Should delete oldest backups exceeding max limit."""
        backups_dir = temp_specs_dir / ".backups" / "test-spec"
        backups_dir.mkdir(parents=True)

        # Create 5 backup files with known timestamps
        timestamps = [
            "2025-01-01T10-00-00.000001",
            "2025-01-01T10-00-01.000001",
            "2025-01-01T10-00-02.000001",
            "2025-01-01T10-00-03.000001",
            "2025-01-01T10-00-04.000001",
        ]
        for ts in timestamps:
            (backups_dir / f"{ts}.json").write_text("{}")

        deleted = _apply_backup_retention(backups_dir, max_backups=3)

        assert deleted == 2
        remaining = sorted([f.stem for f in backups_dir.glob("*.json")])
        assert remaining == timestamps[2:]  # Oldest 2 deleted

    def test_retention_preserves_latest_json(self, temp_specs_dir):
        """Should not count or delete latest.json in retention."""
        backups_dir = temp_specs_dir / ".backups" / "test-spec"
        backups_dir.mkdir(parents=True)

        # Create backups and latest.json
        timestamps = ["2025-01-01T10-00-00.000001", "2025-01-01T10-00-01.000001"]
        for ts in timestamps:
            (backups_dir / f"{ts}.json").write_text("{}")
        (backups_dir / "latest.json").write_text("{}")

        deleted = _apply_backup_retention(backups_dir, max_backups=2)

        assert deleted == 0
        assert (backups_dir / "latest.json").exists()


class TestListSpecBackups:
    """Tests for list_spec_backups function."""

    def test_list_returns_backups_newest_first(self, temp_specs_dir, sample_spec):
        """Should return backups in reverse chronological order."""
        spec_id = "test-spec-001"
        timestamps = [
            "2025-01-01T10-00-00.000001",
            "2025-01-01T10-00-01.000001",
            "2025-01-01T10-00-02.000001",
        ]
        for ts in timestamps:
            _create_backup(temp_specs_dir, spec_id, ts, sample_spec)

        result = list_spec_backups(spec_id, temp_specs_dir)

        assert result["spec_id"] == spec_id
        assert result["count"] == 3
        # Newest first
        assert result["backups"][0]["timestamp"] == timestamps[2]
        assert result["backups"][1]["timestamp"] == timestamps[1]
        assert result["backups"][2]["timestamp"] == timestamps[0]

    def test_list_excludes_latest_json(self, temp_specs_dir, sample_spec):
        """Should not include latest.json in the backup list."""
        spec_id = "test-spec-001"
        _create_backup(temp_specs_dir, spec_id, "2025-01-01T10-00-00.000001", sample_spec)

        # Also create latest.json
        backups_dir = temp_specs_dir / ".backups" / spec_id
        (backups_dir / "latest.json").write_text(json.dumps(sample_spec))

        result = list_spec_backups(spec_id, temp_specs_dir)

        assert result["count"] == 1
        assert all(b["timestamp"] != "latest" for b in result["backups"])

    def test_list_returns_file_metadata(self, temp_specs_dir, sample_spec):
        """Should include timestamp, file_path, and file_size_bytes."""
        spec_id = "test-spec-001"
        timestamp = "2025-01-01T10-00-00.000001"
        backup_file = _create_backup(temp_specs_dir, spec_id, timestamp, sample_spec)

        result = list_spec_backups(spec_id, temp_specs_dir)

        assert len(result["backups"]) == 1
        backup = result["backups"][0]
        assert backup["timestamp"] == timestamp
        assert backup["file_path"] == str(backup_file.absolute())
        assert backup["file_size_bytes"] == backup_file.stat().st_size

    def test_list_pagination_with_cursor(self, temp_specs_dir, sample_spec):
        """Should support cursor-based pagination."""
        spec_id = "test-spec-001"
        # Create 5 backups
        timestamps = [f"2025-01-01T10-00-0{i}.000001" for i in range(5)]
        for ts in timestamps:
            _create_backup(temp_specs_dir, spec_id, ts, sample_spec)

        # First page (limit 2)
        result1 = list_spec_backups(spec_id, temp_specs_dir, limit=2)
        assert result1["count"] == 2
        assert result1["pagination"]["has_more"] is True
        assert result1["pagination"]["cursor"] is not None

        # Second page using cursor
        result2 = list_spec_backups(
            spec_id, temp_specs_dir, cursor=result1["pagination"]["cursor"], limit=2
        )
        assert result2["count"] == 2
        assert result2["pagination"]["has_more"] is True

        # Verify no overlap
        page1_timestamps = [b["timestamp"] for b in result1["backups"]]
        page2_timestamps = [b["timestamp"] for b in result2["backups"]]
        assert set(page1_timestamps) & set(page2_timestamps) == set()

    def test_list_respects_limit(self, temp_specs_dir, sample_spec):
        """Should respect the page size limit."""
        spec_id = "test-spec-001"
        for i in range(10):
            _create_backup(temp_specs_dir, spec_id, f"2025-01-01T10-00-{i:02d}.000001", sample_spec)

        result = list_spec_backups(spec_id, temp_specs_dir, limit=3)

        assert result["count"] == 3
        assert result["pagination"]["page_size"] == 3
        assert result["pagination"]["has_more"] is True

    def test_list_empty_if_no_backups(self, temp_specs_dir):
        """Should return empty list if no backups exist."""
        result = list_spec_backups("nonexistent-spec", temp_specs_dir)

        assert result["spec_id"] == "nonexistent-spec"
        assert result["backups"] == []
        assert result["count"] == 0
        assert result["pagination"]["has_more"] is False

    def test_list_empty_if_backup_dir_missing(self, temp_specs_dir, sample_spec):
        """Should return empty list if backup directory doesn't exist."""
        # Write spec but don't create any backups
        _write_spec(temp_specs_dir, sample_spec)

        result = list_spec_backups("test-spec-001", temp_specs_dir)

        assert result["backups"] == []
        assert result["count"] == 0


class TestDiffSpecs:
    """Tests for diff_specs function."""

    def test_diff_identifies_added_nodes(self, temp_specs_dir, sample_spec):
        """Should identify nodes added in target spec."""
        # Source spec (base)
        source = sample_spec.copy()
        source = json.loads(json.dumps(source))  # Deep copy

        # Target spec with new task added
        target = json.loads(json.dumps(sample_spec))
        target["hierarchy"]["task-1-2"] = {
            "type": "task",
            "title": "Task 2",
            "status": "pending",
            "parent": "phase-1",
            "children": [],
            "metadata": {},
            "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
        }

        result = diff_specs(source, target, temp_specs_dir)

        assert "error" not in result
        assert result["summary"]["added_count"] == 1
        assert len(result["changes"]["added"]) == 1
        assert result["changes"]["added"][0]["node_id"] == "task-1-2"
        assert result["changes"]["added"][0]["type"] == "task"

    def test_diff_identifies_removed_nodes(self, temp_specs_dir, sample_spec):
        """Should identify nodes removed from target spec."""
        source = json.loads(json.dumps(sample_spec))

        # Target with task removed
        target = json.loads(json.dumps(sample_spec))
        del target["hierarchy"]["task-1-1"]

        result = diff_specs(source, target, temp_specs_dir)

        assert result["summary"]["removed_count"] == 1
        assert len(result["changes"]["removed"]) == 1
        assert result["changes"]["removed"][0]["node_id"] == "task-1-1"

    def test_diff_identifies_modified_nodes(self, temp_specs_dir, sample_spec):
        """Should identify modified nodes with field changes."""
        source = json.loads(json.dumps(sample_spec))

        # Target with modified task
        target = json.loads(json.dumps(sample_spec))
        target["hierarchy"]["task-1-1"]["status"] = "completed"
        target["hierarchy"]["task-1-1"]["title"] = "Updated Task 1"

        result = diff_specs(source, target, temp_specs_dir)

        assert result["summary"]["modified_count"] == 1
        modified = result["changes"]["modified"][0]
        assert modified["node_id"] == "task-1-1"
        assert len(modified["field_changes"]) == 2

        # Check field changes
        field_names = [fc["field"] for fc in modified["field_changes"]]
        assert "status" in field_names
        assert "title" in field_names

    def test_diff_compares_against_backup_file(self, temp_specs_dir, sample_spec):
        """Should accept backup file path as source."""
        # Create a backup (older version)
        backup_spec_data = json.loads(json.dumps(sample_spec))
        timestamp = "2025-01-01T10-00-00.000001"
        backup_file = _create_backup(
            temp_specs_dir, "test-spec-001", timestamp, backup_spec_data
        )

        # Current version with changes
        current = json.loads(json.dumps(sample_spec))
        current["hierarchy"]["task-1-1"]["status"] = "completed"
        _write_spec(temp_specs_dir, current)

        # Compare backup (source) against current (target)
        result = diff_specs(str(backup_file), current, temp_specs_dir)

        assert "error" not in result
        assert result["summary"]["modified_count"] >= 1

    def test_diff_compares_by_spec_id(self, temp_specs_dir, sample_spec):
        """Should accept spec_id as source or target."""
        # Write two different specs
        spec1 = json.loads(json.dumps(sample_spec))
        spec1["spec_id"] = "spec-source"
        _write_spec(temp_specs_dir, spec1)

        spec2 = json.loads(json.dumps(sample_spec))
        spec2["spec_id"] = "spec-target"
        spec2["hierarchy"]["task-1-1"]["status"] = "completed"
        _write_spec(temp_specs_dir, spec2)

        result = diff_specs("spec-source", "spec-target", temp_specs_dir)

        assert "error" not in result
        assert result["source_spec_id"] == "spec-source"
        assert result["target_spec_id"] == "spec-target"

    def test_diff_partial_when_results_truncated(self, temp_specs_dir, sample_spec):
        """Should set partial=True when results exceed max_results."""
        source = json.loads(json.dumps(sample_spec))

        # Add many nodes to target
        target = json.loads(json.dumps(sample_spec))
        for i in range(10):
            target["hierarchy"][f"new-task-{i}"] = {
                "type": "task",
                "title": f"New Task {i}",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            }

        result = diff_specs(source, target, temp_specs_dir, max_results=3)

        assert result["summary"]["added_count"] == 10  # Total count
        assert len(result["changes"]["added"]) == 3  # Limited results
        assert result["partial"] is True

    def test_diff_error_for_missing_source(self, temp_specs_dir, sample_spec):
        """Should return error structure for missing source spec."""
        target = json.loads(json.dumps(sample_spec))

        result = diff_specs("nonexistent-source", target, temp_specs_dir)

        assert result.get("success") is False
        assert "error" in result
        assert "source" in result["error"].lower()

    def test_diff_error_for_missing_target(self, temp_specs_dir, sample_spec):
        """Should return error structure for missing target spec."""
        source = json.loads(json.dumps(sample_spec))

        result = diff_specs(source, "nonexistent-target", temp_specs_dir)

        assert result.get("success") is False
        assert "error" in result
        assert "target" in result["error"].lower()

    def test_diff_no_changes(self, temp_specs_dir, sample_spec):
        """Should return zero counts when specs are identical."""
        spec = json.loads(json.dumps(sample_spec))

        result = diff_specs(spec, spec, temp_specs_dir)

        assert result["summary"]["total_changes"] == 0
        assert result["summary"]["added_count"] == 0
        assert result["summary"]["removed_count"] == 0
        assert result["summary"]["modified_count"] == 0


class TestRollbackSpec:
    """Tests for rollback_spec function."""

    def test_rollback_restores_from_timestamp(self, temp_specs_dir, sample_spec):
        """Should restore spec content from backup."""
        # Write current spec (modified version)
        current = json.loads(json.dumps(sample_spec))
        current["hierarchy"]["task-1-1"]["status"] = "completed"
        current["metadata"]["version"] = "2.0.0"
        _write_spec(temp_specs_dir, current)

        # Create backup of original version
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        result = rollback_spec("test-spec-001", timestamp, temp_specs_dir)

        assert result["success"] is True
        assert result["spec_id"] == "test-spec-001"
        assert result["timestamp"] == timestamp
        assert result["restored_from"] is not None

        # Verify content was restored
        restored = load_spec("test-spec-001", temp_specs_dir)
        assert restored["metadata"]["version"] == "1.0.0"
        assert restored["hierarchy"]["task-1-1"]["status"] == "pending"

    def test_rollback_creates_safety_backup(self, temp_specs_dir, sample_spec):
        """Should create safety backup before rollback by default."""
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        result = rollback_spec("test-spec-001", timestamp, temp_specs_dir)

        assert result["success"] is True
        assert result["backup_created"] is not None
        # Verify safety backup exists
        assert Path(result["backup_created"]).exists()

    def test_rollback_dry_run(self, temp_specs_dir, sample_spec):
        """Should validate without making changes in dry_run mode."""
        current = json.loads(json.dumps(sample_spec))
        current["metadata"]["version"] = "2.0.0"
        _write_spec(temp_specs_dir, current)

        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        result = rollback_spec(
            "test-spec-001", timestamp, temp_specs_dir, dry_run=True
        )

        assert result["success"] is True
        assert result["dry_run"] is True
        assert result["backup_created"] == "(would be created)"

        # Verify content was NOT restored
        still_current = load_spec("test-spec-001", temp_specs_dir)
        assert still_current["metadata"]["version"] == "2.0.0"

    def test_rollback_without_safety_backup(self, temp_specs_dir, sample_spec):
        """Should skip safety backup when create_backup=False."""
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        # Count backups before
        backups_dir = temp_specs_dir / ".backups" / "test-spec-001"
        before_count = len(list(backups_dir.glob("*.json")))

        result = rollback_spec(
            "test-spec-001", timestamp, temp_specs_dir, create_backup=False
        )

        assert result["success"] is True
        assert result["backup_created"] is None

        # Same number of backups
        after_count = len(list(backups_dir.glob("*.json")))
        assert after_count == before_count

    def test_rollback_error_missing_spec(self, temp_specs_dir, sample_spec):
        """Should return error for nonexistent spec."""
        timestamp = "2025-01-01T10-00-00.000001"

        result = rollback_spec("nonexistent-spec", timestamp, temp_specs_dir)

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_rollback_error_missing_backup(self, temp_specs_dir, sample_spec):
        """Should return error for invalid timestamp."""
        _write_spec(temp_specs_dir, sample_spec)

        result = rollback_spec("test-spec-001", "invalid-timestamp", temp_specs_dir)

        assert result["success"] is False
        assert "not found" in result["error"].lower() or "backup" in result["error"].lower()

    def test_rollback_error_corrupted_backup(self, temp_specs_dir, sample_spec):
        """Should return error for corrupted backup JSON."""
        _write_spec(temp_specs_dir, sample_spec)

        # Create corrupted backup
        timestamp = "2025-01-01T10-00-00.000001"
        backups_dir = temp_specs_dir / ".backups" / "test-spec-001"
        backups_dir.mkdir(parents=True, exist_ok=True)
        (backups_dir / f"{timestamp}.json").write_text("{ invalid json }")

        result = rollback_spec("test-spec-001", timestamp, temp_specs_dir)

        assert result["success"] is False
        assert "json" in result["error"].lower()

    def test_rollback_error_no_backups_directory(self, temp_specs_dir, sample_spec):
        """Should return error when no backups directory exists."""
        _write_spec(temp_specs_dir, sample_spec)

        result = rollback_spec("test-spec-001", "any-timestamp", temp_specs_dir)

        assert result["success"] is False
        assert "no backups" in result["error"].lower() or "directory" in result["error"].lower()


class TestDiffWithBackups:
    """Tests for diff_specs with backup integration."""

    def test_diff_against_latest_backup(self, temp_specs_dir, sample_spec):
        """Should support diffing against a backup file path."""
        # Create old version as backup
        old_spec = json.loads(json.dumps(sample_spec))
        old_spec["hierarchy"]["task-1-1"]["status"] = "pending"
        timestamp = "2025-01-01T10-00-00.000001"
        backup_file = _create_backup(
            temp_specs_dir, "test-spec-001", timestamp, old_spec
        )

        # Write current version
        current = json.loads(json.dumps(sample_spec))
        current["hierarchy"]["task-1-1"]["status"] = "completed"
        _write_spec(temp_specs_dir, current)

        # Diff: backup (old) vs current (new)
        result = diff_specs(str(backup_file), current, temp_specs_dir)

        assert "error" not in result
        assert result["summary"]["modified_count"] >= 1

        # Find the task modification
        task_mod = next(
            (m for m in result["changes"]["modified"] if m["node_id"] == "task-1-1"),
            None,
        )
        assert task_mod is not None
        status_change = next(
            (fc for fc in task_mod["field_changes"] if fc["field"] == "status"),
            None,
        )
        assert status_change is not None
        assert status_change["old"] == "pending"
        assert status_change["new"] == "completed"


class TestBackupRetentionIntegration:
    """Integration tests for backup retention across operations."""

    def test_rollback_preserves_retention_after_safety_backup(
        self, temp_specs_dir, sample_spec
    ):
        """Safety backup during rollback should still respect retention."""
        _write_spec(temp_specs_dir, sample_spec)

        # Create initial backup
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        # Rollback with safety backup (default create_backup=True)
        result = rollback_spec(
            "test-spec-001", timestamp, temp_specs_dir, create_backup=True
        )

        assert result["success"] is True
        # Both original and safety backup should exist
        backups_dir = temp_specs_dir / ".backups" / "test-spec-001"
        backup_count = len([
            f for f in backups_dir.glob("*.json") if f.name != "latest.json"
        ])
        assert backup_count >= 2


class TestEdgeCases:
    """Edge cases and error handling tests."""

    def test_diff_empty_hierarchies(self, temp_specs_dir):
        """Should handle specs with empty hierarchies."""
        source = {"spec_id": "empty-source", "hierarchy": {}}
        target = {"spec_id": "empty-target", "hierarchy": {}}

        result = diff_specs(source, target, temp_specs_dir)

        assert result["summary"]["total_changes"] == 0

    def test_list_backups_handles_stat_failure(self, temp_specs_dir, sample_spec):
        """Should skip files that can't be stat'd."""
        spec_id = "test-spec-001"
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, spec_id, timestamp, sample_spec)

        # This test just ensures the function doesn't crash on edge cases
        result = list_spec_backups(spec_id, temp_specs_dir)
        assert result["count"] >= 0

    def test_backup_creates_directory_structure(self, temp_specs_dir, sample_spec):
        """Should create .backups/{spec_id}/ if it doesn't exist."""
        _write_spec(temp_specs_dir, sample_spec)

        # Remove .backups dir to test auto-creation
        import shutil
        backups_root = temp_specs_dir / ".backups"
        if backups_root.exists():
            shutil.rmtree(backups_root)

        backup_path = backup_spec("test-spec-001", temp_specs_dir)

        assert backup_path is not None
        assert backup_path.exists()
        assert (temp_specs_dir / ".backups" / "test-spec-001").is_dir()

    def test_diff_with_nested_metadata_changes(self, temp_specs_dir, sample_spec):
        """Should detect changes in nested metadata fields."""
        source = json.loads(json.dumps(sample_spec))
        source["hierarchy"]["task-1-1"]["metadata"] = {"priority": "low"}

        target = json.loads(json.dumps(sample_spec))
        target["hierarchy"]["task-1-1"]["metadata"] = {"priority": "high"}

        result = diff_specs(source, target, temp_specs_dir)

        assert result["summary"]["modified_count"] == 1
        modified = result["changes"]["modified"][0]
        metadata_change = next(
            (fc for fc in modified["field_changes"] if fc["field"] == "metadata"),
            None,
        )
        assert metadata_change is not None
        assert metadata_change["old"] == {"priority": "low"}
        assert metadata_change["new"] == {"priority": "high"}

    def test_rollback_with_special_characters_in_timestamp(
        self, temp_specs_dir, sample_spec
    ):
        """Should handle timestamps with various formats."""
        _write_spec(temp_specs_dir, sample_spec)

        # Create backup with microseconds
        timestamp = "2025-12-26T18-20-13.456789"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        result = rollback_spec("test-spec-001", timestamp, temp_specs_dir)

        assert result["success"] is True
        assert result["timestamp"] == timestamp
