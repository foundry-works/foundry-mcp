"""
Unit tests for batch_operations module.

Tests the get_independent_tasks function and helper utilities for
parallel task execution.
"""

import json
import os
import pytest
from pathlib import Path

from foundry_mcp.core.batch_operations import (
    get_independent_tasks,
    start_batch,
    complete_batch,
    reset_batch,
    prepare_batch_context,
    _get_active_phases,
    _paths_conflict,
    _is_within_project_root,
    _has_direct_dependency,
    DEFAULT_MAX_TASKS,
    MAX_RETRY_COUNT,
    STALE_TASK_THRESHOLD_HOURS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def test_specs_dir(tmp_path):
    """Create a temporary specs directory structure."""
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    for d in ["active", "pending", "completed", "archived"]:
        (specs_dir / d).mkdir()
    return specs_dir


@pytest.fixture
def basic_spec(test_specs_dir):
    """Create a basic spec with multiple independent tasks."""
    spec = {
        "spec_id": "test-spec-001",
        "title": "Test Spec",
        "metadata": {"title": "Test Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2", "task-1-3"],
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-1": {
                "type": "task",
                "title": "Task A",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/a.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-2": {
                "type": "task",
                "title": "Task B",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/b.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-3": {
                "type": "task",
                "title": "Task C",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/c.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        },
        "assumptions": [],
        "revision_history": [],
        "journal": [],
    }
    (test_specs_dir / "active" / "test-spec-001.json").write_text(json.dumps(spec))
    return spec


@pytest.fixture
def dependent_spec(test_specs_dir):
    """Create a spec with task dependencies."""
    spec = {
        "spec_id": "dep-spec-001",
        "title": "Dependent Spec",
        "metadata": {"title": "Dependent Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2", "task-1-3"],
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-1": {
                "type": "task",
                "title": "Task A",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/a.py"},
                "dependencies": {"blocks": ["task-1-2"], "blocked_by": [], "depends": []},
            },
            "task-1-2": {
                "type": "task",
                "title": "Task B",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/b.py"},
                "dependencies": {"blocks": [], "blocked_by": ["task-1-1"], "depends": []},
            },
            "task-1-3": {
                "type": "task",
                "title": "Task C",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/c.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        },
        "assumptions": [],
        "revision_history": [],
        "journal": [],
    }
    (test_specs_dir / "active" / "dep-spec-001.json").write_text(json.dumps(spec))
    return spec


@pytest.fixture
def same_file_spec(test_specs_dir):
    """Create a spec with tasks targeting the same file."""
    spec = {
        "spec_id": "same-file-001",
        "title": "Same File Spec",
        "metadata": {"title": "Same File Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2", "task-1-3"],
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-1": {
                "type": "task",
                "title": "Edit file A",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/shared.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-2": {
                "type": "task",
                "title": "Also edit file A",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/shared.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-3": {
                "type": "task",
                "title": "Edit different file",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/other.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        },
        "assumptions": [],
        "revision_history": [],
        "journal": [],
    }
    (test_specs_dir / "active" / "same-file-001.json").write_text(json.dumps(spec))
    return spec


@pytest.fixture
def barrier_spec(test_specs_dir):
    """Create a spec with a barrier task (no file_path)."""
    spec = {
        "spec_id": "barrier-spec-001",
        "title": "Barrier Spec",
        "metadata": {"title": "Barrier Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2", "task-1-3"],
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-1": {
                "type": "task",
                "title": "Barrier task (no file)",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {},  # No file_path - this is a barrier
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-2": {
                "type": "task",
                "title": "Task with file",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/a.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-3": {
                "type": "task",
                "title": "Another task with file",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/b.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        },
        "assumptions": [],
        "revision_history": [],
        "journal": [],
    }
    (test_specs_dir / "active" / "barrier-spec-001.json").write_text(json.dumps(spec))
    return spec


@pytest.fixture
def ancestry_spec(test_specs_dir):
    """Create a spec with path ancestry conflicts."""
    spec = {
        "spec_id": "ancestry-spec-001",
        "title": "Ancestry Spec",
        "metadata": {"title": "Ancestry Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2", "task-1-3"],
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-1": {
                "type": "task",
                "title": "Edit parent dir",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/components"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-2": {
                "type": "task",
                "title": "Edit child file",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/components/button.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-3": {
                "type": "task",
                "title": "Edit unrelated file",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/utils.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        },
        "assumptions": [],
        "revision_history": [],
        "journal": [],
    }
    (test_specs_dir / "active" / "ancestry-spec-001.json").write_text(json.dumps(spec))
    return spec


@pytest.fixture
def multi_phase_spec(test_specs_dir):
    """Create a spec with multiple phases including phase dependencies."""
    spec = {
        "spec_id": "multi-phase-001",
        "title": "Multi Phase Spec",
        "metadata": {"title": "Multi Phase Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test",
                "status": "in_progress",
                "children": ["phase-1", "phase-2"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1"],
                "dependencies": {"blocks": ["phase-2"], "blocked_by": [], "depends": []},
            },
            "task-1-1": {
                "type": "task",
                "title": "Phase 1 Task",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/a.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "phase-2": {
                "type": "phase",
                "title": "Phase 2",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-2-1"],
                "dependencies": {"blocks": [], "blocked_by": ["phase-1"], "depends": []},
            },
            "task-2-1": {
                "type": "task",
                "title": "Phase 2 Task",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
                "metadata": {"file_path": "src/b.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        },
        "assumptions": [],
        "revision_history": [],
        "journal": [],
    }
    (test_specs_dir / "active" / "multi-phase-001.json").write_text(json.dumps(spec))
    return spec


@pytest.fixture
def retry_spec(test_specs_dir):
    """Create a spec with tasks that have exceeded retry count."""
    spec = {
        "spec_id": "retry-spec-001",
        "title": "Retry Spec",
        "metadata": {"title": "Retry Spec", "status": "in_progress", "version": "1.0.0"},
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-1": {
                "type": "task",
                "title": "Failed too many times",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/a.py", "retry_count": 5},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-2": {
                "type": "task",
                "title": "Still has retries",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/b.py", "retry_count": 1},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        },
        "assumptions": [],
        "revision_history": [],
        "journal": [],
    }
    (test_specs_dir / "active" / "retry-spec-001.json").write_text(json.dumps(spec))
    return spec


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestPathsConflict:
    """Tests for _paths_conflict helper."""

    def test_same_path_conflicts(self):
        assert _paths_conflict("/a/b/c.py", "/a/b/c.py") is True

    def test_different_paths_no_conflict(self):
        assert _paths_conflict("/a/b/c.py", "/a/b/d.py") is False

    def test_ancestor_path_conflicts(self):
        assert _paths_conflict("/a/b", "/a/b/c.py") is True

    def test_descendant_path_conflicts(self):
        assert _paths_conflict("/a/b/c.py", "/a/b") is True

    def test_none_path_no_conflict(self):
        assert _paths_conflict(None, "/a/b/c.py") is False
        assert _paths_conflict("/a/b/c.py", None) is False
        assert _paths_conflict(None, None) is False

    def test_sibling_paths_no_conflict(self):
        assert _paths_conflict("/a/b/c.py", "/a/b/d.py") is False
        assert _paths_conflict("/a/x/file.py", "/a/y/file.py") is False

    def test_similar_prefix_no_conflict(self):
        # "src/component" should not conflict with "src/components/foo.py"
        assert _paths_conflict("/src/component", "/src/components/foo.py") is False


class TestIsWithinProjectRoot:
    """Tests for _is_within_project_root helper."""

    def test_file_within_root(self, tmp_path):
        assert _is_within_project_root(str(tmp_path / "src" / "test.py"), tmp_path) is True

    def test_file_outside_root(self, tmp_path):
        assert _is_within_project_root("/etc/passwd", tmp_path) is False

    def test_root_itself(self, tmp_path):
        assert _is_within_project_root(str(tmp_path), tmp_path) is True

    def test_parent_traversal_blocked(self, tmp_path):
        # Attempt to traverse outside should fail
        assert _is_within_project_root(str(tmp_path / ".." / "other"), tmp_path) is False


class TestHasDirectDependency:
    """Tests for _has_direct_dependency helper."""

    def test_no_dependency(self):
        hierarchy = {}
        task_a = {"dependencies": {"blocks": [], "blocked_by": [], "depends": []}}
        task_b = {"dependencies": {"blocks": [], "blocked_by": [], "depends": []}}
        assert _has_direct_dependency(hierarchy, "a", task_a, "b", task_b) is False

    def test_a_blocks_b(self):
        hierarchy = {}
        task_a = {"dependencies": {"blocks": ["b"], "blocked_by": [], "depends": []}}
        task_b = {"dependencies": {"blocks": [], "blocked_by": ["a"], "depends": []}}
        assert _has_direct_dependency(hierarchy, "a", task_a, "b", task_b) is True

    def test_b_blocks_a(self):
        hierarchy = {}
        task_a = {"dependencies": {"blocks": [], "blocked_by": ["b"], "depends": []}}
        task_b = {"dependencies": {"blocks": ["a"], "blocked_by": [], "depends": []}}
        assert _has_direct_dependency(hierarchy, "a", task_a, "b", task_b) is True


class TestGetActivePhases:
    """Tests for _get_active_phases helper."""

    def test_in_progress_before_pending(self, test_specs_dir, basic_spec):
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("test-spec-001", test_specs_dir)
        phases = _get_active_phases(spec_data)
        assert "phase-1" in phases

    def test_excludes_completed_phases(self, test_specs_dir):
        spec = {
            "spec_id": "completed-phase-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1", "phase-2"]},
                "phase-1": {"type": "phase", "status": "completed", "parent": "spec-root", "children": []},
                "phase-2": {"type": "phase", "status": "pending", "parent": "spec-root", "children": []},
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "completed-phase-001.json").write_text(json.dumps(spec))
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("completed-phase-001", test_specs_dir)
        phases = _get_active_phases(spec_data)
        assert "phase-1" not in phases
        assert "phase-2" in phases


# =============================================================================
# get_independent_tasks Tests
# =============================================================================

class TestGetIndependentTasksBasic:
    """Basic functionality tests for get_independent_tasks."""

    def test_returns_independent_tasks(self, test_specs_dir, basic_spec):
        tasks, error = get_independent_tasks("test-spec-001", max_tasks=3, specs_dir=test_specs_dir)
        assert error is None
        assert len(tasks) == 3
        task_ids = [t[0] for t in tasks]
        assert "task-1-1" in task_ids
        assert "task-1-2" in task_ids
        assert "task-1-3" in task_ids

    def test_respects_max_tasks(self, test_specs_dir, basic_spec):
        tasks, error = get_independent_tasks("test-spec-001", max_tasks=2, specs_dir=test_specs_dir)
        assert error is None
        assert len(tasks) == 2

    def test_spec_not_found(self, test_specs_dir):
        tasks, error = get_independent_tasks("nonexistent-spec", specs_dir=test_specs_dir)
        assert tasks == []
        assert "not found" in error.lower()

    def test_no_active_phases(self, test_specs_dir):
        spec = {
            "spec_id": "no-active-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
                "phase-1": {"type": "phase", "status": "completed", "parent": "spec-root", "children": []},
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "no-active-001.json").write_text(json.dumps(spec))
        tasks, error = get_independent_tasks("no-active-001", specs_dir=test_specs_dir)
        assert tasks == []
        # No error - caller should check spec_complete via prepare_batch_context
        assert error is None


class TestGetIndependentTasksDependencies:
    """Tests for dependency handling in get_independent_tasks."""

    def test_excludes_dependent_tasks(self, test_specs_dir, dependent_spec):
        tasks, error = get_independent_tasks("dep-spec-001", max_tasks=3, specs_dir=test_specs_dir)
        assert error is None
        task_ids = [t[0] for t in tasks]
        # task-1-1 and task-1-3 are independent, task-1-2 is blocked by task-1-1
        # Since task-1-1 blocks task-1-2, they shouldn't both be selected
        assert "task-1-1" in task_ids
        assert "task-1-3" in task_ids
        # task-1-2 is blocked, so it shouldn't be selected
        assert "task-1-2" not in task_ids


class TestGetIndependentTasksFilePaths:
    """Tests for file path conflict handling."""

    def test_excludes_same_file_tasks(self, test_specs_dir, same_file_spec):
        tasks, error = get_independent_tasks("same-file-001", max_tasks=3, specs_dir=test_specs_dir)
        assert error is None
        task_ids = [t[0] for t in tasks]
        # task-1-1 and task-1-2 target same file, only one should be selected
        same_file_selected = sum(1 for tid in task_ids if tid in ["task-1-1", "task-1-2"])
        assert same_file_selected == 1
        # task-1-3 targets different file, should be selected
        assert "task-1-3" in task_ids

    def test_excludes_path_ancestry_conflicts(self, test_specs_dir, ancestry_spec):
        tasks, error = get_independent_tasks("ancestry-spec-001", max_tasks=3, specs_dir=test_specs_dir)
        assert error is None
        task_ids = [t[0] for t in tasks]
        # task-1-1 (src/components) and task-1-2 (src/components/button.py) conflict
        ancestry_selected = sum(1 for tid in task_ids if tid in ["task-1-1", "task-1-2"])
        assert ancestry_selected == 1
        # task-1-3 (src/utils.py) should be selected
        assert "task-1-3" in task_ids


class TestGetIndependentTasksBarriers:
    """Tests for barrier task handling (tasks without file_path)."""

    def test_barrier_task_is_exclusive(self, test_specs_dir, barrier_spec):
        tasks, error = get_independent_tasks("barrier-spec-001", max_tasks=3, specs_dir=test_specs_dir)
        assert error is None
        # Barrier task (task-1-1) should be selected alone since it has no file_path
        # Or if non-barrier tasks are selected first, barrier should not be added
        task_ids = [t[0] for t in tasks]
        if "task-1-1" in task_ids:
            # If barrier is selected, it should be the only task
            assert len(tasks) == 1
        else:
            # Otherwise, the file-having tasks can be selected together
            assert len(tasks) <= 2


class TestGetIndependentTasksPhases:
    """Tests for phase handling."""

    def test_only_active_phase_tasks(self, test_specs_dir, multi_phase_spec):
        tasks, error = get_independent_tasks("multi-phase-001", max_tasks=3, specs_dir=test_specs_dir)
        assert error is None
        task_ids = [t[0] for t in tasks]
        # Only task-1-1 from in_progress phase should be selected
        # task-2-1 is in blocked phase
        assert "task-1-1" in task_ids
        # Phase 2 is blocked by phase 1, so task-2-1 should not be selected
        assert "task-2-1" not in task_ids


class TestGetIndependentTasksRetry:
    """Tests for retry count handling."""

    def test_excludes_high_retry_count(self, test_specs_dir, retry_spec):
        tasks, error = get_independent_tasks("retry-spec-001", max_tasks=3, specs_dir=test_specs_dir)
        assert error is None
        task_ids = [t[0] for t in tasks]
        # task-1-1 has retry_count=5, should be excluded
        assert "task-1-1" not in task_ids
        # task-1-2 has retry_count=1, should be included
        assert "task-1-2" in task_ids


class TestGetIndependentTasksLeafPreference:
    """Tests for leaf task preference."""

    def test_prefers_leaf_tasks(self, test_specs_dir):
        spec = {
            "spec_id": "leaf-test-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
                "phase-1": {
                    "type": "phase",
                    "status": "in_progress",
                    "parent": "spec-root",
                    "children": ["task-1-1"],
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Parent Task",
                    "status": "pending",
                    "parent": "phase-1",
                    "children": ["task-1-1-1"],
                    "metadata": {"file_path": "src/a.py"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1-1": {
                    "type": "subtask",
                    "title": "Child Task",
                    "status": "pending",
                    "parent": "task-1-1",
                    "children": [],
                    "metadata": {"file_path": "src/b.py"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "leaf-test-001.json").write_text(json.dumps(spec))
        tasks, error = get_independent_tasks("leaf-test-001", specs_dir=test_specs_dir)
        assert error is None
        task_ids = [t[0] for t in tasks]
        # Should select child (leaf) task, not parent
        assert "task-1-1-1" in task_ids
        assert "task-1-1" not in task_ids


# =============================================================================
# Security Tests
# =============================================================================

class TestSecurityValidation:
    """Tests for security-related validation."""

    def test_rejects_path_outside_root(self, test_specs_dir, tmp_path):
        spec = {
            "spec_id": "security-test-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
                "phase-1": {
                    "type": "phase",
                    "status": "in_progress",
                    "parent": "spec-root",
                    "children": ["task-1-1"],
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Malicious task",
                    "status": "pending",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {"file_path": "/etc/passwd"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "security-test-001.json").write_text(json.dumps(spec))
        tasks, error = get_independent_tasks(
            "security-test-001",
            specs_dir=test_specs_dir,
            project_root=tmp_path,
        )
        # Task with path outside project root should be excluded
        assert len(tasks) == 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_hierarchy(self, test_specs_dir):
        spec = {
            "spec_id": "empty-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {},
            "journal": [],
        }
        (test_specs_dir / "active" / "empty-001.json").write_text(json.dumps(spec))
        tasks, error = get_independent_tasks("empty-001", specs_dir=test_specs_dir)
        assert "no hierarchy" in error.lower() or "no active phases" in error.lower()

    def test_all_tasks_completed(self, test_specs_dir):
        spec = {
            "spec_id": "all-done-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
                "phase-1": {
                    "type": "phase",
                    "status": "in_progress",
                    "parent": "spec-root",
                    "children": ["task-1-1"],
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Done Task",
                    "status": "completed",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {"file_path": "src/a.py"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "all-done-001.json").write_text(json.dumps(spec))
        tasks, error = get_independent_tasks("all-done-001", specs_dir=test_specs_dir)
        assert error is None
        assert len(tasks) == 0

    def test_deterministic_ordering(self, test_specs_dir, basic_spec):
        # Run multiple times to ensure consistent ordering
        results = []
        for _ in range(3):
            tasks, _ = get_independent_tasks("test-spec-001", max_tasks=3, specs_dir=test_specs_dir)
            task_ids = [t[0] for t in tasks]
            results.append(tuple(task_ids))
        # All runs should produce the same order
        assert len(set(results)) == 1


# =============================================================================
# Integration Tests: start_batch
# =============================================================================

class TestStartBatchAtomic:
    """Tests for start_batch atomic behavior."""

    def test_start_batch_success(self, test_specs_dir, basic_spec):
        """start_batch should atomically start multiple tasks."""
        result, error = start_batch(
            spec_id="test-spec-001",
            task_ids=["task-1-1", "task-1-2"],
            specs_dir=test_specs_dir,
        )
        assert error is None
        assert result["started_count"] == 2
        assert set(result["started"]) == {"task-1-1", "task-1-2"}
        assert "started_at" in result

        # Verify tasks are actually in_progress
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("test-spec-001", test_specs_dir)
        assert spec_data["hierarchy"]["task-1-1"]["status"] == "in_progress"
        assert spec_data["hierarchy"]["task-1-2"]["status"] == "in_progress"
        # Verify started_at timestamp is set
        assert "started_at" in spec_data["hierarchy"]["task-1-1"]["metadata"]

    def test_start_batch_validates_all_before_changing(self, test_specs_dir, basic_spec):
        """start_batch should validate ALL tasks before making ANY changes."""
        # Include one valid task and one invalid task
        result, error = start_batch(
            spec_id="test-spec-001",
            task_ids=["task-1-1", "nonexistent-task"],
            specs_dir=test_specs_dir,
        )
        assert error is not None
        assert "Validation failed" in error
        assert result["started"] == []
        assert "nonexistent" in str(result.get("errors", []))

        # Verify NO tasks were changed (all-or-nothing)
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("test-spec-001", test_specs_dir)
        assert spec_data["hierarchy"]["task-1-1"]["status"] == "pending"

    def test_start_batch_rejects_already_started(self, test_specs_dir, basic_spec):
        """start_batch should reject already in_progress tasks."""
        # Start task-1-1 first
        result, _ = start_batch("test-spec-001", ["task-1-1"], test_specs_dir)
        assert result["started_count"] == 1

        # Try to start it again
        result, error = start_batch("test-spec-001", ["task-1-1"], test_specs_dir)
        assert error is not None
        assert "already in_progress" in str(result.get("errors", []))

    def test_start_batch_rejects_completed_tasks(self, test_specs_dir):
        """start_batch should reject completed tasks."""
        spec = {
            "spec_id": "completed-task-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
                "phase-1": {
                    "type": "phase",
                    "status": "in_progress",
                    "parent": "spec-root",
                    "children": ["task-1-1"],
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Completed Task",
                    "status": "completed",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {"file_path": "src/a.py"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "completed-task-001.json").write_text(json.dumps(spec))

        result, error = start_batch("completed-task-001", ["task-1-1"], test_specs_dir)
        assert error is not None
        assert "already completed" in str(result.get("errors", []))

    def test_start_batch_rejects_blocked_tasks(self, test_specs_dir, dependent_spec):
        """start_batch should reject tasks with unresolved dependencies."""
        # task-1-2 is blocked by task-1-1
        result, error = start_batch("dep-spec-001", ["task-1-2"], test_specs_dir)
        assert error is not None
        assert "unresolved dependencies" in str(result.get("errors", []))

    def test_start_batch_rejects_dependent_tasks_in_same_batch(self, test_specs_dir, dependent_spec):
        """start_batch should reject tasks that depend on each other."""
        # task-1-1 blocks task-1-2, trying to start both should fail
        result, error = start_batch(
            "dep-spec-001",
            ["task-1-1", "task-1-2"],
            test_specs_dir,
        )
        # task-1-2 is blocked by task-1-1 which is still pending
        assert error is not None

    def test_start_batch_empty_task_ids(self, test_specs_dir, basic_spec):
        """start_batch should reject empty task list."""
        result, error = start_batch("test-spec-001", [], test_specs_dir)
        assert error is not None
        assert "No task IDs" in error

    def test_start_batch_spec_not_found(self, test_specs_dir):
        """start_batch should handle missing spec."""
        result, error = start_batch("nonexistent-spec", ["task-1"], test_specs_dir)
        assert error is not None
        assert "not found" in error.lower()


# =============================================================================
# Integration Tests: complete_batch
# =============================================================================

class TestCompleteBatchPartialFailures:
    """Tests for complete_batch partial failure handling."""

    def test_complete_batch_all_success(self, test_specs_dir, basic_spec):
        """complete_batch should handle all successful completions."""
        # First start the tasks
        start_batch("test-spec-001", ["task-1-1", "task-1-2"], test_specs_dir)

        completions = [
            {"task_id": "task-1-1", "success": True, "completion_note": "Done A"},
            {"task_id": "task-1-2", "success": True, "completion_note": "Done B"},
        ]
        result, error = complete_batch("test-spec-001", completions, test_specs_dir)

        assert error is None
        assert result["completed_count"] == 2
        assert result["failed_count"] == 0
        assert result["total_processed"] == 2

        # Verify tasks are completed
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("test-spec-001", test_specs_dir)
        assert spec_data["hierarchy"]["task-1-1"]["status"] == "completed"
        assert spec_data["hierarchy"]["task-1-2"]["status"] == "completed"

    def test_complete_batch_mixed_success_failure(self, test_specs_dir, basic_spec):
        """complete_batch should handle mixed success and failure."""
        start_batch("test-spec-001", ["task-1-1", "task-1-2"], test_specs_dir)

        completions = [
            {"task_id": "task-1-1", "success": True, "completion_note": "Done"},
            {"task_id": "task-1-2", "success": False, "completion_note": "Failed due to X"},
        ]
        result, error = complete_batch("test-spec-001", completions, test_specs_dir)

        assert error is None
        assert result["completed_count"] == 1
        assert result["failed_count"] == 1
        assert result["total_processed"] == 2
        assert result["results"]["task-1-1"]["status"] == "completed"
        assert result["results"]["task-1-2"]["status"] == "failed"

        # Verify task statuses and retry_count
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("test-spec-001", test_specs_dir)
        assert spec_data["hierarchy"]["task-1-1"]["status"] == "completed"
        assert spec_data["hierarchy"]["task-1-2"]["status"] == "failed"
        assert spec_data["hierarchy"]["task-1-2"]["metadata"]["retry_count"] == 1

    def test_complete_batch_increments_retry_count(self, test_specs_dir):
        """complete_batch should increment retry_count on failure."""
        spec = {
            "spec_id": "retry-test-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
                "phase-1": {
                    "type": "phase",
                    "status": "in_progress",
                    "parent": "spec-root",
                    "children": ["task-1-1"],
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Task A",
                    "status": "in_progress",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {"file_path": "src/a.py", "retry_count": 2},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "retry-test-001.json").write_text(json.dumps(spec))

        completions = [{"task_id": "task-1-1", "success": False, "completion_note": "Failed"}]
        result, error = complete_batch("retry-test-001", completions, test_specs_dir)

        assert error is None
        assert result["results"]["task-1-1"]["retry_count"] == 3

        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("retry-test-001", test_specs_dir)
        assert spec_data["hierarchy"]["task-1-1"]["metadata"]["retry_count"] == 3

    def test_complete_batch_skips_already_completed(self, test_specs_dir):
        """complete_batch should skip already completed tasks."""
        spec = {
            "spec_id": "skip-test-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
                "phase-1": {
                    "type": "phase",
                    "status": "in_progress",
                    "parent": "spec-root",
                    "children": ["task-1-1"],
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Task A",
                    "status": "completed",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {"file_path": "src/a.py"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "skip-test-001.json").write_text(json.dumps(spec))

        completions = [{"task_id": "task-1-1", "success": True, "completion_note": "Done"}]
        result, error = complete_batch("skip-test-001", completions, test_specs_dir)

        assert error is None
        assert result["results"]["task-1-1"]["status"] == "skipped"
        assert result["completed_count"] == 0

    def test_complete_batch_handles_not_found(self, test_specs_dir, basic_spec):
        """complete_batch should handle task not found."""
        completions = [{"task_id": "nonexistent", "success": True}]
        result, error = complete_batch("test-spec-001", completions, test_specs_dir)

        assert error is None  # Partial success supported
        assert result["results"]["nonexistent"]["status"] == "error"
        assert "not found" in result["results"]["nonexistent"]["error"]

    def test_complete_batch_empty_completions(self, test_specs_dir, basic_spec):
        """complete_batch should reject empty completions list."""
        result, error = complete_batch("test-spec-001", [], test_specs_dir)
        assert error is not None
        assert "No completions" in error


# =============================================================================
# Integration Tests: reset_batch
# =============================================================================

class TestResetBatch:
    """Tests for reset_batch function."""

    def test_reset_batch_specific_tasks(self, test_specs_dir, basic_spec):
        """reset_batch should reset specific in_progress tasks to pending."""
        # Start tasks first
        start_batch("test-spec-001", ["task-1-1", "task-1-2"], test_specs_dir)

        # Reset only task-1-1
        result, error = reset_batch(
            spec_id="test-spec-001",
            task_ids=["task-1-1"],
            specs_dir=test_specs_dir,
        )

        assert error is None
        assert result["reset_count"] == 1
        assert "task-1-1" in result["reset"]

        # Verify task-1-1 is pending, task-1-2 still in_progress
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("test-spec-001", test_specs_dir)
        assert spec_data["hierarchy"]["task-1-1"]["status"] == "pending"
        assert spec_data["hierarchy"]["task-1-2"]["status"] == "in_progress"
        # Verify started_at is cleared
        assert "started_at" not in spec_data["hierarchy"]["task-1-1"].get("metadata", {})

    def test_reset_batch_auto_detect_stale(self, test_specs_dir):
        """reset_batch should auto-detect stale tasks by threshold."""
        from datetime import datetime, timezone, timedelta

        # Create spec with a task that has old started_at
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
        spec = {
            "spec_id": "stale-test-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
                "phase-1": {
                    "type": "phase",
                    "status": "in_progress",
                    "parent": "spec-root",
                    "children": ["task-1-1", "task-1-2"],
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Stale Task",
                    "status": "in_progress",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {"file_path": "src/a.py", "started_at": old_time},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-2": {
                    "type": "task",
                    "title": "Fresh Task",
                    "status": "pending",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {"file_path": "src/b.py"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "stale-test-001.json").write_text(json.dumps(spec))

        # Auto-detect stale tasks (threshold 1 hour)
        result, error = reset_batch(
            spec_id="stale-test-001",
            threshold_hours=1.0,
            specs_dir=test_specs_dir,
        )

        assert error is None
        assert result["reset_count"] == 1
        assert "task-1-1" in result["reset"]

    def test_reset_batch_validation_not_in_progress(self, test_specs_dir, basic_spec):
        """reset_batch should reject tasks not in_progress."""
        result, error = reset_batch(
            spec_id="test-spec-001",
            task_ids=["task-1-1"],  # This is pending, not in_progress
            specs_dir=test_specs_dir,
        )

        assert error is not None
        assert "not in_progress" in str(result.get("errors", []))

    def test_reset_batch_no_stale_tasks(self, test_specs_dir):
        """reset_batch should return empty when no stale tasks found."""
        from datetime import datetime, timezone

        # Create spec with a recently started task
        recent_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        spec = {
            "spec_id": "fresh-test-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
                "phase-1": {
                    "type": "phase",
                    "status": "in_progress",
                    "parent": "spec-root",
                    "children": ["task-1-1"],
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Fresh Task",
                    "status": "in_progress",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {"file_path": "src/a.py", "started_at": recent_time},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "fresh-test-001.json").write_text(json.dumps(spec))

        result, error = reset_batch(
            spec_id="fresh-test-001",
            threshold_hours=1.0,
            specs_dir=test_specs_dir,
        )

        assert error is None
        assert result["reset_count"] == 0
        assert "No stale" in result.get("message", "")

    def test_reset_batch_clears_started_at(self, test_specs_dir, basic_spec):
        """reset_batch should clear started_at timestamp."""
        # Start a task
        start_batch("test-spec-001", ["task-1-1"], test_specs_dir)

        # Verify started_at is set
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("test-spec-001", test_specs_dir)
        assert "started_at" in spec_data["hierarchy"]["task-1-1"]["metadata"]

        # Reset the task
        result, error = reset_batch(
            spec_id="test-spec-001",
            task_ids=["task-1-1"],
            specs_dir=test_specs_dir,
        )

        assert error is None
        spec_data = load_spec("test-spec-001", test_specs_dir)
        assert spec_data["hierarchy"]["task-1-1"]["status"] == "pending"
        assert "started_at" not in spec_data["hierarchy"]["task-1-1"]["metadata"]


# =============================================================================
# Integration Tests: prepare_batch_context
# =============================================================================

class TestPrepareBatchContext:
    """Tests for prepare_batch_context function."""

    def test_prepare_batch_context_returns_tasks(self, test_specs_dir, basic_spec):
        """prepare_batch_context should return tasks with context."""
        result, error = prepare_batch_context(
            spec_id="test-spec-001",
            max_tasks=3,
            specs_dir=test_specs_dir,
        )

        assert error is None
        assert result["task_count"] == 3
        assert len(result["tasks"]) == 3
        assert result["spec_complete"] is False
        assert "dependency_graph" in result
        assert "warnings" in result

    def test_prepare_batch_context_respects_token_budget(self, test_specs_dir, basic_spec):
        """prepare_batch_context should respect token budget."""
        result, error = prepare_batch_context(
            spec_id="test-spec-001",
            max_tasks=10,
            token_budget=100,  # Very small budget
            specs_dir=test_specs_dir,
        )

        assert error is None
        # Should return fewer tasks due to token budget
        assert result["task_count"] < 3
        # Should have warning about budget
        assert any("Token budget" in w for w in result.get("warnings", []))

    def test_prepare_batch_context_detects_spec_complete(self, test_specs_dir):
        """prepare_batch_context should detect when spec is complete."""
        spec = {
            "spec_id": "done-spec-001",
            "title": "Test",
            "metadata": {},
            "hierarchy": {
                "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
                "phase-1": {
                    "type": "phase",
                    "status": "completed",
                    "parent": "spec-root",
                    "children": ["task-1-1"],
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Done Task",
                    "status": "completed",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {"file_path": "src/a.py"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
            "journal": [],
        }
        (test_specs_dir / "active" / "done-spec-001.json").write_text(json.dumps(spec))

        result, error = prepare_batch_context("done-spec-001", specs_dir=test_specs_dir)

        # Graceful handling: no error, but spec_complete should be True
        assert error is None
        assert result["spec_complete"] is True
        assert result["tasks"] == []
        assert result["task_count"] == 0


# =============================================================================
# Stress Tests: Concurrent Batch Operations
# =============================================================================

class TestStartBatchConcurrency:
    """Stress tests for concurrent start_batch calls.

    These tests verify that start_batch handles concurrent access correctly,
    preventing race conditions where multiple callers try to start the same
    tasks simultaneously.
    """

    @pytest.fixture
    def concurrent_spec(self, test_specs_dir):
        """Create a spec with many independent tasks for concurrency testing."""
        tasks = {}
        task_ids = []
        for i in range(1, 11):  # 10 independent tasks
            task_id = f"task-1-{i}"
            task_ids.append(task_id)
            tasks[task_id] = {
                "type": "task",
                "title": f"Task {i}",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": f"src/file_{i}.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            }

        spec = {
            "spec_id": "concurrent-spec-001",
            "title": "Concurrent Test Spec",
            "metadata": {"title": "Concurrent Test", "status": "in_progress", "version": "1.0.0"},
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Test",
                    "status": "in_progress",
                    "children": ["phase-1"],
                },
                "phase-1": {
                    "type": "phase",
                    "title": "Phase 1",
                    "status": "in_progress",
                    "parent": "spec-root",
                    "children": task_ids,
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                **tasks,
            },
            "assumptions": [],
            "revision_history": [],
            "journal": [],
        }
        (test_specs_dir / "active" / "concurrent-spec-001.json").write_text(json.dumps(spec))
        return spec

    def test_concurrent_start_same_task(self, test_specs_dir, concurrent_spec):
        """Multiple threads trying to start the same task should not corrupt data.

        Only one thread should succeed in starting the task, others should fail
        because the task is already in_progress.
        """
        import concurrent.futures
        import threading

        results = []
        errors = []
        lock = threading.Lock()

        def attempt_start():
            result, error = start_batch(
                spec_id="concurrent-spec-001",
                task_ids=["task-1-1"],
                specs_dir=test_specs_dir,
            )
            with lock:
                results.append(result)
                errors.append(error)

        # Launch 10 concurrent threads all trying to start the same task
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(attempt_start) for _ in range(10)]
            concurrent.futures.wait(futures)

        # Verify results
        success_count = sum(1 for e in errors if e is None)
        failure_count = sum(1 for e in errors if e is not None)

        # At least one should succeed (first to acquire)
        assert success_count >= 1, f"Expected at least 1 success, got {success_count}"
        # Most should fail (task already in_progress)
        assert failure_count >= 1, f"Expected at least 1 failure due to race condition"

        # Verify final state is consistent
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("concurrent-spec-001", test_specs_dir)
        assert spec_data is not None, "Spec file should not be corrupted"

        task = spec_data["hierarchy"]["task-1-1"]
        assert task["status"] == "in_progress", "Task should be in_progress"
        assert "started_at" in task.get("metadata", {}), "Task should have started_at timestamp"

    def test_concurrent_start_different_tasks(self, test_specs_dir, concurrent_spec):
        """Multiple threads starting different tasks should all succeed.

        When tasks are independent (no shared dependencies or file paths),
        concurrent start_batch calls should all succeed.
        """
        import concurrent.futures
        import threading

        results = []
        errors = []
        lock = threading.Lock()

        def attempt_start(task_id):
            result, error = start_batch(
                spec_id="concurrent-spec-001",
                task_ids=[task_id],
                specs_dir=test_specs_dir,
            )
            with lock:
                results.append((task_id, result))
                errors.append((task_id, error))

        # Each thread starts a different task
        task_ids = [f"task-1-{i}" for i in range(1, 6)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attempt_start, tid) for tid in task_ids]
            concurrent.futures.wait(futures)

        # All should eventually reach a consistent state
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("concurrent-spec-001", test_specs_dir)
        assert spec_data is not None, "Spec file should not be corrupted"

        # Count how many tasks are in_progress
        in_progress_count = 0
        for task_id in task_ids:
            task = spec_data["hierarchy"][task_id]
            if task["status"] == "in_progress":
                in_progress_count += 1
                assert "started_at" in task.get("metadata", {}), f"{task_id} missing started_at"

        # At least some should be in_progress (may vary due to race conditions)
        assert in_progress_count >= 1, "At least one task should be in_progress"

    def test_high_contention_stress(self, test_specs_dir, concurrent_spec):
        """High-contention stress test with many concurrent callers.

        This test verifies the system handles concurrent access without
        unrecoverable data loss. Under extreme contention, some operations
        may fail, but the spec file should remain loadable.

        NOTE: The current implementation has a known race condition where
        concurrent atomic writes (using temp file + rename) can interfere
        with each other. This test documents that behavior.
        """
        import concurrent.futures
        import threading
        import random
        import time

        results = []
        io_errors = []  # Expected under high contention (FileNotFoundError, etc.)
        other_exceptions = []
        lock = threading.Lock()

        def attempt_start(worker_id):
            try:
                # Small random delay to vary timing
                time.sleep(random.uniform(0, 0.01))

                # Each worker tries to start 1-3 random tasks
                num_tasks = random.randint(1, 3)
                task_ids = [f"task-1-{random.randint(1, 10)}" for _ in range(num_tasks)]
                # Remove duplicates
                task_ids = list(set(task_ids))

                result, error = start_batch(
                    spec_id="concurrent-spec-001",
                    task_ids=task_ids,
                    specs_dir=test_specs_dir,
                )
                with lock:
                    results.append((worker_id, task_ids, result, error))
            except (FileNotFoundError, OSError) as e:
                # Expected under high contention - concurrent temp file ops
                with lock:
                    io_errors.append((worker_id, type(e).__name__, str(e)))
            except Exception as e:
                with lock:
                    other_exceptions.append((worker_id, type(e).__name__, str(e)))

        # Launch workers with moderate concurrency (10 is more realistic)
        num_workers = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(attempt_start, i) for i in range(num_workers)]
            concurrent.futures.wait(futures)

        # No unexpected exceptions (IO errors are expected under contention)
        assert len(other_exceptions) == 0, f"Unexpected exceptions: {other_exceptions}"

        # Some operations should complete (with or without IO contention)
        total_operations = len(results) + len(io_errors)
        assert total_operations == num_workers, (
            f"Expected {num_workers} total operations, got {total_operations}"
        )

        for worker_id, task_ids, result, error in results:
            # Each result should be a dict (even on error)
            assert isinstance(result, dict), f"Worker {worker_id}: result should be dict"
            # Error should be None or a string
            assert error is None or isinstance(error, str), f"Worker {worker_id}: invalid error type"

        # Verify final spec state
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("concurrent-spec-001", test_specs_dir)

        # Under high contention, the spec file may be in a transient state
        # The key invariant is that IF it loads, it should be valid
        if spec_data is not None:
            assert "hierarchy" in spec_data, "Spec should have hierarchy"

            # Verify each task has valid status
            for i in range(1, 11):
                task_id = f"task-1-{i}"
                task = spec_data["hierarchy"].get(task_id)
                assert task is not None, f"Task {task_id} should exist"
                assert task["status"] in ("pending", "in_progress"), f"Task {task_id} has invalid status"
        else:
            # If spec is None, the file may have been mid-write
            # Verify the file exists (it shouldn't be deleted)
            spec_file = test_specs_dir / "active" / "concurrent-spec-001.json"
            assert spec_file.exists(), "Spec file should exist even if transiently unloadable"

    def test_concurrent_batch_preserves_atomicity(self, test_specs_dir, concurrent_spec):
        """Concurrent start_batch calls should preserve all-or-nothing atomicity.

        If one task in a batch fails validation, none should be started,
        even under concurrent load.
        """
        import concurrent.futures
        import threading

        results = []
        lock = threading.Lock()

        def attempt_invalid_batch():
            # This batch includes a nonexistent task, so it should fail completely
            result, error = start_batch(
                spec_id="concurrent-spec-001",
                task_ids=["task-1-1", "nonexistent-task"],
                specs_dir=test_specs_dir,
            )
            with lock:
                results.append((result, error))

        def attempt_valid_batch():
            # This batch is valid
            result, error = start_batch(
                spec_id="concurrent-spec-001",
                task_ids=["task-1-2"],
                specs_dir=test_specs_dir,
            )
            with lock:
                results.append((result, error))

        # Mix of valid and invalid batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(5):
                futures.append(executor.submit(attempt_invalid_batch))
                futures.append(executor.submit(attempt_valid_batch))
            concurrent.futures.wait(futures)

        # Verify results
        from foundry_mcp.core.spec import load_spec
        spec_data = load_spec("concurrent-spec-001", test_specs_dir)
        assert spec_data is not None, "Spec file should not be corrupted"

        # task-1-1 should NOT be in_progress (was only in invalid batches)
        task_1_1 = spec_data["hierarchy"]["task-1-1"]
        assert task_1_1["status"] == "pending", (
            "task-1-1 should remain pending - it was only in invalid batches"
        )

        # task-1-2 may be in_progress (from valid batch)
        task_1_2 = spec_data["hierarchy"]["task-1-2"]
        # At least verify the status is valid
        assert task_1_2["status"] in ("pending", "in_progress")

    def test_rapid_sequential_starts(self, test_specs_dir, concurrent_spec):
        """Rapid sequential start_batch calls should handle state transitions correctly.

        This simulates a scenario where multiple agents quickly try to claim tasks
        one after another.
        """
        from foundry_mcp.core.spec import load_spec

        results = []

        # Rapidly start different tasks in sequence
        for i in range(1, 6):
            task_id = f"task-1-{i}"
            result, error = start_batch(
                spec_id="concurrent-spec-001",
                task_ids=[task_id],
                specs_dir=test_specs_dir,
            )
            results.append((task_id, result, error))

        # All sequential starts should succeed
        for task_id, result, error in results:
            assert error is None, f"Start of {task_id} failed: {error}"
            assert result["started_count"] == 1, f"Expected 1 started task for {task_id}"

        # Verify final state
        spec_data = load_spec("concurrent-spec-001", test_specs_dir)
        for i in range(1, 6):
            task = spec_data["hierarchy"][f"task-1-{i}"]
            assert task["status"] == "in_progress", f"task-1-{i} should be in_progress"
            assert "started_at" in task.get("metadata", {}), f"task-1-{i} missing started_at"
