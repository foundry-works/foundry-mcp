"""Tests for spec_adapter: hierarchy → phases-view conversion."""

from __future__ import annotations

from foundry_mcp.core.autonomy.spec_adapter import ensure_phases_view

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_hierarchy_spec(
    *,
    spec_id: str = "test-spec-001",
    phases: list | None = None,
) -> dict:
    """Build a hierarchy-format spec (production format)."""
    if phases is None:
        phases = [
            {
                "id": "phase-1",
                "title": "Phase 1",
                "children": ["task-1-1", "verify-1-1"],
                "metadata": {"purpose": "testing"},
            },
        ]

    hierarchy: dict = {
        "spec-root": {
            "type": "spec",
            "title": "Test Spec",
            "status": "pending",
            "parent": None,
            "children": [p["id"] for p in phases],
            "total_tasks": 0,
            "completed_tasks": 0,
            "metadata": {},
            "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
        },
    }

    for seq_idx, phase in enumerate(phases):
        phase_id = phase["id"]
        children = phase.get("children", [])
        hierarchy[phase_id] = {
            "type": "phase",
            "title": phase.get("title", f"Phase {seq_idx + 1}"),
            "status": "pending",
            "parent": "spec-root",
            "children": children,
            "total_tasks": len(children),
            "completed_tasks": 0,
            "metadata": phase.get("metadata", {}),
            "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
        }

        # Auto-create task/verify leaf nodes
        for child_id in children:
            node_type = "verify" if child_id.startswith("verify-") else "task"
            hierarchy[child_id] = {
                "type": node_type,
                "title": f"{node_type.title()} {child_id}",
                "status": "pending",
                "parent": phase_id,
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            }

    return {"spec_id": spec_id, "hierarchy": hierarchy}


def _make_legacy_spec() -> dict:
    """Build a legacy phases-array spec (test fixture format)."""
    return {
        "spec_id": "legacy-001",
        "phases": [
            {
                "id": "phase-1",
                "title": "Phase 1",
                "sequence_index": 0,
                "tasks": [
                    {"id": "task-1", "title": "Task 1", "type": "task", "status": "pending"},
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests: pass-through for legacy / empty specs
# ---------------------------------------------------------------------------


class TestPassThrough:
    def test_empty_dict(self):
        assert ensure_phases_view({}) == {}

    def test_none_safe(self):
        # Should not crash; returns falsy input unchanged
        assert ensure_phases_view(None) is None  # type: ignore[arg-type]

    def test_legacy_phases_preserved(self):
        spec = _make_legacy_spec()
        original_phases = spec["phases"]
        result = ensure_phases_view(spec)
        assert result["phases"] is original_phases  # same object, not rebuilt

    def test_no_hierarchy_no_phases(self):
        spec = {"spec_id": "bare"}
        result = ensure_phases_view(spec)
        assert "phases" not in result  # nothing to build from


# ---------------------------------------------------------------------------
# Tests: hierarchy → phases conversion
# ---------------------------------------------------------------------------


class TestHierarchyConversion:
    def test_single_phase_with_tasks(self):
        spec = _make_hierarchy_spec()
        result = ensure_phases_view(spec)

        phases = result["phases"]
        assert len(phases) == 1

        phase = phases[0]
        assert phase["id"] == "phase-1"
        assert phase["title"] == "Phase 1"
        assert phase["sequence_index"] == 0
        assert phase["metadata"] == {"purpose": "testing"}

        tasks = phase["tasks"]
        assert len(tasks) == 2

        task_ids = [t["id"] for t in tasks]
        assert "task-1-1" in task_ids
        assert "verify-1-1" in task_ids

        # Check types are mapped correctly
        task_map = {t["id"]: t for t in tasks}
        assert task_map["task-1-1"]["type"] == "task"
        assert task_map["verify-1-1"]["type"] == "verify"

    def test_multi_phase(self):
        spec = _make_hierarchy_spec(
            phases=[
                {"id": "phase-1", "title": "Core", "children": ["task-1-1"]},
                {"id": "phase-2", "title": "Polish", "children": ["task-2-1", "verify-2-1"]},
            ],
        )
        result = ensure_phases_view(spec)

        phases = result["phases"]
        assert len(phases) == 2
        assert phases[0]["id"] == "phase-1"
        assert phases[0]["sequence_index"] == 0
        assert len(phases[0]["tasks"]) == 1
        assert phases[1]["id"] == "phase-2"
        assert phases[1]["sequence_index"] == 1
        assert len(phases[1]["tasks"]) == 2

    def test_task_status_preserved(self):
        spec = _make_hierarchy_spec()
        # Mark the task as completed in the hierarchy
        spec["hierarchy"]["task-1-1"]["status"] = "completed"
        result = ensure_phases_view(spec)

        task = [t for t in result["phases"][0]["tasks"] if t["id"] == "task-1-1"][0]
        assert task["status"] == "completed"

    def test_dependencies_mapped(self):
        spec = _make_hierarchy_spec()
        # Add blocked_by dependency
        spec["hierarchy"]["verify-1-1"]["dependencies"]["blocked_by"] = ["task-1-1"]
        result = ensure_phases_view(spec)

        verify = [t for t in result["phases"][0]["tasks"] if t["id"] == "verify-1-1"][0]
        assert verify["depends"] == ["task-1-1"]
        assert verify["dependencies"] == ["task-1-1"]

    def test_hierarchy_preserved(self):
        """Original hierarchy key should not be removed."""
        spec = _make_hierarchy_spec()
        result = ensure_phases_view(spec)
        assert "hierarchy" in result
        assert "spec-root" in result["hierarchy"]

    def test_idempotent(self):
        """Calling ensure_phases_view twice should not change the result."""
        spec = _make_hierarchy_spec()
        first = ensure_phases_view(spec)
        second = ensure_phases_view(first)
        # After first call, phases exist so second call returns as-is
        assert first["phases"] is second["phases"]

    def test_empty_phase(self):
        spec = _make_hierarchy_spec(
            phases=[{"id": "phase-1", "title": "Empty", "children": []}],
        )
        result = ensure_phases_view(spec)
        assert len(result["phases"]) == 1
        assert result["phases"][0]["tasks"] == []


# ---------------------------------------------------------------------------
# Tests: nested subtasks
# ---------------------------------------------------------------------------


class TestNestedSubtasks:
    def test_subtask_collected_as_leaf(self):
        """Subtask children of a parent task should be collected, not the parent."""
        spec = _make_hierarchy_spec(
            phases=[
                {"id": "phase-1", "title": "Phase 1", "children": ["task-1-1"]},
            ],
        )
        # Give task-1-1 subtask children
        spec["hierarchy"]["task-1-1"]["children"] = ["task-1-1-1", "task-1-1-2"]
        # Create subtask nodes
        for subtask_id in ["task-1-1-1", "task-1-1-2"]:
            spec["hierarchy"][subtask_id] = {
                "type": "subtask",
                "title": f"Subtask {subtask_id}",
                "status": "pending",
                "parent": "task-1-1",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            }

        result = ensure_phases_view(spec)
        tasks = result["phases"][0]["tasks"]

        # Should have 2 subtask leaves, not the parent task-1-1
        assert len(tasks) == 2
        task_ids = {t["id"] for t in tasks}
        assert task_ids == {"task-1-1-1", "task-1-1-2"}

        # Subtasks should be mapped to type "task"
        for t in tasks:
            assert t["type"] == "task"

    def test_group_nodes_recursed(self):
        """Group nodes should be transparent — their children are collected."""
        spec = _make_hierarchy_spec(
            phases=[
                {"id": "phase-1", "title": "Phase 1", "children": ["group-1"]},
            ],
        )
        # Replace auto-created group-1 with actual group
        spec["hierarchy"]["group-1"] = {
            "type": "group",
            "title": "Feature Group",
            "status": "pending",
            "parent": "phase-1",
            "children": ["task-g-1", "task-g-2"],
            "total_tasks": 2,
            "completed_tasks": 0,
            "metadata": {},
            "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
        }
        for task_id in ["task-g-1", "task-g-2"]:
            spec["hierarchy"][task_id] = {
                "type": "task",
                "title": f"Task {task_id}",
                "status": "pending",
                "parent": "group-1",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            }

        result = ensure_phases_view(spec)
        tasks = result["phases"][0]["tasks"]
        assert len(tasks) == 2
        task_ids = {t["id"] for t in tasks}
        assert task_ids == {"task-g-1", "task-g-2"}


# ---------------------------------------------------------------------------
# Tests: spec_structure_hash integration
# ---------------------------------------------------------------------------


class TestHashIntegration:
    def test_hierarchy_spec_produces_valid_hash(self):
        """A hierarchy spec should produce the same hash as an equivalent phases spec."""
        from foundry_mcp.core.autonomy.spec_hash import compute_spec_structure_hash

        hierarchy_spec = _make_hierarchy_spec(
            spec_id="hash-test",
            phases=[
                {"id": "phase-1", "title": "Phase 1", "children": ["task-1-1"]},
            ],
        )
        converted = ensure_phases_view(hierarchy_spec)

        # Build equivalent legacy spec
        legacy_spec = {
            "spec_id": "hash-test",
            "phases": [
                {
                    "id": "phase-1",
                    "title": "Phase 1",
                    "sequence_index": 0,
                    "tasks": [
                        {"id": "task-1-1", "title": "Task task-1-1", "type": "task", "status": "pending"},
                    ],
                },
            ],
        }

        hash_from_hierarchy = compute_spec_structure_hash(converted)
        hash_from_legacy = compute_spec_structure_hash(legacy_spec)
        assert hash_from_hierarchy == hash_from_legacy

    def test_non_empty_hash(self):
        from foundry_mcp.core.autonomy.spec_hash import compute_spec_structure_hash

        spec = ensure_phases_view(_make_hierarchy_spec())
        h = compute_spec_structure_hash(spec)
        assert len(h) == 64  # SHA-256 hex digest
