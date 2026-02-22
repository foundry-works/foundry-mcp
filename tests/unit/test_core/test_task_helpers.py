"""Tests for shared task helper utilities."""

from foundry_mcp.core.task._helpers import check_all_blocked


def test_check_all_blocked_phase_only_unblocked_task_returns_false():
    spec_data = {
        "phases": [
            {
                "id": "phase-1",
                "tasks": [
                    {"id": "task-1", "type": "task", "status": "pending"},
                ],
            },
        ],
    }

    assert check_all_blocked(spec_data) is False


def test_check_all_blocked_phase_only_all_pending_tasks_blocked_returns_true():
    spec_data = {
        "phases": [
            {
                "id": "phase-1",
                "tasks": [
                    {"id": "task-1", "type": "task", "status": "pending", "depends": ["task-2"]},
                    {"id": "task-2", "type": "task", "status": "pending", "depends": ["task-1"]},
                ],
            },
        ],
    }

    assert check_all_blocked(spec_data) is True


def test_check_all_blocked_phase_only_no_pending_tasks_returns_false():
    spec_data = {
        "phases": [
            {
                "id": "phase-1",
                "tasks": [
                    {"id": "task-1", "type": "task", "status": "completed"},
                ],
            },
        ],
    }

    assert check_all_blocked(spec_data) is False


def test_check_all_blocked_uses_hierarchy_when_pending_nodes_exist():
    spec_data = {
        "hierarchy": {
            "spec-root": {"type": "spec", "status": "in_progress", "children": ["phase-1"]},
            "phase-1": {
                "type": "phase",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1"],
                "dependencies": {"blocked_by": []},
            },
            "task-1": {
                "type": "task",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "dependencies": {"blocked_by": []},
            },
        },
        "phases": [],
    }

    assert check_all_blocked(spec_data) is False
