"""Unit tests for CLI review helper functions."""

import pytest

from foundry_mcp.cli.commands import review


@pytest.fixture
def keyed_spec_data() -> dict:
    """Return spec data that uses the keyed hierarchy format."""
    return {
        "spec_id": "spec-keyed",
        "title": "Keyed Hierarchy Spec",
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Keyed Hierarchy Spec",
                "status": "in_progress",
                "parent": None,
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "completed",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "file_path": "src/task_one.py",
                    "details": ["Implements feature one"],
                },
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 2",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "file_path": "src/task_two.py",
                    "details": ["Adds follow-up validation"],
                },
            },
        },
    }


def test_build_spec_requirements_lists_children(keyed_spec_data):
    """_build_spec_requirements should include child tasks for keyed hierarchy phases."""
    output = review._build_spec_requirements(
        keyed_spec_data, task_id=None, phase_id="phase-1"
    )

    assert "Phase: Phase 1" in output
    assert "task-1-1" in output
    assert "task-1-2" in output


def test_build_implementation_artifacts_reads_phase_files(
    keyed_spec_data, tmp_path, monkeypatch
):
    """_build_implementation_artifacts should resolve child file paths by ID."""
    project_root = tmp_path / "project"
    (project_root / "src").mkdir(parents=True)
    file_path = project_root / "src" / "task_one.py"
    file_path.write_text("print('cli')", encoding="utf-8")

    monkeypatch.chdir(project_root)

    output = review._build_implementation_artifacts(
        keyed_spec_data,
        task_id=None,
        phase_id="phase-1",
        files=None,
        incremental=False,
        base_branch="main",
    )

    assert "src/task_one.py" in output
    assert "print('cli')" in output
