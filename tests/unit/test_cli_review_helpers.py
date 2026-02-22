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
    output = review._build_spec_requirements(keyed_spec_data, task_id=None, phase_id="phase-1")

    assert "Phase: Phase 1" in output
    assert "task-1-1" in output
    assert "task-1-2" in output


def test_build_implementation_artifacts_reads_phase_files(keyed_spec_data, tmp_path, monkeypatch):
    """_build_implementation_artifacts should list child file paths (not contents)."""
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
    assert "src/task_two.py" in output
    # File paths are listed with existence markers, not file contents
    assert "[+]" in output  # task_one.py exists
    assert "[-]" in output  # task_two.py does not exist


def test_build_implementation_artifacts_full_spec_review(keyed_spec_data, tmp_path, monkeypatch):
    """_build_implementation_artifacts should list all task files for full spec review."""
    project_root = tmp_path / "project"
    (project_root / "src").mkdir(parents=True)
    (project_root / "src" / "task_one.py").write_text("# task one", encoding="utf-8")
    (project_root / "src" / "task_two.py").write_text("# task two", encoding="utf-8")

    monkeypatch.chdir(project_root)

    # Full spec review: no task_id, no phase_id, no files
    output = review._build_implementation_artifacts(
        keyed_spec_data,
        task_id=None,
        phase_id=None,
        files=None,
        incremental=False,
        base_branch="main",
    )

    # Should list paths from all tasks (not contents)
    assert "src/task_one.py" in output
    assert "src/task_two.py" in output
    assert "[+]" in output  # Both files exist


def test_build_implementation_artifacts_full_spec_review_no_artifacts():
    """_build_implementation_artifacts returns placeholder when spec has no file_path metadata."""
    spec_data = {
        "spec_id": "spec-empty",
        "title": "Empty Spec",
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Empty Spec",
                "status": "pending",
                "parent": None,
                "children": ["task-1"],
            },
            "task-1": {
                "type": "task",
                "title": "Task without file_path",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "metadata": {},  # No file_path
            },
        },
    }

    output = review._build_implementation_artifacts(
        spec_data,
        task_id=None,
        phase_id=None,
        files=None,
        incremental=False,
        base_branch="main",
    )

    assert "*No implementation artifacts available*" in output
