"""Tests for check_spec_completeness and detect_duplicate_tasks functions."""

import json
import tempfile
from pathlib import Path

import pytest

from foundry_mcp.core.spec import check_spec_completeness, detect_duplicate_tasks


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        specs_dir = (Path(tmpdir) / "specs").resolve()
        for folder in ["pending", "active", "completed", "archived"]:
            (specs_dir / folder).mkdir(parents=True)
        yield specs_dir


@pytest.fixture
def complete_spec():
    """A spec with all fields filled in."""
    return {
        "spec_id": "complete-spec-001",
        "metadata": {"title": "Complete Spec"},
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Complete Spec",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Implementation Phase",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Implement feature A",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "description": "Implement the main feature",
                    "estimated_hours": 2.0,
                    "task_category": "implementation",
                    "file_path": "src/feature.py",
                },
            },
            "task-1-2": {
                "type": "task",
                "title": "Implement feature B",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "description": "Implement secondary feature",
                    "estimated_hours": 1.5,
                    "task_category": "implementation",
                    "file_path": "src/feature_b.py",
                },
            },
        },
        "journal": [],
    }


@pytest.fixture
def incomplete_spec():
    """A spec with missing fields."""
    return {
        "spec_id": "incomplete-spec-001",
        "metadata": {"title": "Incomplete Spec"},
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Incomplete Spec",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "",  # Empty title
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Task without description",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    # No description
                    "task_category": "implementation",
                    # No file_path
                },
            },
            "task-1-2": {
                "type": "task",
                "title": "Task without estimate",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "description": "Has description but no estimate",
                    # No estimated_hours
                },
            },
        },
        "journal": [],
    }


@pytest.fixture
def duplicate_spec():
    """A spec with duplicate tasks."""
    return {
        "spec_id": "duplicate-spec-001",
        "metadata": {"title": "Duplicate Spec"},
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Duplicate Spec",
                "status": "in_progress",
                "children": ["phase-1", "phase-2"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
            },
            "phase-2": {
                "type": "phase",
                "title": "Phase 2",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-2-1", "verify-2-1"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Implement user authentication",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"description": "Add login functionality"},
            },
            "task-1-2": {
                "type": "task",
                "title": "Add user authentication",  # Similar to task-1-1
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"description": "Implement login system"},
            },
            "task-2-1": {
                "type": "task",
                "title": "Completely different task",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
                "metadata": {"description": "Something unrelated"},
            },
            "verify-2-1": {
                "type": "verify",
                "title": "Run tests",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
                "metadata": {},
            },
        },
        "journal": [],
    }


class TestCheckSpecCompleteness:
    """Tests for check_spec_completeness function."""

    def test_complete_spec_scores_100(self, temp_specs_dir, complete_spec):
        """A spec with all fields should score 100."""
        spec_file = temp_specs_dir / "active" / "complete-spec-001.json"
        spec_file.write_text(json.dumps(complete_spec))

        result, error = check_spec_completeness(
            "complete-spec-001", specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["completeness_score"] == 100
        assert result["issue_count"] == 0

    def test_incomplete_spec_flags_issues(self, temp_specs_dir, incomplete_spec):
        """A spec with missing fields should flag issues."""
        spec_file = temp_specs_dir / "active" / "incomplete-spec-001.json"
        spec_file.write_text(json.dumps(incomplete_spec))

        result, error = check_spec_completeness(
            "incomplete-spec-001", specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["completeness_score"] < 100
        assert result["issue_count"] > 0

        # Check specific issues are flagged
        issues = result["issues"]
        categories = {i["category"] for i in issues}
        assert "titles" in categories  # Empty phase title
        assert "descriptions" in categories  # Missing description
        assert "file_paths" in categories  # Missing file_path for impl task
        assert "estimates" in categories  # Missing estimated_hours

    def test_category_scores(self, temp_specs_dir, incomplete_spec):
        """Category scores should reflect completion."""
        spec_file = temp_specs_dir / "active" / "incomplete-spec-001.json"
        spec_file.write_text(json.dumps(incomplete_spec))

        result, error = check_spec_completeness(
            "incomplete-spec-001", specs_dir=temp_specs_dir
        )

        assert error is None
        categories = result["categories"]

        # Titles: 2/3 complete (phase-1 empty)
        assert categories["titles"]["total"] == 3
        assert categories["titles"]["complete"] == 2

        # Descriptions: 1/2 tasks have descriptions
        assert categories["descriptions"]["total"] == 2
        assert categories["descriptions"]["complete"] == 1

    def test_spec_not_found(self, temp_specs_dir):
        """Should return error for non-existent spec."""
        result, error = check_spec_completeness(
            "nonexistent-spec", specs_dir=temp_specs_dir
        )

        assert result is None
        assert "not found" in error.lower()

    def test_empty_hierarchy(self, temp_specs_dir):
        """Spec with empty hierarchy should score 100."""
        spec = {"spec_id": "empty-001", "metadata": {}, "hierarchy": {}}
        spec_file = temp_specs_dir / "active" / "empty-001.json"
        spec_file.write_text(json.dumps(spec))

        result, error = check_spec_completeness("empty-001", specs_dir=temp_specs_dir)

        assert error is None
        assert result["completeness_score"] == 100
        assert result["message"] == "No hierarchy nodes to check"


class TestDetectDuplicateTasks:
    """Tests for detect_duplicate_tasks function."""

    def test_finds_similar_titles(self, temp_specs_dir, duplicate_spec):
        """Should find tasks with similar titles."""
        spec_file = temp_specs_dir / "active" / "duplicate-spec-001.json"
        spec_file.write_text(json.dumps(duplicate_spec))

        result, error = detect_duplicate_tasks(
            "duplicate-spec-001",
            scope="titles",
            threshold=0.6,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["duplicate_count"] > 0

        # task-1-1 and task-1-2 should be similar
        node_pairs = [(d["node_a"], d["node_b"]) for d in result["duplicates"]]
        found_auth_pair = any(
            ("task-1-1" in pair and "task-1-2" in pair) for pair in node_pairs
        )
        assert found_auth_pair

    def test_threshold_filtering(self, temp_specs_dir, duplicate_spec):
        """Higher threshold should find fewer duplicates."""
        spec_file = temp_specs_dir / "active" / "duplicate-spec-001.json"
        spec_file.write_text(json.dumps(duplicate_spec))

        result_low, _ = detect_duplicate_tasks(
            "duplicate-spec-001",
            threshold=0.5,
            specs_dir=temp_specs_dir,
        )

        result_high, _ = detect_duplicate_tasks(
            "duplicate-spec-001",
            threshold=0.95,
            specs_dir=temp_specs_dir,
        )

        assert result_low["duplicate_count"] >= result_high["duplicate_count"]

    def test_scope_descriptions(self, temp_specs_dir, duplicate_spec):
        """Should compare descriptions when scope is descriptions."""
        spec_file = temp_specs_dir / "active" / "duplicate-spec-001.json"
        spec_file.write_text(json.dumps(duplicate_spec))

        result, error = detect_duplicate_tasks(
            "duplicate-spec-001",
            scope="descriptions",
            threshold=0.5,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["scope"] == "descriptions"

    def test_scope_both(self, temp_specs_dir, duplicate_spec):
        """Scope 'both' should check titles and descriptions."""
        spec_file = temp_specs_dir / "active" / "duplicate-spec-001.json"
        spec_file.write_text(json.dumps(duplicate_spec))

        result, error = detect_duplicate_tasks(
            "duplicate-spec-001",
            scope="both",
            threshold=0.6,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["scope"] == "both"

    def test_max_pairs_truncation(self, temp_specs_dir, duplicate_spec):
        """Should truncate results when max_pairs is exceeded."""
        spec_file = temp_specs_dir / "active" / "duplicate-spec-001.json"
        spec_file.write_text(json.dumps(duplicate_spec))

        result, error = detect_duplicate_tasks(
            "duplicate-spec-001",
            threshold=0.0,  # Match everything
            max_pairs=2,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["duplicate_count"] <= 2
        assert result.get("truncated") is True
        assert "warnings" in result

    def test_invalid_scope(self, temp_specs_dir, duplicate_spec):
        """Should return error for invalid scope."""
        spec_file = temp_specs_dir / "active" / "duplicate-spec-001.json"
        spec_file.write_text(json.dumps(duplicate_spec))

        result, error = detect_duplicate_tasks(
            "duplicate-spec-001",
            scope="invalid",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "scope must be" in error

    def test_invalid_threshold(self, temp_specs_dir, duplicate_spec):
        """Should return error for invalid threshold."""
        spec_file = temp_specs_dir / "active" / "duplicate-spec-001.json"
        spec_file.write_text(json.dumps(duplicate_spec))

        result, error = detect_duplicate_tasks(
            "duplicate-spec-001",
            threshold=1.5,  # Invalid
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "threshold" in error

    def test_spec_not_found(self, temp_specs_dir):
        """Should return error for non-existent spec."""
        result, error = detect_duplicate_tasks(
            "nonexistent-spec", specs_dir=temp_specs_dir
        )

        assert result is None
        assert "not found" in error.lower()

    def test_no_duplicates_high_threshold(self, temp_specs_dir, complete_spec):
        """Spec with distinct tasks should have no duplicates at high threshold."""
        spec_file = temp_specs_dir / "active" / "complete-spec-001.json"
        spec_file.write_text(json.dumps(complete_spec))

        result, error = detect_duplicate_tasks(
            "complete-spec-001",
            threshold=0.95,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["duplicate_count"] == 0

    def test_result_structure(self, temp_specs_dir, duplicate_spec):
        """Result should have all expected fields."""
        spec_file = temp_specs_dir / "active" / "duplicate-spec-001.json"
        spec_file.write_text(json.dumps(duplicate_spec))

        result, error = detect_duplicate_tasks(
            "duplicate-spec-001", specs_dir=temp_specs_dir
        )

        assert error is None
        assert "spec_id" in result
        assert "duplicates" in result
        assert "duplicate_count" in result
        assert "scope" in result
        assert "threshold" in result
        assert "nodes_checked" in result
        assert "pairs_compared" in result
