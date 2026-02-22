"""Tests for documentation_helpers including fidelity-verify filtering, limit removal, and new context builders."""

from pathlib import Path

from foundry_mcp.tools.unified.documentation_helpers import (
    _build_implementation_artifacts,
    _build_journal_entries,
    _build_plan_context,
    _build_spec_overview,
    _build_spec_requirements,
    _build_subsequent_phases,
    _build_test_results,
    _is_fidelity_verify_node,
)

# ---------------------------------------------------------------------------
# Fixture: spec_data with a mix of regular and fidelity-verify nodes
# ---------------------------------------------------------------------------


def _make_spec_data():
    """Build a minimal spec with one phase containing a regular task and a fidelity-verify node."""
    return {
        "title": "Test Spec",
        "hierarchy": {
            "phase-1": {
                "id": "phase-1",
                "type": "phase",
                "title": "Phase One",
                "status": "in_progress",
                "metadata": {
                    "description": "Initial implementation of core features",
                    "purpose": "Establish the foundation module",
                },
                "children": ["task-1", "verify-fidelity-1"],
            },
            "task-1": {
                "id": "task-1",
                "type": "task",
                "title": "Implement feature",
                "status": "in_progress",
                "metadata": {
                    "description": "Create the greet() function in feature module",
                    "task_category": "implementation",
                    "file_path": "src/feature.py",
                    "details": ["Add the feature"],
                },
            },
            "verify-fidelity-1": {
                "id": "verify-fidelity-1",
                "type": "verify",
                "title": "Fidelity review",
                "status": "pending",
                "metadata": {
                    "verification_type": "fidelity",
                    "file_path": "src/feature.py",
                },
            },
        },
        "journal": [
            {
                "task_id": "task-1",
                "title": "Implemented feature",
                "entry_type": "progress",
                "timestamp": "2025-01-01T00:00:00Z",
                "content": "Done",
            },
            {
                "task_id": "verify-fidelity-1",
                "title": "Fidelity verification started",
                "entry_type": "verify",
                "timestamp": "2025-01-02T00:00:00Z",
                "content": "Running fidelity check",
            },
            {
                "title": "Phase-level test results",
                "entry_type": "note",
                "timestamp": "2025-01-03T00:00:00Z",
                "content": "All tests passed",
                "metadata": {"phase_id": "phase-1"},
            },
        ],
    }


def _make_multi_phase_spec():
    """Build a spec with 3 phases and a spec-root for subsequent-phase testing."""
    return {
        "title": "Multi-Phase Spec",
        "description": "A spec with multiple phases",
        "mission": "Deliver the project",
        "category": "feature",
        "complexity": "high",
        "status": "in_progress",
        "progress": {"percentage": 33},
        "objectives": ["Build phase 1", "Build phase 2"],
        "assumptions": [
            {"text": "Python 3.10+"},
            "No external deps",
        ],
        "hierarchy": {
            "spec-root": {
                "id": "spec-root",
                "type": "root",
                "children": ["phase-1", "phase-2", "phase-3"],
            },
            "phase-1": {
                "id": "phase-1",
                "type": "phase",
                "title": "Foundation",
                "status": "completed",
                "children": ["task-1a"],
            },
            "phase-2": {
                "id": "phase-2",
                "type": "phase",
                "title": "Core Features",
                "status": "in_progress",
                "description": "Build the core",
                "children": ["task-2a", "task-2b"],
            },
            "phase-3": {
                "id": "phase-3",
                "type": "phase",
                "title": "Polish",
                "status": "pending",
                "purpose": "Final polish and docs",
                "children": ["task-3a"],
            },
            "task-1a": {
                "id": "task-1a",
                "type": "task",
                "title": "Setup project",
                "status": "completed",
                "metadata": {"file_path": "setup.py"},
            },
            "task-2a": {
                "id": "task-2a",
                "type": "task",
                "title": "Implement API",
                "status": "in_progress",
                "metadata": {"file_path": "src/api.py"},
            },
            "task-2b": {
                "id": "task-2b",
                "type": "task",
                "title": "Implement CLI",
                "status": "pending",
                "metadata": {"file_path": "src/cli.py"},
            },
            "task-3a": {
                "id": "task-3a",
                "type": "task",
                "title": "Write docs",
                "status": "pending",
                "metadata": {"file_path": "docs/README.md"},
            },
        },
        "journal": [
            {
                "task_id": "task-1a",
                "title": "Setup test results",
                "entry_type": "note",
                "timestamp": "2025-01-01T00:00:00Z",
                "content": "Setup done",
            },
            {
                "task_id": "task-2a",
                "title": "API test verification",
                "entry_type": "note",
                "timestamp": "2025-01-02T00:00:00Z",
                "content": "API tests pass",
            },
            {
                "task_id": "task-2a",
                "title": "API implementation note",
                "entry_type": "progress",
                "timestamp": "2025-01-03T00:00:00Z",
                "content": "Endpoints added",
            },
        ],
    }


# ---------------------------------------------------------------------------
# _is_fidelity_verify_node
# ---------------------------------------------------------------------------


class TestIsFidelityVerifyNode:
    def test_positive_match(self):
        node = {"type": "verify", "metadata": {"verification_type": "fidelity"}}
        assert _is_fidelity_verify_node(node) is True

    def test_wrong_type(self):
        node = {"type": "task", "metadata": {"verification_type": "fidelity"}}
        assert _is_fidelity_verify_node(node) is False

    def test_wrong_verification_type(self):
        node = {"type": "verify", "metadata": {"verification_type": "manual"}}
        assert _is_fidelity_verify_node(node) is False

    def test_no_metadata(self):
        node = {"type": "verify"}
        assert _is_fidelity_verify_node(node) is False

    def test_empty_node(self):
        assert _is_fidelity_verify_node({}) is False

    def test_verify_without_verification_type(self):
        node = {"type": "verify", "metadata": {}}
        assert _is_fidelity_verify_node(node) is False


# ---------------------------------------------------------------------------
# _build_spec_requirements
# ---------------------------------------------------------------------------


class TestBuildSpecRequirements:
    def test_phase_includes_all_by_default(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, None, "phase-1")
        assert "task-1" in result
        assert "verify-fidelity-1" in result

    def test_phase_excludes_fidelity_verify(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, None, "phase-1", exclude_fidelity_verify=True)
        assert "task-1" in result
        assert "verify-fidelity-1" not in result

    def test_task_id_fidelity_node_excluded(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, "verify-fidelity-1", None, exclude_fidelity_verify=True)
        assert "excluded from review context" in result

    def test_task_id_fidelity_node_included_by_default(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, "verify-fidelity-1", None)
        assert "Fidelity review" in result

    def test_regular_task_unaffected_by_flag(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, "task-1", None, exclude_fidelity_verify=True)
        assert "Implement feature" in result

    def test_task_scope_includes_description(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, "task-1", None)
        assert "Create the greet() function in feature module" in result

    def test_task_scope_no_description_is_graceful(self):
        spec = _make_spec_data()
        del spec["hierarchy"]["task-1"]["metadata"]["description"]
        result = _build_spec_requirements(spec, "task-1", None)
        assert "Implement feature" in result
        assert "Description" not in result

    def test_phase_scope_includes_child_description(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, None, "phase-1")
        assert "Description: Create the greet() function" in result

    def test_phase_scope_includes_child_details(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, None, "phase-1")
        assert "Detail: Add the feature" in result

    def test_phase_scope_includes_child_file_path(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, None, "phase-1")
        assert "File: src/feature.py" in result

    def test_phase_scope_missing_metadata_is_graceful(self):
        spec = _make_spec_data()
        spec["hierarchy"]["task-1"]["metadata"] = {}
        spec["hierarchy"]["verify-fidelity-1"]["metadata"] = {}
        del spec["hierarchy"]["phase-1"]["metadata"]
        result = _build_spec_requirements(spec, None, "phase-1")
        assert "task-1" in result
        assert "Description:" not in result
        assert "Detail:" not in result
        assert "File:" not in result
        assert "Category:" not in result
        assert "Purpose:" not in result

    def test_task_scope_includes_category(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, "task-1", None)
        assert "Category:** implementation" in result

    def test_task_scope_no_category_is_graceful(self):
        spec = _make_spec_data()
        del spec["hierarchy"]["task-1"]["metadata"]["task_category"]
        result = _build_spec_requirements(spec, "task-1", None)
        assert "Category" not in result

    def test_phase_scope_includes_child_category(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, None, "phase-1")
        assert "Category: implementation" in result

    def test_phase_scope_includes_phase_description(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, None, "phase-1")
        assert "Initial implementation of core features" in result

    def test_phase_scope_includes_phase_purpose(self):
        spec = _make_spec_data()
        result = _build_spec_requirements(spec, None, "phase-1")
        assert "Establish the foundation module" in result

    def test_phase_scope_no_phase_metadata_is_graceful(self):
        spec = _make_spec_data()
        del spec["hierarchy"]["phase-1"]["metadata"]
        result = _build_spec_requirements(spec, None, "phase-1")
        assert "Phase One" in result
        assert "Purpose:" not in result

    def test_all_assumptions_shown(self):
        """Verify that all assumptions are included (no [:5] cap)."""
        spec = {
            "title": "Big Spec",
            "assumptions": [f"Assumption {i}" for i in range(10)],
            "hierarchy": {},
        }
        result = _build_spec_requirements(spec, None, None)
        for i in range(10):
            assert f"Assumption {i}" in result


# ---------------------------------------------------------------------------
# _build_implementation_artifacts
# ---------------------------------------------------------------------------


class TestBuildImplementationArtifacts:
    def test_phase_includes_all_file_paths_by_default(self):
        spec = _make_spec_data()
        result = _build_implementation_artifacts(spec, None, "phase-1", None, False, "main")
        # Both nodes reference src/feature.py — should appear
        assert "src/feature.py" in result

    def test_phase_excludes_fidelity_verify_file_paths(self):
        """When fidelity-verify node has a unique file, it should be excluded."""
        spec = _make_spec_data()
        # Give the fidelity node a different file so we can distinguish
        spec["hierarchy"]["verify-fidelity-1"]["metadata"]["file_path"] = "src/fidelity_only.py"
        result = _build_implementation_artifacts(
            spec, None, "phase-1", None, False, "main", exclude_fidelity_verify=True
        )
        assert "src/feature.py" in result
        assert "src/fidelity_only.py" not in result

    def test_full_spec_excludes_fidelity_verify(self):
        spec = _make_spec_data()
        spec["hierarchy"]["verify-fidelity-1"]["metadata"]["file_path"] = "src/fidelity_only.py"
        result = _build_implementation_artifacts(spec, None, None, None, False, "main", exclude_fidelity_verify=True)
        assert "src/feature.py" in result
        assert "src/fidelity_only.py" not in result

    def test_outputs_path_listing_not_file_contents(self):
        """Artifacts should list paths with markers, not inline file contents."""
        spec = _make_spec_data()
        result = _build_implementation_artifacts(spec, None, "phase-1", None, False, "main")
        # Should be a bullet list, not a code block
        assert "```" not in result
        assert "- [" in result
        assert "`src/feature.py`" in result

    def test_no_file_cap(self):
        """All files should be listed, not just the first 5."""
        spec = {
            "title": "Many Files",
            "hierarchy": {
                "phase-1": {
                    "id": "phase-1",
                    "type": "phase",
                    "title": "Phase",
                    "children": [f"task-{i}" for i in range(10)],
                },
                **{
                    f"task-{i}": {
                        "id": f"task-{i}",
                        "type": "task",
                        "title": f"Task {i}",
                        "metadata": {"file_path": f"src/file_{i}.py"},
                    }
                    for i in range(10)
                },
            },
        }
        result = _build_implementation_artifacts(spec, None, "phase-1", None, False, "main")
        for i in range(10):
            assert f"src/file_{i}.py" in result


# ---------------------------------------------------------------------------
# _build_test_results
# ---------------------------------------------------------------------------


class TestBuildTestResults:
    def test_includes_fidelity_verify_entries_by_default(self):
        spec = _make_spec_data()
        # The fidelity journal entry has "verification" in title
        result = _build_test_results(spec, None, "phase-1")
        assert "Fidelity verification" in result

    def test_excludes_fidelity_verify_entries(self):
        spec = _make_spec_data()
        result = _build_test_results(spec, None, "phase-1", exclude_fidelity_verify=True)
        assert "Fidelity verification" not in result

    def test_scopes_to_phase_tasks(self):
        """When phase_id is set, only entries from phase tasks are included."""
        spec = _make_multi_phase_spec()
        result = _build_test_results(spec, None, "phase-2")
        assert "API test verification" in result
        # phase-1 entry should not be included
        assert "Setup test" not in result

    def test_scopes_to_task(self):
        """When task_id is set, only entries from that task are included."""
        spec = _make_multi_phase_spec()
        result = _build_test_results(spec, "task-2a", None)
        assert "API test verification" in result
        assert "Setup test" not in result

    def test_no_limit_on_entries(self):
        """All matching entries should be returned, not just last 3."""
        spec = {
            "title": "Many Tests",
            "hierarchy": {},
            "journal": [
                {
                    "task_id": f"t-{i}",
                    "title": f"Test run {i} verification",
                    "entry_type": "note",
                    "timestamp": f"2025-01-{i+1:02d}T00:00:00Z",
                    "content": f"Result {i}",
                }
                for i in range(10)
            ],
        }
        result = _build_test_results(spec, None, None)
        for i in range(10):
            assert f"Test run {i} verification" in result

    def test_no_content_truncation(self):
        """Content should not be truncated at 500 chars."""
        long_content = "x" * 1000
        spec = {
            "title": "Long Content",
            "hierarchy": {},
            "journal": [
                {
                    "title": "Test verification",
                    "entry_type": "note",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "content": long_content,
                }
            ],
        }
        result = _build_test_results(spec, None, None)
        assert long_content in result
        assert "..." not in result


# ---------------------------------------------------------------------------
# _build_journal_entries
# ---------------------------------------------------------------------------


class TestBuildJournalEntries:
    def test_phase_includes_all_by_default(self):
        spec = _make_spec_data()
        result = _build_journal_entries(spec, None, "phase-1")
        assert "Implemented feature" in result
        assert "Fidelity verification" in result

    def test_phase_excludes_fidelity_verify(self):
        spec = _make_spec_data()
        result = _build_journal_entries(spec, None, "phase-1", exclude_fidelity_verify=True)
        assert "Implemented feature" in result
        assert "Fidelity verification" not in result

    def test_phase_keeps_phase_level_entries(self):
        spec = _make_spec_data()
        result = _build_journal_entries(spec, None, "phase-1", exclude_fidelity_verify=True)
        assert "Phase-level test results" in result

    def test_task_id_fidelity_node_returns_empty(self):
        spec = _make_spec_data()
        result = _build_journal_entries(spec, "verify-fidelity-1", None, exclude_fidelity_verify=True)
        assert result == "*No journal entries found*"

    def test_task_id_fidelity_node_included_by_default(self):
        spec = _make_spec_data()
        result = _build_journal_entries(spec, "verify-fidelity-1", None)
        assert "Fidelity verification" in result

    def test_regular_task_unaffected_by_flag(self):
        spec = _make_spec_data()
        result = _build_journal_entries(spec, "task-1", None, exclude_fidelity_verify=True)
        assert "Implemented feature" in result

    def test_no_limit_on_entries(self):
        """All journal entries should be returned, not just last 5."""
        spec = {
            "title": "Many Entries",
            "hierarchy": {},
            "journal": [
                {
                    "title": f"Entry {i}",
                    "entry_type": "note",
                    "timestamp": f"2025-01-{i+1:02d}T00:00:00Z",
                    "content": f"Content {i}",
                }
                for i in range(10)
            ],
        }
        result = _build_journal_entries(spec, None, None)
        assert "10 journal entries found" in result
        for i in range(10):
            assert f"Entry {i}" in result

    def test_no_content_truncation(self):
        """Content should not be truncated at 500 chars."""
        long_content = "y" * 1000
        spec = {
            "title": "Long Content",
            "hierarchy": {},
            "journal": [
                {
                    "title": "Big entry",
                    "entry_type": "note",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "content": long_content,
                }
            ],
        }
        result = _build_journal_entries(spec, None, None)
        assert long_content in result
        assert "..." not in result


# ---------------------------------------------------------------------------
# _build_spec_overview
# ---------------------------------------------------------------------------


class TestBuildSpecOverview:
    def test_basic_fields(self):
        spec = _make_multi_phase_spec()
        result = _build_spec_overview(spec)
        assert "Multi-Phase Spec" in result
        assert "A spec with multiple phases" in result
        assert "Deliver the project" in result
        assert "feature" in result
        assert "high" in result
        assert "in_progress" in result
        assert "33%" in result

    def test_objectives(self):
        spec = _make_multi_phase_spec()
        result = _build_spec_overview(spec)
        assert "Build phase 1" in result
        assert "Build phase 2" in result

    def test_assumptions(self):
        spec = _make_multi_phase_spec()
        result = _build_spec_overview(spec)
        assert "Python 3.10+" in result
        assert "No external deps" in result

    def test_metadata_fallback(self):
        """Should fall back to metadata dict for fields not at top level."""
        spec = {
            "title": "Top Title",
            "metadata": {
                "mission": "Meta mission",
                "category": "refactor",
            },
            "hierarchy": {},
        }
        result = _build_spec_overview(spec)
        assert "Top Title" in result
        assert "Meta mission" in result
        assert "refactor" in result

    def test_empty_spec(self):
        result = _build_spec_overview({"hierarchy": {}})
        assert "Unknown" in result


# ---------------------------------------------------------------------------
# _build_subsequent_phases
# ---------------------------------------------------------------------------


class TestBuildSubsequentPhases:
    def test_lists_phases_after_current(self):
        spec = _make_multi_phase_spec()
        result = _build_subsequent_phases(spec, "phase-1")
        assert "not yet expected to be implemented" in result
        assert "Core Features" in result
        assert "Polish" in result
        assert "Implement API" in result
        assert "Write docs" in result

    def test_last_phase_returns_empty(self):
        spec = _make_multi_phase_spec()
        result = _build_subsequent_phases(spec, "phase-3")
        assert result == ""

    def test_middle_phase_only_shows_later(self):
        spec = _make_multi_phase_spec()
        result = _build_subsequent_phases(spec, "phase-2")
        assert "Polish" in result
        assert "Foundation" not in result
        assert "Core Features" not in result

    def test_no_phase_id_returns_empty(self):
        spec = _make_multi_phase_spec()
        result = _build_subsequent_phases(spec, None)
        assert result == ""

    def test_unknown_phase_id_returns_empty(self):
        spec = _make_multi_phase_spec()
        result = _build_subsequent_phases(spec, "nonexistent")
        assert result == ""

    def test_no_spec_root_returns_empty(self):
        spec = {"hierarchy": {"phase-1": {"type": "phase", "title": "P1"}}}
        result = _build_subsequent_phases(spec, "phase-1")
        assert result == ""

    def test_includes_description_and_purpose(self):
        spec = _make_multi_phase_spec()
        result = _build_subsequent_phases(spec, "phase-1")
        # phase-2 has "description", phase-3 has "purpose"
        assert "Build the core" in result
        assert "Final polish and docs" in result

    def test_includes_task_count(self):
        spec = _make_multi_phase_spec()
        result = _build_subsequent_phases(spec, "phase-1")
        assert "Tasks (2)" in result  # phase-2 has 2 tasks
        assert "Tasks (1)" in result  # phase-3 has 1 task


# ---------------------------------------------------------------------------
# _build_spec_overview — plan linkage fields
# ---------------------------------------------------------------------------


class TestBuildSpecOverviewPlanLinkage:
    def test_includes_plan_path_when_present(self):
        spec = {
            "title": "Linked Spec",
            "metadata": {"plan_path": ".plans/my-feature.md"},
            "hierarchy": {},
        }
        result = _build_spec_overview(spec)
        assert "Plan:" in result
        assert ".plans/my-feature.md" in result

    def test_includes_plan_review_path_when_present(self):
        spec = {
            "title": "Linked Spec",
            "metadata": {"plan_review_path": ".plan-reviews/my-feature-review-full.md"},
            "hierarchy": {},
        }
        result = _build_spec_overview(spec)
        assert "Plan Review:" in result
        assert ".plan-reviews/my-feature-review-full.md" in result

    def test_omits_plan_path_when_absent(self):
        spec = {"title": "No Plan", "metadata": {}, "hierarchy": {}}
        result = _build_spec_overview(spec)
        assert "Plan:" not in result
        assert "Plan Review:" not in result

    def test_omits_plan_path_when_empty(self):
        spec = {
            "title": "Empty Plan",
            "metadata": {"plan_path": "", "plan_review_path": "  "},
            "hierarchy": {},
        }
        result = _build_spec_overview(spec)
        assert "Plan:" not in result
        assert "Plan Review:" not in result


# ---------------------------------------------------------------------------
# _build_plan_context
# ---------------------------------------------------------------------------


class TestBuildPlanContext:
    def test_returns_content_when_plan_file_exists(self, tmp_path):
        plan_file = tmp_path / ".plans" / "feature.md"
        plan_file.parent.mkdir(parents=True)
        plan_file.write_text("# My Plan\n\nBuild the thing.\n")
        spec = {"metadata": {"plan_path": ".plans/feature.md"}}
        result = _build_plan_context(spec, tmp_path)
        assert "### Original Plan" in result
        assert "Build the thing." in result

    def test_returns_empty_when_plan_file_missing(self, tmp_path):
        spec = {"metadata": {"plan_path": ".plans/nonexistent.md"}}
        result = _build_plan_context(spec, tmp_path)
        assert result == ""

    def test_returns_empty_when_plan_path_absent(self):
        spec = {"metadata": {}}
        result = _build_plan_context(spec, Path("/some/root"))
        assert result == ""

    def test_returns_empty_when_plan_path_is_empty_string(self):
        spec = {"metadata": {"plan_path": ""}}
        result = _build_plan_context(spec, Path("/some/root"))
        assert result == ""

    def test_returns_empty_when_workspace_root_is_none(self):
        spec = {"metadata": {"plan_path": ".plans/feature.md"}}
        result = _build_plan_context(spec, None)
        assert result == ""

    def test_returns_empty_when_no_metadata(self):
        spec = {}
        result = _build_plan_context(spec, Path("/some/root"))
        assert result == ""

    def test_returns_empty_when_metadata_not_dict(self):
        spec = {"metadata": "not a dict"}
        result = _build_plan_context(spec, Path("/some/root"))
        assert result == ""

    def test_returns_empty_for_empty_file(self, tmp_path):
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("")
        spec = {"metadata": {"plan_path": "plan.md"}}
        result = _build_plan_context(spec, tmp_path)
        assert result == ""
