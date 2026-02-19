"""Tests for exclude_fidelity_verify filtering in documentation_helpers."""

from foundry_mcp.tools.unified.documentation_helpers import (
    _build_implementation_artifacts,
    _build_journal_entries,
    _build_spec_requirements,
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
                "children": ["task-1", "verify-fidelity-1"],
            },
            "task-1": {
                "id": "task-1",
                "type": "task",
                "title": "Implement feature",
                "status": "in_progress",
                "metadata": {
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

    def test_no_filtering_without_phase(self):
        """Without a phase_id, exclude flag has no effect on the broad journal."""
        spec = _make_spec_data()
        result = _build_test_results(spec, None, None, exclude_fidelity_verify=True)
        # Fidelity entry still present since no phase scoping
        assert "Fidelity verification" in result


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
