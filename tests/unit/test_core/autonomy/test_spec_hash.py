"""Tests for spec structure hashing and diff computation.

Covers:
- Structural hash stability (same input -> same hash)
- Structural hash ignores non-structural changes (titles, descriptions, metadata)
- Structural diff computation (added/removed phases/tasks)
- SpecFileMetadata retrieval (mtime optimization skip)
"""

import json
from pathlib import Path

import pytest

from foundry_mcp.core.autonomy.spec_hash import (
    SpecFileMetadata,
    StructuralDiff,
    compute_spec_structure_hash,
    compute_structural_diff,
    get_spec_file_metadata,
)


# =============================================================================
# Structural Hash Stability
# =============================================================================


class TestComputeSpecStructureHash:
    """Test compute_spec_structure_hash determinism and correctness."""

    def test_same_input_produces_same_hash(self):
        spec = {
            "phases": [
                {"id": "p1", "sequence_index": 0, "tasks": [{"id": "t1"}, {"id": "t2"}]},
                {"id": "p2", "sequence_index": 1, "tasks": [{"id": "t3"}]},
            ]
        }
        h1 = compute_spec_structure_hash(spec)
        h2 = compute_spec_structure_hash(spec)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_structures_produce_different_hashes(self):
        spec_a = {"phases": [{"id": "p1", "tasks": [{"id": "t1"}]}]}
        spec_b = {"phases": [{"id": "p1", "tasks": [{"id": "t1"}, {"id": "t2"}]}]}
        assert compute_spec_structure_hash(spec_a) != compute_spec_structure_hash(spec_b)

    def test_ignores_title_changes(self):
        spec_a = {"phases": [{"id": "p1", "title": "Old Title", "tasks": [{"id": "t1"}]}]}
        spec_b = {"phases": [{"id": "p1", "title": "New Title", "tasks": [{"id": "t1"}]}]}
        assert compute_spec_structure_hash(spec_a) == compute_spec_structure_hash(spec_b)

    def test_ignores_description_changes(self):
        spec_a = {
            "phases": [
                {
                    "id": "p1",
                    "tasks": [{"id": "t1", "description": "Do stuff"}],
                }
            ]
        }
        spec_b = {
            "phases": [
                {
                    "id": "p1",
                    "tasks": [{"id": "t1", "description": "Do different stuff"}],
                }
            ]
        }
        assert compute_spec_structure_hash(spec_a) == compute_spec_structure_hash(spec_b)

    def test_ignores_metadata_changes(self):
        spec_a = {
            "phases": [
                {
                    "id": "p1",
                    "tasks": [{"id": "t1", "metadata": {"foo": "bar"}}],
                }
            ]
        }
        spec_b = {
            "phases": [
                {
                    "id": "p1",
                    "tasks": [{"id": "t1", "metadata": {"baz": "qux"}}],
                }
            ]
        }
        assert compute_spec_structure_hash(spec_a) == compute_spec_structure_hash(spec_b)

    def test_sensitive_to_phase_id_changes(self):
        spec_a = {"phases": [{"id": "phase-alpha", "tasks": [{"id": "t1"}]}]}
        spec_b = {"phases": [{"id": "phase-beta", "tasks": [{"id": "t1"}]}]}
        assert compute_spec_structure_hash(spec_a) != compute_spec_structure_hash(spec_b)

    def test_sensitive_to_task_phase_mapping_changes(self):
        """Moving a task from one phase to another changes the hash."""
        spec_a = {
            "phases": [
                {"id": "p1", "tasks": [{"id": "t1"}]},
                {"id": "p2", "tasks": []},
            ]
        }
        spec_b = {
            "phases": [
                {"id": "p1", "tasks": []},
                {"id": "p2", "tasks": [{"id": "t1"}]},
            ]
        }
        assert compute_spec_structure_hash(spec_a) != compute_spec_structure_hash(spec_b)

    def test_sensitive_to_sequence_index_changes(self):
        spec_a = {"phases": [{"id": "p1", "sequence_index": 0, "tasks": []}]}
        spec_b = {"phases": [{"id": "p1", "sequence_index": 1, "tasks": []}]}
        assert compute_spec_structure_hash(spec_a) != compute_spec_structure_hash(spec_b)

    def test_empty_spec_produces_stable_hash(self):
        h = compute_spec_structure_hash({})
        assert len(h) == 64
        assert h == compute_spec_structure_hash({"phases": []})

    def test_handles_malformed_phases(self):
        """Non-dict items in phases are skipped gracefully."""
        spec = {"phases": [None, "invalid", {"id": "p1", "tasks": [{"id": "t1"}]}]}
        h = compute_spec_structure_hash(spec)
        assert len(h) == 64

    def test_handles_tasks_not_a_list(self):
        spec = {"phases": [{"id": "p1", "tasks": "not a list"}]}
        h = compute_spec_structure_hash(spec)
        assert len(h) == 64

    def test_skips_tasks_without_id(self):
        spec_a = {"phases": [{"id": "p1", "tasks": [{"id": "t1"}, {}]}]}
        spec_b = {"phases": [{"id": "p1", "tasks": [{"id": "t1"}]}]}
        assert compute_spec_structure_hash(spec_a) == compute_spec_structure_hash(spec_b)

    def test_order_independent_for_phase_ids(self):
        """Phase ordering is by sorted ID, so insertion order doesn't matter."""
        spec_a = {
            "phases": [
                {"id": "alpha", "tasks": []},
                {"id": "beta", "tasks": []},
            ]
        }
        spec_b = {
            "phases": [
                {"id": "beta", "tasks": []},
                {"id": "alpha", "tasks": []},
            ]
        }
        assert compute_spec_structure_hash(spec_a) == compute_spec_structure_hash(spec_b)


# =============================================================================
# Structural Diff
# =============================================================================


class TestComputeStructuralDiff:
    """Test compute_structural_diff for human-readable comparisons."""

    def test_no_changes(self):
        spec = {"phases": [{"id": "p1", "tasks": [{"id": "t1"}]}]}
        diff = compute_structural_diff(spec, spec)
        assert not diff.has_changes
        assert diff.added_phases == []
        assert diff.removed_phases == []
        assert diff.added_tasks == []
        assert diff.removed_tasks == []

    def test_added_phase(self):
        old = {"phases": [{"id": "p1", "tasks": []}]}
        new = {"phases": [{"id": "p1", "tasks": []}, {"id": "p2", "tasks": []}]}
        diff = compute_structural_diff(old, new)
        assert diff.has_changes
        assert diff.added_phases == ["p2"]
        assert diff.removed_phases == []

    def test_removed_phase(self):
        old = {"phases": [{"id": "p1", "tasks": []}, {"id": "p2", "tasks": []}]}
        new = {"phases": [{"id": "p1", "tasks": []}]}
        diff = compute_structural_diff(old, new)
        assert diff.has_changes
        assert diff.removed_phases == ["p2"]

    def test_added_task(self):
        old = {"phases": [{"id": "p1", "tasks": [{"id": "t1"}]}]}
        new = {"phases": [{"id": "p1", "tasks": [{"id": "t1"}, {"id": "t2"}]}]}
        diff = compute_structural_diff(old, new)
        assert diff.has_changes
        assert diff.added_tasks == ["t2"]
        assert diff.removed_tasks == []

    def test_removed_task(self):
        old = {"phases": [{"id": "p1", "tasks": [{"id": "t1"}, {"id": "t2"}]}]}
        new = {"phases": [{"id": "p1", "tasks": [{"id": "t1"}]}]}
        diff = compute_structural_diff(old, new)
        assert diff.has_changes
        assert diff.removed_tasks == ["t2"]

    def test_combined_changes(self):
        old = {
            "phases": [
                {"id": "p1", "tasks": [{"id": "t1"}, {"id": "t2"}]},
                {"id": "p2", "tasks": [{"id": "t3"}]},
            ]
        }
        new = {
            "phases": [
                {"id": "p1", "tasks": [{"id": "t1"}]},
                {"id": "p3", "tasks": [{"id": "t4"}]},
            ]
        }
        diff = compute_structural_diff(old, new)
        assert diff.has_changes
        assert diff.added_phases == ["p3"]
        assert diff.removed_phases == ["p2"]
        assert diff.added_tasks == ["t4"]
        assert sorted(diff.removed_tasks) == ["t2", "t3"]

    def test_to_dict(self):
        diff = StructuralDiff(
            added_phases=["p3"],
            removed_phases=["p2"],
            added_tasks=["t4"],
            removed_tasks=["t2"],
        )
        d = diff.to_dict()
        assert d["added_phases"] == ["p3"]
        assert d["removed_tasks"] == ["t2"]

    def test_empty_specs(self):
        diff = compute_structural_diff({}, {})
        assert not diff.has_changes


# =============================================================================
# File Metadata (mtime optimization)
# =============================================================================


class TestGetSpecFileMetadata:
    """Test get_spec_file_metadata for mtime-based optimization."""

    def test_returns_metadata_for_existing_file(self, tmp_path):
        spec_file = tmp_path / "spec.json"
        content = json.dumps({"phases": []})
        spec_file.write_text(content)

        metadata = get_spec_file_metadata(spec_file)
        assert metadata is not None
        assert isinstance(metadata, SpecFileMetadata)
        assert metadata.file_size == len(content)
        assert metadata.mtime > 0

    def test_returns_none_for_missing_file(self, tmp_path):
        spec_file = tmp_path / "nonexistent.json"
        metadata = get_spec_file_metadata(spec_file)
        assert metadata is None

    def test_mtime_changes_on_write(self, tmp_path):
        import time

        spec_file = tmp_path / "spec.json"
        spec_file.write_text('{"phases": []}')
        m1 = get_spec_file_metadata(spec_file)

        # Ensure filesystem timestamp granularity
        time.sleep(0.05)
        spec_file.write_text('{"phases": [{"id": "p1", "tasks": []}]}')
        m2 = get_spec_file_metadata(spec_file)

        assert m2.file_size > m1.file_size
        assert m2.mtime >= m1.mtime

    def test_same_content_same_size(self, tmp_path):
        spec_file = tmp_path / "spec.json"
        content = '{"phases": []}'
        spec_file.write_text(content)

        m1 = get_spec_file_metadata(spec_file)
        m2 = get_spec_file_metadata(spec_file)

        assert m1.file_size == m2.file_size
        assert m1.mtime == m2.mtime
