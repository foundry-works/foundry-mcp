"""Tests for core spec operations."""

import json
import tempfile
from pathlib import Path

import pytest

from foundry_mcp.core.spec import (
    find_specs_directory,
    find_spec_file,
    load_spec,
    save_spec,
    list_specs,
    get_node,
    update_node,
    add_revision,
    update_frontmatter,
)


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Resolve to handle macOS /var -> /private/var symlink
        specs_dir = (Path(tmpdir) / "specs").resolve()

        # Create status directories
        (specs_dir / "pending").mkdir(parents=True)
        (specs_dir / "active").mkdir(parents=True)
        (specs_dir / "completed").mkdir(parents=True)
        (specs_dir / "archived").mkdir(parents=True)

        yield specs_dir


@pytest.fixture
def sample_spec():
    """Create a sample spec data structure."""
    return {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "metadata": {
            "title": "Test Specification",
            "version": "1.0.0",
        },
        "hierarchy": {
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "pending",
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "pending",
                "parent": "phase-1",
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 2",
                "status": "completed",
                "parent": "phase-1",
            },
        }
    }


class TestFindSpecsDirectory:
    """Tests for find_specs_directory function."""

    def test_find_specs_directory_with_explicit_path(self, temp_specs_dir):
        """Should find specs directory when given explicit path."""
        result = find_specs_directory(str(temp_specs_dir))
        assert result == temp_specs_dir

    def test_find_specs_directory_from_parent(self, temp_specs_dir):
        """Should find specs directory from parent path."""
        parent = temp_specs_dir.parent
        result = find_specs_directory(str(parent))
        assert result == temp_specs_dir


class TestFindSpecFile:
    """Tests for find_spec_file function."""

    def test_find_spec_in_active(self, temp_specs_dir, sample_spec):
        """Should find spec in active folder."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result = find_spec_file("test-spec-001", temp_specs_dir)
        assert result == spec_file

    def test_find_spec_in_pending(self, temp_specs_dir, sample_spec):
        """Should find spec in pending folder."""
        spec_file = temp_specs_dir / "pending" / "test-spec-002.json"
        sample_spec["spec_id"] = "test-spec-002"
        spec_file.write_text(json.dumps(sample_spec))

        result = find_spec_file("test-spec-002", temp_specs_dir)
        assert result == spec_file

    def test_spec_not_found(self, temp_specs_dir):
        """Should return None when spec not found."""
        result = find_spec_file("nonexistent-spec", temp_specs_dir)
        assert result is None


class TestLoadSpec:
    """Tests for load_spec function."""

    def test_load_spec_success(self, temp_specs_dir, sample_spec):
        """Should load spec successfully."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result = load_spec("test-spec-001", temp_specs_dir)
        assert result is not None
        assert result["spec_id"] == "test-spec-001"
        assert result["title"] == "Test Specification"

    def test_load_spec_not_found(self, temp_specs_dir):
        """Should return None for nonexistent spec."""
        result = load_spec("nonexistent-spec", temp_specs_dir)
        assert result is None


class TestListSpecs:
    """Tests for list_specs function."""

    def test_list_all_specs(self, temp_specs_dir, sample_spec):
        """Should list all specs across folders."""
        # Create specs in different folders
        (temp_specs_dir / "active" / "spec-1.json").write_text(
            json.dumps({**sample_spec, "spec_id": "spec-1"})
        )
        (temp_specs_dir / "pending" / "spec-2.json").write_text(
            json.dumps({**sample_spec, "spec_id": "spec-2"})
        )

        result = list_specs(specs_dir=temp_specs_dir)
        assert len(result) == 2
        spec_ids = [s["spec_id"] for s in result]
        assert "spec-1" in spec_ids
        assert "spec-2" in spec_ids

    def test_list_specs_by_status(self, temp_specs_dir, sample_spec):
        """Should filter specs by status."""
        (temp_specs_dir / "active" / "spec-1.json").write_text(
            json.dumps({**sample_spec, "spec_id": "spec-1"})
        )
        (temp_specs_dir / "pending" / "spec-2.json").write_text(
            json.dumps({**sample_spec, "spec_id": "spec-2"})
        )

        result = list_specs(specs_dir=temp_specs_dir, status="active")
        assert len(result) == 1
        assert result[0]["spec_id"] == "spec-1"


class TestGetNode:
    """Tests for get_node function."""

    def test_get_existing_node(self, sample_spec):
        """Should return node data for existing node."""
        result = get_node(sample_spec, "task-1-1")
        assert result is not None
        assert result["title"] == "Task 1"
        assert result["status"] == "pending"

    def test_get_nonexistent_node(self, sample_spec):
        """Should return None for nonexistent node."""
        result = get_node(sample_spec, "nonexistent")
        assert result is None


class TestUpdateNode:
    """Tests for update_node function."""

    def test_update_existing_node(self, sample_spec):
        """Should update node and return True."""
        result = update_node(sample_spec, "task-1-1", {"status": "in_progress"})
        assert result is True
        assert sample_spec["hierarchy"]["task-1-1"]["status"] == "in_progress"

    def test_update_nonexistent_node(self, sample_spec):
        """Should return False for nonexistent node."""
        result = update_node(sample_spec, "nonexistent", {"status": "completed"})
        assert result is False

    def test_update_preserves_existing_fields(self, sample_spec):
        """Should preserve fields not being updated."""
        result = update_node(sample_spec, "task-1-1", {"status": "completed"})
        assert result is True
        assert sample_spec["hierarchy"]["task-1-1"]["title"] == "Task 1"
        assert sample_spec["hierarchy"]["task-1-1"]["parent"] == "phase-1"


class TestAddRevision:
    """Tests for add_revision function."""

    def test_add_revision_success(self, temp_specs_dir, sample_spec):
        """Should add revision entry successfully."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_revision(
            "test-spec-001",
            version="1.1",
            changelog="Added new feature",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None
        assert result["spec_id"] == "test-spec-001"
        assert result["version"] == "1.1"
        assert result["changelog"] == "Added new feature"
        assert result["revision_index"] == 1

        # Verify it was persisted
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        revisions = spec_data["metadata"]["revision_history"]
        assert len(revisions) == 1
        assert revisions[0]["version"] == "1.1"
        assert revisions[0]["changelog"] == "Added new feature"
        assert "date" in revisions[0]

    def test_add_revision_with_optional_fields(self, temp_specs_dir, sample_spec):
        """Should include optional fields when provided."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_revision(
            "test-spec-001",
            version="2.0",
            changelog="Major refactor",
            author="Test Author",
            modified_by="sdd-cli",
            review_triggered_by="/path/to/review.md",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["author"] == "Test Author"
        assert result["modified_by"] == "sdd-cli"
        assert result["review_triggered_by"] == "/path/to/review.md"

        # Verify persisted
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        revision = spec_data["metadata"]["revision_history"][0]
        assert revision["author"] == "Test Author"
        assert revision["modified_by"] == "sdd-cli"
        assert revision["review_triggered_by"] == "/path/to/review.md"

    def test_add_multiple_revisions(self, temp_specs_dir, sample_spec):
        """Should append multiple revisions."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        add_revision("test-spec-001", "1.0", "Initial release", specs_dir=temp_specs_dir)
        result, error = add_revision("test-spec-001", "1.1", "Bug fix", specs_dir=temp_specs_dir)

        assert error is None
        assert result["revision_index"] == 2

        spec_data = load_spec("test-spec-001", temp_specs_dir)
        revisions = spec_data["metadata"]["revision_history"]
        assert len(revisions) == 2
        assert revisions[0]["version"] == "1.0"
        assert revisions[1]["version"] == "1.1"

    def test_add_revision_spec_not_found(self, temp_specs_dir):
        """Should return error for nonexistent spec."""
        result, error = add_revision(
            "nonexistent-spec",
            version="1.0",
            changelog="Test",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "not found" in error

    def test_add_revision_empty_version(self, temp_specs_dir, sample_spec):
        """Should reject empty version."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_revision(
            "test-spec-001",
            version="",
            changelog="Test",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "Version is required" in error

    def test_add_revision_empty_changelog(self, temp_specs_dir, sample_spec):
        """Should reject empty changelog."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_revision(
            "test-spec-001",
            version="1.0",
            changelog="",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "Changelog is required" in error

    def test_add_revision_strips_whitespace(self, temp_specs_dir, sample_spec):
        """Should strip whitespace from inputs."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_revision(
            "test-spec-001",
            version="  1.0  ",
            changelog="  Test changelog  ",
            author="  Author  ",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["version"] == "1.0"
        assert result["changelog"] == "Test changelog"

        spec_data = load_spec("test-spec-001", temp_specs_dir)
        revision = spec_data["metadata"]["revision_history"][0]
        assert revision["version"] == "1.0"
        assert revision["changelog"] == "Test changelog"
        assert revision["author"] == "Author"


class TestUpdateFrontmatter:
    """Tests for update_frontmatter function."""

    def test_update_frontmatter_success(self, temp_specs_dir, sample_spec):
        """Should update metadata field successfully."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="description",
            value="Updated description",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result is not None
        assert result["spec_id"] == "test-spec-001"
        assert result["key"] == "description"
        assert result["value"] == "Updated description"
        assert result["previous_value"] is None  # Was not set before

        # Verify persisted
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        assert spec_data["metadata"]["description"] == "Updated description"

    def test_update_frontmatter_with_previous_value(self, temp_specs_dir, sample_spec):
        """Should return previous value when updating existing field."""
        sample_spec["metadata"]["owner"] = "Original Owner"
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="owner",
            value="New Owner",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["previous_value"] == "Original Owner"
        assert result["value"] == "New Owner"

    def test_update_frontmatter_top_level_sync(self, temp_specs_dir, sample_spec):
        """Should sync title/status to top-level fields."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="title",
            value="New Title",
            specs_dir=temp_specs_dir
        )

        assert error is None
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        # Both metadata and top-level should be updated
        assert spec_data["metadata"]["title"] == "New Title"
        assert spec_data["title"] == "New Title"

    def test_update_frontmatter_numeric_value(self, temp_specs_dir, sample_spec):
        """Should handle numeric values."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="estimated_hours",
            value=42,
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["value"] == 42

        spec_data = load_spec("test-spec-001", temp_specs_dir)
        assert spec_data["metadata"]["estimated_hours"] == 42

    def test_update_frontmatter_list_value(self, temp_specs_dir, sample_spec):
        """Should handle list values for objectives."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="objectives",
            value=["Objective 1", "Objective 2"],
            specs_dir=temp_specs_dir
        )

        assert error is None
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        assert spec_data["metadata"]["objectives"] == ["Objective 1", "Objective 2"]

    def test_update_frontmatter_spec_not_found(self, temp_specs_dir):
        """Should return error for nonexistent spec."""
        result, error = update_frontmatter(
            "nonexistent-spec",
            key="title",
            value="Test",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "not found" in error

    def test_update_frontmatter_empty_key(self, temp_specs_dir, sample_spec):
        """Should reject empty key."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="",
            value="Test",
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "Key is required" in error

    def test_update_frontmatter_none_value(self, temp_specs_dir, sample_spec):
        """Should reject None value."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="description",
            value=None,
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "Value cannot be None" in error

    def test_update_frontmatter_blocks_assumptions(self, temp_specs_dir, sample_spec):
        """Should block direct update of assumptions array."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="assumptions",
            value=["new assumption"],
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "dedicated function" in error

    def test_update_frontmatter_blocks_revision_history(self, temp_specs_dir, sample_spec):
        """Should block direct update of revision_history array."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="revision_history",
            value=[{"version": "1.0"}],
            specs_dir=temp_specs_dir
        )

        assert result is None
        assert "dedicated function" in error

    def test_update_frontmatter_strips_whitespace(self, temp_specs_dir, sample_spec):
        """Should strip whitespace from string values."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="  description  ",
            value="  Trimmed value  ",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["key"] == "description"
        assert result["value"] == "Trimmed value"

    def test_update_frontmatter_allows_empty_string(self, temp_specs_dir, sample_spec):
        """Should allow empty string as value (to clear a field)."""
        sample_spec["metadata"]["description"] = "Original"
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="description",
            value="",
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["value"] == ""
        assert result["previous_value"] == "Original"

    def test_update_frontmatter_allows_zero(self, temp_specs_dir, sample_spec):
        """Should allow zero as numeric value."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="progress_percentage",
            value=0,
            specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["value"] == 0
