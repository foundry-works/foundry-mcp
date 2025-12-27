"""Tests for core spec operations."""

import json
import tempfile
from pathlib import Path

import pytest

from foundry_mcp.core.spec import (
    PHASE_TEMPLATES,
    apply_phase_template,
    find_specs_directory,
    find_spec_file,
    get_node,
    get_phase_template_structure,
    list_specs,
    load_spec,
    update_node,
    add_revision,
    update_frontmatter,
    add_phase,
    remove_phase,
    move_phase,
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
        },
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


class TestAddPhase:
    """Tests for add_phase helper."""

    def _write_spec(self, temp_specs_dir, spec_id: str = "test-spec-phase") -> Path:
        spec_data = {
            "spec_id": spec_id,
            "title": "Test Spec",
            "metadata": {
                "estimated_hours": 5,
                "status": "pending",
            },
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Test Spec",
                    "status": "pending",
                    "parent": None,
                    "children": ["phase-1"],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "metadata": {"purpose": "", "category": "implementation"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "phase-1": {
                    "type": "phase",
                    "title": "Existing Phase",
                    "status": "pending",
                    "parent": "spec-root",
                    "children": [],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "metadata": {"purpose": "Initial work"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
        }
        spec_path = temp_specs_dir / "pending" / f"{spec_id}.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec_data))
        return spec_path

    def test_add_phase_appends_and_links_to_previous(self, temp_specs_dir):
        """add_phase should append a phase, scaffold verifications, and link dependencies."""
        spec_id = "phase-spec"
        self._write_spec(temp_specs_dir, spec_id)

        result, error = add_phase(
            spec_id=spec_id,
            title="Implementation",
            description="Async orchestrator",
            purpose="Core work",
            estimated_hours=3,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["phase_id"] == "phase-2"
        assert result["linked_previous"] == "phase-1"
        assert result["verify_tasks"] == ["verify-2-1", "verify-2-2"]

        spec = load_spec(spec_id, temp_specs_dir)
        hierarchy = spec["hierarchy"]
        spec_root = hierarchy["spec-root"]
        assert spec_root["children"][-1] == "phase-2"
        assert spec_root["total_tasks"] == 2  # verification tasks added

        phase_two = hierarchy["phase-2"]
        assert phase_two["metadata"]["description"] == "Async orchestrator"
        assert phase_two["metadata"]["estimated_hours"] == 3
        assert phase_two["children"] == ["verify-2-1", "verify-2-2"]
        assert hierarchy["verify-2-1"]["parent"] == "phase-2"
        assert hierarchy["verify-2-2"]["dependencies"]["blocked_by"] == ["verify-2-1"]
        assert hierarchy["phase-1"]["dependencies"]["blocks"] == ["phase-2"]
        assert hierarchy["phase-2"]["dependencies"]["blocked_by"] == ["phase-1"]
        assert spec["metadata"]["estimated_hours"] == 8

    def test_add_phase_inserts_at_custom_position_without_link(self, temp_specs_dir):
        """add_phase should support insertion at specific index without linking."""
        spec_id = "phase-spec-position"
        self._write_spec(temp_specs_dir, spec_id)

        result, error = add_phase(
            spec_id=spec_id,
            title="Prep",
            position=0,
            link_previous=False,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["position"] == 0
        assert result["linked_previous"] is None

        spec = load_spec(spec_id, temp_specs_dir)
        spec_root_children = spec["hierarchy"]["spec-root"]["children"]
        assert spec_root_children[0] == result["phase_id"]
        # Original phase remains second
        assert spec_root_children[1] == "phase-1"
        # No automatic block linkage when inserted at beginning
        assert spec["hierarchy"]["phase-1"]["dependencies"]["blocked_by"] == []

    def test_add_phase_validates_inputs(self, temp_specs_dir):
        """add_phase should validate required fields and numeric ranges."""
        # Missing spec_id
        result, error = add_phase(spec_id="", title="New")
        assert result is None
        assert error == "Specification ID is required"

        # Negative estimated hours
        spec_id = "phase-spec-invalid"
        self._write_spec(temp_specs_dir, spec_id)
        result, error = add_phase(
            spec_id=spec_id,
            title="Negative",
            estimated_hours=-1,
            specs_dir=temp_specs_dir,
        )
        assert result is None
        assert error == "estimated_hours must be non-negative"


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
            specs_dir=temp_specs_dir,
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
            specs_dir=temp_specs_dir,
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

        add_revision(
            "test-spec-001", "1.0", "Initial release", specs_dir=temp_specs_dir
        )
        result, error = add_revision(
            "test-spec-001", "1.1", "Bug fix", specs_dir=temp_specs_dir
        )

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
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "not found" in error

    def test_add_revision_empty_version(self, temp_specs_dir, sample_spec):
        """Should reject empty version."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_revision(
            "test-spec-001", version="", changelog="Test", specs_dir=temp_specs_dir
        )

        assert result is None
        assert "Version is required" in error

    def test_add_revision_empty_changelog(self, temp_specs_dir, sample_spec):
        """Should reject empty changelog."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = add_revision(
            "test-spec-001", version="1.0", changelog="", specs_dir=temp_specs_dir
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
            specs_dir=temp_specs_dir,
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
            specs_dir=temp_specs_dir,
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

    def test_update_frontmatter_mission_field(self, temp_specs_dir, sample_spec):
        """Should allow writing the new mission metadata field."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="mission",
            value="Align labelers on student correctness goals",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["value"] == "Align labelers on student correctness goals"

        spec_data = load_spec("test-spec-001", temp_specs_dir)
        assert (
            spec_data["metadata"]["mission"]
            == "Align labelers on student correctness goals"
        )

    def test_update_frontmatter_with_previous_value(self, temp_specs_dir, sample_spec):
        """Should return previous value when updating existing field."""
        sample_spec["metadata"]["owner"] = "Original Owner"
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001", key="owner", value="New Owner", specs_dir=temp_specs_dir
        )

        assert error is None
        assert result["previous_value"] == "Original Owner"
        assert result["value"] == "New Owner"

    def test_update_frontmatter_top_level_sync(self, temp_specs_dir, sample_spec):
        """Should sync title/status to top-level fields."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001", key="title", value="New Title", specs_dir=temp_specs_dir
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
            "test-spec-001", key="estimated_hours", value=42, specs_dir=temp_specs_dir
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
            specs_dir=temp_specs_dir,
        )

        assert error is None
        spec_data = load_spec("test-spec-001", temp_specs_dir)
        assert spec_data["metadata"]["objectives"] == ["Objective 1", "Objective 2"]

    def test_update_frontmatter_spec_not_found(self, temp_specs_dir):
        """Should return error for nonexistent spec."""
        result, error = update_frontmatter(
            "nonexistent-spec", key="title", value="Test", specs_dir=temp_specs_dir
        )

        assert result is None
        assert "not found" in error

    def test_update_frontmatter_empty_key(self, temp_specs_dir, sample_spec):
        """Should reject empty key."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001", key="", value="Test", specs_dir=temp_specs_dir
        )

        assert result is None
        assert "Key is required" in error

    def test_update_frontmatter_none_value(self, temp_specs_dir, sample_spec):
        """Should reject None value."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001", key="description", value=None, specs_dir=temp_specs_dir
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
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "dedicated function" in error

    def test_update_frontmatter_blocks_revision_history(
        self, temp_specs_dir, sample_spec
    ):
        """Should block direct update of revision_history array."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        result, error = update_frontmatter(
            "test-spec-001",
            key="revision_history",
            value=[{"version": "1.0"}],
            specs_dir=temp_specs_dir,
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
            specs_dir=temp_specs_dir,
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
            "test-spec-001", key="description", value="", specs_dir=temp_specs_dir
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
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["value"] == 0


class TestRemovePhase:
    """Tests for remove_phase function."""

    def _create_spec_with_phases(
        self,
        temp_specs_dir,
        spec_id: str = "test-remove-phase",
        num_phases: int = 3,
        add_tasks: bool = True,
        task_status: str = "completed",
        link_phases: bool = True,
    ) -> Path:
        """Helper to create a spec with multiple phases for testing."""
        hierarchy = {
            "spec-root": {
                "type": "spec",
                "title": "Test Spec",
                "status": "pending",
                "parent": None,
                "children": [],
                "total_tasks": 0,
                "completed_tasks": 0,
                "metadata": {"purpose": "", "category": "implementation"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        }

        prev_phase_id = None
        total_tasks = 0
        completed_tasks = 0

        for i in range(1, num_phases + 1):
            phase_id = f"phase-{i}"
            hierarchy["spec-root"]["children"].append(phase_id)

            phase = {
                "type": "phase",
                "title": f"Phase {i}",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "total_tasks": 0,
                "completed_tasks": 0,
                "metadata": {"purpose": f"Work for phase {i}"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            }

            # Link phases if requested
            if link_phases and prev_phase_id:
                phase["dependencies"]["blocked_by"].append(prev_phase_id)
                hierarchy[prev_phase_id]["dependencies"]["blocks"].append(phase_id)

            if add_tasks:
                task_id = f"task-{i}-1"
                phase["children"].append(task_id)
                phase["total_tasks"] = 1
                total_tasks += 1

                hierarchy[task_id] = {
                    "type": "task",
                    "title": f"Task {i}.1",
                    "status": task_status,
                    "parent": phase_id,
                    "children": [],
                    "metadata": {},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                }

                if task_status == "completed":
                    phase["completed_tasks"] = 1
                    completed_tasks += 1

            hierarchy[phase_id] = phase
            prev_phase_id = phase_id

        hierarchy["spec-root"]["total_tasks"] = total_tasks
        hierarchy["spec-root"]["completed_tasks"] = completed_tasks

        spec_data = {
            "spec_id": spec_id,
            "title": "Test Spec",
            "metadata": {
                "title": "Test Spec",
                "version": "1.0.0",
                "estimated_hours": 10,
            },
            "hierarchy": hierarchy,
        }

        spec_path = temp_specs_dir / "active" / f"{spec_id}.json"
        spec_path.write_text(json.dumps(spec_data))
        return spec_path

    def test_remove_phase_success(self, temp_specs_dir):
        """Should successfully remove a phase with completed tasks."""
        self._create_spec_with_phases(
            temp_specs_dir,
            spec_id="test-remove",
            num_phases=3,
            task_status="completed",
        )

        result, error = remove_phase(
            spec_id="test-remove",
            phase_id="phase-2",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert result["spec_id"] == "test-remove"
        assert result["phase_id"] == "phase-2"
        assert result["phase_title"] == "Phase 2"
        assert result["children_removed"] == 1  # task-2-1
        assert result["total_tasks_removed"] == 1
        assert result["force"] is False

        # Verify phase and task removed from hierarchy
        spec = load_spec("test-remove", temp_specs_dir)
        assert "phase-2" not in spec["hierarchy"]
        assert "task-2-1" not in spec["hierarchy"]
        assert "phase-1" in spec["hierarchy"]
        assert "phase-3" in spec["hierarchy"]

    def test_remove_middle_phase_relinks_adjacent(self, temp_specs_dir):
        """Removing middle phase should re-link prev to next in dependency chain."""
        self._create_spec_with_phases(
            temp_specs_dir,
            spec_id="test-relink",
            num_phases=3,
            task_status="completed",
            link_phases=True,
        )

        result, error = remove_phase(
            spec_id="test-relink",
            phase_id="phase-2",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert "relinked" in result
        assert result["relinked"]["from"] == "phase-1"
        assert result["relinked"]["to"] == "phase-3"

        # Verify re-linking
        spec = load_spec("test-relink", temp_specs_dir)
        phase1 = spec["hierarchy"]["phase-1"]
        phase3 = spec["hierarchy"]["phase-3"]

        # phase-1 should now block phase-3
        assert "phase-3" in phase1["dependencies"]["blocks"]
        # phase-2 reference should be cleaned
        assert "phase-2" not in phase1["dependencies"]["blocks"]

        # phase-3 should now be blocked by phase-1
        assert "phase-1" in phase3["dependencies"]["blocked_by"]
        assert "phase-2" not in phase3["dependencies"]["blocked_by"]

    def test_remove_first_phase_clears_successor(self, temp_specs_dir):
        """Removing first phase should clear blocked_by in second phase."""
        self._create_spec_with_phases(
            temp_specs_dir,
            spec_id="test-first",
            num_phases=3,
            task_status="completed",
            link_phases=True,
        )

        result, error = remove_phase(
            spec_id="test-first",
            phase_id="phase-1",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        # No re-linking since there's no predecessor
        assert "relinked" not in result

        spec = load_spec("test-first", temp_specs_dir)
        phase2 = spec["hierarchy"]["phase-2"]

        # phase-1 reference should be removed from phase-2's blocked_by
        assert "phase-1" not in phase2["dependencies"]["blocked_by"]

    def test_remove_last_phase_clears_predecessor(self, temp_specs_dir):
        """Removing last phase should clear blocks in predecessor."""
        self._create_spec_with_phases(
            temp_specs_dir,
            spec_id="test-last",
            num_phases=3,
            task_status="completed",
            link_phases=True,
        )

        result, error = remove_phase(
            spec_id="test-last",
            phase_id="phase-3",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        # No re-linking since there's no successor
        assert "relinked" not in result

        spec = load_spec("test-last", temp_specs_dir)
        phase2 = spec["hierarchy"]["phase-2"]

        # phase-3 reference should be removed from phase-2's blocks
        assert "phase-3" not in phase2["dependencies"]["blocks"]

    def test_blocked_with_active_tasks(self, temp_specs_dir):
        """Should refuse to remove phase with pending/in_progress tasks without force."""
        self._create_spec_with_phases(
            temp_specs_dir,
            spec_id="test-active",
            num_phases=2,
            task_status="pending",  # Active work
        )

        result, error = remove_phase(
            spec_id="test-active",
            phase_id="phase-1",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "non-completed task" in error
        assert "force=True" in error
        assert "task-1-1" in error

        # Verify phase still exists
        spec = load_spec("test-active", temp_specs_dir)
        assert "phase-1" in spec["hierarchy"]

    def test_force_with_active_tasks(self, temp_specs_dir):
        """Should remove phase with active tasks when force=True."""
        self._create_spec_with_phases(
            temp_specs_dir,
            spec_id="test-force",
            num_phases=2,
            task_status="in_progress",
        )

        result, error = remove_phase(
            spec_id="test-force",
            phase_id="phase-1",
            force=True,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["force"] is True
        assert result["total_tasks_removed"] == 1

        # Verify phase removed
        spec = load_spec("test-force", temp_specs_dir)
        assert "phase-1" not in spec["hierarchy"]

    def test_updates_spec_root_counts(self, temp_specs_dir):
        """Should update spec-root total_tasks and completed_tasks."""
        self._create_spec_with_phases(
            temp_specs_dir,
            spec_id="test-counts",
            num_phases=3,
            task_status="completed",
        )

        # Before removal: 3 total, 3 completed
        spec_before = load_spec("test-counts", temp_specs_dir)
        assert spec_before["hierarchy"]["spec-root"]["total_tasks"] == 3
        assert spec_before["hierarchy"]["spec-root"]["completed_tasks"] == 3

        result, error = remove_phase(
            spec_id="test-counts",
            phase_id="phase-2",
            specs_dir=temp_specs_dir,
        )

        assert error is None

        # After removal: 2 total, 2 completed
        spec_after = load_spec("test-counts", temp_specs_dir)
        assert spec_after["hierarchy"]["spec-root"]["total_tasks"] == 2
        assert spec_after["hierarchy"]["spec-root"]["completed_tasks"] == 2

    def test_cleans_dependency_references(self, temp_specs_dir):
        """Should clean all dependency references to removed nodes."""
        # Create spec with cross-dependencies
        spec_data = {
            "spec_id": "test-deps",
            "title": "Test Deps",
            "metadata": {"title": "Test Deps"},
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Test Deps",
                    "status": "pending",
                    "parent": None,
                    "children": ["phase-1", "phase-2"],
                    "total_tasks": 2,
                    "completed_tasks": 2,
                    "metadata": {},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "phase-1": {
                    "type": "phase",
                    "title": "Phase 1",
                    "status": "completed",
                    "parent": "spec-root",
                    "children": ["task-1-1"],
                    "total_tasks": 1,
                    "completed_tasks": 1,
                    "metadata": {},
                    "dependencies": {"blocks": ["phase-2"], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Task 1.1",
                    "status": "completed",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {},
                    "dependencies": {"blocks": ["task-2-1"], "blocked_by": [], "depends": []},
                },
                "phase-2": {
                    "type": "phase",
                    "title": "Phase 2",
                    "status": "pending",
                    "parent": "spec-root",
                    "children": ["task-2-1"],
                    "total_tasks": 1,
                    "completed_tasks": 1,
                    "metadata": {},
                    "dependencies": {"blocks": [], "blocked_by": ["phase-1"], "depends": []},
                },
                "task-2-1": {
                    "type": "task",
                    "title": "Task 2.1",
                    "status": "completed",
                    "parent": "phase-2",
                    "children": [],
                    "metadata": {},
                    "dependencies": {"blocks": [], "blocked_by": ["task-1-1"], "depends": []},
                },
            },
        }

        spec_path = temp_specs_dir / "active" / "test-deps.json"
        spec_path.write_text(json.dumps(spec_data))

        result, error = remove_phase(
            spec_id="test-deps",
            phase_id="phase-1",
            specs_dir=temp_specs_dir,
        )

        assert error is None

        # Verify all references to phase-1 and task-1-1 cleaned
        spec = load_spec("test-deps", temp_specs_dir)
        phase2 = spec["hierarchy"]["phase-2"]
        task2 = spec["hierarchy"]["task-2-1"]

        assert "phase-1" not in phase2["dependencies"]["blocked_by"]
        assert "task-1-1" not in task2["dependencies"]["blocked_by"]

    def test_nonexistent_phase_error(self, temp_specs_dir):
        """Should return error for nonexistent phase."""
        self._create_spec_with_phases(
            temp_specs_dir,
            spec_id="test-missing",
            num_phases=1,
        )

        result, error = remove_phase(
            spec_id="test-missing",
            phase_id="phase-99",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "not found" in error

    def test_non_phase_node_error(self, temp_specs_dir):
        """Should return error when trying to remove a non-phase node."""
        self._create_spec_with_phases(
            temp_specs_dir,
            spec_id="test-type",
            num_phases=1,
            task_status="completed",
        )

        result, error = remove_phase(
            spec_id="test-type",
            phase_id="task-1-1",  # This is a task, not a phase
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "not a phase" in error

    def test_validates_empty_spec_id(self, temp_specs_dir):
        """Should return error for empty spec_id."""
        result, error = remove_phase(
            spec_id="",
            phase_id="phase-1",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "Specification ID is required" in error

    def test_validates_empty_phase_id(self, temp_specs_dir):
        """Should return error for empty phase_id."""
        self._create_spec_with_phases(
            temp_specs_dir,
            spec_id="test-empty",
            num_phases=1,
        )

        result, error = remove_phase(
            spec_id="test-empty",
            phase_id="",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "Phase ID is required" in error

    def test_spec_not_found_error(self, temp_specs_dir):
        """Should return error for nonexistent spec."""
        result, error = remove_phase(
            spec_id="nonexistent-spec",
            phase_id="phase-1",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "not found" in error


class TestPhaseTemplates:
    """Tests for phase template functions."""

    def test_phase_templates_constant(self):
        """Should define valid phase templates."""
        assert PHASE_TEMPLATES == (
            "planning",
            "implementation",
            "testing",
            "security",
            "documentation",
        )

    def test_get_phase_template_structure_planning(self):
        """Should return correct structure for planning template."""
        result = get_phase_template_structure("planning")

        assert result["template_name"] == "planning"
        assert result["title"] == "Planning & Discovery"
        assert result["estimated_hours"] == 4
        assert result["includes_verification"] is True
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["title"] == "Define requirements"
        assert result["tasks"][1]["title"] == "Design solution approach"

    def test_get_phase_template_structure_implementation(self):
        """Should return correct structure for implementation template."""
        result = get_phase_template_structure("implementation")

        assert result["template_name"] == "implementation"
        assert result["title"] == "Implementation"
        assert result["estimated_hours"] == 8
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["title"] == "Implement core functionality"

    def test_get_phase_template_structure_testing(self):
        """Should return correct structure for testing template."""
        result = get_phase_template_structure("testing")

        assert result["template_name"] == "testing"
        assert result["title"] == "Testing & Validation"
        assert result["estimated_hours"] == 6
        assert len(result["tasks"]) == 2
        # Testing tasks should have investigation category
        assert result["tasks"][0]["task_category"] == "investigation"

    def test_get_phase_template_structure_security(self):
        """Should return correct structure for security template."""
        result = get_phase_template_structure("security")

        assert result["template_name"] == "security"
        assert result["title"] == "Security Review"
        assert result["estimated_hours"] == 6
        assert len(result["tasks"]) == 2

    def test_get_phase_template_structure_documentation(self):
        """Should return correct structure for documentation template."""
        result = get_phase_template_structure("documentation")

        assert result["template_name"] == "documentation"
        assert result["title"] == "Documentation"
        assert result["estimated_hours"] == 4
        assert len(result["tasks"]) == 2
        # Documentation tasks should have research category
        assert result["tasks"][0]["task_category"] == "research"

    def test_get_phase_template_structure_with_category_override(self):
        """Planning template tasks use investigation category (not overridable)."""
        result = get_phase_template_structure("planning", category="refactoring")

        # Planning tasks always use investigation (hardcoded for validation)
        assert result["tasks"][0]["task_category"] == "investigation"
        assert result["tasks"][1]["task_category"] == "investigation"

    def test_get_phase_template_structure_invalid_template(self):
        """Should raise ValueError for invalid template name."""
        with pytest.raises(ValueError) as exc_info:
            get_phase_template_structure("invalid-template")

        assert "Invalid phase template" in str(exc_info.value)
        assert "invalid-template" in str(exc_info.value)

    def test_all_templates_have_required_fields(self):
        """Should ensure all templates have required fields."""
        required_fields = [
            "title",
            "description",
            "purpose",
            "estimated_hours",
            "tasks",
            "includes_verification",
            "template_name",
        ]

        for template_name in PHASE_TEMPLATES:
            result = get_phase_template_structure(template_name)
            for field in required_fields:
                assert field in result, f"{template_name} missing field: {field}"


class TestApplyPhaseTemplate:
    """Tests for apply_phase_template function."""

    def _create_base_spec(self, temp_specs_dir, spec_id="test-spec"):
        """Create a minimal spec for testing."""
        spec_data = {
            "spec_id": spec_id,
            "title": "Test Spec",
            "metadata": {
                "title": "Test Spec",
                "version": "1.0.0",
            },
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Test Spec",
                    "status": "pending",
                    "parent": None,
                    "children": [],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "metadata": {"purpose": "Testing", "category": "implementation"},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
        }
        spec_path = temp_specs_dir / "active" / f"{spec_id}.json"
        spec_path.write_text(json.dumps(spec_data))
        return spec_path

    def test_apply_phase_template_success(self, temp_specs_dir):
        """Should successfully apply a phase template."""
        self._create_base_spec(temp_specs_dir, spec_id="test-apply")

        result, error = apply_phase_template(
            spec_id="test-apply",
            template="planning",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert result["template_applied"] == "planning"
        assert result["template_title"] == "Planning & Discovery"
        assert "phase_id" in result
        assert result["total_tasks"] == 4  # 2 tasks from template + 2 verify tasks

    def test_apply_phase_template_creates_tasks(self, temp_specs_dir):
        """Should create tasks from the template."""
        self._create_base_spec(temp_specs_dir, spec_id="test-tasks")

        result, error = apply_phase_template(
            spec_id="test-tasks",
            template="implementation",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["total_tasks"] == 4  # 2 tasks from template + 2 verify tasks

        # Verify spec was updated
        spec = load_spec("test-tasks", temp_specs_dir)
        phase_id = result["phase_id"]
        assert phase_id in spec["hierarchy"]
        assert spec["hierarchy"][phase_id]["title"] == "Implementation"

        # Verify verification tasks were created
        phase_children = spec["hierarchy"][phase_id]["children"]
        verify_tasks = [
            tid for tid in phase_children
            if spec["hierarchy"][tid]["type"] == "verify"
        ]
        assert len(verify_tasks) == 2

    def test_apply_phase_template_invalid_template(self, temp_specs_dir):
        """Should return error for invalid template."""
        self._create_base_spec(temp_specs_dir, spec_id="test-invalid")

        result, error = apply_phase_template(
            spec_id="test-invalid",
            template="nonexistent",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "Invalid phase template" in error

    def test_apply_phase_template_spec_not_found(self, temp_specs_dir):
        """Should return error for nonexistent spec."""
        result, error = apply_phase_template(
            spec_id="nonexistent-spec",
            template="planning",
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None

    def test_apply_phase_template_with_category(self, temp_specs_dir):
        """Should apply custom category to tasks."""
        self._create_base_spec(temp_specs_dir, spec_id="test-category")

        result, error = apply_phase_template(
            spec_id="test-category",
            template="planning",
            category="refactoring",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None

    def test_apply_phase_template_with_position(self, temp_specs_dir):
        """Should respect position parameter."""
        self._create_base_spec(temp_specs_dir, spec_id="test-position")

        result, error = apply_phase_template(
            spec_id="test-position",
            template="planning",
            position=0,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None

    def test_apply_phase_template_without_linking(self, temp_specs_dir):
        """Should respect link_previous=False."""
        self._create_base_spec(temp_specs_dir, spec_id="test-no-link")

        result, error = apply_phase_template(
            spec_id="test-no-link",
            template="planning",
            link_previous=False,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None


class TestMovePhase:
    """Tests for move_phase function."""

    def _create_spec_with_phases(
        self,
        temp_specs_dir,
        spec_id: str = "test-move-phase",
        num_phases: int = 3,
        link_phases: bool = True,
    ) -> Path:
        """Helper to create a spec with multiple phases for testing."""
        hierarchy = {
            "spec-root": {
                "type": "spec",
                "title": "Test Spec",
                "status": "pending",
                "parent": None,
                "children": [],
                "total_tasks": 0,
                "completed_tasks": 0,
                "metadata": {"purpose": "", "category": "implementation"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        }

        prev_phase_id = None
        for i in range(1, num_phases + 1):
            phase_id = f"phase-{i}"
            hierarchy["spec-root"]["children"].append(phase_id)

            phase = {
                "type": "phase",
                "title": f"Phase {i}",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "total_tasks": 0,
                "completed_tasks": 0,
                "metadata": {"purpose": f"Work for phase {i}"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            }

            # Link phases if requested (each phase blocked by predecessor)
            if link_phases and prev_phase_id:
                phase["dependencies"]["blocked_by"].append(prev_phase_id)
                hierarchy[prev_phase_id]["dependencies"]["blocks"].append(phase_id)

            hierarchy[phase_id] = phase
            prev_phase_id = phase_id

        spec_data = {
            "spec_id": spec_id,
            "title": "Test Spec",
            "metadata": {"title": "Test Spec", "version": "1.0.0"},
            "hierarchy": hierarchy,
        }

        spec_path = temp_specs_dir / "active" / f"{spec_id}.json"
        spec_path.write_text(json.dumps(spec_data))
        return spec_path

    def test_move_phase_forward(self, temp_specs_dir):
        """Should move phase from earlier to later position."""
        self._create_spec_with_phases(temp_specs_dir, spec_id="test-forward")

        result, error = move_phase(
            spec_id="test-forward",
            phase_id="phase-1",
            position=3,  # Move from position 1 to 3
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert result["phase_id"] == "phase-1"
        assert result["old_position"] == 1
        assert result["new_position"] == 3
        assert result["moved"] is True

        # Verify order in spec
        spec = load_spec("test-forward", temp_specs_dir)
        children = spec["hierarchy"]["spec-root"]["children"]
        assert children == ["phase-2", "phase-3", "phase-1"]

    def test_move_phase_backward(self, temp_specs_dir):
        """Should move phase from later to earlier position."""
        self._create_spec_with_phases(temp_specs_dir, spec_id="test-backward")

        result, error = move_phase(
            spec_id="test-backward",
            phase_id="phase-3",
            position=1,  # Move from position 3 to 1
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert result["old_position"] == 3
        assert result["new_position"] == 1
        assert result["moved"] is True

        # Verify order in spec
        spec = load_spec("test-backward", temp_specs_dir)
        children = spec["hierarchy"]["spec-root"]["children"]
        assert children == ["phase-3", "phase-1", "phase-2"]

    def test_move_phase_same_position(self, temp_specs_dir):
        """Should return no-op when already at target position."""
        self._create_spec_with_phases(temp_specs_dir, spec_id="test-same")

        result, error = move_phase(
            spec_id="test-same",
            phase_id="phase-2",
            position=2,  # Already at position 2
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert result["moved"] is False
        assert "already at" in result.get("message", "")

    def test_move_phase_updates_dependencies(self, temp_specs_dir):
        """Should update dependency chain when link_previous=True."""
        self._create_spec_with_phases(
            temp_specs_dir, spec_id="test-deps", link_phases=True
        )

        # Initially: phase-1 -> phase-2 -> phase-3
        # Move phase-2 to end: phase-1 -> phase-3 -> phase-2
        result, error = move_phase(
            spec_id="test-deps",
            phase_id="phase-2",
            position=3,
            link_previous=True,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None

        # Verify dependency updates were made
        spec = load_spec("test-deps", temp_specs_dir)
        phase1 = spec["hierarchy"]["phase-1"]
        phase2 = spec["hierarchy"]["phase-2"]
        phase3 = spec["hierarchy"]["phase-3"]

        # phase-1 should now block phase-3 (not phase-2)
        assert "phase-3" in phase1["dependencies"]["blocks"]
        assert "phase-2" not in phase1["dependencies"]["blocks"]

        # phase-3 should be blocked by phase-1 (not phase-2)
        assert "phase-1" in phase3["dependencies"]["blocked_by"]

        # phase-2 (now last) should be blocked by phase-3
        assert "phase-3" in phase2["dependencies"]["blocked_by"]

    def test_move_phase_preserves_dependencies(self, temp_specs_dir):
        """Should preserve existing deps when link_previous=False."""
        self._create_spec_with_phases(
            temp_specs_dir, spec_id="test-preserve", link_phases=True
        )

        # Get original dependencies
        spec_before = load_spec("test-preserve", temp_specs_dir)
        phase2_before = spec_before["hierarchy"]["phase-2"]["dependencies"].copy()

        result, error = move_phase(
            spec_id="test-preserve",
            phase_id="phase-2",
            position=3,
            link_previous=False,  # Don't update dependencies
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None

        # Dependencies should be unchanged
        spec = load_spec("test-preserve", temp_specs_dir)
        phase2 = spec["hierarchy"]["phase-2"]
        assert phase2["dependencies"]["blocked_by"] == phase2_before["blocked_by"]

    def test_move_phase_invalid_position_zero(self, temp_specs_dir):
        """Should reject position 0 (must be 1-based)."""
        self._create_spec_with_phases(temp_specs_dir, spec_id="test-zero")

        result, error = move_phase(
            spec_id="test-zero",
            phase_id="phase-1",
            position=0,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "positive integer" in error.lower()

    def test_move_phase_invalid_position_too_high(self, temp_specs_dir):
        """Should reject position beyond phase count."""
        self._create_spec_with_phases(temp_specs_dir, spec_id="test-high")

        result, error = move_phase(
            spec_id="test-high",
            phase_id="phase-1",
            position=10,  # Only 3 phases
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "invalid position" in error.lower()

    def test_move_phase_not_found(self, temp_specs_dir):
        """Should return error for nonexistent phase."""
        self._create_spec_with_phases(temp_specs_dir, spec_id="test-missing")

        result, error = move_phase(
            spec_id="test-missing",
            phase_id="phase-99",
            position=1,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "not found" in error.lower()

    def test_move_non_phase_node_error(self, temp_specs_dir):
        """Should reject moving non-phase nodes."""
        # Create spec with a task
        spec_data = {
            "spec_id": "test-non-phase",
            "title": "Test",
            "metadata": {"title": "Test"},
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Test",
                    "status": "pending",
                    "parent": None,
                    "children": ["phase-1"],
                    "total_tasks": 1,
                    "completed_tasks": 0,
                    "metadata": {},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "phase-1": {
                    "type": "phase",
                    "title": "Phase 1",
                    "status": "pending",
                    "parent": "spec-root",
                    "children": ["task-1-1"],
                    "total_tasks": 1,
                    "completed_tasks": 0,
                    "metadata": {},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
                "task-1-1": {
                    "type": "task",
                    "title": "Task 1",
                    "status": "pending",
                    "parent": "phase-1",
                    "children": [],
                    "metadata": {},
                    "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
                },
            },
        }
        spec_path = temp_specs_dir / "active" / "test-non-phase.json"
        spec_path.write_text(json.dumps(spec_data))

        result, error = move_phase(
            spec_id="test-non-phase",
            phase_id="task-1-1",  # This is a task, not a phase
            position=1,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "not a phase" in error.lower()

    def test_move_phase_spec_not_found(self, temp_specs_dir):
        """Should return error for nonexistent spec."""
        result, error = move_phase(
            spec_id="nonexistent-spec",
            phase_id="phase-1",
            position=1,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "not found" in error.lower()

    def test_move_phase_empty_spec_id(self, temp_specs_dir):
        """Should reject empty spec_id."""
        result, error = move_phase(
            spec_id="",
            phase_id="phase-1",
            position=1,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "Specification ID is required" in error

    def test_move_phase_empty_phase_id(self, temp_specs_dir):
        """Should reject empty phase_id."""
        self._create_spec_with_phases(temp_specs_dir, spec_id="test-empty-phase")

        result, error = move_phase(
            spec_id="test-empty-phase",
            phase_id="",
            position=1,
            specs_dir=temp_specs_dir,
        )

        assert result is None
        assert error is not None
        assert "Phase ID is required" in error

    def test_move_phase_dry_run(self, temp_specs_dir):
        """Dry run should preview without saving."""
        self._create_spec_with_phases(temp_specs_dir, spec_id="test-dry-run")

        result, error = move_phase(
            spec_id="test-dry-run",
            phase_id="phase-1",
            position=3,
            dry_run=True,
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result is not None
        assert result["dry_run"] is True
        assert result["new_position"] == 3

        # Verify NOT saved (order unchanged)
        spec = load_spec("test-dry-run", temp_specs_dir)
        children = spec["hierarchy"]["spec-root"]["children"]
        assert children == ["phase-1", "phase-2", "phase-3"]  # Unchanged
