"""Tests for find_replace_in_spec function."""

import json
import tempfile
from pathlib import Path

import pytest

from foundry_mcp.core.spec import find_replace_in_spec, load_spec


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        specs_dir = (Path(tmpdir) / "specs").resolve()
        for folder in ["pending", "active", "completed", "archived"]:
            (specs_dir / folder).mkdir(parents=True)
        yield specs_dir


@pytest.fixture
def fr_spec():
    """Create a spec for find/replace testing."""
    return {
        "spec_id": "fr-test-001",
        "title": "Find Replace Test",
        "metadata": {"title": "Find Replace Test"},
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "FR Test",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Implement FooBar feature",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
                "metadata": {"description": "FooBar implementation phase"},
            },
            "task-1-1": {
                "type": "task",
                "title": "Add FooBar component",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"description": "Create the FooBar component with tests"},
            },
            "task-1-2": {
                "type": "task",
                "title": "Update documentation",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"description": "Document FooBar API"},
            },
        },
        "journal": [],
    }


class TestFindReplaceLiteral:
    """Tests for literal find/replace."""

    def test_replace_in_titles(self, temp_specs_dir, fr_spec):
        """Should replace text in titles."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "FooBar", "BazQux",
            scope="titles", specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["total_replacements"] == 2
        spec = load_spec("fr-test-001", temp_specs_dir)
        assert "BazQux" in spec["hierarchy"]["phase-1"]["title"]
        assert "BazQux" in spec["hierarchy"]["task-1-1"]["title"]

    def test_replace_in_descriptions(self, temp_specs_dir, fr_spec):
        """Should replace text in descriptions."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "FooBar", "NewName",
            scope="descriptions", specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["total_replacements"] == 3
        spec = load_spec("fr-test-001", temp_specs_dir)
        assert "NewName" in spec["hierarchy"]["task-1-1"]["metadata"]["description"]

    def test_replace_in_all(self, temp_specs_dir, fr_spec):
        """Should replace text in both titles and descriptions."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "FooBar", "Updated",
            scope="all", specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["total_replacements"] == 5  # 2 titles + 3 descriptions


class TestFindReplaceRegex:
    """Tests for regex find/replace."""

    def test_regex_pattern(self, temp_specs_dir, fr_spec):
        """Should support regex patterns."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", r"Foo\w+", "Widget",
            use_regex=True, scope="all", specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["total_replacements"] >= 1

    def test_regex_backreference(self, temp_specs_dir, fr_spec):
        """Should support regex backreferences."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", r"(Foo)(Bar)", r"\2\1",
            use_regex=True, scope="titles", specs_dir=temp_specs_dir,
        )

        assert error is None
        spec = load_spec("fr-test-001", temp_specs_dir)
        assert "BarFoo" in spec["hierarchy"]["phase-1"]["title"]

    def test_invalid_regex(self, temp_specs_dir, fr_spec):
        """Should reject invalid regex."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "[invalid", "test",
            use_regex=True, specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "Invalid regex" in error


class TestFindReplaceCaseSensitive:
    """Tests for case sensitivity."""

    def test_case_sensitive_default(self, temp_specs_dir, fr_spec):
        """Should be case-sensitive by default."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "foobar", "test",
            case_sensitive=True, specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["total_replacements"] == 0  # No match (wrong case)

    def test_case_insensitive(self, temp_specs_dir, fr_spec):
        """Should support case-insensitive matching."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "foobar", "Widget",
            case_sensitive=False, specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["total_replacements"] >= 1


class TestFindReplaceDryRun:
    """Tests for dry_run mode."""

    def test_dry_run_no_changes(self, temp_specs_dir, fr_spec):
        """Dry run should not modify spec."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "FooBar", "Changed",
            dry_run=True, specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["dry_run"] is True
        assert result["total_replacements"] >= 1

        # Verify no changes persisted
        spec = load_spec("fr-test-001", temp_specs_dir)
        assert "FooBar" in spec["hierarchy"]["phase-1"]["title"]

    def test_dry_run_shows_changes(self, temp_specs_dir, fr_spec):
        """Dry run should show planned changes."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "FooBar", "NewText",
            dry_run=True, specs_dir=temp_specs_dir,
        )

        assert error is None
        assert "changes" in result
        assert len(result["changes"]) > 0
        change = result["changes"][0]
        assert "old" in change
        assert "new" in change


class TestFindReplaceValidation:
    """Tests for validation errors."""

    def test_empty_find(self, temp_specs_dir, fr_spec):
        """Should reject empty find pattern."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "", "test", specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "non-empty" in error.lower()

    def test_invalid_scope(self, temp_specs_dir, fr_spec):
        """Should reject invalid scope."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "test", "new",
            scope="invalid", specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "scope must be" in error.lower()

    def test_spec_not_found(self, temp_specs_dir):
        """Should error for nonexistent spec."""
        result, error = find_replace_in_spec(
            "nonexistent", "test", "new", specs_dir=temp_specs_dir,
        )

        assert result is None
        assert "not found" in error.lower()

    def test_no_matches(self, temp_specs_dir, fr_spec):
        """Should return zero replacements when no matches."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "NonexistentText", "new",
            specs_dir=temp_specs_dir,
        )

        assert error is None
        assert result["total_replacements"] == 0
        assert result["message"] == "No matches found"

    def test_delete_matches(self, temp_specs_dir, fr_spec):
        """Should allow empty replace string to delete matches."""
        spec_file = temp_specs_dir / "active" / "fr-test-001.json"
        spec_file.write_text(json.dumps(fr_spec))

        result, error = find_replace_in_spec(
            "fr-test-001", "FooBar ", "",
            scope="titles", specs_dir=temp_specs_dir,
        )

        assert error is None
        spec = load_spec("fr-test-001", temp_specs_dir)
        # "Implement FooBar feature" -> "Implement feature"
        assert spec["hierarchy"]["phase-1"]["title"] == "Implement feature"
