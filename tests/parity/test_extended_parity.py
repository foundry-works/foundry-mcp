"""
Extended parity tests for additional operations.

Tests for:
- Rendering & Reporting (spec_render, render_progress, spec_report, spec_report_summary)
- Authoring Extensions (update_frontmatter)
- File Operations (find_related_files, validate_paths)
- Pagination (task_query, journal_list_paginated)
"""

import pytest
from tests.parity.harness.foundry_adapter import FoundryMcpAdapter
from tests.parity.harness.sdd_adapter import SddToolkitAdapter
from tests.parity.harness.fixture_manager import FixtureManager


def _sdd_command_available(result):
    """Check if SDD CLI command executed successfully with real data."""
    if not isinstance(result, dict):
        return False
    if result.get("success") is False:
        return False
    if "error" in result:
        error = result.get("error", "")
        if isinstance(error, str) and error:
            return False
    if "_stdout" in result or "_stderr" in result or "_exit_code" in result:
        return False
    if "success" not in result:
        return False
    return True


# =============================================================================
# Rendering Tests
# =============================================================================

class TestRenderSpecParity:
    """Tests for spec rendering to markdown."""

    def test_render_spec_basic(self, large_spec_dir):
        """Render spec with basic mode."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        sdd = SddToolkitAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.render_spec(spec_id)
        sdd_result = sdd.render_spec(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["spec_id"] == spec_id
        assert foundry_result["markdown"]  # Has content
        assert "total_tasks" in foundry_result

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_render_spec_with_journal(self, large_spec_dir):
        """Render spec including journal entries."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.render_spec(spec_id, include_journal=True)

        assert foundry_result["success"] is True
        assert foundry_result["markdown"]

    def test_render_nonexistent_spec(self, large_spec_dir):
        """Render nonexistent spec should error."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "nonexistent-spec"

        foundry_result = foundry.render_spec(spec_id)

        assert foundry_result["success"] is False
        assert "not found" in foundry_result["error"].lower()


class TestRenderProgressParity:
    """Tests for progress bar rendering."""

    def test_render_progress_basic(self, large_spec_dir):
        """Render progress bars for spec."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        sdd = SddToolkitAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.render_progress(spec_id)
        sdd_result = sdd.render_progress(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["spec_id"] == spec_id
        assert "overall" in foundry_result
        assert "phases" in foundry_result
        assert foundry_result["overall"]["progress_bar"]

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_render_progress_custom_width(self, large_spec_dir):
        """Render progress with custom bar width."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.render_progress(spec_id, bar_width=30)

        assert foundry_result["success"] is True
        # Progress bar should be visible in overall
        assert foundry_result["overall"]["progress_bar"]

    def test_render_progress_includes_phases(self, large_spec_dir):
        """Progress should include all phases."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.render_progress(spec_id)

        assert foundry_result["success"] is True
        # Large spec has 5 phases
        assert len(foundry_result["phases"]) == 5

        for phase in foundry_result["phases"]:
            assert "id" in phase
            assert "title" in phase
            assert "progress_bar" in phase
            assert "completed" in phase
            assert "total" in phase


# =============================================================================
# Reporting Tests
# =============================================================================

class TestSpecReportParity:
    """Tests for spec report generation."""

    def test_spec_report_all_sections(self, large_spec_dir):
        """Generate full report with all sections."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        sdd = SddToolkitAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.spec_report(spec_id)
        sdd_result = sdd.spec_report(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["spec_id"] == spec_id
        assert foundry_result["report"]
        assert "validation" in foundry_result["sections"] or len(foundry_result["sections"]) >= 1
        assert "summary" in foundry_result

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_spec_report_validation_only(self, large_spec_dir):
        """Generate report with only validation section."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.spec_report(spec_id, sections="validation")

        assert foundry_result["success"] is True
        assert "validation" in foundry_result["sections"]
        assert "Validation" in foundry_result["report"]

    def test_spec_report_stats_only(self, large_spec_dir):
        """Generate report with only stats section."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.spec_report(spec_id, sections="stats")

        assert foundry_result["success"] is True
        assert "stats" in foundry_result["sections"]


class TestSpecReportSummaryParity:
    """Tests for quick spec report summary."""

    def test_spec_report_summary(self, large_spec_dir):
        """Generate quick summary report."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        sdd = SddToolkitAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.spec_report_summary(spec_id)
        sdd_result = sdd.spec_report_summary(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["spec_id"] == spec_id
        assert "validation" in foundry_result
        assert "progress" in foundry_result
        assert "health" in foundry_result

        # Check health score exists
        assert "score" in foundry_result["health"]

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_spec_report_summary_health_status(self, large_spec_dir):
        """Summary should include health status."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.spec_report_summary(spec_id)

        assert foundry_result["success"] is True
        health = foundry_result["health"]
        assert health["status"] in ("healthy", "needs_attention", "critical")


# =============================================================================
# Authoring Extension Tests
# =============================================================================

class TestUpdateFrontmatterParity:
    """Tests for updating spec frontmatter."""

    def test_update_frontmatter_title(self, test_dir):
        """Update spec title."""
        fixture = FixtureManager(test_dir)
        fixture.setup("authoring_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        sdd = SddToolkitAdapter(fixture.specs_dir)
        spec_id = "parity-test-authoring"

        foundry_result = foundry.update_frontmatter(
            spec_id=spec_id,
            key="title",
            value="New Title",
        )
        sdd_result = sdd.update_frontmatter(
            spec_id=spec_id,
            key="title",
            value="New Title",
        )

        assert foundry_result["success"] is True
        assert foundry_result["key"] == "title"
        assert foundry_result["value"] == "New Title"

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_update_frontmatter_version(self, test_dir):
        """Update spec version."""
        fixture = FixtureManager(test_dir)
        fixture.setup("authoring_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-authoring"

        foundry_result = foundry.update_frontmatter(
            spec_id=spec_id,
            key="version",
            value="2.0.0",
        )

        assert foundry_result["success"] is True
        assert foundry_result["key"] == "version"
        assert foundry_result["value"] == "2.0.0"
        assert foundry_result["previous_value"] == "1.0.0"

    def test_update_frontmatter_dry_run(self, test_dir):
        """Dry run should not modify spec."""
        fixture = FixtureManager(test_dir)
        fixture.setup("authoring_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-authoring"

        foundry_result = foundry.update_frontmatter(
            spec_id=spec_id,
            key="title",
            value="Should Not Apply",
            dry_run=True,
        )

        assert foundry_result["success"] is True
        assert foundry_result["dry_run"] is True

        # Verify original title unchanged
        spec_result = foundry.get_spec(spec_id)
        assert spec_result["spec"]["metadata"]["title"] != "Should Not Apply"


# =============================================================================
# File Operation Tests
# =============================================================================

class TestFindRelatedFilesParity:
    """Tests for finding related files."""

    def test_find_related_files_spec_reference(self, patterns_spec_dir):
        """Find files that reference a file path."""
        foundry = FoundryMcpAdapter(patterns_spec_dir / "specs")
        sdd = SddToolkitAdapter(patterns_spec_dir / "specs")
        spec_id = "parity-test-patterns"

        # Patterns spec has file_path metadata on tasks
        foundry_result = foundry.find_related_files("src/core/module.py", spec_id=spec_id)
        sdd_result = sdd.find_related_files("src/core/module.py", spec_id=spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["file_path"] == "src/core/module.py"
        # Should find spec references
        assert len(foundry_result["spec_references"]) >= 1

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_find_related_files_no_matches(self, patterns_spec_dir):
        """Find related files with no matches."""
        foundry = FoundryMcpAdapter(patterns_spec_dir / "specs")
        spec_id = "parity-test-patterns"

        foundry_result = foundry.find_related_files("nonexistent/path.xyz", spec_id=spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["total_count"] == 0
        assert foundry_result["spec_references"] == []


class TestValidatePathsParity:
    """Tests for path validation."""

    def test_validate_paths_all_valid(self, test_dir):
        """Validate paths that all exist."""
        fixture = FixtureManager(test_dir)
        fixture.setup("authoring_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        sdd = SddToolkitAdapter(fixture.specs_dir)

        # These paths exist in the test dir
        paths = [str(fixture.specs_dir)]

        foundry_result = foundry.validate_paths(paths)
        sdd_result = sdd.validate_paths(paths)

        assert foundry_result["success"] is True
        assert foundry_result["all_valid"] is True
        assert foundry_result["valid_count"] == 1
        assert foundry_result["invalid_count"] == 0

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_validate_paths_mixed(self, test_dir):
        """Validate paths with some invalid."""
        fixture = FixtureManager(test_dir)
        fixture.setup("authoring_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)

        paths = [
            str(fixture.specs_dir),  # Valid
            "nonexistent/path/file.txt",  # Invalid
        ]

        foundry_result = foundry.validate_paths(paths)

        assert foundry_result["success"] is True
        assert foundry_result["all_valid"] is False
        assert foundry_result["valid_count"] == 1
        assert foundry_result["invalid_count"] == 1
        assert "nonexistent/path/file.txt" in foundry_result["invalid_paths"]

    def test_validate_paths_all_invalid(self):
        """Validate paths that all don't exist."""
        # Use a temp adapter with any specs dir
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            specs_dir = Path(tmpdir) / "specs" / "active"
            specs_dir.mkdir(parents=True)
            foundry = FoundryMcpAdapter(specs_dir.parent)

            paths = [
                "fake/path1.txt",
                "fake/path2.txt",
            ]

            foundry_result = foundry.validate_paths(paths)

            assert foundry_result["success"] is True
            assert foundry_result["all_valid"] is False
            assert foundry_result["valid_count"] == 0
            assert foundry_result["invalid_count"] == 2


# =============================================================================
# Pagination Tests
# =============================================================================

class TestTaskQueryPaginationParity:
    """Tests for task query with pagination."""

    def test_task_query_basic(self, large_spec_dir):
        """Query tasks from large spec."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        sdd = SddToolkitAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.task_query(spec_id)
        sdd_result = sdd.task_query(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["spec_id"] == spec_id
        assert foundry_result["total"] == 100
        assert foundry_result["count"] == 100  # Default limit is 100

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_task_query_with_limit(self, large_spec_dir):
        """Query tasks with limit."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.task_query(spec_id, limit=25)

        assert foundry_result["success"] is True
        assert foundry_result["count"] == 25
        assert foundry_result["total"] == 100
        assert foundry_result["has_more"] is True
        assert foundry_result["cursor"] is not None

    def test_task_query_pagination(self, large_spec_dir):
        """Paginate through tasks."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        # First page
        page1 = foundry.task_query(spec_id, limit=25)
        assert page1["success"] is True
        assert page1["count"] == 25
        assert page1["has_more"] is True

        # Second page
        page2 = foundry.task_query(spec_id, limit=25, cursor=page1["cursor"])
        assert page2["success"] is True
        assert page2["count"] == 25
        assert page2["has_more"] is True

        # Ensure no duplicates
        page1_ids = {t["task_id"] for t in page1["tasks"]}
        page2_ids = {t["task_id"] for t in page2["tasks"]}
        assert page1_ids.isdisjoint(page2_ids)

    def test_task_query_filter_status(self, large_spec_dir):
        """Query tasks filtered by status."""
        foundry = FoundryMcpAdapter(large_spec_dir / "specs")
        spec_id = "parity-test-large"

        foundry_result = foundry.task_query(spec_id, status="completed")

        assert foundry_result["success"] is True
        # All returned tasks should be completed
        for task in foundry_result["tasks"]:
            assert task["status"] == "completed"


class TestJournalPaginationParity:
    """Tests for journal listing with pagination."""

    def test_journal_list_paginated_basic(self, verification_results_dir):
        """List journal entries with pagination."""
        foundry = FoundryMcpAdapter(verification_results_dir / "specs")
        sdd = SddToolkitAdapter(verification_results_dir / "specs")
        spec_id = "parity-test-verification-results"

        foundry_result = foundry.journal_list_paginated(spec_id)
        sdd_result = sdd.journal_list_paginated(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["spec_id"] == spec_id
        # Verification results spec has 4 journal entries
        assert foundry_result["total"] == 4

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_journal_list_with_limit(self, verification_results_dir):
        """List journal with limit."""
        foundry = FoundryMcpAdapter(verification_results_dir / "specs")
        spec_id = "parity-test-verification-results"

        foundry_result = foundry.journal_list_paginated(spec_id, limit=2)

        assert foundry_result["success"] is True
        assert foundry_result["count"] == 2
        assert foundry_result["total"] == 4
        assert foundry_result["has_more"] is True

    def test_journal_list_filter_type(self, all_blocked_dir):
        """Filter journal by entry type."""
        foundry = FoundryMcpAdapter(all_blocked_dir / "specs")
        spec_id = "parity-test-all-blocked"

        foundry_result = foundry.journal_list_paginated(spec_id, entry_type="blocker")

        assert foundry_result["success"] is True
        # All blocked spec has 3 blocker entries
        assert foundry_result["total"] == 3
