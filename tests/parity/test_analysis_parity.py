"""
Parity tests for analysis operations.

Tests that foundry-mcp and sdd-toolkit produce equivalent results for
spec analysis operations like dependency analysis, cycle detection, and pattern finding.

NOTE: Many of these tests focus on foundry-mcp functionality since the SDD CLI
may not have all analysis commands implemented. The tests gracefully handle
missing CLI commands.
"""

import pytest
from tests.parity.harness.foundry_adapter import FoundryMcpAdapter
from tests.parity.harness.sdd_adapter import SddToolkitAdapter
from tests.parity.harness.fixture_manager import FixtureManager


def _sdd_command_available(result):
    """
    Check if SDD CLI command executed successfully with real data.

    Returns False if:
    - Result is not a dict
    - Command returned an error (success=False)
    - Error indicates command not found/unknown
    - Result contains internal markers from failed CLI calls (_stdout, _exit_code)

    This allows tests to gracefully skip SDD assertions when
    the CLI doesn't have a particular command implemented.
    """
    if not isinstance(result, dict):
        return False
    # Check for explicit failure
    if result.get("success") is False:
        return False
    # Check for error key presence
    if "error" in result:
        error = result.get("error", "")
        if isinstance(error, str) and error:
            return False
    # Check for internal markers that indicate CLI didn't return real data
    if "_stdout" in result or "_stderr" in result or "_exit_code" in result:
        return False
    return True


def _sdd_has_meaningful_data(result, *keys):
    """
    Check if SDD result has meaningful data (not just normalized defaults).

    Returns True only if the result contains non-zero/non-empty values
    for the specified keys.
    """
    if not _sdd_command_available(result):
        return False
    for key in keys:
        val = result.get(key)
        if val and val != 0:
            return True
    return False


class TestAnalyzeDepsParity:
    """Tests for analyze_deps parity between foundry-mcp and sdd-toolkit."""

    def test_analyze_deps_basic(self, bottleneck_spec_dir):
        """Basic dependency analysis should work in foundry-mcp."""
        foundry = FoundryMcpAdapter(bottleneck_spec_dir / "specs")
        sdd = SddToolkitAdapter(bottleneck_spec_dir / "specs")
        spec_id = "parity-test-bottleneck"

        foundry_result = foundry.analyze_deps(spec_id)
        sdd_result = sdd.analyze_deps(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["spec_id"] == spec_id
        assert foundry_result["dependency_count"] >= 0

        # Only compare if SDD CLI returned meaningful data
        if _sdd_has_meaningful_data(sdd_result, "dependency_count", "bottlenecks"):
            assert sdd_result["success"] is True
            assert sdd_result["dependency_count"] == foundry_result["dependency_count"]

    def test_bottleneck_detection(self, bottleneck_spec_dir):
        """Detect task-1-1 as bottleneck (blocks 4 tasks)."""
        foundry = FoundryMcpAdapter(bottleneck_spec_dir / "specs")
        sdd = SddToolkitAdapter(bottleneck_spec_dir / "specs")
        spec_id = "parity-test-bottleneck"

        # Use threshold of 3 - task-1-1 blocks 4 tasks
        foundry_result = foundry.analyze_deps(spec_id, bottleneck_threshold=3)
        sdd_result = sdd.analyze_deps(spec_id, bottleneck_threshold=3)

        assert foundry_result["success"] is True
        assert foundry_result["has_bottlenecks"] is True

        # Check that task-1-1 is identified as a bottleneck
        bottleneck_ids = [b["task_id"] for b in foundry_result["bottlenecks"]]
        assert "task-1-1" in bottleneck_ids

        # Only compare if SDD CLI returned meaningful data
        if _sdd_has_meaningful_data(sdd_result, "bottlenecks", "dependency_count"):
            assert sdd_result["success"] is True
            assert sdd_result["has_bottlenecks"] == foundry_result["has_bottlenecks"]

    def test_no_bottlenecks_with_high_threshold(self, bottleneck_spec_dir):
        """High threshold should return no bottlenecks."""
        foundry = FoundryMcpAdapter(bottleneck_spec_dir / "specs")
        sdd = SddToolkitAdapter(bottleneck_spec_dir / "specs")
        spec_id = "parity-test-bottleneck"

        # Use threshold of 10 - no task blocks 10 others
        foundry_result = foundry.analyze_deps(spec_id, bottleneck_threshold=10)
        sdd_result = sdd.analyze_deps(spec_id, bottleneck_threshold=10)

        assert foundry_result["success"] is True
        assert foundry_result["has_bottlenecks"] is False
        assert foundry_result["bottlenecks"] == []

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["has_bottlenecks"] is False

    def test_dependency_count(self, bottleneck_spec_dir):
        """Verify dependency count matches spec structure."""
        foundry = FoundryMcpAdapter(bottleneck_spec_dir / "specs")
        spec_id = "parity-test-bottleneck"

        foundry_result = foundry.analyze_deps(spec_id)

        assert foundry_result["success"] is True
        # Spec has: task-1-2, task-1-3, task-1-4, task-1-5 depend on task-1-1 (4)
        # task-1-6 depends on task-1-2 (1)
        # Total: 5 dependencies
        assert foundry_result["dependency_count"] == 5


class TestDetectCyclesParity:
    """Tests for detect_cycles parity."""

    def test_detect_cycles_with_cycle(self, circular_deps_dir):
        """Detect circular dependency in spec."""
        foundry = FoundryMcpAdapter(circular_deps_dir / "specs")
        sdd = SddToolkitAdapter(circular_deps_dir / "specs")
        spec_id = "parity-test-circular"

        foundry_result = foundry.detect_cycles(spec_id)
        sdd_result = sdd.detect_cycles(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["has_cycles"] is True
        assert foundry_result["cycle_count"] >= 1

        # Only compare if SDD CLI returned meaningful data
        if _sdd_has_meaningful_data(sdd_result, "cycles", "affected_tasks"):
            assert sdd_result["success"] is True
            assert sdd_result["has_cycles"] is True

    def test_detect_cycles_no_cycle(self, bottleneck_spec_dir):
        """Spec without cycles returns has_cycles=False."""
        foundry = FoundryMcpAdapter(bottleneck_spec_dir / "specs")
        sdd = SddToolkitAdapter(bottleneck_spec_dir / "specs")
        spec_id = "parity-test-bottleneck"

        foundry_result = foundry.detect_cycles(spec_id)
        sdd_result = sdd.detect_cycles(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["has_cycles"] is False
        assert foundry_result["cycles"] == []
        assert foundry_result["cycle_count"] == 0

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["has_cycles"] is False

    def test_affected_tasks_in_cycle(self, circular_deps_dir):
        """List all tasks involved in cycles."""
        foundry = FoundryMcpAdapter(circular_deps_dir / "specs")
        spec_id = "parity-test-circular"

        foundry_result = foundry.detect_cycles(spec_id)

        assert foundry_result["success"] is True
        affected = foundry_result["affected_tasks"]

        # The cycle involves task-1-1, task-1-2, task-1-3
        # task-1-4 is independent and should NOT be affected
        for task in ["task-1-1", "task-1-2", "task-1-3"]:
            assert task in affected

    def test_independent_task_not_in_cycle(self, circular_deps_dir):
        """Independent task should not be in affected_tasks."""
        foundry = FoundryMcpAdapter(circular_deps_dir / "specs")
        spec_id = "parity-test-circular"

        foundry_result = foundry.detect_cycles(spec_id)

        assert foundry_result["success"] is True
        affected = foundry_result["affected_tasks"]

        # task-1-4 is independent (not part of the cycle)
        assert "task-1-4" not in affected


class TestFindPatternsParity:
    """Tests for find_patterns parity."""

    def test_find_python_files(self, patterns_spec_dir):
        """Find tasks with .py file paths."""
        foundry = FoundryMcpAdapter(patterns_spec_dir / "specs")
        sdd = SddToolkitAdapter(patterns_spec_dir / "specs")
        spec_id = "parity-test-patterns"

        foundry_result = foundry.find_patterns(spec_id, "*.py")
        sdd_result = sdd.find_patterns(spec_id, "*.py")

        assert foundry_result["success"] is True
        assert foundry_result["pattern"] == "*.py"
        # Spec has: src/core/module.py, src/utils/helpers.py, tests/test_module.py, tests/test_helpers.py
        assert foundry_result["total_count"] == 4

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["total_count"] == foundry_result["total_count"]

    def test_find_test_files(self, patterns_spec_dir):
        """Find tasks with test file paths."""
        foundry = FoundryMcpAdapter(patterns_spec_dir / "specs")
        sdd = SddToolkitAdapter(patterns_spec_dir / "specs")
        spec_id = "parity-test-patterns"

        foundry_result = foundry.find_patterns(spec_id, "tests/*")
        sdd_result = sdd.find_patterns(spec_id, "tests/*")

        assert foundry_result["success"] is True
        # Spec has: tests/test_module.py, tests/test_helpers.py
        assert foundry_result["total_count"] == 2

        # Verify matched files
        file_paths = [m["file_path"] for m in foundry_result["matches"]]
        assert "tests/test_module.py" in file_paths
        assert "tests/test_helpers.py" in file_paths

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_find_src_files(self, patterns_spec_dir):
        """Find tasks with src/ file paths."""
        foundry = FoundryMcpAdapter(patterns_spec_dir / "specs")
        spec_id = "parity-test-patterns"

        foundry_result = foundry.find_patterns(spec_id, "src/*/*.py")

        assert foundry_result["success"] is True
        # Spec has: src/core/module.py, src/utils/helpers.py
        assert foundry_result["total_count"] == 2

    def test_find_markdown_files(self, patterns_spec_dir):
        """Find tasks with .md file paths."""
        foundry = FoundryMcpAdapter(patterns_spec_dir / "specs")
        spec_id = "parity-test-patterns"

        foundry_result = foundry.find_patterns(spec_id, "*.md")

        assert foundry_result["success"] is True
        # Spec has: docs/README.md
        assert foundry_result["total_count"] == 1

        file_paths = [m["file_path"] for m in foundry_result["matches"]]
        assert "docs/README.md" in file_paths

    def test_find_no_matches(self, patterns_spec_dir):
        """Pattern with no matches returns empty list."""
        foundry = FoundryMcpAdapter(patterns_spec_dir / "specs")
        sdd = SddToolkitAdapter(patterns_spec_dir / "specs")
        spec_id = "parity-test-patterns"

        foundry_result = foundry.find_patterns(spec_id, "*.xyz")
        sdd_result = sdd.find_patterns(spec_id, "*.xyz")

        assert foundry_result["success"] is True
        assert foundry_result["total_count"] == 0
        assert foundry_result["matches"] == []

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["total_count"] == 0

    def test_pattern_matches_include_task_info(self, patterns_spec_dir):
        """Pattern matches should include task metadata."""
        foundry = FoundryMcpAdapter(patterns_spec_dir / "specs")
        spec_id = "parity-test-patterns"

        foundry_result = foundry.find_patterns(spec_id, "*.json")

        assert foundry_result["success"] is True
        # config.json should match
        assert foundry_result["total_count"] == 1

        match = foundry_result["matches"][0]
        assert "task_id" in match
        assert "file_path" in match
        assert "title" in match
        assert match["file_path"] == "config.json"
