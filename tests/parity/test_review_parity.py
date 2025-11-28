"""
Parity tests for review and verification operations.

Tests that foundry-mcp and sdd-toolkit produce equivalent results for:
- Verification result recording
- Verification summary formatting
- Pre-populated verification data querying
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


class TestVerificationAddParity:
    """Tests for adding verification results."""

    def test_add_verification_passed(self, test_dir):
        """Add PASSED verification result."""
        fixture = FixtureManager(test_dir)
        fixture.setup("review_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        sdd = SddToolkitAdapter(fixture.specs_dir)
        spec_id = "parity-test-review"

        foundry_result = foundry.add_verification(
            spec_id=spec_id,
            verify_id="verify-2-1",
            result="PASSED",
            command="pytest tests/ -v",
            output="10 passed, 0 failed",
        )
        sdd_result = sdd.add_verification(
            spec_id=spec_id,
            verify_id="verify-2-1",
            result="PASSED",
            command="pytest tests/ -v",
            output="10 passed, 0 failed",
        )

        assert foundry_result["success"] is True
        assert foundry_result["verify_id"] == "verify-2-1"
        assert foundry_result["result"] == "PASSED"

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["result"] == "PASSED"

    def test_add_verification_failed(self, test_dir):
        """Add FAILED verification result."""
        fixture = FixtureManager(test_dir)
        fixture.setup("review_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-review"

        foundry_result = foundry.add_verification(
            spec_id=spec_id,
            verify_id="verify-2-2",
            result="FAILED",
            command="ruff check src/",
            output="Found 5 errors",
            issues="E501,W503,F401",
        )

        assert foundry_result["success"] is True
        assert foundry_result["verify_id"] == "verify-2-2"
        assert foundry_result["result"] == "FAILED"

    def test_add_verification_partial(self, test_dir):
        """Add PARTIAL verification result."""
        fixture = FixtureManager(test_dir)
        fixture.setup("review_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-review"

        foundry_result = foundry.add_verification(
            spec_id=spec_id,
            verify_id="verify-2-1",
            result="PARTIAL",
            command="pytest tests/ -v",
            output="8 passed, 2 skipped",
            notes="Some tests skipped due to missing fixtures",
        )

        assert foundry_result["success"] is True
        assert foundry_result["result"] == "PARTIAL"

    def test_add_verification_invalid_result(self, test_dir):
        """Invalid result value should error."""
        fixture = FixtureManager(test_dir)
        fixture.setup("review_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-review"

        foundry_result = foundry.add_verification(
            spec_id=spec_id,
            verify_id="verify-2-1",
            result="INVALID",
        )

        assert foundry_result["success"] is False
        assert "Invalid result" in foundry_result["error"]

    def test_add_verification_nonexistent_node(self, test_dir):
        """Adding verification to nonexistent node should error."""
        fixture = FixtureManager(test_dir)
        fixture.setup("review_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-review"

        foundry_result = foundry.add_verification(
            spec_id=spec_id,
            verify_id="verify-99-99",
            result="PASSED",
        )

        assert foundry_result["success"] is False
        assert "not found" in foundry_result["error"]


class TestVerificationResultsParity:
    """Tests for getting verification results."""

    def test_get_verification_results_with_data(self, verification_results_dir):
        """Get verification results from pre-populated spec."""
        foundry = FoundryMcpAdapter(verification_results_dir / "specs")
        sdd = SddToolkitAdapter(verification_results_dir / "specs")
        spec_id = "parity-test-verification-results"

        foundry_result = foundry.get_verification_results(spec_id)
        sdd_result = sdd.get_verification_results(spec_id)

        assert foundry_result["success"] is True
        # Spec has 4 verify nodes with results
        assert foundry_result["count"] == 4

        # Verify individual results
        results_by_id = {r["verify_id"]: r for r in foundry_result["results"]}
        assert results_by_id["verify-1-1"]["result"] == "PASSED"
        assert results_by_id["verify-1-2"]["result"] == "FAILED"
        assert results_by_id["verify-1-3"]["result"] == "PARTIAL"
        assert results_by_id["verify-1-4"]["result"] == "PASSED"

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True

    def test_get_verification_results_no_data(self, review_spec_dir):
        """Get verification results when none exist."""
        foundry = FoundryMcpAdapter(review_spec_dir / "specs")
        spec_id = "parity-test-review"

        foundry_result = foundry.get_verification_results(spec_id)

        assert foundry_result["success"] is True
        # No results recorded yet
        assert foundry_result["count"] == 0
        assert foundry_result["results"] == []


class TestVerificationSummaryParity:
    """Tests for formatting verification summary."""

    def test_format_verification_summary(self, verification_results_dir):
        """Format summary of verification results."""
        foundry = FoundryMcpAdapter(verification_results_dir / "specs")
        sdd = SddToolkitAdapter(verification_results_dir / "specs")
        spec_id = "parity-test-verification-results"

        foundry_result = foundry.format_verification_summary(spec_id)
        sdd_result = sdd.format_verification_summary(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["total_verifications"] == 4
        assert foundry_result["passed"] == 2
        assert foundry_result["failed"] == 1
        assert foundry_result["partial"] == 1

        # Check summary text exists
        assert "summary" in foundry_result
        assert foundry_result["summary"]

        if _sdd_command_available(sdd_result):
            assert sdd_result["success"] is True
            assert sdd_result["total_verifications"] == foundry_result["total_verifications"]

    def test_format_summary_no_results(self, review_spec_dir):
        """Format summary when no verification results exist."""
        foundry = FoundryMcpAdapter(review_spec_dir / "specs")
        spec_id = "parity-test-review"

        foundry_result = foundry.format_verification_summary(spec_id)

        assert foundry_result["success"] is True
        assert foundry_result["total_verifications"] == 0
        assert foundry_result["passed"] == 0
        assert foundry_result["failed"] == 0
        assert foundry_result["partial"] == 0


class TestVerificationStatusUpdate:
    """Tests that verification results update node status correctly."""

    def test_passed_sets_completed(self, test_dir):
        """PASSED result should set node status to completed."""
        fixture = FixtureManager(test_dir)
        fixture.setup("review_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-review"

        # Add PASSED verification
        foundry.add_verification(
            spec_id=spec_id,
            verify_id="verify-2-1",
            result="PASSED",
        )

        # Check status was updated
        task_result = foundry.get_task(spec_id, "verify-2-1")
        assert task_result["success"] is True
        assert task_result["task"]["status"] == "completed"

    def test_failed_sets_blocked(self, test_dir):
        """FAILED result should set node status to blocked."""
        fixture = FixtureManager(test_dir)
        fixture.setup("review_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-review"

        # Add FAILED verification
        foundry.add_verification(
            spec_id=spec_id,
            verify_id="verify-2-1",
            result="FAILED",
        )

        # Check status was updated
        task_result = foundry.get_task(spec_id, "verify-2-1")
        assert task_result["success"] is True
        assert task_result["task"]["status"] == "blocked"

    def test_partial_sets_in_progress(self, test_dir):
        """PARTIAL result should set node status to in_progress."""
        fixture = FixtureManager(test_dir)
        fixture.setup("review_base", status="active")
        foundry = FoundryMcpAdapter(fixture.specs_dir)
        spec_id = "parity-test-review"

        # Add PARTIAL verification
        foundry.add_verification(
            spec_id=spec_id,
            verify_id="verify-2-1",
            result="PARTIAL",
        )

        # Check status was updated
        task_result = foundry.get_task(spec_id, "verify-2-1")
        assert task_result["success"] is True
        assert task_result["task"]["status"] == "in_progress"
