"""
Parity tests for validation operations.

Tests validate_spec, fix_spec, and spec_stats operations.
"""

import pytest

from .harness.normalizers import normalize_for_comparison
from .harness.comparators import ResultComparator


class TestValidateSpec:
    """Parity tests for spec validation."""

    @pytest.mark.parity
    def test_validate_spec_parity(self, both_adapters):
        """Test that validate_spec returns equivalent results."""
        foundry, sdd = both_adapters

        foundry_result = foundry.validate_spec("parity-test-simple")
        sdd_result = sdd.validate_spec("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "validate_spec")

        # Check is_valid matches
        ResultComparator.assert_key_match(
            foundry_result, sdd_result, "is_valid", "validate_spec"
        )

    @pytest.mark.parity
    def test_validate_spec_valid_parity(self, both_adapters):
        """Test validation of a known-valid spec."""
        foundry, sdd = both_adapters

        foundry_result = foundry.validate_spec("parity-test-simple")
        sdd_result = sdd.validate_spec("parity-test-simple")

        # Both should report valid
        ResultComparator.assert_success(foundry_result, sdd_result, "validate_spec(valid)")
        assert foundry_result.get("is_valid") is True

    @pytest.mark.parity
    def test_validate_nonexistent_spec_parity(self, both_adapters):
        """Test validation of non-existent spec."""
        foundry, sdd = both_adapters

        foundry_result = foundry.validate_spec("nonexistent-spec-xyz")
        sdd_result = sdd.validate_spec("nonexistent-spec-xyz")

        # Both should return error
        ResultComparator.assert_both_error(
            foundry_result, sdd_result, "validate_spec(nonexistent)"
        )


class TestFixSpec:
    """Parity tests for spec auto-fix."""

    @pytest.mark.parity
    def test_fix_spec_parity(self, isolated_adapters):
        """Test that fix_spec produces equivalent results."""
        foundry, sdd = isolated_adapters

        foundry_result = foundry.fix_spec("parity-test-simple")
        sdd_result = sdd.fix_spec("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "fix_spec")

    @pytest.mark.parity
    def test_fix_spec_already_valid_parity(self, isolated_adapters):
        """Test fix_spec on already valid spec."""
        foundry, sdd = isolated_adapters

        foundry_result = foundry.fix_spec("parity-test-simple")
        sdd_result = sdd.fix_spec("parity-test-simple")

        # Both should succeed with no/minimal fixes
        ResultComparator.assert_success(
            foundry_result, sdd_result, "fix_spec(already valid)"
        )


class TestSpecStats:
    """Parity tests for spec statistics."""

    @pytest.mark.parity
    def test_spec_stats_parity(self, both_adapters):
        """Test that spec_stats returns equivalent results."""
        foundry, sdd = both_adapters

        foundry_result = foundry.spec_stats("parity-test-simple")
        sdd_result = sdd.spec_stats("parity-test-simple")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "spec_stats")

        # Check spec_id matches
        ResultComparator.assert_key_match(
            foundry_result, sdd_result, "spec_id", "spec_stats"
        )

    @pytest.mark.parity
    def test_spec_stats_multi_phase_parity(self, fixture_manager):
        """Test spec_stats for multi-phase spec."""
        from .harness.foundry_adapter import FoundryMcpAdapter
        from .harness.sdd_adapter import SddToolkitAdapter

        fixture_manager.setup("multi_phase_spec", status="active")
        specs_dir = fixture_manager.specs_dir

        foundry = FoundryMcpAdapter(specs_dir)
        sdd = SddToolkitAdapter(specs_dir)

        foundry_result = foundry.spec_stats("parity-test-multi-phase")
        sdd_result = sdd.spec_stats("parity-test-multi-phase")

        # Check both succeeded
        ResultComparator.assert_success(foundry_result, sdd_result, "spec_stats(multi-phase)")


# Standalone tests for foundry adapter
class TestFoundryValidationOperations:
    """Tests for foundry adapter validation operations."""

    def test_foundry_validate_spec(self, foundry_adapter):
        """Test foundry adapter validate_spec works."""
        result = foundry_adapter.validate_spec("parity-test-simple")
        assert result.get("success") is True
        assert "is_valid" in result

    def test_foundry_fix_spec(self, foundry_adapter):
        """Test foundry adapter fix_spec works."""
        result = foundry_adapter.fix_spec("parity-test-simple")
        assert result.get("success") is True

    def test_foundry_spec_stats(self, foundry_adapter):
        """Test foundry adapter spec_stats works."""
        result = foundry_adapter.spec_stats("parity-test-simple")
        assert result.get("success") is True
        assert result.get("spec_id") == "parity-test-simple"
