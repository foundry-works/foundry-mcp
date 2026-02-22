"""Tests for plan linkage in spec creation (plan_path and plan_review_path)."""

import json

import pytest

from foundry_mcp.core.spec import create_spec, generate_spec_data


class TestGenerateSpecDataPlanFields:
    def test_stores_plan_path_in_metadata(self):
        spec_data, error = generate_spec_data(
            name="test-feature",
            mission="Build feature",
            plan_path=".plans/test-feature.md",
        )
        assert error is None
        assert spec_data is not None
        assert spec_data["metadata"]["plan_path"] == ".plans/test-feature.md"

    def test_stores_plan_review_path_in_metadata(self):
        spec_data, error = generate_spec_data(
            name="test-feature",
            mission="Build feature",
            plan_review_path=".plan-reviews/test-feature-review-full.md",
        )
        assert error is None
        assert spec_data is not None
        assert spec_data["metadata"]["plan_review_path"] == ".plan-reviews/test-feature-review-full.md"

    def test_stores_both_paths(self):
        spec_data, error = generate_spec_data(
            name="test-feature",
            mission="Build feature",
            plan_path=".plans/test-feature.md",
            plan_review_path=".plan-reviews/test-feature-review-full.md",
        )
        assert error is None
        assert spec_data["metadata"]["plan_path"] == ".plans/test-feature.md"
        assert spec_data["metadata"]["plan_review_path"] == ".plan-reviews/test-feature-review-full.md"

    def test_omits_plan_path_when_not_provided(self):
        spec_data, error = generate_spec_data(
            name="test-feature",
            mission="Build feature",
        )
        assert error is None
        assert "plan_path" not in spec_data["metadata"]
        assert "plan_review_path" not in spec_data["metadata"]

    def test_strips_whitespace_from_paths(self):
        spec_data, error = generate_spec_data(
            name="test-feature",
            mission="Build feature",
            plan_path="  .plans/test.md  ",
            plan_review_path="  .plan-reviews/test.md  ",
        )
        assert error is None
        assert spec_data["metadata"]["plan_path"] == ".plans/test.md"
        assert spec_data["metadata"]["plan_review_path"] == ".plan-reviews/test.md"


class TestCreateSpecPlanFileValidation:
    def test_fails_when_plan_path_file_missing(self, tmp_path):
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        result, error = create_spec(
            name="test-feature",
            mission="Build feature",
            plan_path=".plans/nonexistent.md",
            specs_dir=specs_dir,
        )
        assert result is None
        assert "Plan file not found" in error

    def test_fails_when_plan_review_path_file_missing(self, tmp_path):
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        # Create the plan file but not the review
        plans_dir = specs_dir / ".plans"
        plans_dir.mkdir()
        (plans_dir / "feature.md").write_text("# Plan")
        result, error = create_spec(
            name="test-feature",
            mission="Build feature",
            plan_path=".plans/feature.md",
            plan_review_path=".plan-reviews/missing-review.md",
            specs_dir=specs_dir,
        )
        assert result is None
        assert "Plan review file not found" in error

    def test_succeeds_when_both_files_exist(self, tmp_path):
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        plans_dir = specs_dir / ".plans"
        plans_dir.mkdir()
        (plans_dir / "feature.md").write_text("# Plan\nBuild it.")
        reviews_dir = specs_dir / ".plan-reviews"
        reviews_dir.mkdir()
        (reviews_dir / "feature-review.md").write_text("# Review\nLooks good.")
        result, error = create_spec(
            name="test-feature",
            mission="Build feature",
            plan_path=".plans/feature.md",
            plan_review_path=".plan-reviews/feature-review.md",
            specs_dir=specs_dir,
        )
        assert error is None
        assert result is not None
        assert result["spec_id"] is not None
        # Verify the spec file contains the plan paths
        spec_path = result["spec_path"]
        with open(spec_path) as f:
            spec_data = json.load(f)
        assert spec_data["metadata"]["plan_path"] == ".plans/feature.md"
        assert spec_data["metadata"]["plan_review_path"] == ".plan-reviews/feature-review.md"

    def test_succeeds_without_plan_paths(self, tmp_path):
        """Backward compat: create_spec works without plan paths."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        result, error = create_spec(
            name="test-feature",
            mission="Build feature",
            specs_dir=specs_dir,
        )
        assert error is None
        assert result is not None
