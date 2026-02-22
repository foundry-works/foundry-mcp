"""Tests for spec review persistence and metadata update (Fixes 2 & 3)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_config(specs_dir):
    config = MagicMock()
    config.specs_dir = Path(specs_dir)
    return config


def _make_ai_result(success=True, plan_enhanced=False, response="", title="Test Spec"):
    """Build a mock result dict matching asdict(success_response(...)) structure."""
    data = {
        "plan_enhanced": plan_enhanced,
        "response": response,
        "title": title,
        "template_id": "SPEC_REVIEW_V1",
        "ai_provider": "test-provider",
    }
    return {
        "success": success,
        "data": data,
        "error": None,
        "meta": {"version": "response-v2"},
    }


class TestSpecReviewPersistence:
    """Tests for always-persist spec review (Fix 2)."""

    def test_standalone_review_persisted(self, tmp_path):
        """A standalone review (no plan) should be persisted to .spec-reviews/."""
        from foundry_mcp.tools.unified.review import _handle_spec_review

        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        # Create a minimal spec file
        pending_dir = specs_dir / "pending"
        pending_dir.mkdir()
        spec_data = {"spec_id": "test-spec", "title": "Test Spec", "metadata": {}, "hierarchy": {}}
        (pending_dir / "test-spec.json").write_text(json.dumps(spec_data))

        review_json = json.dumps({"verdict": "pass", "summary": "Looks good"})
        ai_result = _make_ai_result(success=True, plan_enhanced=False, response=review_json)

        config = _make_config(specs_dir)

        with patch("foundry_mcp.tools.unified.review._get_llm_status", return_value="available"):
            with patch("foundry_mcp.tools.unified.review._build_plan_context", return_value=None):
                with patch("foundry_mcp.tools.unified.review._run_ai_review", return_value=ai_result):
                    with patch("foundry_mcp.tools.unified.review.update_frontmatter") as mock_fm:
                        result = _handle_spec_review(
                            config=config,
                            payload={
                                "spec_id": "test-spec",
                                "path": str(specs_dir),
                                "dry_run": False,
                                "ai_timeout": 60.0,
                                "consultation_cache": True,
                            },
                        )

        # Review should be persisted
        review_file = specs_dir / ".spec-reviews" / "test-spec-spec-review.md"
        assert review_file.exists()
        content = review_file.read_text()
        assert "standalone spec review" in content
        assert "Foundry MCP Spec Review" in content
        assert result.get("review_path") == str(review_file)

        # Frontmatter should be updated
        mock_fm.assert_called_once_with(
            "test-spec", "spec_review_path", ".spec-reviews/test-spec-spec-review.md", specs_dir=specs_dir
        )

    def test_plan_enhanced_review_still_persisted(self, tmp_path):
        """A plan-enhanced review should still be persisted with plan-specific labels."""
        from foundry_mcp.tools.unified.review import _handle_spec_review

        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        pending_dir = specs_dir / "pending"
        pending_dir.mkdir()
        spec_data = {"spec_id": "test-spec", "title": "Test Spec", "metadata": {}, "hierarchy": {}}
        (pending_dir / "test-spec.json").write_text(json.dumps(spec_data))

        review_json = json.dumps({"verdict": "warn", "summary": "Minor issues"})
        ai_result = _make_ai_result(success=True, plan_enhanced=True, response=review_json)

        config = _make_config(specs_dir)

        with patch("foundry_mcp.tools.unified.review._get_llm_status", return_value="available"):
            with patch("foundry_mcp.tools.unified.review._build_plan_context", return_value="# Plan context"):
                with patch("foundry_mcp.tools.unified.review._run_ai_review", return_value=ai_result):
                    with patch("foundry_mcp.tools.unified.review.update_frontmatter"):
                        result = _handle_spec_review(
                            config=config,
                            payload={
                                "spec_id": "test-spec",
                                "path": str(specs_dir),
                                "dry_run": False,
                                "ai_timeout": 60.0,
                                "consultation_cache": True,
                            },
                        )

        review_file = specs_dir / ".spec-reviews" / "test-spec-spec-review.md"
        assert review_file.exists()
        content = review_file.read_text()
        assert "plan-enhanced full review" in content
        assert "Foundry MCP Spec-vs-Plan Review" in content

    def test_dry_run_review_not_persisted(self, tmp_path):
        """A dry-run review should NOT be persisted to disk."""
        from foundry_mcp.tools.unified.review import _handle_spec_review

        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        pending_dir = specs_dir / "pending"
        pending_dir.mkdir()
        spec_data = {"spec_id": "test-spec", "title": "Test Spec", "metadata": {}, "hierarchy": {}}
        (pending_dir / "test-spec.json").write_text(json.dumps(spec_data))

        review_json = json.dumps({"verdict": "pass"})
        ai_result = _make_ai_result(success=True, plan_enhanced=False, response=review_json)

        config = _make_config(specs_dir)

        with patch("foundry_mcp.tools.unified.review._get_llm_status", return_value="available"):
            with patch("foundry_mcp.tools.unified.review._build_plan_context", return_value=None):
                with patch("foundry_mcp.tools.unified.review._run_ai_review", return_value=ai_result):
                    with patch("foundry_mcp.tools.unified.review.update_frontmatter") as mock_fm:
                        result = _handle_spec_review(
                            config=config,
                            payload={
                                "spec_id": "test-spec",
                                "path": str(specs_dir),
                                "dry_run": True,
                                "ai_timeout": 60.0,
                                "consultation_cache": True,
                            },
                        )

        review_file = specs_dir / ".spec-reviews" / "test-spec-spec-review.md"
        assert not review_file.exists()
        mock_fm.assert_not_called()

    def test_failed_review_not_persisted(self, tmp_path):
        """A failed review should NOT be persisted to disk."""
        from foundry_mcp.tools.unified.review import _handle_spec_review

        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        pending_dir = specs_dir / "pending"
        pending_dir.mkdir()
        spec_data = {"spec_id": "test-spec", "title": "Test Spec", "metadata": {}, "hierarchy": {}}
        (pending_dir / "test-spec.json").write_text(json.dumps(spec_data))

        ai_result = _make_ai_result(success=False)

        config = _make_config(specs_dir)

        with patch("foundry_mcp.tools.unified.review._get_llm_status", return_value="available"):
            with patch("foundry_mcp.tools.unified.review._build_plan_context", return_value=None):
                with patch("foundry_mcp.tools.unified.review._run_ai_review", return_value=ai_result):
                    with patch("foundry_mcp.tools.unified.review.update_frontmatter") as mock_fm:
                        result = _handle_spec_review(
                            config=config,
                            payload={
                                "spec_id": "test-spec",
                                "path": str(specs_dir),
                                "dry_run": False,
                                "ai_timeout": 60.0,
                                "consultation_cache": True,
                            },
                        )

        review_file = specs_dir / ".spec-reviews" / "test-spec-spec-review.md"
        assert not review_file.exists()
        mock_fm.assert_not_called()


class TestSpecReviewMetadataUpdate:
    """Tests for writing spec_review_path to spec metadata (Fix 3)."""

    def test_frontmatter_called_with_relative_path(self, tmp_path):
        """update_frontmatter should be called with a relative path."""
        from foundry_mcp.tools.unified.review import _handle_spec_review

        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        pending_dir = specs_dir / "pending"
        pending_dir.mkdir()
        spec_data = {"spec_id": "my-spec", "title": "My Spec", "metadata": {}, "hierarchy": {}}
        (pending_dir / "my-spec.json").write_text(json.dumps(spec_data))

        review_json = json.dumps({"verdict": "pass"})
        ai_result = _make_ai_result(success=True, response=review_json)

        config = _make_config(specs_dir)

        with patch("foundry_mcp.tools.unified.review._get_llm_status", return_value="available"):
            with patch("foundry_mcp.tools.unified.review._build_plan_context", return_value=None):
                with patch("foundry_mcp.tools.unified.review._run_ai_review", return_value=ai_result):
                    with patch("foundry_mcp.tools.unified.review.update_frontmatter") as mock_fm:
                        _handle_spec_review(
                            config=config,
                            payload={
                                "spec_id": "my-spec",
                                "path": str(specs_dir),
                                "dry_run": False,
                                "ai_timeout": 60.0,
                                "consultation_cache": True,
                            },
                        )

        mock_fm.assert_called_once()
        args = mock_fm.call_args
        # Positional: spec_id, key, value
        assert args[0][0] == "my-spec"
        assert args[0][1] == "spec_review_path"
        # Value should be relative, not absolute
        rel_path = args[0][2]
        assert rel_path == ".spec-reviews/my-spec-spec-review.md"
        assert not Path(rel_path).is_absolute()

    def test_frontmatter_failure_does_not_break_review(self, tmp_path):
        """If update_frontmatter raises, the review should still succeed."""
        from foundry_mcp.tools.unified.review import _handle_spec_review

        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        pending_dir = specs_dir / "pending"
        pending_dir.mkdir()
        spec_data = {"spec_id": "test-spec", "title": "Test Spec", "metadata": {}, "hierarchy": {}}
        (pending_dir / "test-spec.json").write_text(json.dumps(spec_data))

        review_json = json.dumps({"verdict": "pass"})
        ai_result = _make_ai_result(success=True, response=review_json)

        config = _make_config(specs_dir)

        with patch("foundry_mcp.tools.unified.review._get_llm_status", return_value="available"):
            with patch("foundry_mcp.tools.unified.review._build_plan_context", return_value=None):
                with patch("foundry_mcp.tools.unified.review._run_ai_review", return_value=ai_result):
                    with patch(
                        "foundry_mcp.tools.unified.review.update_frontmatter",
                        side_effect=RuntimeError("disk full"),
                    ):
                        result = _handle_spec_review(
                            config=config,
                            payload={
                                "spec_id": "test-spec",
                                "path": str(specs_dir),
                                "dry_run": False,
                                "ai_timeout": 60.0,
                                "consultation_cache": True,
                            },
                        )

        # Review should still have succeeded
        assert result.get("success") is True or result.get("review_path") is not None
        # File should still be written
        review_file = specs_dir / ".spec-reviews" / "test-spec-spec-review.md"
        assert review_file.exists()
