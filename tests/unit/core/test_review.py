"""Tests for core/review.py - non-LLM review logic."""

from unittest.mock import patch

from foundry_mcp.core.review import (
    ReviewFinding,
    QuickReviewResult,
    get_llm_status,
    prepare_review_context,
    quick_review,
    review_type_requires_llm,
    QUICK_REVIEW_TYPES,
    LLM_REQUIRED_TYPES,
)


class TestReviewTypeRequiresLLM:
    """Tests for review_type_requires_llm function."""

    def test_quick_does_not_require_llm(self):
        """Quick review type should not require LLM."""
        assert review_type_requires_llm("quick") is False

    def test_full_requires_llm(self):
        """Full review type should require LLM."""
        assert review_type_requires_llm("full") is True

    def test_security_requires_llm(self):
        """Security review type should require LLM."""
        assert review_type_requires_llm("security") is True

    def test_feasibility_requires_llm(self):
        """Feasibility review type should require LLM."""
        assert review_type_requires_llm("feasibility") is True

    def test_unknown_type_does_not_require_llm(self):
        """Unknown review types should not require LLM (default behavior)."""
        assert review_type_requires_llm("unknown") is False


class TestGetLLMStatus:
    """Tests for get_llm_status function."""

    def test_returns_dict(self):
        """Should return a dict with configured status."""
        status = get_llm_status()
        assert isinstance(status, dict)
        assert "configured" in status

    def test_handles_import_error(self):
        """Should handle import error gracefully."""
        with patch.dict("sys.modules", {"foundry_mcp.core.llm_config": None}):
            # Force reimport
            import importlib
            from foundry_mcp.core import review

            importlib.reload(review)
            # Even with import issues, should return dict
            status = review.get_llm_status()
            assert isinstance(status, dict)


class TestQuickReviewResult:
    """Tests for QuickReviewResult dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        result = QuickReviewResult(
            spec_id="test-spec",
            title="Test Spec",
        )
        assert result.spec_id == "test-spec"
        assert result.title == "Test Spec"
        assert result.review_type == "quick"
        assert result.is_valid is True
        assert result.findings == []
        assert result.error_count == 0
        assert result.warning_count == 0
        assert result.info_count == 0


class TestReviewFinding:
    """Tests for ReviewFinding dataclass."""

    def test_creation(self):
        """Should create finding with all fields."""
        finding = ReviewFinding(
            code="TEST_ISSUE",
            message="Test message",
            severity="warning",
            category="quality",
            location="task-1",
            suggestion="Fix it",
        )
        assert finding.code == "TEST_ISSUE"
        assert finding.severity == "warning"
        assert finding.category == "quality"
        assert finding.location == "task-1"
        assert finding.suggestion == "Fix it"

    def test_optional_fields(self):
        """Should allow optional fields to be None."""
        finding = ReviewFinding(
            code="TEST",
            message="Test",
            severity="info",
            category="structure",
        )
        assert finding.location is None
        assert finding.suggestion is None


def make_minimal_spec(spec_id: str, title: str) -> dict:
    """Create a minimal valid spec for testing."""
    return {
        "spec_id": spec_id,
        "title": title,
        "generated": "2025-01-01T00:00:00Z",
        "last_updated": "2025-01-01T00:00:00Z",
        "hierarchy": {
            "spec-root": {
                "type": "group",
                "title": "Root",
                "status": "pending",
                "parent": None,
                "children": [],
                "total_tasks": 0,
                "completed_tasks": 0,
                "metadata": {},
            }
        },
        "journal": [],
    }


class TestPrepareReviewContext:
    """Tests for prepare_review_context function."""

    def test_returns_none_for_nonexistent_spec(self, tmp_path):
        """Should return None if spec not found."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        (specs_dir / "active").mkdir()

        context = prepare_review_context("nonexistent-spec", specs_dir)
        assert context is None

    def test_returns_context_for_valid_spec(self, tmp_path):
        """Should return ReviewContext for valid spec."""
        import json

        specs_dir = tmp_path / "specs"
        (specs_dir / "active").mkdir(parents=True)

        spec_data = make_minimal_spec("test-spec-2025-01-01-001", "Test Spec")

        spec_file = specs_dir / "active" / "test-spec-2025-01-01-001.json"
        spec_file.write_text(json.dumps(spec_data))

        context = prepare_review_context("test-spec-2025-01-01-001", specs_dir)
        assert context is not None
        assert context.spec_id == "test-spec-2025-01-01-001"
        assert context.title == "Test Spec"
        assert isinstance(context.progress, dict)
        assert isinstance(context.phases, list)


def make_spec_with_task(spec_id: str, title: str, task_metadata: dict = None) -> dict:
    """Create a valid spec with one task for testing."""
    task_meta = task_metadata or {"estimated_hours": 1, "file_path": "src/test.py"}
    return {
        "spec_id": spec_id,
        "title": title,
        "generated": "2025-01-01T00:00:00Z",
        "last_updated": "2025-01-01T00:00:00Z",
        "hierarchy": {
            "spec-root": {
                "type": "group",
                "title": "Root",
                "status": "pending",
                "parent": None,
                "children": ["phase-1"],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-1"],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": {},
            },
            "task-1": {
                "type": "task",
                "title": "Task 1",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "total_tasks": 1,
                "completed_tasks": 0,
                "metadata": task_meta,
            },
        },
        "journal": [],
    }


class TestQuickReview:
    """Tests for quick_review function."""

    def test_returns_not_found_for_missing_spec(self, tmp_path):
        """Should return error result for missing spec."""
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()
        (specs_dir / "active").mkdir()

        result = quick_review("nonexistent", specs_dir)

        assert result.is_valid is False
        assert result.error_count == 1
        assert any(f.code == "SPEC_NOT_FOUND" for f in result.findings)

    def test_returns_valid_result_for_valid_spec(self, tmp_path):
        """Should return QuickReviewResult for valid spec."""
        import json

        specs_dir = tmp_path / "specs"
        (specs_dir / "active").mkdir(parents=True)

        spec_data = make_spec_with_task(
            "test-spec-2025-01-01-001",
            "Test Spec",
            {"estimated_hours": 1, "file_path": "src/test.py"},
        )

        spec_file = specs_dir / "active" / "test-spec-2025-01-01-001.json"
        spec_file.write_text(json.dumps(spec_data))

        result = quick_review("test-spec-2025-01-01-001", specs_dir)

        # Check result type name (avoid isinstance issue with reimported class)
        assert type(result).__name__ == "QuickReviewResult"
        assert result.spec_id == "test-spec-2025-01-01-001"
        assert result.title == "Test Spec"
        assert result.review_type == "quick"
        assert result.summary != ""

    def test_detects_empty_phase(self, tmp_path):
        """Should detect empty phases."""
        import json

        specs_dir = tmp_path / "specs"
        (specs_dir / "active").mkdir(parents=True)

        spec_data = {
            "spec_id": "test-spec-2025-01-01-001",
            "title": "Test Spec",
            "generated": "2025-01-01T00:00:00Z",
            "last_updated": "2025-01-01T00:00:00Z",
            "hierarchy": {
                "spec-root": {
                    "type": "group",
                    "title": "Root",
                    "status": "pending",
                    "parent": None,
                    "children": ["phase-1"],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "metadata": {},
                },
                "phase-1": {
                    "type": "phase",
                    "title": "Empty Phase",
                    "status": "pending",
                    "parent": "spec-root",
                    "children": [],
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "metadata": {},
                },
            },
            "journal": [],
        }

        spec_file = specs_dir / "active" / "test-spec-2025-01-01-001.json"
        spec_file.write_text(json.dumps(spec_data))

        result = quick_review("test-spec-2025-01-01-001", specs_dir)

        assert any(f.code == "EMPTY_PHASE" for f in result.findings)

    def test_detects_missing_estimates(self, tmp_path):
        """Should detect tasks without estimates."""
        import json

        specs_dir = tmp_path / "specs"
        (specs_dir / "active").mkdir(parents=True)

        # Task without estimated_hours in metadata
        spec_data = make_spec_with_task(
            "test-spec-2025-01-01-001",
            "Test Spec",
            {"file_path": "src/test.py"},  # No estimated_hours
        )

        spec_file = specs_dir / "active" / "test-spec-2025-01-01-001.json"
        spec_file.write_text(json.dumps(spec_data))

        result = quick_review("test-spec-2025-01-01-001", specs_dir)

        assert any(f.code == "MISSING_ESTIMATES" for f in result.findings)

    def test_detects_blocked_tasks(self, tmp_path):
        """Should detect blocked tasks."""
        import json

        specs_dir = tmp_path / "specs"
        (specs_dir / "active").mkdir(parents=True)

        spec_data = make_spec_with_task(
            "test-spec-2025-01-01-001",
            "Test Spec",
            {"estimated_hours": 1, "file_path": "src/test.py"},
        )
        # Set task as blocked
        spec_data["hierarchy"]["task-1"]["status"] = "blocked"

        spec_file = specs_dir / "active" / "test-spec-2025-01-01-001.json"
        spec_file.write_text(json.dumps(spec_data))

        result = quick_review("test-spec-2025-01-01-001", specs_dir)

        assert any(f.code == "BLOCKED_TASKS" for f in result.findings)


class TestConstants:
    """Tests for module constants."""

    def test_quick_review_types(self):
        """Should have expected quick review types."""
        assert "quick" in QUICK_REVIEW_TYPES

    def test_llm_required_types(self):
        """Should have expected LLM required types."""
        assert "full" in LLM_REQUIRED_TYPES
        assert "security" in LLM_REQUIRED_TYPES
        assert "feasibility" in LLM_REQUIRED_TYPES
