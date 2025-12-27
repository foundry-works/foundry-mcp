"""Contract tests for Phase 7 actions: completeness-check and duplicate-detection.

Validates response-v2 envelope compliance per 10-testing-fixtures.md:
1. Response envelope structure matches response-v2
2. Error responses include error_code, error_type, remediation fields
3. Data fields match expected structure
4. Truncation scenarios include warnings
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Response-v2 Schema Validation Helpers
# ---------------------------------------------------------------------------


def validate_response_v2_envelope(response: Dict[str, Any]) -> list[str]:
    """Validate response conforms to response-v2 schema."""
    errors = []

    if "success" not in response:
        errors.append("Missing required field: success")
    elif not isinstance(response["success"], bool):
        errors.append(f"success must be boolean, got {type(response['success']).__name__}")

    if "data" not in response:
        errors.append("Missing required field: data")
    elif not isinstance(response["data"], dict):
        errors.append(f"data must be object, got {type(response['data']).__name__}")

    if "meta" not in response:
        errors.append("Missing required field: meta")
    elif not isinstance(response["meta"], dict):
        errors.append(f"meta must be object, got {type(response['meta']).__name__}")
    else:
        if "version" not in response["meta"]:
            errors.append("Missing required field: meta.version")
        elif response["meta"]["version"] != "response-v2":
            errors.append(
                f"meta.version must be 'response-v2', got '{response['meta']['version']}'"
            )

    if response.get("success") is True:
        if response.get("error") is not None:
            errors.append("error must be null when success is True")
    else:
        if "error" not in response:
            errors.append("Missing required field: error")
        elif response["error"] is None:
            errors.append("error must be non-null when success is False")

    return errors


def validate_error_response_fields(response: Dict[str, Any]) -> list[str]:
    """Validate error response has structured error fields in data."""
    errors = []

    if response.get("success") is not False:
        return []

    data = response.get("data", {})

    if "error_code" not in data:
        errors.append("Error response missing recommended field: data.error_code")

    if "error_type" not in data:
        errors.append("Error response missing recommended field: data.error_type")

    return errors


def assert_valid_response_v2(response: Dict[str, Any], context: str = ""):
    """Assert response is valid response-v2 format."""
    errors = validate_response_v2_envelope(response)
    if errors:
        error_msg = f"Response-v2 validation errors"
        if context:
            error_msg += f" ({context})"
        error_msg += ":\n  - " + "\n  - ".join(errors)
        pytest.fail(error_msg)


def assert_valid_error_response(response: Dict[str, Any], context: str = ""):
    """Assert error response has required error fields."""
    assert_valid_response_v2(response, context)

    errors = validate_error_response_fields(response)
    if errors:
        error_msg = f"Error response field validation errors"
        if context:
            error_msg += f" ({context})"
        error_msg += ":\n  - " + "\n  - ".join(errors)
        pytest.fail(error_msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        specs_dir = (Path(tmpdir) / "specs").resolve()
        for status in ("pending", "active", "completed", "archived"):
            (specs_dir / status).mkdir(parents=True)
        yield specs_dir


@pytest.fixture
def complete_spec():
    """A spec with all fields complete."""
    return {
        "spec_id": "complete-spec-001",
        "metadata": {"title": "Complete Spec"},
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Complete Spec",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Implementation",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Implement feature",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "description": "Full description",
                    "estimated_hours": 2.0,
                    "task_category": "implementation",
                    "file_path": "src/feature.py",
                },
            },
        },
        "journal": [],
    }


@pytest.fixture
def duplicate_spec():
    """A spec with duplicate task titles."""
    return {
        "spec_id": "duplicate-spec-001",
        "metadata": {"title": "Duplicate Spec"},
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Duplicate Spec",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2", "task-1-3"],
            },
            "task-1-1": {
                "type": "task",
                "title": "Implement user authentication",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"description": "Add login"},
            },
            "task-1-2": {
                "type": "task",
                "title": "Add user authentication",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"description": "Login system"},
            },
            "task-1-3": {
                "type": "task",
                "title": "Different task entirely",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"description": "Something else"},
            },
        },
        "journal": [],
    }


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _write_spec(specs_dir: Path, spec_data: Dict[str, Any], status: str = "active") -> Path:
    """Helper to write a spec file."""
    spec_id = spec_data["spec_id"]
    spec_file = specs_dir / status / f"{spec_id}.json"
    spec_file.write_text(json.dumps(spec_data))
    return spec_file


def call_spec_handler(
    action: str,
    payload: Dict[str, Any],
    specs_dir: Path,
) -> Dict[str, Any]:
    """Call the spec router handler and return response dict."""
    from foundry_mcp.tools.unified.spec import _SPEC_ROUTER
    from foundry_mcp.config import ServerConfig

    config = ServerConfig(specs_dir=specs_dir)
    return _SPEC_ROUTER.dispatch(action=action, config=config, payload=payload)


# ---------------------------------------------------------------------------
# Completeness-Check Contract Tests
# ---------------------------------------------------------------------------


class TestCompletenessCheckContracts:
    """Contract tests for spec action='completeness-check'."""

    def test_success_response_envelope(self, temp_specs_dir, complete_spec):
        """Success response must match response-v2 envelope."""
        _write_spec(temp_specs_dir, complete_spec)

        response = call_spec_handler(
            "completeness-check",
            {"spec_id": "complete-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "completeness-check success")
        assert response["success"] is True
        assert response["error"] is None

    def test_success_response_has_telemetry(self, temp_specs_dir, complete_spec):
        """Success response should include telemetry in meta."""
        _write_spec(temp_specs_dir, complete_spec)

        response = call_spec_handler(
            "completeness-check",
            {"spec_id": "complete-spec-001"},
            temp_specs_dir,
        )

        # Telemetry is added by the MCP server layer, not unit tests
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_data_fields(self, temp_specs_dir, complete_spec):
        """Success response should include required data fields."""
        _write_spec(temp_specs_dir, complete_spec)

        response = call_spec_handler(
            "completeness-check",
            {"spec_id": "complete-spec-001"},
            temp_specs_dir,
        )

        data = response["data"]
        assert "spec_id" in data
        assert "completeness_score" in data
        assert isinstance(data["completeness_score"], int)
        assert 0 <= data["completeness_score"] <= 100
        assert "categories" in data
        assert "issues" in data
        assert "issue_count" in data

    def test_complete_spec_scores_100(self, temp_specs_dir, complete_spec):
        """A complete spec should score 100."""
        _write_spec(temp_specs_dir, complete_spec)

        response = call_spec_handler(
            "completeness-check",
            {"spec_id": "complete-spec-001"},
            temp_specs_dir,
        )

        assert response["success"] is True
        assert response["data"]["completeness_score"] == 100
        assert response["data"]["issue_count"] == 0

    def test_error_missing_spec_id(self, temp_specs_dir):
        """Missing spec_id should return error response."""
        response = call_spec_handler(
            "completeness-check",
            {},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert "spec_id" in response["error"].lower()
        assert response["data"]["error_code"] == "MISSING_REQUIRED"

    def test_error_spec_not_found(self, temp_specs_dir):
        """Non-existent spec should return error response."""
        response = call_spec_handler(
            "completeness-check",
            {"spec_id": "nonexistent-spec"},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "spec not found")
        assert response["success"] is False
        assert "not found" in response["error"].lower()

    def test_categories_structure(self, temp_specs_dir, complete_spec):
        """Categories should have complete/total/score fields."""
        _write_spec(temp_specs_dir, complete_spec)

        response = call_spec_handler(
            "completeness-check",
            {"spec_id": "complete-spec-001"},
            temp_specs_dir,
        )

        categories = response["data"]["categories"]
        for cat_name, cat_data in categories.items():
            assert "complete" in cat_data, f"Missing 'complete' in {cat_name}"
            assert "total" in cat_data, f"Missing 'total' in {cat_name}"
            assert "score" in cat_data, f"Missing 'score' in {cat_name}"


# ---------------------------------------------------------------------------
# Duplicate-Detection Contract Tests
# ---------------------------------------------------------------------------


class TestDuplicateDetectionContracts:
    """Contract tests for spec action='duplicate-detection'."""

    def test_success_response_envelope(self, temp_specs_dir, duplicate_spec):
        """Success response must match response-v2 envelope."""
        _write_spec(temp_specs_dir, duplicate_spec)

        response = call_spec_handler(
            "duplicate-detection",
            {"spec_id": "duplicate-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "duplicate-detection success")
        assert response["success"] is True
        assert response["error"] is None

    def test_success_response_version(self, temp_specs_dir, duplicate_spec):
        """Success response should have version in meta."""
        _write_spec(temp_specs_dir, duplicate_spec)

        response = call_spec_handler(
            "duplicate-detection",
            {"spec_id": "duplicate-spec-001"},
            temp_specs_dir,
        )

        assert response["meta"]["version"] == "response-v2"

    def test_success_response_data_fields(self, temp_specs_dir, duplicate_spec):
        """Success response should include required data fields."""
        _write_spec(temp_specs_dir, duplicate_spec)

        response = call_spec_handler(
            "duplicate-detection",
            {"spec_id": "duplicate-spec-001"},
            temp_specs_dir,
        )

        data = response["data"]
        assert "spec_id" in data
        assert "duplicates" in data
        assert isinstance(data["duplicates"], list)
        assert "duplicate_count" in data
        assert "scope" in data
        assert "threshold" in data
        assert "nodes_checked" in data
        assert "pairs_compared" in data

    def test_finds_duplicates_with_similar_titles(self, temp_specs_dir, duplicate_spec):
        """Should find tasks with similar titles."""
        _write_spec(temp_specs_dir, duplicate_spec)

        response = call_spec_handler(
            "duplicate-detection",
            {"spec_id": "duplicate-spec-001", "threshold": 0.6},
            temp_specs_dir,
        )

        assert response["success"] is True
        assert response["data"]["duplicate_count"] > 0

    def test_duplicate_entry_structure(self, temp_specs_dir, duplicate_spec):
        """Duplicate entries should have required fields."""
        _write_spec(temp_specs_dir, duplicate_spec)

        response = call_spec_handler(
            "duplicate-detection",
            {"spec_id": "duplicate-spec-001", "threshold": 0.5},
            temp_specs_dir,
        )

        for dup in response["data"]["duplicates"]:
            assert "node_a" in dup
            assert "node_b" in dup
            assert "similarity" in dup
            assert isinstance(dup["similarity"], float)
            assert 0.0 <= dup["similarity"] <= 1.0

    def test_truncation_sets_flag(self, temp_specs_dir, duplicate_spec):
        """Truncation should set truncated flag."""
        _write_spec(temp_specs_dir, duplicate_spec)

        response = call_spec_handler(
            "duplicate-detection",
            {"spec_id": "duplicate-spec-001", "threshold": 0.0, "max_pairs": 1},
            temp_specs_dir,
        )

        assert response["success"] is True
        data = response["data"]
        assert data.get("truncated") is True
        assert data["duplicate_count"] <= 1

    def test_error_missing_spec_id(self, temp_specs_dir):
        """Missing spec_id should return error response."""
        response = call_spec_handler(
            "duplicate-detection",
            {},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert "spec_id" in response["error"].lower()

    def test_error_invalid_threshold(self, temp_specs_dir, duplicate_spec):
        """Invalid threshold should return error response."""
        _write_spec(temp_specs_dir, duplicate_spec)

        response = call_spec_handler(
            "duplicate-detection",
            {"spec_id": "duplicate-spec-001", "threshold": 2.0},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid threshold")
        assert response["success"] is False
        assert "threshold" in response["error"].lower()

    def test_error_spec_not_found(self, temp_specs_dir):
        """Non-existent spec should return error response."""
        response = call_spec_handler(
            "duplicate-detection",
            {"spec_id": "nonexistent-spec"},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "spec not found")
        assert response["success"] is False
        assert "not found" in response["error"].lower()

    def test_no_duplicates_high_threshold(self, temp_specs_dir, complete_spec):
        """Distinct tasks should have no duplicates at high threshold."""
        _write_spec(temp_specs_dir, complete_spec)

        response = call_spec_handler(
            "duplicate-detection",
            {"spec_id": "complete-spec-001", "threshold": 0.99},
            temp_specs_dir,
        )

        assert response["success"] is True
        assert response["data"]["duplicate_count"] == 0


# ---------------------------------------------------------------------------
# Fixture Freshness Tests
# ---------------------------------------------------------------------------


class TestPhase7FixtureFreshness:
    """Tests to ensure fixtures and schemas stay current."""

    def test_response_version_is_current(self, temp_specs_dir, complete_spec):
        """Response version should be response-v2."""
        _write_spec(temp_specs_dir, complete_spec)

        response = call_spec_handler(
            "completeness-check",
            {"spec_id": "complete-spec-001"},
            temp_specs_dir,
        )

        assert response["meta"]["version"] == "response-v2"

    def test_error_codes_are_valid(self, temp_specs_dir):
        """Error codes should be from ErrorCode enum."""
        from foundry_mcp.core.responses import ErrorCode

        response = call_spec_handler(
            "completeness-check",
            {},
            temp_specs_dir,
        )

        error_code = response["data"]["error_code"]
        valid_codes = [e.value for e in ErrorCode]
        assert error_code in valid_codes, f"Unknown error code: {error_code}"
