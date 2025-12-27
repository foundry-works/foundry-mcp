"""Contract tests for Phase 5 actions: metadata-batch and spec-find-replace.

Validates response-v2 envelope compliance per 10-testing-fixtures.md:
1. Response envelope structure matches response-v2
2. Error responses include error_code, error_type, remediation fields
3. Edge cases: partial success, empty matches, dry-run mode
4. Fixtures are fresh and schema-aligned
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pytest

from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    success_response,
    error_response,
)


# ---------------------------------------------------------------------------
# Response-v2 Schema Validation Helpers (reused from test_phase3_contracts.py)
# ---------------------------------------------------------------------------


def validate_response_v2_envelope(response: Dict[str, Any]) -> list[str]:
    """Validate response conforms to response-v2 schema.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    # Required top-level fields
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
        # meta.version is required
        if "version" not in response["meta"]:
            errors.append("Missing required field: meta.version")
        elif response["meta"]["version"] != "response-v2":
            errors.append(
                f"meta.version must be 'response-v2', got '{response['meta']['version']}'"
            )

    # error field rules
    if response.get("success") is True:
        if response.get("error") is not None:
            errors.append("error must be null when success is True")
    else:
        if "error" not in response:
            errors.append("Missing required field: error")
        elif response["error"] is None:
            errors.append("error must be non-null when success is False")
        elif not isinstance(response["error"], str):
            errors.append(f"error must be string, got {type(response['error']).__name__}")

    return errors


def validate_error_response_fields(response: Dict[str, Any]) -> list[str]:
    """Validate error response has structured error fields in data.

    Per mcp_response_schema.md, error responses SHOULD include:
    - error_code: machine-readable code (SCREAMING_SNAKE_CASE)
    - error_type: error category for routing
    - remediation: actionable guidance
    """
    errors = []

    if response.get("success") is not False:
        return []  # Only validate error responses

    data = response.get("data", {})

    # error_code SHOULD be present
    if "error_code" not in data:
        errors.append("Error response missing recommended field: data.error_code")
    elif not isinstance(data["error_code"], str):
        errors.append(
            f"data.error_code must be string, got {type(data['error_code']).__name__}"
        )

    # error_type SHOULD be present
    if "error_type" not in data:
        errors.append("Error response missing recommended field: data.error_type")
    elif not isinstance(data["error_type"], str):
        errors.append(
            f"data.error_type must be string, got {type(data['error_type']).__name__}"
        )

    # remediation SHOULD be present
    if "remediation" not in data:
        errors.append("Error response missing recommended field: data.remediation")
    elif not isinstance(data["remediation"], str):
        errors.append(
            f"data.remediation must be string, got {type(data['remediation']).__name__}"
        )

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
def sample_spec():
    """Create a sample spec with phases and tasks for batch operations."""
    return {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "metadata": {
            "title": "Test Specification",
            "version": "1.0.0",
        },
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Test Spec",
                "status": "in_progress",
                "children": ["phase-1", "phase-2"],
                "total_tasks": 5,
                "completed_tasks": 1,
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1: Implementation",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2", "task-1-3"],
                "completed_tasks": 1,
                "total_tasks": 3,
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            },
            "phase-2": {
                "type": "phase",
                "title": "Phase 2: Testing",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-2-1", "task-2-2"],
                "completed_tasks": 0,
                "total_tasks": 2,
                "dependencies": {
                    "blocks": [],
                    "blocked_by": ["phase-1"],
                    "depends": [],
                },
            },
            "task-1-1": {
                "type": "task",
                "title": "Add feature implementation",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "file_path": "src/feature.py",
                    "description": "Implement the feature module",
                    "estimated_hours": 2.0,
                },
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            },
            "task-1-2": {
                "type": "task",
                "title": "Add feature tests",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "file_path": "tests/test_feature.py",
                    "description": "Write tests for feature",
                },
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            },
            "task-1-3": {
                "type": "task",
                "title": "Add feature documentation",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "file_path": "docs/feature.md",
                },
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            },
            "task-2-1": {
                "type": "task",
                "title": "Run integration tests",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
                "metadata": {},
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            },
            "task-2-2": {
                "type": "task",
                "title": "Run performance tests",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
                "metadata": {},
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            },
        },
        "journal": [],
    }


@pytest.fixture
def spec_with_findable_text(sample_spec):
    """Sample spec with text patterns suitable for find/replace testing."""
    spec = sample_spec.copy()
    spec["hierarchy"] = {k: v.copy() for k, v in sample_spec["hierarchy"].items()}

    # Deep copy metadata
    for task_id in ["task-1-1", "task-1-2", "task-1-3"]:
        if "metadata" in spec["hierarchy"][task_id]:
            spec["hierarchy"][task_id]["metadata"] = (
                spec["hierarchy"][task_id]["metadata"].copy()
            )

    # Add consistent text pattern to search for
    spec["hierarchy"]["task-1-1"]["title"] = "Add FEATURE_V1 implementation"
    spec["hierarchy"]["task-1-2"]["title"] = "Add FEATURE_V1 tests"
    spec["hierarchy"]["task-1-3"]["title"] = "Add FEATURE_V1 documentation"

    return spec


@pytest.fixture
def mock_server_config():
    """Mock ServerConfig for handler testing."""
    from foundry_mcp.config import ServerConfig

    return ServerConfig(workspace=".")


# ---------------------------------------------------------------------------
# Test Helpers to Invoke Handlers
# ---------------------------------------------------------------------------


def call_task_handler(
    action: str,
    payload: Dict[str, Any],
    specs_dir: Path,
    config=None,
) -> Dict[str, Any]:
    """Call the task router handler and return response dict.

    This simulates how the MCP tool would invoke the handlers.
    """
    from foundry_mcp.tools.unified.task import _TASK_ROUTER
    from foundry_mcp.config import ServerConfig

    if config is None:
        config = ServerConfig(specs_dir=specs_dir)

    # Use the action router to dispatch
    return _TASK_ROUTER.dispatch(action=action, config=config, payload=payload)


def call_authoring_handler(
    action: str,
    payload: Dict[str, Any],
    specs_dir: Path,
    config=None,
) -> Dict[str, Any]:
    """Call the authoring router handler and return response dict.

    This simulates how the MCP tool would invoke the handlers.
    """
    from foundry_mcp.tools.unified.authoring import _AUTHORING_ROUTER
    from foundry_mcp.config import ServerConfig

    if config is None:
        config = ServerConfig(specs_dir=specs_dir)

    # Use the action router to dispatch
    return _AUTHORING_ROUTER.dispatch(action=action, config=config, **payload)


# ---------------------------------------------------------------------------
# Metadata-Batch Contract Tests
# ---------------------------------------------------------------------------


class TestMetadataBatchContracts:
    """Contract tests for task action='metadata-batch'."""

    def test_success_response_envelope(self, temp_specs_dir, sample_spec):
        """Success response must match response-v2 envelope."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "description": "Updated via batch",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "metadata-batch success")
        assert response["success"] is True
        assert response["error"] is None
        assert "meta" in response
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_has_request_id(self, temp_specs_dir, sample_spec):
        """Success response should include request_id in meta."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "parent_filter": "phase-1",
                "file_path": "src/updated.py",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        # request_id is SHOULD per spec
        assert "request_id" in response["meta"]

    def test_success_response_has_telemetry(self, temp_specs_dir, sample_spec):
        """Success response should include telemetry in meta."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "estimated_hours": 1.5,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        assert "telemetry" in response["meta"]
        assert "duration_ms" in response["meta"]["telemetry"]

    def test_success_response_data_fields(self, temp_specs_dir, sample_spec):
        """Success response data should contain expected fields."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "description": "Batch updated description",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["spec_id"] == "test-spec-001"
        assert "matched_count" in data
        assert "updated_count" in data
        assert "nodes" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["updated_count"], int)

    def test_dry_run_response_format(self, temp_specs_dir, sample_spec):
        """Dry run should return valid response with dry_run indicator."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "description": "Would update description",
                "dry_run": True,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "dry run")
        assert response["success"] is True
        assert response["data"]["dry_run"] is True
        assert "nodes" in response["data"]

    def test_filter_by_parent(self, temp_specs_dir, sample_spec):
        """Filter by parent_filter should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "parent_filter": "phase-1",
                "category": "implementation",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "parent_filter")
        assert response["success"] is True
        # Phase-1 has 3 tasks
        assert response["data"]["updated_count"] <= 3

    def test_filter_by_pattern(self, temp_specs_dir, sample_spec):
        """Filter by pattern (regex) should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "pattern": ".*test.*",
                "description": "Test-related task",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "pattern filter")
        assert response["success"] is True

    def test_combined_filters(self, temp_specs_dir, sample_spec):
        """Combined filters (AND logic) should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "parent_filter": "phase-1",
                "description": "Pending tasks in phase-1",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "combined filters")
        assert response["success"] is True

    def test_error_missing_spec_id(self, temp_specs_dir, sample_spec):
        """Error: missing spec_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "status_filter": "pending",
                "description": "Some description",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert "spec_id" in response["error"].lower()
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_error_no_filters_provided(self, temp_specs_dir, sample_spec):
        """Error: no filters provided should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "description": "Some description",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "no filters")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_no_metadata_fields(self, temp_specs_dir, sample_spec):
        """Error: no metadata fields provided should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "no metadata fields")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_invalid_status_filter(self, temp_specs_dir, sample_spec):
        """Error: invalid status_filter should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "invalid_status",
                "description": "Some description",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid status_filter")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_invalid_pattern_regex(self, temp_specs_dir, sample_spec):
        """Error: invalid regex pattern should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "pattern": "[invalid(regex",
                "description": "Some description",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid regex pattern")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_invalid_estimated_hours(self, temp_specs_dir, sample_spec):
        """Error: negative estimated_hours should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "estimated_hours": -5,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid estimated_hours")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_invalid_labels_type(self, temp_specs_dir, sample_spec):
        """Error: labels not a dict should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "labels": ["not", "a", "dict"],
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid labels type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_invalid_owners_type(self, temp_specs_dir, sample_spec):
        """Error: owners not a list of strings should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "owners": "not_a_list",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid owners type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_invalid_dry_run_type(self, temp_specs_dir, sample_spec):
        """Error: invalid dry_run type should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "description": "Some description",
                "dry_run": "yes",  # Should be boolean
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid dry_run type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_spec_not_found(self, temp_specs_dir):
        """Error: nonexistent spec should return NOT_FOUND error."""
        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "nonexistent-spec",
                "status_filter": "pending",
                "description": "Some description",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "spec not found")
        assert response["success"] is False
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_parent_not_found(self, temp_specs_dir, sample_spec):
        """Error: nonexistent parent_filter should return NOT_FOUND error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "parent_filter": "nonexistent-phase",
                "description": "Some description",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "parent not found")
        assert response["success"] is False
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_empty_matches_returns_success(self, temp_specs_dir, sample_spec):
        """No matching tasks should return success with zero updated_count."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "pattern": "^nonexistent_pattern_xyz$",
                "description": "Some description",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "empty matches")
        assert response["success"] is True
        assert response["data"]["updated_count"] == 0
        assert response["data"]["matched_count"] == 0
        assert response["data"]["nodes"] == []


# ---------------------------------------------------------------------------
# Spec-Find-Replace Contract Tests
# ---------------------------------------------------------------------------


class TestSpecFindReplaceContracts:
    """Contract tests for authoring action='spec-find-replace'."""

    def test_success_response_envelope(self, temp_specs_dir, spec_with_findable_text):
        """Success response must match response-v2 envelope."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_findable_text))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "FEATURE_V1",
                "replace": "FEATURE_V2",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "spec-find-replace success")
        assert response["success"] is True
        assert response["error"] is None
        assert "meta" in response
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_has_request_id(self, temp_specs_dir, spec_with_findable_text):
        """Success response should include request_id in meta."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_findable_text))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "FEATURE_V1",
                "replace": "FEATURE_V2",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        # request_id is SHOULD per spec
        assert "request_id" in response["meta"]

    def test_success_response_has_telemetry(self, temp_specs_dir, spec_with_findable_text):
        """Success response should include telemetry in meta."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_findable_text))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "FEATURE_V1",
                "replace": "FEATURE_V2",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        assert "telemetry" in response["meta"]
        assert "duration_ms" in response["meta"]["telemetry"]

    def test_success_response_data_fields(self, temp_specs_dir, spec_with_findable_text):
        """Success response data should contain expected fields."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_findable_text))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "FEATURE_V1",
                "replace": "FEATURE_V2",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["spec_id"] == "test-spec-001"
        assert "matches_count" in data or "total_replacements" in data
        assert "affected_nodes" in data or "changes" in data

    def test_dry_run_response_format(self, temp_specs_dir, spec_with_findable_text):
        """Dry run should return valid response with dry_run indicator."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_findable_text))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "FEATURE_V1",
                "replace": "FEATURE_V2",
                "dry_run": True,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "dry run")
        assert response["success"] is True
        assert response["data"]["dry_run"] is True

    def test_scope_titles_only(self, temp_specs_dir, spec_with_findable_text):
        """Scope 'titles' should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_findable_text))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "FEATURE_V1",
                "replace": "FEATURE_V2",
                "scope": "titles",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "scope titles")
        assert response["success"] is True

    def test_scope_descriptions_only(self, temp_specs_dir, sample_spec):
        """Scope 'descriptions' should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "feature",
                "replace": "component",
                "scope": "descriptions",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "scope descriptions")
        assert response["success"] is True

    def test_regex_mode(self, temp_specs_dir, spec_with_findable_text):
        """Regex mode should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_findable_text))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "FEATURE_V\\d+",
                "replace": "FEATURE_V2",
                "use_regex": True,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "regex mode")
        assert response["success"] is True

    def test_case_insensitive(self, temp_specs_dir, spec_with_findable_text):
        """Case insensitive mode should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_findable_text))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "feature_v1",  # lowercase
                "replace": "FEATURE_V2",
                "case_sensitive": False,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "case insensitive")
        assert response["success"] is True

    def test_replace_with_empty_string(self, temp_specs_dir, spec_with_findable_text):
        """Replace with empty string (deletion) should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_findable_text))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "FEATURE_V1 ",
                "replace": "",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "replace with empty")
        assert response["success"] is True

    def test_error_missing_spec_id(self, temp_specs_dir, sample_spec):
        """Error: missing spec_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "find": "old",
                "replace": "new",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert "spec_id" in response["error"].lower()
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_error_missing_find(self, temp_specs_dir, sample_spec):
        """Error: missing find should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "replace": "new",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing find")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_missing_replace(self, temp_specs_dir, sample_spec):
        """Error: missing replace should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "old",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing replace")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_invalid_scope(self, temp_specs_dir, sample_spec):
        """Error: invalid scope should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "old",
                "replace": "new",
                "scope": "invalid_scope",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid scope")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_invalid_use_regex_type(self, temp_specs_dir, sample_spec):
        """Error: invalid use_regex type should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "old",
                "replace": "new",
                "use_regex": "yes",  # Should be boolean
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid use_regex type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_invalid_case_sensitive_type(self, temp_specs_dir, sample_spec):
        """Error: invalid case_sensitive type should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "old",
                "replace": "new",
                "case_sensitive": "no",  # Should be boolean
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid case_sensitive type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_invalid_dry_run_type(self, temp_specs_dir, sample_spec):
        """Error: invalid dry_run type should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "old",
                "replace": "new",
                "dry_run": "true",  # Should be boolean
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid dry_run type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_spec_not_found(self, temp_specs_dir):
        """Error: nonexistent spec should return NOT_FOUND error."""
        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "nonexistent-spec",
                "find": "old",
                "replace": "new",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "spec not found")
        assert response["success"] is False
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_empty_matches_returns_success(self, temp_specs_dir, sample_spec):
        """No matches found should return success with zero count."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "nonexistent_text_xyz123",
                "replace": "replacement",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "empty matches")
        assert response["success"] is True
        # Should have zero matches/replacements
        data = response["data"]
        count_field = data.get("matches_count") or data.get("total_replacements") or 0
        assert count_field == 0


# ---------------------------------------------------------------------------
# Edge Case Contract Tests
# ---------------------------------------------------------------------------


class TestPhase5EdgeCases:
    """Test edge cases for Phase 5 response contract compliance."""

    def test_metadata_batch_whitespace_spec_id(self, temp_specs_dir, sample_spec):
        """Whitespace-only spec_id should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "   ",
                "status_filter": "pending",
                "description": "Some description",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace spec_id")
        assert response["success"] is False

    def test_metadata_batch_whitespace_parent_filter(self, temp_specs_dir, sample_spec):
        """Whitespace-only parent_filter should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "parent_filter": "   ",
                "description": "Some description",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace parent_filter")
        assert response["success"] is False

    def test_metadata_batch_whitespace_pattern(self, temp_specs_dir, sample_spec):
        """Whitespace-only pattern should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "pattern": "   ",
                "description": "Some description",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace pattern")
        assert response["success"] is False

    def test_find_replace_whitespace_spec_id(self, temp_specs_dir, sample_spec):
        """Whitespace-only spec_id should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "   ",
                "find": "old",
                "replace": "new",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace spec_id")
        assert response["success"] is False

    def test_find_replace_empty_find(self, temp_specs_dir, sample_spec):
        """Empty find string should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "spec-find-replace",
            {
                "spec_id": "test-spec-001",
                "find": "",
                "replace": "new",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "empty find")
        assert response["success"] is False

    def test_metadata_batch_with_update_metadata(self, temp_specs_dir, sample_spec):
        """Custom update_metadata dict should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "update_metadata": {
                    "custom_field": "custom_value",
                    "priority": "high",
                },
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "update_metadata dict")
        assert response["success"] is True

    def test_metadata_batch_with_labels(self, temp_specs_dir, sample_spec):
        """Valid labels dict should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "labels": {
                    "team": "backend",
                    "priority": "p1",
                },
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "labels dict")
        assert response["success"] is True

    def test_metadata_batch_with_owners(self, temp_specs_dir, sample_spec):
        """Valid owners list should produce valid response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "metadata-batch",
            {
                "spec_id": "test-spec-001",
                "status_filter": "pending",
                "owners": ["alice", "bob"],
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "owners list")
        assert response["success"] is True


# ---------------------------------------------------------------------------
# Fixture Freshness Contract Tests
# ---------------------------------------------------------------------------


class TestPhase5FixtureFreshness:
    """Test that test fixtures use current response-v2 version."""

    def test_sample_spec_is_valid(self, sample_spec):
        """Sample spec fixture should be valid JSON structure."""
        assert "spec_id" in sample_spec
        assert "hierarchy" in sample_spec
        assert "spec-root" in sample_spec["hierarchy"]
        # Phase 5 specific: has multiple phases and tasks
        assert "phase-1" in sample_spec["hierarchy"]
        assert "phase-2" in sample_spec["hierarchy"]
        assert "task-1-1" in sample_spec["hierarchy"]
        assert "task-1-2" in sample_spec["hierarchy"]
        assert "task-1-3" in sample_spec["hierarchy"]

    def test_spec_with_findable_text_is_valid(self, spec_with_findable_text):
        """Spec with findable text should have expected patterns."""
        assert "FEATURE_V1" in spec_with_findable_text["hierarchy"]["task-1-1"]["title"]
        assert "FEATURE_V1" in spec_with_findable_text["hierarchy"]["task-1-2"]["title"]
        assert "FEATURE_V1" in spec_with_findable_text["hierarchy"]["task-1-3"]["title"]

    def test_response_version_is_current(self):
        """Response helpers should use response-v2 version."""
        response = asdict(success_response())
        assert response["meta"]["version"] == "response-v2"

        error = asdict(error_response("test"))
        assert error["meta"]["version"] == "response-v2"

    def test_error_code_enum_values_phase5(self):
        """ErrorCode enum should have Phase 5 expected values."""
        # Verify error codes used in Phase 5 actions exist
        assert ErrorCode.MISSING_REQUIRED.value == "MISSING_REQUIRED"
        assert ErrorCode.INVALID_FORMAT.value == "INVALID_FORMAT"
        assert ErrorCode.NOT_FOUND.value == "NOT_FOUND"
        assert ErrorCode.VALIDATION_ERROR.value == "VALIDATION_ERROR"

    def test_error_type_enum_values(self):
        """ErrorType enum should have expected values."""
        assert ErrorType.VALIDATION.value == "validation"
        assert ErrorType.NOT_FOUND.value == "not_found"
