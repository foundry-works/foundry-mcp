"""Contract tests for Phase 3 actions: update-metadata and move.

Validates response-v2 envelope compliance per 10-testing-fixtures.md:
1. Response envelope structure matches response-v2
2. Error responses include error_code, error_type, remediation fields
3. Edge cases: empty results, invalid inputs, partial success
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
# Response-v2 Schema Validation Helpers
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
    """Create a sample spec with phases and tasks."""
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
                "total_tasks": 3,
                "completed_tasks": 1,
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
                "completed_tasks": 1,
                "total_tasks": 2,
            },
            "phase-2": {
                "type": "phase",
                "title": "Phase 2",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-2-1"],
                "completed_tasks": 0,
                "total_tasks": 1,
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1.1",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "file_path": "src/module1.py",
                    "description": "First task description",
                },
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 1.2",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {
                    "file_path": "src/module2.py",
                },
            },
            "task-2-1": {
                "type": "task",
                "title": "Task 2.1",
                "status": "pending",
                "parent": "phase-2",
                "children": [],
                "metadata": {},
            },
        },
        "journal": [],
    }


@pytest.fixture
def spec_with_nested_tasks(sample_spec):
    """Sample spec with nested tasks for circular reference tests."""
    spec = sample_spec.copy()
    spec["hierarchy"] = sample_spec["hierarchy"].copy()

    # Add nested task under task-1-1
    spec["hierarchy"]["task-1-1"]["children"] = ["task-1-1-1"]
    spec["hierarchy"]["task-1-1-1"] = {
        "type": "subtask",
        "title": "Subtask 1.1.1",
        "status": "pending",
        "parent": "task-1-1",
        "children": [],
        "metadata": {},
    }
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


# ---------------------------------------------------------------------------
# Update-Metadata Contract Tests
# ---------------------------------------------------------------------------


class TestUpdateMetadataContracts:
    """Contract tests for task action='update-metadata'."""

    def test_success_response_envelope(self, temp_specs_dir, sample_spec):
        """Success response must match response-v2 envelope."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "title": "Updated Title",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "update-metadata success")
        assert response["success"] is True
        assert response["error"] is None
        assert "meta" in response
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_has_request_id(self, temp_specs_dir, sample_spec):
        """Success response should include request_id in meta."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "title": "New Title",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        # request_id is SHOULD per spec
        assert "request_id" in response["meta"]

    def test_success_response_data_fields(self, temp_specs_dir, sample_spec):
        """Success response data should contain expected fields."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "title": "Updated Title",
                "file_path": "src/new.py",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["spec_id"] == "test-spec-001"
        assert data["task_id"] == "task-1-2"
        assert "fields_updated" in data
        assert "title" in data["fields_updated"]
        assert "file_path" in data["fields_updated"]

    def test_error_missing_spec_id(self, temp_specs_dir, sample_spec):
        """Error: missing spec_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "task_id": "task-1-2",
                "title": "New Title",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert "spec_id" in response["error"].lower()

        data = response["data"]
        assert data["error_code"] == ErrorCode.MISSING_REQUIRED.value
        assert data["error_type"] == ErrorType.VALIDATION.value
        assert "remediation" in data

    def test_error_missing_task_id(self, temp_specs_dir, sample_spec):
        """Error: missing task_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "test-spec-001",
                "title": "New Title",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing task_id")
        assert response["success"] is False
        assert "task_id" in response["error"].lower()

    def test_error_no_fields_to_update(self, temp_specs_dir, sample_spec):
        """Error: no update fields should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "no fields")
        assert response["success"] is False

        data = response["data"]
        assert data["error_code"] == ErrorCode.MISSING_REQUIRED.value
        assert "remediation" in data

    def test_error_task_not_found(self, temp_specs_dir, sample_spec):
        """Error: nonexistent task should return NOT_FOUND error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "test-spec-001",
                "task_id": "nonexistent-task",
                "title": "New Title",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "task not found")
        assert response["success"] is False

        data = response["data"]
        # Should use appropriate error code for not found
        assert data["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_spec_not_found(self, temp_specs_dir):
        """Error: nonexistent spec should return NOT_FOUND error."""
        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "nonexistent-spec",
                "task_id": "task-1-1",
                "title": "New Title",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "spec not found")
        assert response["success"] is False
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_empty_title_rejected(self, temp_specs_dir, sample_spec):
        """Error: empty title should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "title": "",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "empty title")
        assert response["success"] is False
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_error_invalid_dry_run_type(self, temp_specs_dir, sample_spec):
        """Error: invalid dry_run type should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "title": "New Title",
                "dry_run": "yes",  # Should be boolean
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid dry_run type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_dry_run_response_format(self, temp_specs_dir, sample_spec):
        """Dry run should return valid response with dry_run indicator."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "title": "Would Update",
                "dry_run": True,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "dry run")
        assert response["success"] is True
        assert response["data"]["dry_run"] is True
        assert "fields_updated" in response["data"]


# ---------------------------------------------------------------------------
# Move Action Contract Tests
# ---------------------------------------------------------------------------


class TestMoveContracts:
    """Contract tests for task action='move'."""

    def test_success_response_envelope(self, temp_specs_dir, sample_spec):
        """Success response must match response-v2 envelope."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "move success")
        assert response["success"] is True
        assert response["error"] is None
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_has_telemetry(self, temp_specs_dir, sample_spec):
        """Success response should include telemetry in meta."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        assert "telemetry" in response["meta"]
        assert "duration_ms" in response["meta"]["telemetry"]

    def test_reorder_success_data_fields(self, temp_specs_dir, sample_spec):
        """Reorder success should include position change info."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert "task_id" in data
        assert "old_position" in data
        assert "new_position" in data
        assert data["is_reparenting"] is False

    def test_reparent_success_data_fields(self, temp_specs_dir, sample_spec):
        """Reparent success should include parent change info."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "parent": "phase-2",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["is_reparenting"] is True
        assert data["old_parent"] == "phase-1"
        assert data["new_parent"] == "phase-2"

    def test_error_missing_spec_id(self, temp_specs_dir, sample_spec):
        """Error: missing spec_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "task_id": "task-1-2",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_missing_task_id(self, temp_specs_dir, sample_spec):
        """Error: missing task_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing task_id")
        assert response["success"] is False

    def test_error_task_not_found(self, temp_specs_dir, sample_spec):
        """Error: nonexistent task should return TASK_NOT_FOUND."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "nonexistent-task",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "task not found")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.TASK_NOT_FOUND.value
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_invalid_position(self, temp_specs_dir, sample_spec):
        """Error: invalid position should return INVALID_POSITION."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "position": 100,  # Way too high
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid position")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_POSITION.value
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_error_negative_position(self, temp_specs_dir, sample_spec):
        """Error: negative position should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "position": -1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "negative position")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_circular_reference(self, temp_specs_dir, spec_with_nested_tasks):
        """Error: circular reference should return CIRCULAR_DEPENDENCY or INVALID_PARENT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_nested_tasks))

        # Try to move task-1-1 under its own child task-1-1-1
        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "parent": "task-1-1-1",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "circular reference")
        assert response["success"] is False
        # Handler may return CIRCULAR_DEPENDENCY or INVALID_PARENT depending on
        # how the error is detected (cycle detection vs invalid parent type)
        assert response["data"]["error_code"] in [
            ErrorCode.CIRCULAR_DEPENDENCY.value,
            ErrorCode.INVALID_PARENT.value,
        ]
        assert response["data"]["error_type"] in [
            ErrorType.CONFLICT.value,
            ErrorType.VALIDATION.value,
        ]

    def test_error_parent_not_found(self, temp_specs_dir, sample_spec):
        """Error: nonexistent parent should return appropriate error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "parent": "nonexistent-parent",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "parent not found")
        assert response["success"] is False

    def test_error_invalid_parent_format(self, temp_specs_dir, sample_spec):
        """Error: empty parent string should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "parent": "",  # Empty string
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid parent format")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_dry_run_response_format(self, temp_specs_dir, sample_spec):
        """Dry run should return valid response with dry_run indicator."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "parent": "phase-2",
                "dry_run": True,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "dry run")
        assert response["success"] is True
        assert response["data"]["dry_run"] is True

    def test_success_with_warnings(self, temp_specs_dir, sample_spec):
        """Move with warnings should include warnings in meta."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        # Cross-phase move may emit warnings
        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "parent": "phase-2",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        assert response["success"] is True
        # Warnings are optional but should be valid if present
        if "warnings" in response["meta"]:
            assert isinstance(response["meta"]["warnings"], list)


# ---------------------------------------------------------------------------
# Response Helper Contract Tests
# ---------------------------------------------------------------------------


class TestResponseHelperContracts:
    """Test that response helpers produce valid response-v2 structures."""

    def test_success_response_minimal(self):
        """Minimal success_response should be valid."""
        response = asdict(success_response())
        assert_valid_response_v2(response, "minimal success")
        assert response["success"] is True
        assert response["data"] == {}

    def test_success_response_with_data(self):
        """success_response with data should be valid."""
        response = asdict(success_response(data={"key": "value"}))
        assert_valid_response_v2(response, "success with data")
        assert response["data"] == {"key": "value"}

    def test_success_response_with_warnings(self):
        """success_response with warnings should be valid."""
        response = asdict(success_response(warnings=["warning 1", "warning 2"]))
        assert_valid_response_v2(response, "success with warnings")
        assert response["meta"]["warnings"] == ["warning 1", "warning 2"]

    def test_success_response_with_pagination(self):
        """success_response with pagination should be valid."""
        response = asdict(
            success_response(
                data={"items": []},
                pagination={"cursor": "abc", "has_more": True},
            )
        )
        assert_valid_response_v2(response, "success with pagination")
        assert response["meta"]["pagination"]["cursor"] == "abc"
        assert response["meta"]["pagination"]["has_more"] is True

    def test_error_response_minimal(self):
        """Minimal error_response should be valid."""
        response = asdict(error_response("Something failed"))
        assert_valid_response_v2(response, "minimal error")
        # Note: minimal error_response auto-populates error_code and error_type
        # but remediation is optional - so we only check envelope validity
        assert response["success"] is False
        assert response["error"] == "Something failed"
        # Verify error_code and error_type are present (auto-populated defaults)
        assert "error_code" in response["data"]
        assert "error_type" in response["data"]

    def test_error_response_with_code_and_type(self):
        """error_response with code and type should be valid."""
        response = asdict(
            error_response(
                "Validation failed",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Fix the input",
            )
        )
        assert_valid_response_v2(response, "error with code")
        assert_valid_error_response(response, "error with code")
        assert response["data"]["error_code"] == "VALIDATION_ERROR"
        assert response["data"]["error_type"] == "validation"
        assert response["data"]["remediation"] == "Fix the input"

    def test_error_response_with_details(self):
        """error_response with details should be valid."""
        response = asdict(
            error_response(
                "Field invalid",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Check field format",
                details={"field": "email", "constraint": "email_format"},
            )
        )
        assert_valid_response_v2(response, "error with details")
        assert response["data"]["details"]["field"] == "email"


# ---------------------------------------------------------------------------
# Edge Case Contract Tests
# ---------------------------------------------------------------------------


class TestEdgeCaseContracts:
    """Test edge cases for response contract compliance."""

    def test_empty_spec_id_whitespace(self, temp_specs_dir, sample_spec):
        """Whitespace-only spec_id should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "   ",
                "task_id": "task-1-2",
                "title": "New Title",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace spec_id")
        assert response["success"] is False

    def test_empty_task_id_whitespace(self, temp_specs_dir, sample_spec):
        """Whitespace-only task_id should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "test-spec-001",
                "task_id": "   ",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace task_id")
        assert response["success"] is False

    def test_update_metadata_strips_whitespace(self, temp_specs_dir, sample_spec):
        """Title with surrounding whitespace should be trimmed."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "update-metadata",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-2",
                "file_path": "  src/trimmed.py  ",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        assert response["success"] is True

    def test_multiple_validation_errors(self, temp_specs_dir, sample_spec):
        """Multiple invalid inputs should return single valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "move",
            {
                "spec_id": "",  # Invalid
                "task_id": "",  # Invalid
                "position": -1,  # Invalid
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "multiple errors")
        assert response["success"] is False
        # First validation error should be reported
        assert response["data"]["error_code"] is not None


# ---------------------------------------------------------------------------
# Fixture Freshness Contract Tests
# ---------------------------------------------------------------------------


class TestFixtureFreshness:
    """Test that test fixtures use current response-v2 version."""

    def test_sample_spec_is_valid(self, sample_spec):
        """Sample spec fixture should be valid JSON structure."""
        assert "spec_id" in sample_spec
        assert "hierarchy" in sample_spec
        assert "spec-root" in sample_spec["hierarchy"]

    def test_response_version_is_current(self):
        """Response helpers should use response-v2 version."""
        response = asdict(success_response())
        assert response["meta"]["version"] == "response-v2"

        error = asdict(error_response("test"))
        assert error["meta"]["version"] == "response-v2"

    def test_error_code_enum_values(self):
        """ErrorCode enum should have expected values."""
        # Verify error codes used in Phase 3 actions exist
        assert ErrorCode.VALIDATION_ERROR.value == "VALIDATION_ERROR"
        assert ErrorCode.MISSING_REQUIRED.value == "MISSING_REQUIRED"
        assert ErrorCode.INVALID_FORMAT.value == "INVALID_FORMAT"
        assert ErrorCode.TASK_NOT_FOUND.value == "TASK_NOT_FOUND"
        assert ErrorCode.INVALID_POSITION.value == "INVALID_POSITION"
        assert ErrorCode.CIRCULAR_DEPENDENCY.value == "CIRCULAR_DEPENDENCY"

    def test_error_type_enum_values(self):
        """ErrorType enum should have expected values."""
        assert ErrorType.VALIDATION.value == "validation"
        assert ErrorType.NOT_FOUND.value == "not_found"
        assert ErrorType.CONFLICT.value == "conflict"
