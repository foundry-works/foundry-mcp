"""Contract tests for Phase 4 actions: add-dependency, remove-dependency, add-requirement, phase-move.

Validates response-v2 envelope compliance per 10-testing-fixtures.md:
1. Response envelope structure matches response-v2
2. Error responses include error_code, error_type, remediation fields
3. Edge cases: empty results, invalid inputs, circular dependencies
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
                "children": ["phase-1", "phase-2", "phase-3"],
                "total_tasks": 4,
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
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            },
            "phase-2": {
                "type": "phase",
                "title": "Phase 2",
                "status": "pending",
                "parent": "spec-root",
                "children": ["task-2-1"],
                "completed_tasks": 0,
                "total_tasks": 1,
                "dependencies": {
                    "blocks": [],
                    "blocked_by": ["phase-1"],
                    "depends": [],
                },
            },
            "phase-3": {
                "type": "phase",
                "title": "Phase 3",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
                "completed_tasks": 0,
                "total_tasks": 0,
                "dependencies": {
                    "blocks": [],
                    "blocked_by": ["phase-2"],
                    "depends": [],
                },
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
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
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
                "dependencies": {
                    "blocks": [],
                    "blocked_by": [],
                    "depends": [],
                },
            },
            "task-2-1": {
                "type": "task",
                "title": "Task 2.1",
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
def spec_with_existing_dependency(sample_spec):
    """Sample spec with existing dependency for remove-dependency tests."""
    spec = sample_spec.copy()
    spec["hierarchy"] = sample_spec["hierarchy"].copy()

    # Create deep copies to avoid mutation
    spec["hierarchy"]["task-1-1"] = sample_spec["hierarchy"]["task-1-1"].copy()
    spec["hierarchy"]["task-1-2"] = sample_spec["hierarchy"]["task-1-2"].copy()
    spec["hierarchy"]["task-1-1"]["dependencies"] = {
        "blocks": ["task-1-2"],
        "blocked_by": [],
        "depends": [],
    }
    spec["hierarchy"]["task-1-2"]["dependencies"] = {
        "blocks": [],
        "blocked_by": ["task-1-1"],
        "depends": [],
    }
    return spec


@pytest.fixture
def spec_with_requirements(sample_spec):
    """Sample spec with existing requirements for add-requirement tests."""
    spec = sample_spec.copy()
    spec["hierarchy"] = sample_spec["hierarchy"].copy()

    # Create deep copy
    spec["hierarchy"]["task-1-1"] = sample_spec["hierarchy"]["task-1-1"].copy()
    spec["hierarchy"]["task-1-1"]["metadata"] = sample_spec["hierarchy"]["task-1-1"]["metadata"].copy()
    spec["hierarchy"]["task-1-1"]["metadata"]["requirements"] = [
        {"id": "req-1", "type": "acceptance", "text": "Existing requirement"},
    ]
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
# Add-Dependency Contract Tests
# ---------------------------------------------------------------------------


class TestAddDependencyContracts:
    """Contract tests for task action='add-dependency'."""

    def test_success_response_envelope(self, temp_specs_dir, sample_spec):
        """Success response must match response-v2 envelope."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "add-dependency success")
        assert response["success"] is True
        assert response["error"] is None
        assert "meta" in response
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_has_request_id(self, temp_specs_dir, sample_spec):
        """Success response should include request_id in meta."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
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
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
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
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["spec_id"] == "test-spec-001"
        assert data["source_task"] == "task-1-1"
        assert data["target_task"] == "task-1-2"
        assert data["dependency_type"] == "blocks"
        assert data["action"] == "add"
        assert "source_dependencies" in data
        assert "target_dependencies" in data

    def test_dry_run_response_format(self, temp_specs_dir, sample_spec):
        """Dry run should return valid response with dry_run indicator."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
                "dry_run": True,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "dry run")
        assert response["success"] is True
        assert response["data"]["dry_run"] is True
        assert "message" in response["data"]

    def test_error_missing_spec_id(self, temp_specs_dir, sample_spec):
        """Error: missing spec_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert "spec_id" in response["error"].lower()
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_error_missing_task_id(self, temp_specs_dir, sample_spec):
        """Error: missing task_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing task_id")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_missing_target_id(self, temp_specs_dir, sample_spec):
        """Error: missing target_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing target_id")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_invalid_dependency_type(self, temp_specs_dir, sample_spec):
        """Error: invalid dependency_type should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "invalid_type",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid dependency_type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_self_reference(self, temp_specs_dir, sample_spec):
        """Error: self-referencing dependency should return SELF_REFERENCE."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-1",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "self reference")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.SELF_REFERENCE.value
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_error_task_not_found(self, temp_specs_dir, sample_spec):
        """Error: nonexistent task should return TASK_NOT_FOUND."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "nonexistent-task",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "task not found")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.TASK_NOT_FOUND.value
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_duplicate_dependency(self, temp_specs_dir, spec_with_existing_dependency):
        """Error: duplicate dependency should return DUPLICATE_ENTRY."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_existing_dependency))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "duplicate dependency")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.DUPLICATE_ENTRY.value
        assert response["data"]["error_type"] == ErrorType.CONFLICT.value

    def test_error_spec_not_found(self, temp_specs_dir):
        """Error: nonexistent spec should return error."""
        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "nonexistent-spec",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "spec not found")
        assert response["success"] is False
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_invalid_dry_run_type(self, temp_specs_dir, sample_spec):
        """Error: invalid dry_run type should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
                "dry_run": "yes",  # Should be boolean
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid dry_run type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value


# ---------------------------------------------------------------------------
# Remove-Dependency Contract Tests
# ---------------------------------------------------------------------------


class TestRemoveDependencyContracts:
    """Contract tests for task action='remove-dependency'."""

    def test_success_response_envelope(self, temp_specs_dir, spec_with_existing_dependency):
        """Success response must match response-v2 envelope."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_existing_dependency))

        response = call_task_handler(
            "remove-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "remove-dependency success")
        assert response["success"] is True
        assert response["error"] is None
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_data_fields(self, temp_specs_dir, spec_with_existing_dependency):
        """Success response data should contain expected fields."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_existing_dependency))

        response = call_task_handler(
            "remove-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["spec_id"] == "test-spec-001"
        assert data["source_task"] == "task-1-1"
        assert data["target_task"] == "task-1-2"
        assert data["dependency_type"] == "blocks"
        assert data["action"] == "remove"

    def test_error_missing_spec_id(self, temp_specs_dir, spec_with_existing_dependency):
        """Error: missing spec_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_existing_dependency))

        response = call_task_handler(
            "remove-dependency",
            {
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_dependency_not_found(self, temp_specs_dir, sample_spec):
        """Error: dependency that doesn't exist should return DEPENDENCY_NOT_FOUND."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "remove-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "dependency not found")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.DEPENDENCY_NOT_FOUND.value
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_dry_run_response_format(self, temp_specs_dir, spec_with_existing_dependency):
        """Dry run should return valid response with dry_run indicator."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(spec_with_existing_dependency))

        response = call_task_handler(
            "remove-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
                "dry_run": True,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "dry run")
        assert response["success"] is True
        assert response["data"]["dry_run"] is True


# ---------------------------------------------------------------------------
# Add-Requirement Contract Tests
# ---------------------------------------------------------------------------


class TestAddRequirementContracts:
    """Contract tests for task action='add-requirement'."""

    def test_success_response_envelope(self, temp_specs_dir, sample_spec):
        """Success response must match response-v2 envelope."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "requirement_type": "acceptance",
                "text": "The function should return a valid response",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "add-requirement success")
        assert response["success"] is True
        assert response["error"] is None
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_has_telemetry(self, temp_specs_dir, sample_spec):
        """Success response should include telemetry in meta."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "requirement_type": "technical",
                "text": "Must handle edge cases",
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
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "requirement_type": "constraint",
                "text": "Response time must be under 100ms",
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["spec_id"] == "test-spec-001"
        assert data["task_id"] == "task-1-1"
        assert data["action"] == "add"
        assert "requirement" in data
        assert data["requirement"]["type"] == "constraint"
        assert "id" in data["requirement"]

    def test_dry_run_response_format(self, temp_specs_dir, sample_spec):
        """Dry run should return valid response with dry_run indicator."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "requirement_type": "acceptance",
                "text": "Test requirement",
                "dry_run": True,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "dry run")
        assert response["success"] is True
        assert response["data"]["dry_run"] is True

    def test_error_missing_spec_id(self, temp_specs_dir, sample_spec):
        """Error: missing spec_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-requirement",
            {
                "task_id": "task-1-1",
                "requirement_type": "acceptance",
                "text": "Test requirement",
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
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "requirement_type": "acceptance",
                "text": "Test requirement",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing task_id")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_missing_requirement_type(self, temp_specs_dir, sample_spec):
        """Error: missing requirement_type should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "text": "Test requirement",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing requirement_type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_invalid_requirement_type(self, temp_specs_dir, sample_spec):
        """Error: invalid requirement_type should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "requirement_type": "invalid_type",
                "text": "Test requirement",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid requirement_type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_missing_text(self, temp_specs_dir, sample_spec):
        """Error: missing text should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "requirement_type": "acceptance",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing text")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_empty_text(self, temp_specs_dir, sample_spec):
        """Error: empty text should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "requirement_type": "acceptance",
                "text": "",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "empty text")
        assert response["success"] is False

    def test_error_task_not_found(self, temp_specs_dir, sample_spec):
        """Error: nonexistent task should return TASK_NOT_FOUND."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "task_id": "nonexistent-task",
                "requirement_type": "acceptance",
                "text": "Test requirement",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "task not found")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.TASK_NOT_FOUND.value
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value


# ---------------------------------------------------------------------------
# Phase-Move Contract Tests
# ---------------------------------------------------------------------------


class TestPhaseMoveContracts:
    """Contract tests for authoring action='phase-move'."""

    def test_success_response_envelope(self, temp_specs_dir, sample_spec):
        """Success response must match response-v2 envelope."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "phase-2",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "phase-move success")
        assert response["success"] is True
        assert response["error"] is None
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_has_telemetry(self, temp_specs_dir, sample_spec):
        """Success response should include telemetry in meta."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "phase-2",
                "position": 1,
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

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "phase-2",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["spec_id"] == "test-spec-001"
        assert data["phase_id"] == "phase-2"
        assert "old_position" in data
        assert "new_position" in data

    def test_dry_run_response_format(self, temp_specs_dir, sample_spec):
        """Dry run should return valid response with dry_run indicator."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "phase-2",
                "position": 1,
                "dry_run": True,
            },
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "dry run")
        assert response["success"] is True
        assert response["data"]["dry_run"] is True

    def test_error_missing_spec_id(self, temp_specs_dir, sample_spec):
        """Error: missing spec_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "phase_id": "phase-2",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_error_missing_phase_id(self, temp_specs_dir, sample_spec):
        """Error: missing phase_id should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing phase_id")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_missing_position(self, temp_specs_dir, sample_spec):
        """Error: missing position should return valid error response."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "phase-2",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing position")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_invalid_position_type(self, temp_specs_dir, sample_spec):
        """Error: non-integer position should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "phase-2",
                "position": "first",  # Should be integer
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid position type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_negative_position(self, temp_specs_dir, sample_spec):
        """Error: negative position should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "phase-2",
                "position": -1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "negative position")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_zero_position(self, temp_specs_dir, sample_spec):
        """Error: zero position should return INVALID_FORMAT (1-based)."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "phase-2",
                "position": 0,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "zero position")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value

    def test_error_phase_not_found(self, temp_specs_dir, sample_spec):
        """Error: nonexistent phase should return PHASE_NOT_FOUND."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "nonexistent-phase",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "phase not found")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.PHASE_NOT_FOUND.value
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_spec_not_found(self, temp_specs_dir):
        """Error: nonexistent spec should return SPEC_NOT_FOUND."""
        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "nonexistent-spec",
                "phase_id": "phase-1",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "spec not found")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.SPEC_NOT_FOUND.value
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_not_a_phase(self, temp_specs_dir, sample_spec):
        """Error: moving a non-phase node should return VALIDATION_ERROR."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "task-1-1",  # This is a task, not a phase
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "not a phase")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.VALIDATION_ERROR.value
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_error_position_out_of_bounds(self, temp_specs_dir, sample_spec):
        """Error: position out of bounds should return error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        # sample_spec has 3 phases, so position 100 is out of bounds
        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "phase-2",
                "position": 100,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "position out of bounds")
        assert response["success"] is False

    def test_error_invalid_dry_run_type(self, temp_specs_dir, sample_spec):
        """Error: invalid dry_run type should return INVALID_FORMAT."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "phase-2",
                "position": 1,
                "dry_run": "yes",  # Should be boolean
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid dry_run type")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.INVALID_FORMAT.value


# ---------------------------------------------------------------------------
# Edge Case Contract Tests
# ---------------------------------------------------------------------------


class TestPhase4EdgeCases:
    """Test edge cases for Phase 4 response contract compliance."""

    def test_add_dependency_empty_spec_id_whitespace(self, temp_specs_dir, sample_spec):
        """Whitespace-only spec_id should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "   ",
                "task_id": "task-1-1",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace spec_id")
        assert response["success"] is False

    def test_add_dependency_empty_task_id_whitespace(self, temp_specs_dir, sample_spec):
        """Whitespace-only task_id should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-dependency",
            {
                "spec_id": "test-spec-001",
                "task_id": "   ",
                "target_id": "task-1-2",
                "dependency_type": "blocks",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace task_id")
        assert response["success"] is False

    def test_add_requirement_whitespace_text(self, temp_specs_dir, sample_spec):
        """Whitespace-only text should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_task_handler(
            "add-requirement",
            {
                "spec_id": "test-spec-001",
                "task_id": "task-1-1",
                "requirement_type": "acceptance",
                "text": "   ",
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace text")
        assert response["success"] is False

    def test_phase_move_empty_phase_id_whitespace(self, temp_specs_dir, sample_spec):
        """Whitespace-only phase_id should return validation error."""
        spec_file = temp_specs_dir / "active" / "test-spec-001.json"
        spec_file.write_text(json.dumps(sample_spec))

        response = call_authoring_handler(
            "phase-move",
            {
                "spec_id": "test-spec-001",
                "phase_id": "   ",
                "position": 1,
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace phase_id")
        assert response["success"] is False

    def test_add_dependency_all_dependency_types(self, temp_specs_dir, sample_spec):
        """All valid dependency types should produce valid response."""
        for dep_type in ("blocks", "blocked_by", "depends"):
            # Create fresh spec for each iteration
            spec_file = temp_specs_dir / "active" / "test-spec-001.json"
            spec_file.write_text(json.dumps(sample_spec))

            response = call_task_handler(
                "add-dependency",
                {
                    "spec_id": "test-spec-001",
                    "task_id": "task-1-1",
                    "target_id": "task-1-2",
                    "dependency_type": dep_type,
                },
                temp_specs_dir,
            )

            assert_valid_response_v2(response, f"dependency type: {dep_type}")
            assert response["success"] is True
            assert response["data"]["dependency_type"] == dep_type

    def test_add_requirement_all_requirement_types(self, temp_specs_dir, sample_spec):
        """All valid requirement types should produce valid response."""
        for req_type in ("acceptance", "technical", "constraint"):
            # Create fresh spec for each iteration
            spec_file = temp_specs_dir / "active" / "test-spec-001.json"
            spec_file.write_text(json.dumps(sample_spec))

            response = call_task_handler(
                "add-requirement",
                {
                    "spec_id": "test-spec-001",
                    "task_id": "task-1-1",
                    "requirement_type": req_type,
                    "text": f"Test {req_type} requirement",
                },
                temp_specs_dir,
            )

            assert_valid_response_v2(response, f"requirement type: {req_type}")
            assert response["success"] is True
            assert response["data"]["requirement"]["type"] == req_type


# ---------------------------------------------------------------------------
# Fixture Freshness Contract Tests
# ---------------------------------------------------------------------------


class TestPhase4FixtureFreshness:
    """Test that test fixtures use current response-v2 version."""

    def test_sample_spec_is_valid(self, sample_spec):
        """Sample spec fixture should be valid JSON structure."""
        assert "spec_id" in sample_spec
        assert "hierarchy" in sample_spec
        assert "spec-root" in sample_spec["hierarchy"]
        # Phase 4 specific: phases exist
        assert "phase-1" in sample_spec["hierarchy"]
        assert "phase-2" in sample_spec["hierarchy"]
        # Dependencies are structured
        assert "dependencies" in sample_spec["hierarchy"]["task-1-1"]

    def test_response_version_is_current(self):
        """Response helpers should use response-v2 version."""
        response = asdict(success_response())
        assert response["meta"]["version"] == "response-v2"

        error = asdict(error_response("test"))
        assert error["meta"]["version"] == "response-v2"

    def test_error_code_enum_values_phase4(self):
        """ErrorCode enum should have Phase 4 expected values."""
        # Verify error codes used in Phase 4 actions exist
        assert ErrorCode.SELF_REFERENCE.value == "SELF_REFERENCE"
        assert ErrorCode.DUPLICATE_ENTRY.value == "DUPLICATE_ENTRY"
        assert ErrorCode.CIRCULAR_DEPENDENCY.value == "CIRCULAR_DEPENDENCY"
        assert ErrorCode.DEPENDENCY_NOT_FOUND.value == "DEPENDENCY_NOT_FOUND"
        assert ErrorCode.PHASE_NOT_FOUND.value == "PHASE_NOT_FOUND"
        assert ErrorCode.TASK_NOT_FOUND.value == "TASK_NOT_FOUND"
        assert ErrorCode.SPEC_NOT_FOUND.value == "SPEC_NOT_FOUND"

    def test_error_type_enum_values(self):
        """ErrorType enum should have expected values."""
        assert ErrorType.VALIDATION.value == "validation"
        assert ErrorType.NOT_FOUND.value == "not_found"
        assert ErrorType.CONFLICT.value == "conflict"
