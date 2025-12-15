"""
Contract tests for MCP response schema validation.

Uses JSON Schema validation to ensure all tool responses conform to the
response-v2 contract defined in docs/codebase_standards/mcp_response_schema.md.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pytest

try:
    import jsonschema
    from jsonschema import Draft7Validator, ValidationError
except ImportError:
    pytest.skip("jsonschema not installed", allow_module_level=True)

from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    ToolResponse,
    conflict_error,
    error_response,
    forbidden_error,
    internal_error,
    not_found_error,
    rate_limit_error,
    success_response,
    unauthorized_error,
    unavailable_error,
    validation_error,
)


# Load schema once at module level
SCHEMA_PATH = Path(__file__).parent / "response_schema.json"


@pytest.fixture(scope="module")
def response_schema() -> Dict[str, Any]:
    """Load the response schema from JSON file."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def validator(response_schema: Dict[str, Any]) -> Draft7Validator:
    """Create a reusable JSON Schema validator."""
    Draft7Validator.check_schema(response_schema)
    return Draft7Validator(response_schema)


def validate_response(response: ToolResponse, validator: Draft7Validator) -> None:
    """Validate a ToolResponse against the schema, raising on failure."""
    response_dict = asdict(response)
    errors = list(validator.iter_errors(response_dict))
    if errors:
        error_messages = [f"- {e.message} at {list(e.absolute_path)}" for e in errors]
        pytest.fail(
            "Schema validation failed:\n" + "\n".join(error_messages) +
            f"\n\nResponse: {json.dumps(response_dict, indent=2)}"
        )


class TestSchemaValidity:
    """Tests to ensure the schema itself is valid."""

    def test_schema_is_valid_draft7(self, response_schema: Dict[str, Any]):
        """Verify the schema conforms to JSON Schema Draft-07."""
        Draft7Validator.check_schema(response_schema)

    def test_schema_has_required_definitions(self, response_schema: Dict[str, Any]):
        """Verify schema contains all expected definitions."""
        definitions = response_schema.get("definitions", {})
        expected = {"metadata", "pagination", "rate_limit", "telemetry", "error_data"}
        assert expected.issubset(definitions.keys()), (
            f"Missing definitions: {expected - definitions.keys()}"
        )


class TestSuccessResponseContract:
    """Contract tests for success responses."""

    def test_minimal_success_response(self, validator: Draft7Validator):
        """Test minimal valid success response."""
        response = success_response()
        validate_response(response, validator)

    def test_success_with_data_payload(self, validator: Draft7Validator):
        """Test success response with data payload."""
        response = success_response(
            data={"spec_id": "test-spec", "count": 5, "tasks": ["task-1", "task-2"]}
        )
        validate_response(response, validator)

    def test_success_with_nested_data(self, validator: Draft7Validator):
        """Test success response with deeply nested data."""
        response = success_response(
            data={
                "spec": {
                    "id": "test-spec",
                    "metadata": {"version": "1.0", "author": "test"},
                },
                "phases": [
                    {"id": "phase-1", "tasks": [{"id": "task-1-1"}]},
                    {"id": "phase-2", "tasks": []},
                ],
            }
        )
        validate_response(response, validator)

    def test_success_with_request_id(self, validator: Draft7Validator):
        """Test success response with request_id in meta."""
        response = success_response(request_id="req_abc123")
        validate_response(response, validator)
        assert response.meta["request_id"] == "req_abc123"

    def test_success_with_warnings(self, validator: Draft7Validator):
        """Test success response with warnings array."""
        response = success_response(
            data={"result": "partial"},
            warnings=["Some items skipped", "Rate limit approaching"],
        )
        validate_response(response, validator)
        assert response.meta["warnings"] == ["Some items skipped", "Rate limit approaching"]

    def test_success_with_pagination(self, validator: Draft7Validator):
        """Test success response with pagination metadata."""
        response = success_response(
            data={"items": [1, 2, 3]},
            pagination={
                "cursor": "abc123",
                "has_more": True,
                "total_count": 100,
                "page_size": 10,
            },
        )
        validate_response(response, validator)
        assert response.meta["pagination"]["has_more"] is True

    def test_success_with_rate_limit(self, validator: Draft7Validator):
        """Test success response with rate limit metadata."""
        response = success_response(
            data={"result": "ok"},
            rate_limit={
                "limit": 100,
                "remaining": 95,
                "reset_at": "2025-01-01T00:00:00Z",
                "period": "minute",
            },
        )
        validate_response(response, validator)
        assert response.meta["rate_limit"]["remaining"] == 95

    def test_success_with_telemetry(self, validator: Draft7Validator):
        """Test success response with telemetry metadata."""
        response = success_response(
            data={"result": "ok"},
            telemetry={
                "duration_ms": 45.2,
                "downstream_calls": 3,
                "cache_hit": False,
            },
        )
        validate_response(response, validator)
        assert response.meta["telemetry"]["duration_ms"] == 45.2

    def test_success_with_all_metadata(self, validator: Draft7Validator):
        """Test success response with all metadata fields populated."""
        response = success_response(
            data={"items": ["a", "b", "c"]},
            request_id="req_full_test",
            warnings=["Warning 1"],
            pagination={"cursor": "next", "has_more": True},
            rate_limit={"limit": 100, "remaining": 50},
            telemetry={"duration_ms": 100},
        )
        validate_response(response, validator)
        assert response.meta["version"] == "response-v2"
        assert "request_id" in response.meta
        assert "warnings" in response.meta
        assert "pagination" in response.meta
        assert "rate_limit" in response.meta
        assert "telemetry" in response.meta


class TestErrorResponseContract:
    """Contract tests for error responses."""

    def test_minimal_error_response(self, validator: Draft7Validator):
        """Test minimal valid error response."""
        response = error_response("Something went wrong")
        validate_response(response, validator)
        assert response.success is False
        assert response.error == "Something went wrong"

    def test_error_with_error_code(self, validator: Draft7Validator):
        """Test error response with machine-readable error code."""
        response = error_response(
            "Validation failed",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
        validate_response(response, validator)
        assert response.data["error_code"] == "VALIDATION_ERROR"

    def test_error_with_error_type(self, validator: Draft7Validator):
        """Test error response with error type category."""
        response = error_response(
            "Resource not found",
            error_type=ErrorType.NOT_FOUND,
        )
        validate_response(response, validator)
        assert response.data["error_type"] == "not_found"

    def test_error_with_remediation(self, validator: Draft7Validator):
        """Test error response with remediation guidance."""
        response = error_response(
            "Invalid input",
            remediation="Provide a valid spec_id parameter",
        )
        validate_response(response, validator)
        assert response.data["remediation"] == "Provide a valid spec_id parameter"

    def test_error_with_details(self, validator: Draft7Validator):
        """Test error response with nested error details."""
        response = error_response(
            "Validation failed: spec_id is required",
            error_code=ErrorCode.MISSING_REQUIRED,
            error_type=ErrorType.VALIDATION,
            details={"field": "spec_id", "constraint": "required", "received": None},
        )
        validate_response(response, validator)
        assert response.data["details"]["field"] == "spec_id"

    def test_error_with_full_context(self, validator: Draft7Validator):
        """Test error response with all error context fields."""
        response = error_response(
            "Rate limit exceeded: 100 requests per minute",
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            error_type=ErrorType.RATE_LIMIT,
            remediation="Wait 45 seconds before retrying",
            details={"retry_after": 45},
            request_id="req_ratelimit",
            rate_limit={"limit": 100, "remaining": 0, "retry_after": 45},
        )
        validate_response(response, validator)


class TestSpecializedErrorHelpers:
    """Contract tests for specialized error helper functions."""

    def test_validation_error_helper(self, validator: Draft7Validator):
        """Test validation_error helper produces valid response."""
        response = validation_error(
            "Invalid email format",
            field="email",
            details={"constraint": "email_format"},
            remediation="Provide email in format: user@domain.com",
        )
        validate_response(response, validator)
        assert response.data["error_code"] == "VALIDATION_ERROR"
        assert response.data["error_type"] == "validation"

    def test_not_found_error_helper(self, validator: Draft7Validator):
        """Test not_found_error helper produces valid response."""
        response = not_found_error("Spec", "my-spec-001")
        validate_response(response, validator)
        assert response.data["error_code"] == "NOT_FOUND"
        assert response.data["error_type"] == "not_found"
        assert response.data["resource_type"] == "Spec"
        assert response.data["resource_id"] == "my-spec-001"

    def test_rate_limit_error_helper(self, validator: Draft7Validator):
        """Test rate_limit_error helper produces valid response."""
        response = rate_limit_error(100, "minute", 45)
        validate_response(response, validator)
        assert response.data["error_code"] == "RATE_LIMIT_EXCEEDED"
        assert response.data["error_type"] == "rate_limit"
        assert response.meta["rate_limit"]["limit"] == 100

    def test_unauthorized_error_helper(self, validator: Draft7Validator):
        """Test unauthorized_error helper produces valid response."""
        response = unauthorized_error("Invalid API key")
        validate_response(response, validator)
        assert response.data["error_code"] == "UNAUTHORIZED"
        assert response.data["error_type"] == "authentication"

    def test_forbidden_error_helper(self, validator: Draft7Validator):
        """Test forbidden_error helper produces valid response."""
        response = forbidden_error(
            "Cannot delete project",
            required_permission="project:delete",
        )
        validate_response(response, validator)
        assert response.data["error_code"] == "FORBIDDEN"
        assert response.data["error_type"] == "authorization"

    def test_conflict_error_helper(self, validator: Draft7Validator):
        """Test conflict_error helper produces valid response."""
        response = conflict_error(
            "Resource already exists",
            details={"existing_id": "spec-001"},
        )
        validate_response(response, validator)
        assert response.data["error_code"] == "CONFLICT"
        assert response.data["error_type"] == "conflict"

    def test_internal_error_helper(self, validator: Draft7Validator):
        """Test internal_error helper produces valid response."""
        response = internal_error(request_id="req_internal")
        validate_response(response, validator)
        assert response.data["error_code"] == "INTERNAL_ERROR"
        assert response.data["error_type"] == "internal"

    def test_unavailable_error_helper(self, validator: Draft7Validator):
        """Test unavailable_error helper produces valid response."""
        response = unavailable_error(
            "Database maintenance in progress",
            retry_after_seconds=300,
        )
        validate_response(response, validator)
        assert response.data["error_code"] == "UNAVAILABLE"
        assert response.data["error_type"] == "unavailable"
        assert response.data["retry_after_seconds"] == 300


class TestEnvelopeInvariants:
    """Tests for envelope invariants that must always hold."""

    def test_success_true_requires_null_error(self, validator: Draft7Validator):
        """When success=True, error must be null."""
        response = success_response(data={"result": "ok"})
        assert response.success is True
        assert response.error is None
        validate_response(response, validator)

    def test_success_false_requires_error_message(self, validator: Draft7Validator):
        """When success=False, error must be non-empty string."""
        response = error_response("Something failed")
        assert response.success is False
        assert response.error is not None
        assert len(response.error) > 0
        validate_response(response, validator)

    def test_meta_always_has_version(self, validator: Draft7Validator):
        """Meta must always contain version field."""
        response = success_response()
        assert response.meta["version"] == "response-v2"
        validate_response(response, validator)

        response = error_response("Error")
        assert response.meta["version"] == "response-v2"
        validate_response(response, validator)

    def test_data_is_always_object(self, validator: Draft7Validator):
        """Data must always be an object (dict), never array or primitive."""
        response = success_response()
        assert isinstance(response.data, dict)
        validate_response(response, validator)

        response = error_response("Error")
        assert isinstance(response.data, dict)
        validate_response(response, validator)


class TestSchemaRejection:
    """Tests that invalid responses are rejected by the schema."""

    def test_rejects_missing_success(self, validator: Draft7Validator):
        """Schema rejects response without success field."""
        invalid = {"data": {}, "error": None, "meta": {"version": "response-v2"}}
        errors = list(validator.iter_errors(invalid))
        assert len(errors) > 0
        assert any("success" in str(e.message) for e in errors)

    def test_rejects_missing_data(self, validator: Draft7Validator):
        """Schema rejects response without data field."""
        invalid = {"success": True, "error": None, "meta": {"version": "response-v2"}}
        errors = list(validator.iter_errors(invalid))
        assert len(errors) > 0
        assert any("data" in str(e.message) for e in errors)

    def test_rejects_missing_error(self, validator: Draft7Validator):
        """Schema rejects response without error field."""
        invalid = {"success": True, "data": {}, "meta": {"version": "response-v2"}}
        errors = list(validator.iter_errors(invalid))
        assert len(errors) > 0
        assert any("error" in str(e.message) for e in errors)

    def test_rejects_missing_meta(self, validator: Draft7Validator):
        """Schema rejects response without meta field."""
        invalid = {"success": True, "data": {}, "error": None}
        errors = list(validator.iter_errors(invalid))
        assert len(errors) > 0
        assert any("meta" in str(e.message) for e in errors)

    def test_rejects_wrong_version(self, validator: Draft7Validator):
        """Schema rejects response with wrong version string."""
        invalid = {
            "success": True,
            "data": {},
            "error": None,
            "meta": {"version": "response-v1"},  # Wrong version
        }
        errors = list(validator.iter_errors(invalid))
        assert len(errors) > 0

    def test_rejects_success_true_with_error_string(self, validator: Draft7Validator):
        """Schema rejects success=True when error is a non-null string."""
        invalid = {
            "success": True,
            "data": {},
            "error": "This should not exist",  # Invalid: should be null
            "meta": {"version": "response-v2"},
        }
        errors = list(validator.iter_errors(invalid))
        assert len(errors) > 0

    def test_rejects_success_false_with_null_error(self, validator: Draft7Validator):
        """Schema rejects success=False when error is null."""
        invalid = {
            "success": False,
            "data": {},
            "error": None,  # Invalid: should have message
            "meta": {"version": "response-v2"},
        }
        errors = list(validator.iter_errors(invalid))
        assert len(errors) > 0

    def test_rejects_extra_top_level_fields(self, validator: Draft7Validator):
        """Schema rejects responses with extra top-level fields."""
        invalid = {
            "success": True,
            "data": {},
            "error": None,
            "meta": {"version": "response-v2"},
            "extra_field": "not allowed",
        }
        errors = list(validator.iter_errors(invalid))
        assert len(errors) > 0


class TestRealWorldScenarios:
    """Tests modeling real-world response scenarios."""

    def test_spec_list_response(self, validator: Draft7Validator):
        """Test a typical spec listing response."""
        response = success_response(
            data={
                "specs": [
                    {"id": "spec-1", "title": "Feature A", "status": "active"},
                    {"id": "spec-2", "title": "Feature B", "status": "completed"},
                ],
                "count": 2,
            },
            pagination={"has_more": False, "total_count": 2},
        )
        validate_response(response, validator)

    def test_task_query_with_empty_results(self, validator: Draft7Validator):
        """Test a query that returns no results (still success)."""
        response = success_response(
            data={"tasks": [], "count": 0},
            warnings=["No tasks match the specified criteria"],
        )
        validate_response(response, validator)
        assert response.success is True

    def test_blocked_task_check(self, validator: Draft7Validator):
        """Test dependency check for a blocked task."""
        response = success_response(
            data={
                "task_id": "task-1-2",
                "can_start": False,
                "blocked_by": [{"id": "task-1-1", "status": "pending"}],
            },
            warnings=["Task currently blocked by 1 dependency"],
        )
        validate_response(response, validator)

    def test_rate_limited_request(self, validator: Draft7Validator):
        """Test a rate-limited error response."""
        response = rate_limit_error(
            limit=100,
            period="minute",
            retry_after_seconds=30,
            remediation="Wait 30 seconds or batch your requests",
        )
        validate_response(response, validator)
        assert response.success is False
        assert response.meta["rate_limit"]["retry_after"] == 30

    def test_validation_failure_with_multiple_fields(self, validator: Draft7Validator):
        """Test validation error affecting multiple fields."""
        response = error_response(
            "Multiple validation errors",
            error_code=ErrorCode.VALIDATION_ERROR,
            error_type=ErrorType.VALIDATION,
            details={
                "errors": [
                    {"field": "spec_id", "message": "Required"},
                    {"field": "title", "message": "Too short"},
                ]
            },
            remediation="Fix the validation errors and retry",
        )
        validate_response(response, validator)
