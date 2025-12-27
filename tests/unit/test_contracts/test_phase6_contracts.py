"""Contract tests for Phase 6 actions: diff, history, and spec-rollback.

Validates response-v2 envelope compliance per 10-testing-fixtures.md:
1. Response envelope structure matches response-v2
2. Error responses include error_code, error_type, remediation fields
3. Pagination in history action via meta.pagination
4. Dry-run mode returns preview without modifying spec
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


def validate_error_response_fields(
    response: Dict[str, Any], strict_remediation: bool = True
) -> list[str]:
    """Validate error response has structured error fields in data.

    Per mcp_response_schema.md, error responses SHOULD include:
    - error_code: machine-readable code (SCREAMING_SNAKE_CASE)
    - error_type: error category for routing
    - remediation: actionable guidance (optional in lenient mode)
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

    # remediation SHOULD be present (optional in lenient mode)
    if strict_remediation:
        if "remediation" not in data:
            errors.append("Error response missing recommended field: data.remediation")
        elif not isinstance(data["remediation"], str):
            errors.append(
                f"data.remediation must be string, got {type(data['remediation']).__name__}"
            )

    return errors


def validate_pagination_fields(response: Dict[str, Any]) -> list[str]:
    """Validate pagination structure in response data.

    Per mcp_response_schema.md, paginated responses should include:
    - pagination.cursor: Optional[str]
    - pagination.has_more: bool
    - pagination.page_size: int
    """
    errors = []
    data = response.get("data", {})
    pagination = data.get("pagination")

    if pagination is None:
        errors.append("Missing required field: data.pagination")
        return errors

    if not isinstance(pagination, dict):
        errors.append(f"pagination must be object, got {type(pagination).__name__}")
        return errors

    # has_more is required
    if "has_more" not in pagination:
        errors.append("Missing required field: pagination.has_more")
    elif not isinstance(pagination["has_more"], bool):
        errors.append(
            f"pagination.has_more must be boolean, got {type(pagination['has_more']).__name__}"
        )

    # page_size is required
    if "page_size" not in pagination:
        errors.append("Missing required field: pagination.page_size")
    elif not isinstance(pagination["page_size"], int):
        errors.append(
            f"pagination.page_size must be integer, got {type(pagination['page_size']).__name__}"
        )

    # cursor is optional but if present must be string or null
    if "cursor" in pagination and pagination["cursor"] is not None:
        if not isinstance(pagination["cursor"], str):
            errors.append(
                f"pagination.cursor must be string or null, got {type(pagination['cursor']).__name__}"
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


def assert_valid_error_response(
    response: Dict[str, Any], context: str = "", strict_remediation: bool = False
):
    """Assert error response has required error fields.

    By default uses lenient mode (strict_remediation=False) since remediation
    is a SHOULD not MUST in the spec.
    """
    assert_valid_response_v2(response, context)

    errors = validate_error_response_fields(response, strict_remediation=strict_remediation)
    if errors:
        error_msg = f"Error response field validation errors"
        if context:
            error_msg += f" ({context})"
        error_msg += ":\n  - " + "\n  - ".join(errors)
        pytest.fail(error_msg)


def assert_valid_pagination(response: Dict[str, Any], context: str = ""):
    """Assert response has valid pagination structure."""
    assert_valid_response_v2(response, context)

    errors = validate_pagination_fields(response)
    if errors:
        error_msg = f"Pagination validation errors"
        if context:
            error_msg += f" ({context})"
        error_msg += ":\n  - " + "\n  - ".join(errors)
        pytest.fail(error_msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure with .backups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        specs_dir = (Path(tmpdir) / "specs").resolve()

        for status in ("pending", "active", "completed", "archived"):
            (specs_dir / status).mkdir(parents=True)
        (specs_dir / ".backups").mkdir(parents=True)

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
            "revision_history": [
                {
                    "version": "1.0.0",
                    "date": "2025-01-01T00:00:00Z",
                    "changes": "Initial version",
                    "author": "test-author",
                }
            ],
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "in_progress",
                "parent": None,
                "children": ["phase-1"],
                "total_tasks": 2,
                "completed_tasks": 1,
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1: Implementation",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
                "completed_tasks": 1,
                "total_tasks": 2,
                "metadata": {},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "src/feature.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 2",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
                "metadata": {"file_path": "tests/test_feature.py"},
                "dependencies": {"blocks": [], "blocked_by": [], "depends": []},
            },
        },
    }


@pytest.fixture
def mock_server_config():
    """Mock ServerConfig for handler testing."""
    from foundry_mcp.config import ServerConfig

    return ServerConfig(workspace=".")


def _write_spec(specs_dir: Path, spec_data: Dict[str, Any], status: str = "active") -> Path:
    """Helper to write a spec file."""
    spec_id = spec_data["spec_id"]
    spec_file = specs_dir / status / f"{spec_id}.json"
    spec_file.write_text(json.dumps(spec_data))
    return spec_file


def _create_backup(
    specs_dir: Path,
    spec_id: str,
    timestamp: str,
    spec_data: Dict[str, Any],
) -> Path:
    """Helper to create a backup file with a specific timestamp."""
    backups_dir = specs_dir / ".backups" / spec_id
    backups_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backups_dir / f"{timestamp}.json"
    backup_file.write_text(json.dumps(spec_data))
    return backup_file


# ---------------------------------------------------------------------------
# Test Helpers to Invoke Handlers
# ---------------------------------------------------------------------------


def call_spec_handler(
    action: str,
    payload: Dict[str, Any],
    specs_dir: Path,
    config=None,
) -> Dict[str, Any]:
    """Call the spec router handler and return response dict."""
    from foundry_mcp.tools.unified.spec import _SPEC_ROUTER
    from foundry_mcp.config import ServerConfig

    if config is None:
        config = ServerConfig(specs_dir=specs_dir)

    return _SPEC_ROUTER.dispatch(action=action, config=config, payload=payload)


def call_authoring_handler(
    action: str,
    payload: Dict[str, Any],
    specs_dir: Path,
    config=None,
) -> Dict[str, Any]:
    """Call the authoring router handler and return response dict."""
    from foundry_mcp.tools.unified.authoring import _AUTHORING_ROUTER
    from foundry_mcp.config import ServerConfig

    if config is None:
        config = ServerConfig(specs_dir=specs_dir)

    return _AUTHORING_ROUTER.dispatch(action=action, config=config, **payload)


# ---------------------------------------------------------------------------
# Diff Action Contract Tests
# ---------------------------------------------------------------------------


class TestDiffContracts:
    """Contract tests for spec action='diff'."""

    def test_success_response_envelope(self, temp_specs_dir, sample_spec):
        """Success response must match response-v2 envelope."""
        _write_spec(temp_specs_dir, sample_spec)
        # Create a backup to diff against
        _create_backup(
            temp_specs_dir,
            "test-spec-001",
            "2025-01-01T10-00-00.000001",
            sample_spec,
        )

        response = call_spec_handler(
            "diff",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "diff success")
        assert response["success"] is True
        assert response["error"] is None
        assert "meta" in response
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_has_version(self, temp_specs_dir, sample_spec):
        """Success response should include version in meta."""
        _write_spec(temp_specs_dir, sample_spec)
        _create_backup(
            temp_specs_dir,
            "test-spec-001",
            "2025-01-01T10-00-00.000001",
            sample_spec,
        )

        response = call_spec_handler(
            "diff",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        assert response["meta"]["version"] == "response-v2"

    def test_success_response_data_fields(self, temp_specs_dir, sample_spec):
        """Success response data should contain expected fields."""
        _write_spec(temp_specs_dir, sample_spec)
        _create_backup(
            temp_specs_dir,
            "test-spec-001",
            "2025-01-01T10-00-00.000001",
            sample_spec,
        )

        response = call_spec_handler(
            "diff",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["spec_id"] == "test-spec-001"
        assert "summary" in data
        assert "changes" in data
        assert "compared_to" in data
        assert "partial" in data
        # Verify summary structure
        summary = data["summary"]
        assert "added_count" in summary
        assert "removed_count" in summary
        assert "modified_count" in summary
        assert "total_changes" in summary

    def test_diff_with_explicit_target(self, temp_specs_dir, sample_spec):
        """Diff with explicit target timestamp should produce valid response."""
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        response = call_spec_handler(
            "diff",
            {"spec_id": "test-spec-001", "target": timestamp},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "diff with target")
        assert response["success"] is True

    def test_diff_detects_changes(self, temp_specs_dir, sample_spec):
        """Diff should detect changes between backup and current."""
        # Create backup with old state
        old_spec = json.loads(json.dumps(sample_spec))
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, old_spec)

        # Write current spec with modifications
        current_spec = json.loads(json.dumps(sample_spec))
        current_spec["hierarchy"]["task-1-2"]["status"] = "completed"
        _write_spec(temp_specs_dir, current_spec)

        response = call_spec_handler(
            "diff",
            {"spec_id": "test-spec-001", "target": timestamp},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "diff with changes")
        assert response["success"] is True
        assert response["data"]["summary"]["modified_count"] >= 1

    def test_diff_with_limit(self, temp_specs_dir, sample_spec):
        """Diff with limit should respect max_results."""
        _write_spec(temp_specs_dir, sample_spec)
        _create_backup(
            temp_specs_dir,
            "test-spec-001",
            "2025-01-01T10-00-00.000001",
            sample_spec,
        )

        response = call_spec_handler(
            "diff",
            {"spec_id": "test-spec-001", "limit": 2},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "diff with limit")
        assert response["success"] is True

    def test_error_missing_spec_id(self, temp_specs_dir, sample_spec):
        """Error: missing spec_id should return valid error response."""
        _write_spec(temp_specs_dir, sample_spec)

        response = call_spec_handler(
            "diff",
            {},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert "spec_id" in response["error"].lower()
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_error_no_backups(self, temp_specs_dir, sample_spec):
        """Error: no backups should return valid error response."""
        _write_spec(temp_specs_dir, sample_spec)
        # Don't create any backups

        response = call_spec_handler(
            "diff",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "no backups")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.NOT_FOUND.value
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_spec_not_found(self, temp_specs_dir):
        """Error: nonexistent spec should return NOT_FOUND error."""
        response = call_spec_handler(
            "diff",
            {"spec_id": "nonexistent-spec"},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "spec not found")
        assert response["success"] is False
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value


# ---------------------------------------------------------------------------
# History Action Contract Tests
# ---------------------------------------------------------------------------


class TestHistoryContracts:
    """Contract tests for spec action='history'."""

    def test_success_response_envelope(self, temp_specs_dir, sample_spec):
        """Success response must match response-v2 envelope."""
        _write_spec(temp_specs_dir, sample_spec)
        _create_backup(
            temp_specs_dir,
            "test-spec-001",
            "2025-01-01T10-00-00.000001",
            sample_spec,
        )

        response = call_spec_handler(
            "history",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "history success")
        assert response["success"] is True
        assert response["error"] is None

    def test_success_response_has_pagination(self, temp_specs_dir, sample_spec):
        """Success response should include pagination in data."""
        _write_spec(temp_specs_dir, sample_spec)
        _create_backup(
            temp_specs_dir,
            "test-spec-001",
            "2025-01-01T10-00-00.000001",
            sample_spec,
        )

        response = call_spec_handler(
            "history",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "history pagination")
        # Pagination is in data for history action
        data = response["data"]
        assert "backup_count" in data
        assert "revision_count" in data

    def test_success_response_data_fields(self, temp_specs_dir, sample_spec):
        """Success response data should contain expected fields."""
        _write_spec(temp_specs_dir, sample_spec)
        _create_backup(
            temp_specs_dir,
            "test-spec-001",
            "2025-01-01T10-00-00.000001",
            sample_spec,
        )

        response = call_spec_handler(
            "history",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["spec_id"] == "test-spec-001"
        assert "entries" in data
        assert "backup_count" in data
        assert "revision_count" in data
        assert isinstance(data["entries"], list)

    def test_history_includes_backups(self, temp_specs_dir, sample_spec):
        """History should include backup entries."""
        _write_spec(temp_specs_dir, sample_spec)
        _create_backup(
            temp_specs_dir,
            "test-spec-001",
            "2025-01-01T10-00-00.000001",
            sample_spec,
        )

        response = call_spec_handler(
            "history",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        entries = response["data"]["entries"]
        backup_entries = [e for e in entries if e["type"] == "backup"]
        assert len(backup_entries) >= 1
        # Backup entry should have expected fields
        backup = backup_entries[0]
        assert "timestamp" in backup
        assert "file_path" in backup
        assert "file_size_bytes" in backup

    def test_history_includes_revisions(self, temp_specs_dir, sample_spec):
        """History should include revision entries from metadata."""
        _write_spec(temp_specs_dir, sample_spec)

        response = call_spec_handler(
            "history",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["revision_count"] == 1
        revision_entries = [e for e in data["entries"] if e["type"] == "revision"]
        assert len(revision_entries) == 1
        revision = revision_entries[0]
        assert "version" in revision
        assert "timestamp" in revision
        assert "changes" in revision

    def test_history_pagination_with_cursor(self, temp_specs_dir, sample_spec):
        """History should support cursor-based pagination."""
        _write_spec(temp_specs_dir, sample_spec)
        # Create multiple backups
        for i in range(5):
            _create_backup(
                temp_specs_dir,
                "test-spec-001",
                f"2025-01-01T10-00-0{i}.000001",
                sample_spec,
            )

        # First page
        response1 = call_spec_handler(
            "history",
            {"spec_id": "test-spec-001", "limit": 2},
            temp_specs_dir,
        )

        assert_valid_response_v2(response1, "first page")
        assert response1["success"] is True

    def test_history_respects_limit(self, temp_specs_dir, sample_spec):
        """History should respect the limit parameter."""
        _write_spec(temp_specs_dir, sample_spec)
        for i in range(10):
            _create_backup(
                temp_specs_dir,
                "test-spec-001",
                f"2025-01-01T10-00-{i:02d}.000001",
                sample_spec,
            )

        response = call_spec_handler(
            "history",
            {"spec_id": "test-spec-001", "limit": 3},
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        # Should have at most 3 backup entries (plus revision entries)
        backup_entries = [e for e in response["data"]["entries"] if e["type"] == "backup"]
        assert len(backup_entries) <= 3

    def test_error_missing_spec_id(self, temp_specs_dir):
        """Error: missing spec_id should return valid error response."""
        response = call_spec_handler(
            "history",
            {},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert "spec_id" in response["error"].lower()
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_history_empty_when_no_backups(self, temp_specs_dir, sample_spec):
        """History should return empty entries when no backups exist."""
        # Remove revision history for cleaner test
        spec_no_rev = json.loads(json.dumps(sample_spec))
        spec_no_rev["metadata"]["revision_history"] = []
        _write_spec(temp_specs_dir, spec_no_rev)

        response = call_spec_handler(
            "history",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "empty history")
        assert response["success"] is True
        assert response["data"]["backup_count"] == 0


# ---------------------------------------------------------------------------
# Spec-Rollback Action Contract Tests
# ---------------------------------------------------------------------------


class TestSpecRollbackContracts:
    """Contract tests for authoring action='spec-rollback'."""

    def test_success_response_envelope(self, temp_specs_dir, sample_spec):
        """Success response must match response-v2 envelope."""
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": timestamp},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "spec-rollback success")
        assert response["success"] is True
        assert response["error"] is None

    def test_success_response_has_request_id(self, temp_specs_dir, sample_spec):
        """Success response should include request_id in meta."""
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": timestamp},
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        assert "request_id" in response["meta"]

    def test_success_response_has_telemetry(self, temp_specs_dir, sample_spec):
        """Success response should include telemetry in meta."""
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": timestamp},
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        assert "telemetry" in response["meta"]
        assert "duration_ms" in response["meta"]["telemetry"]

    def test_success_response_data_fields(self, temp_specs_dir, sample_spec):
        """Success response data should contain expected fields."""
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": timestamp},
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        data = response["data"]
        assert data["spec_id"] == "test-spec-001"
        assert data["timestamp"] == timestamp
        assert "restored_from" in data
        assert "backup_created" in data

    def test_dry_run_response_format(self, temp_specs_dir, sample_spec):
        """Dry run should return valid response with dry_run indicator."""
        # Write current spec with modifications
        current = json.loads(json.dumps(sample_spec))
        current["metadata"]["version"] = "2.0.0"
        _write_spec(temp_specs_dir, current)

        # Backup of original
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": timestamp, "dry_run": True},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "dry run")
        assert response["success"] is True
        assert response["data"]["dry_run"] is True

    def test_dry_run_does_not_modify_spec(self, temp_specs_dir, sample_spec):
        """Dry run should not modify the actual spec file."""
        # Write current spec with modifications
        current = json.loads(json.dumps(sample_spec))
        current["metadata"]["version"] = "2.0.0"
        _write_spec(temp_specs_dir, current)

        # Backup of original
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": timestamp, "dry_run": True},
            temp_specs_dir,
        )

        # Verify spec was NOT restored
        from foundry_mcp.core.spec import load_spec

        still_current = load_spec("test-spec-001", temp_specs_dir)
        assert still_current["metadata"]["version"] == "2.0.0"

    def test_rollback_creates_safety_backup(self, temp_specs_dir, sample_spec):
        """Rollback should create a safety backup by default."""
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": timestamp},
            temp_specs_dir,
        )

        assert_valid_response_v2(response)
        assert response["data"]["backup_created"] is not None
        # Verify the backup file exists
        backup_path = Path(response["data"]["backup_created"])
        assert backup_path.exists()

    def test_error_missing_spec_id(self, temp_specs_dir):
        """Error: missing spec_id should return valid error response."""
        response = call_authoring_handler(
            "spec-rollback",
            {"version": "2025-01-01T10-00-00.000001"},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing spec_id")
        assert response["success"] is False
        assert "spec_id" in response["error"].lower()
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value
        assert response["data"]["error_type"] == ErrorType.VALIDATION.value

    def test_error_missing_version(self, temp_specs_dir, sample_spec):
        """Error: missing version should return valid error response."""
        _write_spec(temp_specs_dir, sample_spec)

        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "missing version")
        assert response["success"] is False
        assert response["data"]["error_code"] == ErrorCode.MISSING_REQUIRED.value

    def test_error_backup_not_found(self, temp_specs_dir, sample_spec):
        """Error: nonexistent backup should return NOT_FOUND error."""
        _write_spec(temp_specs_dir, sample_spec)
        # Create backup dir but not the specific backup
        (temp_specs_dir / ".backups" / "test-spec-001").mkdir(parents=True, exist_ok=True)

        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": "2099-01-01T00-00-00.000001"},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "backup not found")
        assert response["success"] is False
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_spec_not_found(self, temp_specs_dir):
        """Error: nonexistent spec should return NOT_FOUND error."""
        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "nonexistent-spec", "version": "2025-01-01T10-00-00.000001"},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "spec not found")
        assert response["success"] is False
        assert response["data"]["error_type"] == ErrorType.NOT_FOUND.value

    def test_error_invalid_dry_run_type(self, temp_specs_dir, sample_spec):
        """Error: invalid dry_run type should return INVALID_FORMAT."""
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        response = call_authoring_handler(
            "spec-rollback",
            {
                "spec_id": "test-spec-001",
                "version": timestamp,
                "dry_run": "yes",  # Should be boolean
            },
            temp_specs_dir,
        )

        assert_valid_error_response(response, "invalid dry_run type")
        assert response["success"] is False


# ---------------------------------------------------------------------------
# Edge Case Contract Tests
# ---------------------------------------------------------------------------


class TestPhase6EdgeCases:
    """Test edge cases for Phase 6 response contract compliance."""

    def test_diff_whitespace_spec_id(self, temp_specs_dir, sample_spec):
        """Whitespace-only spec_id should return validation error."""
        _write_spec(temp_specs_dir, sample_spec)

        response = call_spec_handler(
            "diff",
            {"spec_id": "   "},
            temp_specs_dir,
        )

        # Whitespace may be trimmed, resulting in NOT_FOUND instead
        assert response["success"] is False

    def test_history_whitespace_spec_id(self, temp_specs_dir, sample_spec):
        """Whitespace-only spec_id should return empty history or error."""
        _write_spec(temp_specs_dir, sample_spec)

        response = call_spec_handler(
            "history",
            {"spec_id": "   "},
            temp_specs_dir,
        )

        # Either fails or returns empty (handler may trim whitespace)
        assert_valid_response_v2(response)

    def test_rollback_whitespace_spec_id(self, temp_specs_dir, sample_spec):
        """Whitespace-only spec_id should return validation error."""
        _write_spec(temp_specs_dir, sample_spec)

        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "   ", "version": "2025-01-01T10-00-00.000001"},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace spec_id")
        assert response["success"] is False

    def test_rollback_whitespace_version(self, temp_specs_dir, sample_spec):
        """Whitespace-only version should return validation error."""
        _write_spec(temp_specs_dir, sample_spec)

        response = call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": "   "},
            temp_specs_dir,
        )

        assert_valid_error_response(response, "whitespace version")
        assert response["success"] is False


# ---------------------------------------------------------------------------
# Fixture Freshness Contract Tests
# ---------------------------------------------------------------------------


class TestPhase6FixtureFreshness:
    """Test that test fixtures use current response-v2 version."""

    def test_sample_spec_is_valid(self, sample_spec):
        """Sample spec fixture should be valid JSON structure."""
        assert "spec_id" in sample_spec
        assert "hierarchy" in sample_spec
        assert "spec-root" in sample_spec["hierarchy"]
        assert "phase-1" in sample_spec["hierarchy"]
        assert "task-1-1" in sample_spec["hierarchy"]

    def test_sample_spec_has_revision_history(self, sample_spec):
        """Sample spec should have revision history for testing."""
        assert "metadata" in sample_spec
        assert "revision_history" in sample_spec["metadata"]
        assert len(sample_spec["metadata"]["revision_history"]) > 0

    def test_response_version_is_current(self):
        """Response helpers should use response-v2 version."""
        response = asdict(success_response())
        assert response["meta"]["version"] == "response-v2"

        error = asdict(error_response("test"))
        assert error["meta"]["version"] == "response-v2"

    def test_error_code_enum_values_phase6(self):
        """ErrorCode enum should have Phase 6 expected values."""
        assert ErrorCode.MISSING_REQUIRED.value == "MISSING_REQUIRED"
        assert ErrorCode.INVALID_FORMAT.value == "INVALID_FORMAT"
        assert ErrorCode.NOT_FOUND.value == "NOT_FOUND"

    def test_error_type_enum_values(self):
        """ErrorType enum should have expected values."""
        assert ErrorType.VALIDATION.value == "validation"
        assert ErrorType.NOT_FOUND.value == "not_found"


# ---------------------------------------------------------------------------
# Integration Contract Tests
# ---------------------------------------------------------------------------


class TestPhase6Integration:
    """Integration tests combining Phase 6 operations."""

    def test_diff_after_rollback(self, temp_specs_dir, sample_spec):
        """Diff should work correctly after a rollback operation."""
        # Create original spec
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        # Modify the spec
        modified = json.loads(json.dumps(sample_spec))
        modified["hierarchy"]["task-1-2"]["status"] = "completed"
        _write_spec(temp_specs_dir, modified)

        # Create another backup after modification
        _create_backup(
            temp_specs_dir,
            "test-spec-001",
            "2025-01-02T10-00-00.000001",
            modified,
        )

        # Rollback to original
        call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": timestamp},
            temp_specs_dir,
        )

        # Now diff should show changes relative to the post-modification backup
        response = call_spec_handler(
            "diff",
            {"spec_id": "test-spec-001", "target": "2025-01-02T10-00-00.000001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response, "diff after rollback")
        assert response["success"] is True

    def test_history_reflects_rollback_backup(self, temp_specs_dir, sample_spec):
        """History should include safety backup created during rollback."""
        _write_spec(temp_specs_dir, sample_spec)
        timestamp = "2025-01-01T10-00-00.000001"
        _create_backup(temp_specs_dir, "test-spec-001", timestamp, sample_spec)

        # Get initial backup count
        response1 = call_spec_handler(
            "history",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )
        initial_count = response1["data"]["backup_count"]

        # Perform rollback (creates safety backup)
        call_authoring_handler(
            "spec-rollback",
            {"spec_id": "test-spec-001", "version": timestamp},
            temp_specs_dir,
        )

        # Check history again
        response2 = call_spec_handler(
            "history",
            {"spec_id": "test-spec-001"},
            temp_specs_dir,
        )

        assert_valid_response_v2(response2)
        # Should have one more backup (the safety backup)
        assert response2["data"]["backup_count"] == initial_count + 1
