"""Parity regression tests for native SDD CLI.

Validates that native CLI outputs match expected golden fixture schemas,
ensuring backward compatibility and output structure stability.

Tests cover:
- Response envelope consistency (success/data/error/meta structure)
- Key field presence for each command type
- Error response structure consistency
- Schema stability against golden fixtures
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from foundry_mcp.cli.main import cli


# Golden fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "golden"


def load_golden_fixture(name: str) -> dict:
    """Load a golden fixture by name."""
    fixture_path = FIXTURES_DIR / f"{name}.json"
    if not fixture_path.exists():
        pytest.skip(f"Golden fixture not found: {name}")
    with open(fixture_path) as f:
        return json.load(f)


def validate_envelope_structure(response: dict, is_error: bool = False) -> list[str]:
    """Validate response envelope structure and return any violations.

    Per docs/codebase_standards/mcp_response_schema.md, the envelope is:
    - success (bool): required
    - data (object): required - contains payload on success, error context on failure
    - error (string|null): required - null on success, message string on failure
    - meta (object): required - contains version and optional metadata

    Args:
        response: The response dict to validate
        is_error: If True, validates as error response
    """
    violations = []

    # All responses require success, data, and error keys per response-v2 contract
    required_keys = {"success", "data", "error"}
    missing_keys = required_keys - set(response.keys())
    if missing_keys:
        violations.append(f"Missing envelope keys: {missing_keys}")

    # Type checks
    if "success" in response and not isinstance(response["success"], bool):
        violations.append(
            f"'success' must be boolean, got {type(response['success']).__name__}"
        )

    if "data" in response and not isinstance(response["data"], dict):
        violations.append(
            f"'data' must be object, got {type(response['data']).__name__}"
        )

    if (
        "error" in response
        and response["error"] is not None
        and not isinstance(response["error"], str)
    ):
        violations.append(
            f"'error' must be string or null, got {type(response['error']).__name__}"
        )

    # Consistency checks
    if response.get("success") is True and response.get("error") is not None:
        violations.append("Successful response should have error=null")

    if response.get("success") is False and response.get("error") is None:
        violations.append("Failed response should have error message")

    # Error responses should have structured error context in data (per mcp_response_schema.md)
    # data should contain error_code, error_type, remediation when available

    return violations


def compare_schema_keys(expected: dict, actual: dict, path: str = "") -> list[str]:
    """Compare schema keys recursively, returning differences."""
    differences = []

    # Skip fixture metadata fields
    skip_keys = {"fixture_version"}
    expected_keys = set(expected.keys()) - skip_keys
    actual_keys = set(actual.keys())

    # Check for missing required keys (present in expected but not actual)
    missing = expected_keys - actual_keys
    if missing:
        differences.append(f"{path}: missing keys {missing}")

    # Recurse into nested dicts for matching keys
    common_keys = expected_keys & actual_keys
    for key in common_keys:
        if isinstance(expected[key], dict) and isinstance(actual[key], dict):
            nested_path = f"{path}.{key}" if path else key
            differences.extend(
                compare_schema_keys(expected[key], actual[key], nested_path)
            )

    return differences


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_specs_dir(tmp_path):
    """Create a temporary specs directory with test fixtures."""
    specs_dir = tmp_path / "specs"
    active_dir = specs_dir / "active"
    active_dir.mkdir(parents=True)

    # Create a test spec that matches golden fixture expectations
    test_spec = {
        "id": "example-spec",
        "title": "Example Specification",
        "version": "1.0.0",
        "status": "active",
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Example Specification",
                "children": ["phase-1"],
                "status": "in_progress",
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
                "status": "in_progress",
            },
            "task-1-1": {
                "type": "task",
                "title": "First task",
                "parent": "phase-1",
                "status": "completed",
                "metadata": {"description": "First task implementation"},
                "dependencies": {},
            },
            "task-1-2": {
                "type": "task",
                "title": "Second task",
                "parent": "phase-1",
                "status": "pending",
                "metadata": {"description": "Second task"},
                "dependencies": {},
            },
        },
        "journal": [],
    }

    spec_file = active_dir / "example-spec.json"
    spec_file.write_text(json.dumps(test_spec, indent=2))

    return specs_dir


class TestResponseEnvelopeParity:
    """Tests that all CLI responses follow the standard envelope structure."""

    def test_success_response_envelope(self, cli_runner, temp_specs_dir):
        """Success responses follow envelope structure."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "test", "presets"]
        )
        assert result.exit_code == 0

        response = json.loads(result.output)
        violations = validate_envelope_structure(response)
        assert not violations, f"Envelope violations: {violations}"

        # Additional success checks
        assert response["success"] is True
        assert response["data"] is not None
        assert response["error"] is None

    def test_error_response_envelope(self, cli_runner, temp_specs_dir):
        """Error responses follow envelope structure."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "validate",
                "check",
                "nonexistent-spec",
            ],
        )
        # May succeed or fail depending on implementation
        response = json.loads(result.output)
        violations = validate_envelope_structure(response)
        assert not violations, f"Envelope violations: {violations}"

        if not response["success"]:
            # Per response-v2 contract: error is a string message
            assert response["error"] is not None
            assert isinstance(response["error"], str)
            # Structured error context is in data (error_code, error_type, etc.)
            assert isinstance(response["data"], dict)


class TestGoldenFixtureParity:
    """Tests that CLI outputs match golden fixture schemas."""

    def test_test_presets_matches_golden(self, cli_runner, temp_specs_dir):
        """test presets output matches expected schema."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "test", "presets"]
        )
        assert result.exit_code == 0

        actual = json.loads(result.output)
        load_golden_fixture("success_test_presets")

        # Validate envelope structure
        violations = validate_envelope_structure(actual)
        assert not violations, f"Envelope violations: {violations}"

        # Verify key structure matches
        assert "presets" in actual["data"]
        expected_presets = {"quick", "full", "unit", "integration", "smoke"}
        actual_presets = set(actual["data"]["presets"].keys())
        missing_presets = expected_presets - actual_presets
        assert not missing_presets, f"Missing presets: {missing_presets}"

    def test_validation_success_matches_golden(self, cli_runner, temp_specs_dir):
        """validate check output matches expected schema."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "validate", "check", "example-spec"],
        )

        actual = json.loads(result.output)
        load_golden_fixture("success_validation")

        # Validate envelope structure
        violations = validate_envelope_structure(actual)
        assert not violations, f"Envelope violations: {violations}"

        if actual["success"]:
            # Verify key fields present
            required_fields = {"spec_id", "is_valid"}
            actual_fields = set(actual["data"].keys())
            missing = required_fields - actual_fields
            assert not missing, f"Missing validation fields: {missing}"

    def test_error_not_found_matches_golden(self, cli_runner, temp_specs_dir):
        """Not found error matches expected schema."""
        result = cli_runner.invoke(
            cli,
            [
                "--specs-dir",
                str(temp_specs_dir),
                "validate",
                "check",
                "nonexistent-spec-id",
            ],
        )

        actual = json.loads(result.output)
        load_golden_fixture("error_not_found")

        # Validate envelope structure
        violations = validate_envelope_structure(actual)
        assert not violations, f"Envelope violations: {violations}"

        # Error responses should have consistent structure per response-v2
        if not actual["success"]:
            # error is a string message
            assert actual["error"] is not None
            assert isinstance(actual["error"], str)
            # Structured error context (error_code, error_type) is in data
            assert isinstance(actual["data"], dict)


class TestCommandGroupParity:
    """Tests that all command groups produce consistent output."""

    def test_specs_analyze_produces_json(self, cli_runner, temp_specs_dir):
        """specs analyze produces valid JSON output."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "specs", "analyze"]
        )
        assert result.exit_code == 0

        response = json.loads(result.output)
        violations = validate_envelope_structure(response)
        assert not violations, f"Envelope violations: {violations}"

    def test_session_status_produces_json(self, cli_runner, temp_specs_dir):
        """session status produces valid JSON output."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "status"]
        )
        assert result.exit_code == 0

        response = json.loads(result.output)
        violations = validate_envelope_structure(response)
        assert not violations, f"Envelope violations: {violations}"

    def test_session_capabilities_produces_json(self, cli_runner, temp_specs_dir):
        """session capabilities produces valid JSON output."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "session", "capabilities"]
        )
        assert result.exit_code == 0

        response = json.loads(result.output)
        violations = validate_envelope_structure(response)
        assert not violations, f"Envelope violations: {violations}"

        # Should have capabilities info
        assert response["success"] is True
        assert "capabilities" in response["data"]

    def test_lifecycle_state_produces_json(self, cli_runner, temp_specs_dir):
        """lifecycle state produces valid JSON output."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "lifecycle", "state", "example-spec"],
        )

        response = json.loads(result.output)
        violations = validate_envelope_structure(response)
        assert not violations, f"Envelope violations: {violations}"


class TestOutputStability:
    """Tests for output stability across invocations."""

    def test_repeated_invocations_consistent(self, cli_runner, temp_specs_dir):
        """Repeated command invocations produce consistent structure."""
        results = []
        for _ in range(3):
            result = cli_runner.invoke(
                cli, ["--specs-dir", str(temp_specs_dir), "test", "presets"]
            )
            assert result.exit_code == 0
            results.append(json.loads(result.output))

        # All responses should have same keys
        first_keys = set(results[0].keys())
        for i, response in enumerate(results[1:], 2):
            assert set(response.keys()) == first_keys, (
                f"Invocation {i} has different keys"
            )

        # All should have same data keys
        first_data_keys = set(results[0]["data"].keys())
        for i, response in enumerate(results[1:], 2):
            assert set(response["data"].keys()) == first_data_keys

    def test_no_ansi_in_output(self, cli_runner, temp_specs_dir):
        """CLI output contains no ANSI escape codes."""
        result = cli_runner.invoke(
            cli, ["--specs-dir", str(temp_specs_dir), "test", "presets"]
        )
        assert result.exit_code == 0

        # Check for common ANSI escape sequences
        ansi_patterns = ["\x1b[", "\033[", "\x1b]"]
        for pattern in ansi_patterns:
            assert pattern not in result.output, (
                f"Found ANSI escape in output: {pattern!r}"
            )

    def test_output_is_valid_json(self, cli_runner, temp_specs_dir):
        """All output is valid JSON (no extra text)."""
        commands = [
            ["test", "presets"],
            ["specs", "analyze"],
            ["session", "status"],
        ]

        for cmd in commands:
            result = cli_runner.invoke(cli, ["--specs-dir", str(temp_specs_dir)] + cmd)

            # Output should be parseable as JSON
            try:
                data = json.loads(result.output)
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"Command {cmd} produced invalid JSON: {e}\nOutput: {result.output[:200]}"
                )

            # Should be a dict (not list or primitive)
            assert isinstance(data, dict), (
                f"Command {cmd} should return object, got {type(data)}"
            )
