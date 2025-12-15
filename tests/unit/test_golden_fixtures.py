"""Golden fixture comparison tests for CLI output stability.

This module provides a diff harness to compare CLI output against golden fixtures,
ensuring output schema stability across releases.

Usage:
    pytest tests/unit/test_golden_fixtures.py -v
    pytest tests/unit/test_golden_fixtures.py --regenerate-fixtures  # Update fixtures
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from foundry_mcp.cli.main import cli

GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "golden"


def pytest_addoption(parser):
    """Add --regenerate-fixtures option."""
    parser.addoption(
        "--regenerate-fixtures",
        action="store_true",
        default=False,
        help="Regenerate golden fixtures from current CLI output",
    )


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_specs_dir(tmp_path):
    """Create a temporary specs directory with test data."""
    specs_dir = tmp_path / "specs"
    active_dir = specs_dir / "active"
    active_dir.mkdir(parents=True)

    # Create a minimal test spec
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
                "children": ["task-1-1"],
                "status": "in_progress",
            },
            "task-1-1": {
                "type": "task",
                "title": "First Task",
                "parent": "phase-1",
                "status": "completed",
                "metadata": {},
                "dependencies": {},
            },
        },
        "journal": [],
    }

    spec_file = active_dir / "example-spec.json"
    spec_file.write_text(json.dumps(test_spec, indent=2))

    return specs_dir


def normalize_response(data: dict) -> dict:
    """Normalize response for comparison (remove variable fields)."""
    normalized = data.copy()

    # Remove telemetry timing (varies per run)
    if "data" in normalized and isinstance(normalized["data"], dict):
        if "telemetry" in normalized["data"]:
            normalized["data"]["telemetry"] = {"duration_ms": "<normalized>"}

    # Remove timestamps
    if "meta" in normalized and isinstance(normalized["meta"], dict):
        if "timestamp" in normalized["meta"]:
            normalized["meta"]["timestamp"] = "<normalized>"

    return normalized


def compare_schema_structure(actual: dict, expected: dict, path: str = "") -> list[str]:
    """Compare schema structure between actual and expected responses.

    Returns list of differences found.
    """
    differences = []

    # Check top-level keys match
    actual_keys = set(actual.keys()) - {"fixture_version"}
    expected_keys = set(expected.keys()) - {"fixture_version"}

    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys

    if missing:
        differences.append(f"{path}: Missing keys: {missing}")
    if extra:
        differences.append(f"{path}: Extra keys: {extra}")

    # Check value types match for common keys
    for key in actual_keys & expected_keys:
        actual_val = actual[key]
        expected_val = expected[key]
        key_path = f"{path}.{key}" if path else key

        if type(actual_val) != type(expected_val):
            differences.append(
                f"{key_path}: Type mismatch - "
                f"got {type(actual_val).__name__}, expected {type(expected_val).__name__}"
            )
        elif isinstance(actual_val, dict) and isinstance(expected_val, dict):
            differences.extend(compare_schema_structure(actual_val, expected_val, key_path))

    return differences


class TestGoldenFixtures:
    """Tests comparing CLI output against golden fixtures."""

    def test_response_envelope_structure(self, cli_runner, temp_specs_dir):
        """CLI responses follow standard envelope structure."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "test", "presets"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Verify envelope structure
        assert "success" in data, "Missing 'success' field"
        assert "data" in data, "Missing 'data' field"
        assert "error" in data, "Missing 'error' field"

    def test_success_response_schema(self, cli_runner, temp_specs_dir):
        """Success responses match expected schema."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "test", "presets"],
        )

        assert result.exit_code == 0
        actual = json.loads(result.output)

        # Load golden fixture
        golden_path = GOLDEN_DIR / "success_test_presets.json"
        if golden_path.exists():
            expected = json.loads(golden_path.read_text())

            # Compare key presence (not exact values which may vary)
            assert actual["success"] == expected["success"]
            assert "presets" in actual["data"]
            for preset_name in ["quick", "full", "unit", "integration", "smoke"]:
                assert preset_name in actual["data"]["presets"], f"Missing preset: {preset_name}"

    def test_error_response_schema(self, cli_runner, temp_specs_dir):
        """Error responses match expected schema."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "validate", "check", "nonexistent-spec"],
        )

        # Should return error (exit code 1)
        assert result.exit_code == 1
        actual = json.loads(result.output)

        # Verify error structure (response-v2 format)
        assert actual["success"] is False
        assert actual["error"] is not None  # error is now a flat string message
        # Error code and details are in the data object
        assert "error_code" in actual["data"]
        assert "NOT_FOUND" in actual["data"]["error_code"]


class TestSchemaStability:
    """Tests ensuring schema stability across versions."""

    def test_test_presets_schema_stable(self, cli_runner, temp_specs_dir):
        """test presets output schema is stable."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "test", "presets"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Verify expected fields in data
        assert data["success"] is True
        assert "presets" in data["data"]
        assert isinstance(data["data"]["presets"], dict)

    def test_validation_schema_stable(self, cli_runner, temp_specs_dir):
        """validate output schema is stable."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "validate", "check", "example-spec"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Verify expected fields in data
        assert data["success"] is True
        assert "is_valid" in data["data"]
        assert "error_count" in data["data"]
        assert "warning_count" in data["data"]

    def test_dev_check_schema_stable(self, cli_runner, temp_specs_dir):
        """dev check output schema is stable."""
        result = cli_runner.invoke(
            cli,
            ["--specs-dir", str(temp_specs_dir), "dev", "check"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)

        # Verify expected fields in data
        assert data["success"] is True
        assert "tools" in data["data"]
        assert "all_required_available" in data["data"]
