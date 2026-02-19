"""Shared fixtures for CLI command tests."""

import json

import pytest
from click.testing import CliRunner


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def temp_specs_dir(tmp_path):
    """Create a temporary specs directory with a resolvable spec."""
    specs_dir = tmp_path / "specs"
    active_dir = specs_dir / "active"
    active_dir.mkdir(parents=True)

    test_spec = {
        "id": "test-spec-001",
        "title": "Test Specification",
        "version": "1.0.0",
        "status": "active",
        "hierarchy": {
            "spec-root": {
                "type": "root",
                "title": "Test",
                "children": [],
                "status": "in_progress",
            }
        },
        "journal": [],
    }
    spec_file = active_dir / "test-spec-001.json"
    spec_file.write_text(json.dumps(test_spec, indent=2))

    return specs_dir
