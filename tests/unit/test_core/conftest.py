"""Shared fixtures for core unit tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_specs_dir():
    """Create a temporary specs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        specs_dir = (Path(tmpdir) / "specs").resolve()
        for folder in ("pending", "active", "completed", "archived"):
            (specs_dir / folder).mkdir(parents=True)
        yield specs_dir
