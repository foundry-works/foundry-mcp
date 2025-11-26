"""
Shared pytest fixtures for parity tests.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from .harness.fixture_manager import FixtureManager


@pytest.fixture
def test_dir(tmp_path):
    """Create isolated test directory."""
    yield tmp_path


@pytest.fixture
def fixture_manager(test_dir):
    """Create fixture manager for test."""
    return FixtureManager(test_dir)


@pytest.fixture
def simple_spec_dir(fixture_manager):
    """Setup core spec fixtures (active + completed) and return project root."""
    fixture_manager.setup("simple_spec", status="active")
    fixture_manager.setup("completed_spec", status="completed")
    return fixture_manager.specs_dir.parent


@pytest.fixture
def deps_spec_dir(fixture_manager):
    """Setup spec with dependencies and return project root."""
    fixture_manager.setup("with_dependencies", status="active")
    return fixture_manager.specs_dir.parent


@pytest.fixture
def foundry_adapter(simple_spec_dir):
    """Create foundry-mcp adapter with simple spec."""
    from .harness.foundry_adapter import FoundryMcpAdapter
    return FoundryMcpAdapter(simple_spec_dir / "specs")


@pytest.fixture
def sdd_adapter(simple_spec_dir):
    """Create sdd-toolkit adapter with simple spec."""
    from .harness.sdd_adapter import SddToolkitAdapter
    return SddToolkitAdapter(simple_spec_dir / "specs")


@pytest.fixture
def both_adapters(simple_spec_dir):
    """
    Provide both adapters pointing to the SAME test directory.

    This is for read-only comparison tests where both systems
    should see identical state.
    """
    from .harness.foundry_adapter import FoundryMcpAdapter
    from .harness.sdd_adapter import SddToolkitAdapter

    specs_dir = simple_spec_dir / "specs"
    return (
        FoundryMcpAdapter(specs_dir),
        SddToolkitAdapter(specs_dir)
    )


@pytest.fixture
def isolated_adapters(test_dir, fixture_manager):
    """
    Provide both adapters with SEPARATE test directories.

    Use this for tests that modify state (update_status, etc.)
    where each system needs its own copy of the spec.
    """
    from .harness.foundry_adapter import FoundryMcpAdapter
    from .harness.sdd_adapter import SddToolkitAdapter

    # Setup foundry copy
    foundry_root = test_dir / "foundry"
    foundry_fixture = FixtureManager(foundry_root)
    foundry_fixture.setup("simple_spec", status="active")

    # Setup sdd copy
    sdd_root = test_dir / "sdd"
    sdd_fixture = FixtureManager(sdd_root)
    sdd_fixture.setup("simple_spec", status="active")

    return (
        FoundryMcpAdapter(foundry_root / "specs"),
        SddToolkitAdapter(sdd_root / "specs")
    )
