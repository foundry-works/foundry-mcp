"""
Shared pytest fixtures for parity tests.

Fixture scoping strategy:
- Module scope: Read-only fixtures that don't modify spec state
- Function scope: Mutation fixtures (update_status, add_task, etc.)
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from .harness.fixture_manager import FixtureManager


# ============================================================================
# Function-scoped fixtures (for mutation tests)
# ============================================================================

@pytest.fixture
def test_dir(tmp_path):
    """Create isolated test directory for mutation tests."""
    yield tmp_path


@pytest.fixture
def fixture_manager(test_dir):
    """Create fixture manager for mutation tests."""
    return FixtureManager(test_dir)


# ============================================================================
# Module-scoped read-only fixtures
# ============================================================================

@pytest.fixture(scope="module")
def simple_spec_dir(tmp_path_factory):
    """Module-scoped simple spec for read-only tests."""
    tmp_path = tmp_path_factory.mktemp("simple")
    fixture = FixtureManager(tmp_path)
    fixture.setup("simple_spec", status="active")
    fixture.setup("completed_spec", status="completed")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def deps_spec_dir(tmp_path_factory):
    """Module-scoped spec with dependencies."""
    tmp_path = tmp_path_factory.mktemp("deps")
    fixture = FixtureManager(tmp_path)
    fixture.setup("with_dependencies", status="active")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def foundry_adapter(simple_spec_dir):
    """Module-scoped foundry-mcp adapter."""
    from .harness.foundry_adapter import FoundryMcpAdapter
    return FoundryMcpAdapter(simple_spec_dir / "specs")


@pytest.fixture(scope="module")
def sdd_adapter(simple_spec_dir):
    """Module-scoped sdd-toolkit adapter."""
    from .harness.sdd_adapter import SddToolkitAdapter
    return SddToolkitAdapter(simple_spec_dir / "specs")


@pytest.fixture(scope="module")
def both_adapters(simple_spec_dir):
    """
    Module-scoped adapters pointing to the SAME directory.
    For read-only comparison tests.
    """
    from .harness.foundry_adapter import FoundryMcpAdapter
    from .harness.sdd_adapter import SddToolkitAdapter

    specs_dir = simple_spec_dir / "specs"
    return (
        FoundryMcpAdapter(specs_dir),
        SddToolkitAdapter(specs_dir)
    )


@pytest.fixture
def isolated_adapters(test_dir):
    """
    Function-scoped adapters with SEPARATE directories.
    For tests that modify state.
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


# ============================================================================
# Authoring Operations Fixtures (function-scoped for mutations)
# ============================================================================

@pytest.fixture
def authoring_spec_dir(fixture_manager):
    """Function-scoped authoring base spec for mutation tests."""
    fixture_manager.setup("authoring_base", status="active")
    return fixture_manager.specs_dir.parent


@pytest.fixture
def authoring_subtasks_dir(fixture_manager):
    """Function-scoped authoring spec with subtasks."""
    fixture_manager.setup("authoring_with_subtasks", status="active")
    return fixture_manager.specs_dir.parent


@pytest.fixture
def authoring_assumptions_dir(fixture_manager):
    """Function-scoped authoring spec with assumptions."""
    fixture_manager.setup("authoring_with_assumptions", status="active")
    return fixture_manager.specs_dir.parent


@pytest.fixture
def authoring_adapters(test_dir):
    """Function-scoped isolated adapters for authoring mutation tests."""
    from .harness.foundry_adapter import FoundryMcpAdapter
    from .harness.sdd_adapter import SddToolkitAdapter

    foundry_root = test_dir / "foundry"
    foundry_fixture = FixtureManager(foundry_root)
    foundry_fixture.setup("authoring_base", status="active")

    sdd_root = test_dir / "sdd"
    sdd_fixture = FixtureManager(sdd_root)
    sdd_fixture.setup("authoring_base", status="active")

    return (
        FoundryMcpAdapter(foundry_root / "specs"),
        SddToolkitAdapter(sdd_root / "specs")
    )


@pytest.fixture
def authoring_spec(test_dir):
    """Function-scoped authoring_base fixture."""
    fixture = FixtureManager(test_dir)
    fixture.setup("authoring_base", status="active")
    return fixture


@pytest.fixture
def authoring_with_subtasks(test_dir):
    """Function-scoped authoring_with_subtasks fixture."""
    fixture = FixtureManager(test_dir)
    fixture.setup("authoring_with_subtasks", status="active")
    return fixture


@pytest.fixture
def authoring_with_assumptions(test_dir):
    """Function-scoped authoring_with_assumptions fixture."""
    fixture = FixtureManager(test_dir)
    fixture.setup("authoring_with_assumptions", status="active")
    return fixture


# ============================================================================
# Analysis Operations Fixtures (module-scoped, read-only)
# ============================================================================

@pytest.fixture(scope="module")
def bottleneck_spec_dir(tmp_path_factory):
    """Module-scoped bottleneck analysis spec."""
    tmp_path = tmp_path_factory.mktemp("bottleneck")
    fixture = FixtureManager(tmp_path)
    fixture.setup("analysis_bottleneck", status="active")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def circular_deps_dir(tmp_path_factory):
    """Module-scoped circular dependency spec."""
    tmp_path = tmp_path_factory.mktemp("circular")
    fixture = FixtureManager(tmp_path)
    fixture.setup("analysis_circular_deps", status="active")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def patterns_spec_dir(tmp_path_factory):
    """Module-scoped patterns spec with file_path metadata."""
    tmp_path = tmp_path_factory.mktemp("patterns")
    fixture = FixtureManager(tmp_path)
    fixture.setup("analysis_patterns", status="active")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def analysis_adapters(bottleneck_spec_dir):
    """Module-scoped adapters for analysis read-only tests."""
    from .harness.foundry_adapter import FoundryMcpAdapter
    from .harness.sdd_adapter import SddToolkitAdapter

    specs_dir = bottleneck_spec_dir / "specs"
    return (
        FoundryMcpAdapter(specs_dir),
        SddToolkitAdapter(specs_dir)
    )


# ============================================================================
# Edge Case Fixtures (module-scoped, read-only)
# ============================================================================

@pytest.fixture(scope="module")
def deep_nesting_dir(tmp_path_factory):
    """Module-scoped deep nesting spec."""
    tmp_path = tmp_path_factory.mktemp("deep")
    fixture = FixtureManager(tmp_path)
    fixture.setup("edge_deep_nesting", status="active")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def large_spec_dir(tmp_path_factory):
    """Module-scoped large spec (100 tasks) for pagination tests."""
    tmp_path = tmp_path_factory.mktemp("large")
    fixture = FixtureManager(tmp_path)
    fixture.setup("edge_large_spec", status="active")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def empty_spec_dir(tmp_path_factory):
    """Module-scoped empty spec for edge cases."""
    tmp_path = tmp_path_factory.mktemp("empty")
    fixture = FixtureManager(tmp_path)
    fixture.setup("edge_empty_spec", status="active")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def all_blocked_dir(tmp_path_factory):
    """Module-scoped all-blocked spec."""
    tmp_path = tmp_path_factory.mktemp("blocked")
    fixture = FixtureManager(tmp_path)
    fixture.setup("edge_all_blocked", status="active")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def edge_adapters(large_spec_dir):
    """Module-scoped adapters for edge case tests."""
    from .harness.foundry_adapter import FoundryMcpAdapter
    from .harness.sdd_adapter import SddToolkitAdapter

    specs_dir = large_spec_dir / "specs"
    return (
        FoundryMcpAdapter(specs_dir),
        SddToolkitAdapter(specs_dir)
    )


# ============================================================================
# Review/Verification Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def review_spec_dir(tmp_path_factory):
    """Module-scoped review spec for read-only tests."""
    tmp_path = tmp_path_factory.mktemp("review")
    fixture = FixtureManager(tmp_path)
    fixture.setup("review_base", status="active")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def verification_results_dir(tmp_path_factory):
    """Module-scoped spec with pre-populated verification results."""
    tmp_path = tmp_path_factory.mktemp("verify")
    fixture = FixtureManager(tmp_path)
    fixture.setup("review_with_results", status="active")
    return fixture.specs_dir.parent


@pytest.fixture(scope="module")
def review_adapters(verification_results_dir):
    """Module-scoped adapters for read-only review tests."""
    from .harness.foundry_adapter import FoundryMcpAdapter
    from .harness.sdd_adapter import SddToolkitAdapter

    specs_dir = verification_results_dir / "specs"
    return (
        FoundryMcpAdapter(specs_dir),
        SddToolkitAdapter(specs_dir)
    )
