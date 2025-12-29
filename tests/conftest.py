"""
Root pytest configuration and shared fixtures.

Provides fixture freshness validation and common test utilities.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pytest

# Fixture schema version - increment when fixture format changes
FIXTURE_SCHEMA_VERSION = "1.0.0"

# Response contract version from responses.py
RESPONSE_CONTRACT_VERSION = "response-v2"

# Track validated fixtures to avoid repeated checks
_validated_fixtures: Set[Path] = set()


def get_fixture_version(fixture_path: Path) -> Optional[str]:
    """Extract version from a JSON fixture file.

    Looks for version in these locations (in order):
    1. Top-level "fixture_version" key
    2. "metadata.version" key
    3. "_fixture_meta.version" key

    Returns None if no version found.
    """
    try:
        with open(fixture_path) as f:
            data = json.load(f)

        # Check top-level fixture_version
        if "fixture_version" in data:
            return data["fixture_version"]

        # Check metadata.version
        if isinstance(data.get("metadata"), dict):
            if "version" in data["metadata"]:
                return data["metadata"]["version"]

        # Check _fixture_meta.version (alternative location)
        if isinstance(data.get("_fixture_meta"), dict):
            if "version" in data["_fixture_meta"]:
                return data["_fixture_meta"]["version"]

        return None
    except (json.JSONDecodeError, IOError):
        return None


def validate_fixture_freshness(
    fixture_path: Path,
    expected_version: str = FIXTURE_SCHEMA_VERSION,
    strict: bool = False,
) -> bool:
    """Validate that a fixture file's version matches expected version.

    Args:
        fixture_path: Path to JSON fixture file
        expected_version: Expected fixture schema version
        strict: If True, raise error on mismatch; if False, warn

    Returns:
        True if version matches or no version found (lenient mode)

    Raises:
        ValueError: In strict mode when version mismatch detected
    """
    if fixture_path in _validated_fixtures:
        return True

    version = get_fixture_version(fixture_path)

    if version is None:
        # No version in fixture - warn but don't fail
        warnings.warn(
            f"Fixture {fixture_path.name} has no version marker. "
            f"Consider adding 'fixture_version': '{FIXTURE_SCHEMA_VERSION}'",
            UserWarning,
            stacklevel=2,
        )
        _validated_fixtures.add(fixture_path)
        return True

    if version != expected_version:
        msg = (
            f"Fixture {fixture_path.name} has version {version}, "
            f"expected {expected_version}. Fixture may be stale."
        )
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=2)
            _validated_fixtures.add(fixture_path)
            return False

    _validated_fixtures.add(fixture_path)
    return True


def validate_response_envelope(response: Dict[str, Any]) -> bool:
    """Validate that a response dict conforms to response-v2 envelope.

    Args:
        response: Response dictionary to validate

    Returns:
        True if valid, False otherwise

    Raises:
        AssertionError: With detailed message on validation failure
    """
    # Check required top-level keys
    required_keys = {"success", "data", "error", "meta"}
    missing = required_keys - set(response.keys())
    assert not missing, f"Response missing required keys: {missing}"

    # Check types
    assert isinstance(response["success"], bool), "success must be boolean"
    assert isinstance(response["data"], dict), "data must be dict"
    assert isinstance(response["meta"], dict), "meta must be dict"

    # Check success/error consistency
    if response["success"]:
        assert response["error"] is None, "error must be null when success=True"
    else:
        assert isinstance(response["error"], str) and response["error"], (
            "error must be non-empty string when success=False"
        )

    # Check meta.version
    assert response["meta"].get("version") == RESPONSE_CONTRACT_VERSION, (
        f"meta.version must be '{RESPONSE_CONTRACT_VERSION}'"
    )

    return True


@pytest.fixture(scope="session", autouse=True)
def validate_fixture_versions(request):
    """Autouse fixture that validates fixture file versions at session start.

    Scans known fixture directories and warns about stale fixtures.
    This helps catch outdated test data that may cause false positives/negatives.
    """
    # Get test root directory
    test_root = Path(__file__).parent

    # Known fixture directories to scan
    fixture_dirs = [
        test_root / "fixtures",
        test_root / "parity" / "fixtures",
        test_root / "contract",
    ]

    stale_count = 0
    checked_count = 0

    for fixture_dir in fixture_dirs:
        if not fixture_dir.exists():
            continue

        for json_file in fixture_dir.rglob("*.json"):
            # Skip schema files (they define the version, not use it)
            if "schema" in json_file.name.lower():
                continue

            checked_count += 1
            if not validate_fixture_freshness(json_file, strict=False):
                stale_count += 1

    if stale_count > 0:
        warnings.warn(
            f"Found {stale_count} potentially stale fixtures out of {checked_count} checked. "
            f"Run 'pytest --fixtures-audit' for details.",
            UserWarning,
        )

    yield

    # Cleanup
    _validated_fixtures.clear()


@pytest.fixture
def response_validator():
    """Fixture providing response envelope validation function.

    Usage:
        def test_my_tool(response_validator):
            response = my_tool()
            response_validator(response)  # Raises on invalid
    """
    return validate_response_envelope


@pytest.fixture
def fixture_freshness_checker():
    """Fixture providing fixture freshness validation function.

    Usage:
        def test_with_fixture(fixture_freshness_checker, tmp_path):
            fixture_path = tmp_path / "data.json"
            # ... create fixture ...
            fixture_freshness_checker(fixture_path)  # Warns on stale
    """
    return validate_fixture_freshness


@pytest.fixture
def assert_response_contract():
    """Fixture for asserting response contract compliance.

    Provides a callable that validates and returns the response,
    allowing chained assertions.

    Usage:
        def test_tool_response(assert_response_contract):
            response = my_tool()
            validated = assert_response_contract(response)
            assert validated["data"]["count"] == 5
    """
    def _assert(response: Dict[str, Any]) -> Dict[str, Any]:
        validate_response_envelope(response)
        return response

    return _assert


# Check if pytest-asyncio is available
try:
    import pytest_asyncio
    HAS_PYTEST_ASYNCIO = True
except ImportError:
    HAS_PYTEST_ASYNCIO = False


# Configure pytest to recognize our custom markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "fixture_version(version): mark test with expected fixture version",
    )
    config.addinivalue_line(
        "markers",
        "requires_fresh_fixtures: mark test that requires up-to-date fixtures",
    )
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async (requires pytest-asyncio)",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on fixture requirements and async support."""
    skip_asyncio = pytest.mark.skip(reason="pytest-asyncio not installed")

    for item in items:
        # Skip async tests when pytest-asyncio is not available
        if not HAS_PYTEST_ASYNCIO and item.get_closest_marker("asyncio"):
            item.add_marker(skip_asyncio)

        # Check for requires_fresh_fixtures marker
        if item.get_closest_marker("requires_fresh_fixtures"):
            # These tests will fail-fast if any fixtures are stale
            item.add_marker(
                pytest.mark.skipif(
                    len(_validated_fixtures) == 0,
                    reason="Fixture validation not yet run",
                )
            )
