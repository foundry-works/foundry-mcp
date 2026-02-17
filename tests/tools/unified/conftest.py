"""Shared fixtures for unified tool dispatch tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_config():
    """Create a mock ServerConfig."""
    config = MagicMock()
    return config
