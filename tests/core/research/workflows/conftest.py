"""Shared fixtures for research workflow tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_config():
    """Create a base mock ResearchConfig with default_provider."""
    config = MagicMock()
    config.default_provider = "test-provider"
    return config


@pytest.fixture
def mock_memory():
    """Create a base mock ResearchMemory."""
    memory = MagicMock()
    return memory
