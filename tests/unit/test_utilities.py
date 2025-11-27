"""Unit tests for utility tools."""

import pytest
from unittest.mock import patch, MagicMock


class TestSddCacheManage:
    """Tests for sdd-cache-manage tool."""

    @pytest.fixture
    def mock_mcp(self):
        """Create a mock FastMCP instance."""
        mock = MagicMock()
        mock.tool = MagicMock(return_value=lambda f: f)
        return mock

    @pytest.fixture
    def mock_config(self):
        """Create a mock server config."""
        config = MagicMock()
        config.specs_dir = None
        return config

    @pytest.fixture
    def registered_tools(self, mock_mcp, mock_config):
        """Register utility tools and return the sdd_cache_manage function."""
        from foundry_mcp.tools.utilities import register_utility_tools
        register_utility_tools(mock_mcp, mock_config)

        # Extract the registered function from the mock calls
        # The canonical_tool decorator is called, we need to find the function
        return {}

    def test_cache_info_success(self):
        """Test cache info action returns cache statistics."""
        from foundry_mcp.tools.utilities import _run_sdd_command

        with patch('foundry_mcp.tools.utilities.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"total_entries": 5, "active_entries": 3}',
                stderr=''
            )

            result = _run_sdd_command(["cache", "info"])

            assert result["success"] is True
            assert result["data"]["total_entries"] == 5
            assert result["data"]["active_entries"] == 3

    def test_cache_clear_success(self):
        """Test cache clear action returns deleted count."""
        from foundry_mcp.tools.utilities import _run_sdd_command

        with patch('foundry_mcp.tools.utilities.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"entries_deleted": 2}',
                stderr=''
            )

            result = _run_sdd_command(["cache", "clear"])

            assert result["success"] is True
            assert result["data"]["entries_deleted"] == 2

    def test_cache_clear_with_filters(self):
        """Test cache clear with spec_id and review_type filters."""
        from foundry_mcp.tools.utilities import _run_sdd_command

        with patch('foundry_mcp.tools.utilities.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"entries_deleted": 1, "filters": {"spec_id": "test-spec"}}',
                stderr=''
            )

            result = _run_sdd_command(["cache", "clear", "--spec-id", "test-spec"])

            assert result["success"] is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "--spec-id" in call_args
            assert "test-spec" in call_args

    def test_command_timeout(self):
        """Test handling of command timeout."""
        from foundry_mcp.tools.utilities import _run_sdd_command
        import subprocess

        with patch('foundry_mcp.tools.utilities.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sdd", timeout=30)

            result = _run_sdd_command(["cache", "info"])

            assert result["success"] is False
            assert "timed out" in result["error"]

    def test_sdd_not_found(self):
        """Test handling when sdd CLI is not found."""
        from foundry_mcp.tools.utilities import _run_sdd_command

        with patch('foundry_mcp.tools.utilities.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = _run_sdd_command(["cache", "info"])

            assert result["success"] is False
            assert "not found" in result["error"]

    def test_command_failure(self):
        """Test handling of command failure."""
        from foundry_mcp.tools.utilities import _run_sdd_command

        with patch('foundry_mcp.tools.utilities.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout='',
                stderr='Cache is disabled'
            )

            result = _run_sdd_command(["cache", "info"])

            assert result["success"] is False
            assert "Cache is disabled" in result["error"]


class TestUtilityToolRegistration:
    """Tests for utility tool registration."""

    def test_all_utility_tools_registered(self):
        """Test that all utility tools are properly registered."""
        from foundry_mcp.tools.utilities import register_utility_tools
        from unittest.mock import MagicMock

        mock_mcp = MagicMock()
        mock_config = MagicMock()
        mock_config.specs_dir = None

        # Store registered tool names
        registered_tools = []

        def capture_tool(*args, **kwargs):
            canonical_name = kwargs.get('canonical_name', args[1] if len(args) > 1 else None)
            def decorator(func):
                if canonical_name:
                    registered_tools.append(canonical_name)
                return func
            return decorator

        # Patch the canonical_tool decorator
        with patch('foundry_mcp.tools.utilities.canonical_tool', side_effect=capture_tool):
            with patch('foundry_mcp.tools.utilities.mcp_tool', return_value=lambda f: f):
                register_utility_tools(mock_mcp, mock_config)

        assert "sdd-cache-manage" in registered_tools


class TestSpecSchemaExport:
    """Tests for spec-schema-export tool."""

    def test_schema_export_success(self):
        """Test schema export returns full JSON schema."""
        from foundry_mcp.tools.utilities import _run_sdd_command

        with patch('foundry_mcp.tools.utilities.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"$schema": "http://json-schema.org/draft-07/schema#", "title": "SDD Spec Schema"}',
                stderr=''
            )

            result = _run_sdd_command(["schema"])

            assert result["success"] is True
            assert result["data"]["$schema"] == "http://json-schema.org/draft-07/schema#"
            assert result["data"]["title"] == "SDD Spec Schema"

    def test_schema_export_command_called_correctly(self):
        """Test that schema command is called with correct arguments."""
        from foundry_mcp.tools.utilities import _run_sdd_command

        with patch('foundry_mcp.tools.utilities.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"$schema": "test"}',
                stderr=''
            )

            _run_sdd_command(["schema"])

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "sdd" in call_args
            assert "schema" in call_args
            assert "--json" in call_args


class TestCircuitBreaker:
    """Tests for circuit breaker behavior in utility tools."""

    def test_circuit_breaker_exists(self):
        """Test that circuit breaker is configured for utilities."""
        from foundry_mcp.tools.utilities import _utility_breaker

        assert _utility_breaker is not None
        assert _utility_breaker.name == "utilities"
        assert _utility_breaker.failure_threshold == 5
        assert _utility_breaker.recovery_timeout == 30.0
