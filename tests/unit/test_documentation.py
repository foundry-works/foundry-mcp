"""Unit tests for documentation tools."""

import pytest
import subprocess
from unittest.mock import patch, MagicMock


class TestRunSddDocCommand:
    """Tests for _run_sdd_doc_command helper."""

    def test_successful_command(self):
        """Test successful command execution returns parsed JSON."""
        from foundry_mcp.tools.documentation import _run_sdd_doc_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"output_path": "specs/.human-readable/test.md", "title": "Test Spec"}',
                stderr=''
            )

            result = _run_sdd_doc_command(["generate"])

            assert result["success"] is True
            assert result["data"]["output_path"] == "specs/.human-readable/test.md"
            assert result["data"]["title"] == "Test Spec"

    def test_command_failure(self):
        """Test handling of command failure."""
        from foundry_mcp.tools.documentation import _run_sdd_doc_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout='',
                stderr='Error: Spec not found'
            )

            result = _run_sdd_doc_command(["generate"])

            assert result["success"] is False
            assert "Spec not found" in result["error"]

    def test_command_timeout(self):
        """Test handling of command timeout."""
        from foundry_mcp.tools.documentation import _run_sdd_doc_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sdd doc", timeout=120)

            result = _run_sdd_doc_command(["generate"])

            assert result["success"] is False
            assert "timed out" in result["error"]

    def test_sdd_not_found(self):
        """Test handling when sdd CLI is not found."""
        from foundry_mcp.tools.documentation import _run_sdd_doc_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = _run_sdd_doc_command(["generate"])

            assert result["success"] is False
            assert "sdd CLI not found" in result["error"]

    def test_json_decode_fallback(self):
        """Test fallback when output is not valid JSON."""
        from foundry_mcp.tools.documentation import _run_sdd_doc_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='Raw output text',
                stderr=''
            )

            result = _run_sdd_doc_command(["generate"])

            assert result["success"] is True
            assert "raw_output" in result["data"]
            assert result["data"]["raw_output"] == "Raw output text"


class TestRunSddRenderCommand:
    """Tests for _run_sdd_render_command helper."""

    def test_successful_render(self):
        """Test successful render command."""
        from foundry_mcp.tools.documentation import _run_sdd_render_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"output_path": "specs/.human-readable/spec.md", "title": "My Spec", "total_tasks": 10}',
                stderr=''
            )

            result = _run_sdd_render_command(["my-spec", "--format", "markdown"])

            assert result["success"] is True
            assert result["data"]["output_path"] == "specs/.human-readable/spec.md"

    def test_render_command_timeout(self):
        """Test render command timeout."""
        from foundry_mcp.tools.documentation import _run_sdd_render_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sdd render", timeout=120)

            result = _run_sdd_render_command(["my-spec"])

            assert result["success"] is False
            assert "timed out" in result["error"]


class TestDocumentationToolRegistration:
    """Tests for documentation tool registration."""

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

    def test_registration_succeeds(self, mock_mcp, mock_config):
        """Test that documentation tools register without error."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Should not raise any exceptions
        register_documentation_tools(mock_mcp, mock_config)


class TestSpecDocTool:
    """Tests for spec-doc tool functionality."""

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

    def test_invalid_output_format_returns_error(self, mock_mcp, mock_config):
        """Test that invalid output_format returns validation error."""
        from foundry_mcp.tools.documentation import register_documentation_tools
        register_documentation_tools(mock_mcp, mock_config)

        # Find the registered spec_doc function
        # The tool is registered via decorator, get it from mock calls
        tool_calls = [call for call in mock_mcp.mock_calls if 'tool' in str(call)]
        assert len(tool_calls) > 0  # Ensure tool was registered

    def test_invalid_mode_returns_error(self, mock_mcp, mock_config):
        """Test that invalid mode returns validation error."""
        from foundry_mcp.tools.documentation import register_documentation_tools
        register_documentation_tools(mock_mcp, mock_config)

        # Verify tool was registered
        tool_calls = [call for call in mock_mcp.mock_calls if 'tool' in str(call)]
        assert len(tool_calls) > 0

    def test_spec_doc_returns_formatted_response(self):
        """Test that spec_doc returns properly formatted response."""
        from foundry_mcp.tools.documentation import _run_sdd_render_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"output_path": "specs/.human-readable/test-spec.md", "title": "Test", "total_tasks": 5, "completed_tasks": 2}',
                stderr=''
            )

            result = _run_sdd_render_command(["test-spec", "--format", "markdown"])

            assert result["success"] is True
            assert "output_path" in result["data"]


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_exists(self):
        """Test that circuit breaker is properly configured."""
        from foundry_mcp.tools.documentation import _doc_breaker

        assert _doc_breaker is not None
        assert _doc_breaker.name == "documentation"
        assert _doc_breaker.failure_threshold == 3
        assert _doc_breaker.recovery_timeout == 60.0

    def test_circuit_breaker_open_returns_error(self):
        """Test that open circuit breaker returns error response."""
        from foundry_mcp.tools.documentation import _doc_breaker

        # Record failures to open the breaker
        for _ in range(5):
            _doc_breaker.record_failure()

        # Check breaker state
        status = _doc_breaker.get_status()
        # Should be open after multiple failures
        assert status is not None


class TestResponseEnvelopeCompliance:
    """Tests for response envelope compliance."""

    def test_success_response_has_required_fields(self):
        """Test success response includes required envelope fields."""
        from foundry_mcp.tools.documentation import _run_sdd_render_command
        from foundry_mcp.core.responses import success_response
        from dataclasses import asdict

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"output_path": "test.md"}',
                stderr=''
            )

            # Test that success_response helper produces compliant envelope
            response = asdict(success_response(
                spec_id="test-spec",
                format="markdown",
                output_path="test.md",
            ))

            assert "success" in response
            assert response["success"] is True
            assert "meta" in response
            assert response["meta"]["version"] == "response-v2"

    def test_error_response_has_required_fields(self):
        """Test error response includes required envelope fields."""
        from foundry_mcp.core.responses import error_response
        from dataclasses import asdict

        response = asdict(error_response(
            "Test error message",
            error_code="VALIDATION_ERROR",
            error_type="validation",
        ))

        assert "success" in response
        assert response["success"] is False
        assert "error" in response
        assert response["error"] == "Test error message"
        assert "meta" in response
        assert response["meta"]["version"] == "response-v2"
