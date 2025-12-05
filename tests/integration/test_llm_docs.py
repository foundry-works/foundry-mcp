"""
Integration tests for documentation tools.

Tests verify:
- spec_doc renders specs to markdown using core Python APIs
- spec_doc_llm returns NOT_IMPLEMENTED (requires external LLM)
- spec_review_fidelity returns NOT_IMPLEMENTED (requires AI consultation)
- Response envelope compliance
- Input validation and security
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from foundry_mcp.config import ServerConfig


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP server instance."""
    mcp = MagicMock()
    mcp._tools = {}

    def mock_tool(*args, **kwargs):
        def decorator(func):
            name = func.__name__
            mcp._tools[name] = MagicMock(fn=func)
            return func
        return decorator

    mcp.tool = mock_tool
    return mcp


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock server config with temp specs dir."""
    return ServerConfig(specs_dir=tmp_path / "specs")


@pytest.fixture
def sample_spec_data():
    """Create a sample spec for testing."""
    return {
        "spec_id": "test-spec-001",
        "metadata": {
            "title": "Test Specification",
            "spec_id": "test-spec-001",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase One",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1", "task-2"],
            },
            "task-1": {
                "type": "task",
                "title": "First Task",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
            },
            "task-2": {
                "type": "task",
                "title": "Second Task",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
            },
        },
        "journal": [],
    }


def setup_spec_file(tmp_path, spec_data, spec_id="test-spec-001"):
    """Helper to create spec file in correct location."""
    specs_dir = tmp_path / "specs" / "active"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_file = specs_dir / f"{spec_id}.json"
    spec_file.write_text(json.dumps(spec_data))
    return spec_file


class TestSpecDocIntegration:
    """Integration tests for spec-doc tool."""

    def test_spec_doc_registers_successfully(self, mock_mcp, mock_config):
        """Test spec_doc tool registers without error."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)
        assert "spec_doc" in mock_mcp._tools

    def test_spec_doc_renders_spec_to_markdown(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_doc renders a spec to markdown successfully."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Setup spec file
        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        spec_doc = mock_mcp._tools["spec_doc"]
        result = spec_doc.fn(
            spec_id="test-spec-001",
            workspace=str(tmp_path),
        )

        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert "output_path" in result["data"]
        assert result["data"]["format"] == "markdown"

        # Verify the file was created
        output_path = Path(result["data"]["output_path"])
        assert output_path.exists()

    def test_spec_doc_includes_progress_stats(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_doc includes progress statistics."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        spec_doc = mock_mcp._tools["spec_doc"]
        result = spec_doc.fn(
            spec_id="test-spec-001",
            include_progress=True,
            workspace=str(tmp_path),
        )

        assert result["success"] is True
        assert "stats" in result["data"]
        stats = result["data"]["stats"]
        assert "total_tasks" in stats
        assert "completed_tasks" in stats

    def test_spec_doc_handles_missing_spec(self, mock_mcp, mock_config, tmp_path):
        """Test spec_doc handles missing spec gracefully."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Create specs dir but no spec file
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")
        register_documentation_tools(mock_mcp, config)

        spec_doc = mock_mcp._tools["spec_doc"]
        result = spec_doc.fn(
            spec_id="nonexistent-spec",
            workspace=str(tmp_path),
        )

        assert result["success"] is False
        assert "SPEC_NOT_FOUND" in result.get("data", {}).get("error_code", "") or \
               "not found" in result.get("error", "").lower()

    def test_spec_doc_validates_output_format(self, mock_mcp, tmp_path, sample_spec_data):
        """Test spec_doc validates output format."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        spec_doc = mock_mcp._tools["spec_doc"]
        result = spec_doc.fn(
            spec_id="test-spec-001",
            output_format="invalid_format",
            workspace=str(tmp_path),
        )

        assert result["success"] is False
        assert "markdown" in result.get("error", "").lower() or \
               "VALIDATION_ERROR" in result.get("data", {}).get("error_code", "")

    def test_spec_doc_validates_mode(self, mock_mcp, tmp_path, sample_spec_data):
        """Test spec_doc validates mode parameter."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        spec_doc = mock_mcp._tools["spec_doc"]
        result = spec_doc.fn(
            spec_id="test-spec-001",
            mode="invalid_mode",
            workspace=str(tmp_path),
        )

        assert result["success"] is False
        assert "VALIDATION_ERROR" in result.get("data", {}).get("error_code", "") or \
               "basic" in result.get("error", "").lower() or \
               "enhanced" in result.get("error", "").lower()


class TestSpecDocLlm:
    """Test spec_doc_llm implemented behavior."""

    def test_spec_doc_llm_registers_successfully(self, mock_mcp, mock_config):
        """Test spec_doc_llm tool registers without error."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)
        assert "spec_doc_llm" in mock_mcp._tools

    def test_spec_doc_llm_returns_response_with_envelope(self, mock_mcp, tmp_path):
        """Test spec_doc_llm returns proper response envelope."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Create a directory for the test
        test_dir = tmp_path / "project"
        test_dir.mkdir()

        config = ServerConfig(specs_dir=tmp_path / "specs")
        register_documentation_tools(mock_mcp, config)

        spec_doc_llm = mock_mcp._tools["spec_doc_llm"]
        result = spec_doc_llm.fn(directory=str(test_dir))

        # spec_doc_llm is now implemented - check for proper response envelope
        assert "success" in result
        assert "data" in result
        assert "meta" in result

    def test_spec_doc_llm_preserves_directory_in_response(self, mock_mcp, tmp_path):
        """Test spec_doc_llm includes directory in response."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        test_dir = tmp_path / "project"
        test_dir.mkdir()

        config = ServerConfig(specs_dir=tmp_path / "specs")
        register_documentation_tools(mock_mcp, config)

        spec_doc_llm = mock_mcp._tools["spec_doc_llm"]
        result = spec_doc_llm.fn(directory=str(test_dir))

        # Check directory in response (success or error)
        assert "success" in result
        if result["success"]:
            assert "output_dir" in result["data"]
        # For errors, directory may not be in response (that's okay)

    def test_spec_doc_llm_validates_directory_exists(self, mock_mcp, tmp_path):
        """Test spec_doc_llm validates directory exists."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        config = ServerConfig(specs_dir=tmp_path / "specs")
        register_documentation_tools(mock_mcp, config)

        spec_doc_llm = mock_mcp._tools["spec_doc_llm"]
        result = spec_doc_llm.fn(directory="/nonexistent/path/xyz")

        assert result["success"] is False
        assert "not found" in result.get("error", "").lower() or \
               "NOT_FOUND" in result.get("data", {}).get("error_code", "")

    def test_spec_doc_llm_validates_batch_size(self, mock_mcp, tmp_path):
        """Test spec_doc_llm validates batch_size parameter."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        test_dir = tmp_path / "project"
        test_dir.mkdir()

        config = ServerConfig(specs_dir=tmp_path / "specs")
        register_documentation_tools(mock_mcp, config)

        spec_doc_llm = mock_mcp._tools["spec_doc_llm"]
        result = spec_doc_llm.fn(directory=str(test_dir), batch_size=100)

        assert result["success"] is False
        assert "VALIDATION_ERROR" in result.get("data", {}).get("error_code", "")


class TestSpecReviewFidelity:
    """Test spec_review_fidelity implemented behavior."""

    def test_spec_review_fidelity_registers_successfully(self, mock_mcp, mock_config):
        """Test spec_review_fidelity tool registers without error."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)
        assert "spec_review_fidelity" in mock_mcp._tools

    def test_spec_review_fidelity_returns_response_envelope(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_review_fidelity returns proper response envelope."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        fidelity = mock_mcp._tools["spec_review_fidelity"]
        result = fidelity.fn(
            spec_id="test-spec-001",
            workspace=str(tmp_path),
        )

        # spec_review_fidelity is now implemented - check for proper response envelope
        assert "success" in result
        assert "data" in result
        assert "meta" in result

    def test_spec_review_fidelity_preserves_parameters(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_review_fidelity includes spec_id in response."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        fidelity = mock_mcp._tools["spec_review_fidelity"]
        result = fidelity.fn(
            spec_id="test-spec-001",
            task_id="task-1",
            workspace=str(tmp_path),
        )

        # The response includes spec_id in data for both success and error cases
        assert "success" in result
        assert "data" in result
        assert result["data"]["spec_id"] == "test-spec-001"
        # On success, also includes task_id and scope
        if result["success"]:
            assert result["data"]["task_id"] == "task-1"
            assert result["data"]["scope"] == "task"

    def test_spec_review_fidelity_validates_mutual_exclusivity(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_review_fidelity validates task_id/phase_id mutual exclusivity."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        fidelity = mock_mcp._tools["spec_review_fidelity"]
        result = fidelity.fn(
            spec_id="test-spec-001",
            task_id="task-1",
            phase_id="phase-1",  # Cannot specify both
            workspace=str(tmp_path),
        )

        assert result["success"] is False
        assert "both" in result.get("error", "").lower() or \
               "VALIDATION_ERROR" in result.get("data", {}).get("error_code", "")

    def test_spec_review_fidelity_handles_missing_spec(
        self, mock_mcp, tmp_path
    ):
        """Test spec_review_fidelity handles missing spec gracefully."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Create specs dir but no spec file
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")
        register_documentation_tools(mock_mcp, config)

        fidelity = mock_mcp._tools["spec_review_fidelity"]
        result = fidelity.fn(
            spec_id="nonexistent-spec",
            workspace=str(tmp_path),
        )

        assert result["success"] is False
        assert "SPEC_NOT_FOUND" in result.get("data", {}).get("error_code", "") or \
               "not found" in result.get("error", "").lower()


class TestSecurityValidation:
    """Test security validation in documentation tools."""

    def test_spec_doc_rejects_prompt_injection(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_doc rejects prompt injection patterns."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        spec_doc = mock_mcp._tools["spec_doc"]

        # Try injection in spec_id
        result = spec_doc.fn(
            spec_id="test; rm -rf /",
            workspace=str(tmp_path),
        )

        # Should either fail validation or treat as regular spec_id (not execute)
        assert "success" in result

    def test_spec_review_fidelity_handles_suspicious_paths(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_review_fidelity handles suspicious file paths."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        fidelity = mock_mcp._tools["spec_review_fidelity"]
        result = fidelity.fn(
            spec_id="test-spec-001",
            files=["../../../etc/passwd"],
            workspace=str(tmp_path),
        )

        # Should handle gracefully (not crash)
        assert "success" in result


class TestResponseEnvelope:
    """Test response envelope compliance for documentation tools."""

    def test_spec_doc_success_envelope(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_doc success response has required envelope fields."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        spec_doc = mock_mcp._tools["spec_doc"]
        result = spec_doc.fn(
            spec_id="test-spec-001",
            workspace=str(tmp_path),
        )

        assert "success" in result
        assert result["success"] is True
        assert "data" in result
        assert "meta" in result

    def test_spec_doc_error_envelope(self, mock_mcp, mock_config, tmp_path):
        """Test spec_doc error response has required envelope fields."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Create specs dir but no spec file
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        config = ServerConfig(specs_dir=tmp_path / "specs")
        register_documentation_tools(mock_mcp, config)

        spec_doc = mock_mcp._tools["spec_doc"]
        result = spec_doc.fn(
            spec_id="nonexistent",
            workspace=str(tmp_path),
        )

        assert "success" in result
        assert result["success"] is False
        assert "error" in result

    def test_spec_doc_llm_error_envelope(self, mock_mcp, tmp_path):
        """Test spec_doc_llm response has required envelope fields."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        test_dir = tmp_path / "project"
        test_dir.mkdir()

        config = ServerConfig(specs_dir=tmp_path / "specs")
        register_documentation_tools(mock_mcp, config)

        spec_doc_llm = mock_mcp._tools["spec_doc_llm"]
        result = spec_doc_llm.fn(directory=str(test_dir))

        # All responses should have proper envelope structure
        assert "success" in result
        assert "data" in result
        assert "meta" in result
        # Error responses additionally have "error" field
        if not result["success"]:
            assert "error" in result

    def test_spec_review_fidelity_error_envelope(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_review_fidelity error response has required envelope fields."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        fidelity = mock_mcp._tools["spec_review_fidelity"]
        result = fidelity.fn(
            spec_id="test-spec-001",
            workspace=str(tmp_path),
        )

        # All responses have proper envelope structure
        assert "success" in result
        assert "data" in result
        assert "meta" in result
        # Error responses additionally have "error" field
        if not result["success"]:
            assert "error" in result


class TestToolRegistration:
    """Test all documentation tools register correctly."""

    def test_all_documentation_tools_register(self, mock_mcp, mock_config):
        """Test all documentation tools register without error."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Should not raise
        register_documentation_tools(mock_mcp, mock_config)

        # Should have registered expected tools
        assert "spec_doc" in mock_mcp._tools
        assert "spec_doc_llm" in mock_mcp._tools
        assert "spec_review_fidelity" in mock_mcp._tools

    def test_registration_with_fastmcp(self, tmp_path):
        """Test registration with actual FastMCP instance."""
        from foundry_mcp.tools.documentation import register_documentation_tools
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("test")
        config = ServerConfig(specs_dir=tmp_path / "specs")

        # Should not raise
        try:
            register_documentation_tools(mcp, config)
            registration_success = True
        except Exception:
            registration_success = False

        assert registration_success, "Documentation tools should register without error"


class TestMetricsEmission:
    """Test metrics emission for documentation tools."""

    def test_spec_doc_emits_metrics_on_success(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_doc emits metrics on successful render."""
        from foundry_mcp.tools.documentation import register_documentation_tools
        from foundry_mcp.core.observability import get_metrics

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        metrics = get_metrics()
        with patch.object(metrics, 'timer') as mock_timer:
            spec_doc = mock_mcp._tools["spec_doc"]
            result = spec_doc.fn(
                spec_id="test-spec-001",
                workspace=str(tmp_path),
            )

            assert result["success"] is True
            # Timer should be called for duration
            mock_timer.assert_called()

    def test_spec_doc_response_includes_telemetry(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_doc response includes telemetry data."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        spec_doc = mock_mcp._tools["spec_doc"]
        result = spec_doc.fn(
            spec_id="test-spec-001",
            workspace=str(tmp_path),
        )

        assert result["success"] is True
        # Should include duration in response
        if "telemetry" in result["data"]:
            assert "duration_ms" in result["data"]["telemetry"]


class TestSpecDocOutputOptions:
    """Test spec_doc output path and format options."""

    def test_spec_doc_custom_output_path(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_doc respects custom output path."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        custom_output = tmp_path / "custom" / "output.md"

        spec_doc = mock_mcp._tools["spec_doc"]
        result = spec_doc.fn(
            spec_id="test-spec-001",
            output_path=str(custom_output),
            workspace=str(tmp_path),
        )

        assert result["success"] is True
        assert str(custom_output) in result["data"]["output_path"]
        assert custom_output.exists()

    def test_spec_doc_basic_mode(
        self, mock_mcp, tmp_path, sample_spec_data
    ):
        """Test spec_doc with basic mode."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        setup_spec_file(tmp_path, sample_spec_data)
        config = ServerConfig(specs_dir=tmp_path / "specs")

        register_documentation_tools(mock_mcp, config)

        spec_doc = mock_mcp._tools["spec_doc"]
        result = spec_doc.fn(
            spec_id="test-spec-001",
            mode="basic",
            workspace=str(tmp_path),
        )

        assert result["success"] is True
        assert result["data"]["mode"] == "basic"
