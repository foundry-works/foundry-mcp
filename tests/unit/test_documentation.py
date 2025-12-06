"""
Unit tests for foundry_mcp.tools.documentation module.

Tests cover:
- spec-doc tool for generating human-readable documentation
- spec-doc-llm tool (returns NOT_IMPLEMENTED since it requires LLM integration)
- spec-review-fidelity tool (returns NOT_IMPLEMENTED since it requires AI consultation)
- Response contract compliance for all tools
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_mcp():
    """Create a mock FastMCP server instance."""
    mcp = MagicMock()
    mcp._tools = {}

    def mock_tool(*args, **kwargs):
        def decorator(func):
            tool_name = kwargs.get("name", func.__name__)
            mcp._tools[tool_name] = func
            return func

        return decorator

    mcp.tool = mock_tool
    return mcp


@pytest.fixture
def mock_config():
    """Create a mock ServerConfig."""
    config = MagicMock()
    config.project_root = "/test/project"
    config.specs_dir = None
    return config


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project with specs."""
    # Create specs directory structure
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()
    (specs_dir / ".human-readable").mkdir()

    # Create a sample spec
    spec_data = {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "version": "1.0.0",
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "in_progress",
                "children": ["phase-1"],
                "parent": None,
            },
            "phase-1": {
                "type": "phase",
                "title": "Phase 1",
                "status": "in_progress",
                "children": ["task-1-1", "task-1-2"],
                "parent": "spec-root",
            },
            "task-1-1": {
                "type": "task",
                "title": "Task 1",
                "status": "completed",
                "children": [],
                "parent": "phase-1",
                "metadata": {
                    "details": ["Implement foundational structures"],
                    "file_path": "src/task_one.py",
                },
            },
            "task-1-2": {
                "type": "task",
                "title": "Task 2",
                "status": "pending",
                "children": [],
                "parent": "phase-1",
                "metadata": {
                    "details": ["Add follow-up improvements"],
                    "file_path": "src/task_two.py",
                },
            },
        },
    }

    spec_file = specs_dir / "active" / "test-spec-001.json"
    with open(spec_file, "w") as f:
        json.dump(spec_data, f)

    return tmp_path, spec_data


# =============================================================================
# spec-doc Tool Tests
# =============================================================================


class TestSpecDoc:
    """Tests for spec-doc tool."""

    def test_invalid_output_format_returns_error(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Should return validation error for invalid output_format."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec", output_format="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"
        assert result["data"].get("error_type") == "validation"

    def test_invalid_mode_returns_error(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Should return validation error for invalid mode."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec", mode="invalid")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_specs_dir_not_found(
        self, mock_mcp, mock_config, tmp_path, assert_response_contract, monkeypatch
    ):
        """Should handle missing specs directory."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        # Use an empty directory without specs
        monkeypatch.chdir(tmp_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "SPECS_DIR_NOT_FOUND"

    def test_spec_not_found(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """Should handle spec not found."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="nonexistent-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "SPEC_NOT_FOUND"

    def test_successful_documentation_generation(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """Should generate documentation successfully."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec-001")

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["spec_id"] == "test-spec-001"
        assert result["data"]["format"] == "markdown"
        assert "output_path" in result["data"]
        assert "stats" in result["data"]

    def test_custom_output_path(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """Should use custom output path when provided."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        custom_path = str(project_path / "custom_output.md")
        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec-001", output_path=custom_path)

        assert_response_contract(result)
        assert result["success"] is True
        assert result["data"]["output_path"] == custom_path


# =============================================================================
# spec-doc-llm Tool Tests
# =============================================================================


class TestSpecDocLlm:
    """Tests for spec-doc-llm tool."""

    def test_generates_documentation_successfully(
        self, mock_mcp, mock_config, tmp_path, assert_response_contract, monkeypatch
    ):
        """Should generate documentation for a valid directory."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        monkeypatch.chdir(tmp_path)

        # Create a sample Python file
        (tmp_path / "sample.py").write_text("def hello(): pass")

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc_llm = mock_mcp._tools["spec-doc-llm"]
        result = spec_doc_llm(directory=str(tmp_path), use_ai=False)

        assert_response_contract(result)
        assert result["success"] is True
        assert "output_dir" in result["data"]
        assert "files_generated" in result["data"]
        assert "generation" in result["data"]

    def test_includes_ai_generation_settings(
        self, mock_mcp, mock_config, tmp_path, assert_response_contract, monkeypatch
    ):
        """Should include AI generation settings in response."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        monkeypatch.chdir(tmp_path)

        # Create a sample Python file
        (tmp_path / "sample.py").write_text("def hello(): pass")

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc_llm = mock_mcp._tools["spec-doc-llm"]
        result = spec_doc_llm(directory=str(tmp_path), use_ai=False, ai_timeout=60.0)

        assert_response_contract(result)
        assert result["success"] is True
        assert "generation" in result["data"]
        assert result["data"]["generation"]["use_ai"] is False
        assert result["data"]["generation"]["ai_timeout"] == 60.0

    def test_missing_directory_validation(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Should return error for missing directory."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc_llm = mock_mcp._tools["spec-doc-llm"]
        result = spec_doc_llm(directory="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_invalid_batch_size_validation(
        self, mock_mcp, mock_config, tmp_path, assert_response_contract, monkeypatch
    ):
        """Should return error for invalid batch_size."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        monkeypatch.chdir(tmp_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc_llm = mock_mcp._tools["spec-doc-llm"]
        result = spec_doc_llm(directory=str(tmp_path), batch_size=20)

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_directory_not_found(self, mock_mcp, mock_config, assert_response_contract):
        """Should handle directory not found."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc_llm = mock_mcp._tools["spec-doc-llm"]
        result = spec_doc_llm(directory="/nonexistent/path")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "NOT_FOUND"


# =============================================================================
# spec-review-fidelity Tool Tests
# =============================================================================


class TestSpecReviewFidelity:
    """Tests for spec-review-fidelity tool."""

    def test_successful_fidelity_review_with_provider(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """Should successfully run fidelity review when providers are available."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(spec_id="test-spec-001")

        assert_response_contract(result)
        # If providers are available, it should succeed
        # If not, it should return AI_NO_PROVIDER error
        if result["success"]:
            assert "spec_id" in result["data"]
            assert result["data"]["spec_id"] == "test-spec-001"
            assert "verdict" in result["data"]
            assert "consensus" in result["data"]
        else:
            # No providers available - acceptable in test environment
            assert result["data"].get("error_code") in (
                "AI_NO_PROVIDER",
                "AI_NOT_AVAILABLE",
            )

    def test_includes_fidelity_response_fields(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """Should include expected response fields when successful."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(spec_id="test-spec-001")

        assert_response_contract(result)
        # Check response structure for successful reviews
        if result["success"]:
            data = result["data"]
            assert "spec_id" in data
            assert "title" in data
            assert "scope" in data
            assert "verdict" in data
            assert "consensus" in data
            assert "deviations" in data
            assert "recommendations" in data

    def test_missing_spec_id_validation(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Should return error for missing spec_id."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(spec_id="")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "MISSING_REQUIRED"

    def test_mutual_exclusivity_validation(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Should return error when both task_id and phase_id are provided."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(
            spec_id="test-spec",
            task_id="task-1-1",
            phase_id="phase-1",
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_invalid_consensus_threshold_validation(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Should return error for invalid consensus_threshold."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(
            spec_id="test-spec",
            consensus_threshold=10,
        )

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "VALIDATION_ERROR"

    def test_spec_not_found(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """Should handle spec not found."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(spec_id="nonexistent-spec")

        assert_response_contract(result)
        assert result["success"] is False
        assert result["data"].get("error_code") == "SPEC_NOT_FOUND"

    def test_no_provider_error_includes_provider_status(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """AI_NO_PROVIDER error should include provider_status with unavailability reasons."""
        import os
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        # Force test mode to make all providers unavailable
        with patch.dict(os.environ, {"FOUNDRY_PROVIDER_TEST_MODE": "1"}):
            register_documentation_tools(mock_mcp, mock_config)

            spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
            result = spec_review_fidelity(spec_id="test-spec-001")

            assert_response_contract(result)
            # Should fail because no providers are available
            assert result["success"] is False
            assert result["data"].get("error_code") == "AI_NO_PROVIDER"
            # Should include provider_status with unavailability reasons
            assert "provider_status" in result["data"]
            provider_status = result["data"]["provider_status"]
            assert isinstance(provider_status, dict)
            # Each provider should have a reason (string) or None (if available)
            for provider_id, reason in provider_status.items():
                assert reason is None or isinstance(reason, str)

    def test_consultation_failure_returns_error_not_unknown(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """Consultation failure should return error response, not success with 'unknown' verdict."""
        from unittest.mock import MagicMock
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        # Mock the ConsultationOrchestrator to return a failed result
        mock_result = MagicMock()
        mock_result.content = ""
        mock_result.provider_id = "claude"
        mock_result.model_used = "none"
        mock_result.error = "Provider execution failed: Connection timeout"
        mock_result.cache_hit = False

        with patch(
            "foundry_mcp.core.ai_consultation.ConsultationOrchestrator"
        ) as MockOrch:
            mock_orchestrator = MagicMock()
            mock_orchestrator.is_available.return_value = True
            mock_orchestrator.consult.return_value = mock_result
            MockOrch.return_value = mock_orchestrator

            register_documentation_tools(mock_mcp, mock_config)

            spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
            result = spec_review_fidelity(spec_id="test-spec-001")

            assert_response_contract(result)
            # Should return error, NOT success with "unknown" verdict
            assert result["success"] is False
            assert result["data"].get("error_code") == "AI_CONSULTATION_FAILED"
            assert "error_details" in result["data"]
            assert "Connection timeout" in result["data"]["error_details"]
            # Should include provider_status
            assert "provider_status" in result["data"]

    def test_fidelity_review_never_returns_unknown_on_success(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """Successful fidelity review should never have 'unknown' verdict when properly mocked."""
        from unittest.mock import patch, MagicMock
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        # Mock consultation to return a proper response with verdict
        mock_result = MagicMock()
        mock_result.content = (
            '{"verdict": "pass", "deviations": [], "recommendations": []}'
        )
        mock_result.error = None
        mock_result.provider_id = "test-provider"
        mock_result.model_used = "test-model"
        mock_result.cache_hit = True

        with patch(
            "foundry_mcp.core.ai_consultation.ConsultationOrchestrator"
        ) as MockOrch:
            mock_orchestrator = MagicMock()
            mock_orchestrator.is_available.return_value = True
            mock_orchestrator.consult.return_value = mock_result
            MockOrch.return_value = mock_orchestrator

            register_documentation_tools(mock_mcp, mock_config)

            spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
            result = spec_review_fidelity(spec_id="test-spec-001")

            assert_response_contract(result)
            # With proper mock, should succeed with valid verdict
            assert result["success"] is True
            verdict = result["data"].get("verdict")
            assert verdict == "pass", (
                f"Expected 'pass' verdict from mocked response, got: {verdict}"
            )


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestDocumentationToolRegistration:
    """Tests for documentation tool registration."""

    def test_all_tools_registered(self, mock_mcp, mock_config):
        """All documentation tools should be registered."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        expected_tools = [
            "spec-doc",
            "spec-doc-llm",
            "spec-review-fidelity",
        ]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp._tools, f"Tool {tool_name} not registered"

    def test_tools_are_callable(self, mock_mcp, mock_config):
        """All registered tools should be callable functions."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        for tool_name, tool_func in mock_mcp._tools.items():
            assert callable(tool_func), f"Tool {tool_name} is not callable"


# =============================================================================
# Response Contract Compliance Tests
# =============================================================================


class TestResponseContractCompliance:
    """Test that all responses comply with the response-v2 contract."""

    def test_validation_error_response_structure(
        self, mock_mcp, mock_config, assert_response_contract
    ):
        """Validation error responses should have correct structure."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec", mode="invalid")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"

    def test_success_response_has_required_fields(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """Success responses should have all required fields."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_doc = mock_mcp._tools["spec-doc"]
        result = spec_doc(spec_id="test-spec-001")

        # Validate structure
        assert result["success"] is True
        assert result["error"] is None
        assert result["meta"]["version"] == "response-v2"

    def test_fidelity_review_response_structure(
        self, mock_mcp, mock_config, temp_project, assert_response_contract, monkeypatch
    ):
        """Fidelity review responses should have correct structure."""
        from foundry_mcp.tools.documentation import register_documentation_tools

        project_path, _ = temp_project
        monkeypatch.chdir(project_path)

        register_documentation_tools(mock_mcp, mock_config)

        spec_review_fidelity = mock_mcp._tools["spec-review-fidelity"]
        result = spec_review_fidelity(spec_id="test-spec-001")

        # Validate structure
        assert "success" in result
        assert "data" in result
        assert "error" in result
        assert "meta" in result
        assert result["meta"]["version"] == "response-v2"
        # Result can be success or failure depending on provider availability
        if result["success"]:
            assert "spec_id" in result["data"]
        else:
            # Expected error codes when no provider available
            assert result["data"].get("error_code") in (
                "AI_NO_PROVIDER",
                "AI_NOT_AVAILABLE",
            )


# =============================================================================
# Telemetry and Metrics Tests
# =============================================================================


class TestTelemetryMetrics:
    """Tests for telemetry and metrics integration."""

    def test_metrics_collector_accessible(self):
        """Metrics collector should be accessible."""
        from foundry_mcp.core.observability import get_metrics

        metrics = get_metrics()
        assert metrics is not None
        assert hasattr(metrics, "counter")
        assert hasattr(metrics, "timer")
        assert hasattr(metrics, "histogram")

    def test_metrics_collector_accepts_labels(self):
        """Metrics collector should accept labels."""
        from foundry_mcp.core.observability import MetricsCollector

        metrics = MetricsCollector()

        # Counter should accept labels
        metrics.counter("documentation.success", labels={"tool": "spec-doc"})
        metrics.counter(
            "documentation.errors",
            labels={"tool": "spec-doc", "error_type": "not_found"},
        )

        # Timer should also accept labels
        metrics.timer("documentation.spec_doc_time", 100.5, labels={"mode": "basic"})

        # Histogram should mirror timer-style usage
        metrics.histogram(
            "documentation.duration_distribution",
            42.0,
            labels={"mode": "basic"},
        )


# =============================================================================
# Response Envelope Compliance Tests
# =============================================================================


class TestResponseEnvelopeCompliance:
    """Tests for response envelope compliance."""

    def test_success_response_has_required_fields(self):
        """Test success response includes required envelope fields."""
        from foundry_mcp.core.responses import success_response
        from dataclasses import asdict

        response = asdict(
            success_response(
                spec_id="test-spec",
                format="markdown",
                output_path="test.md",
            )
        )

        assert "success" in response
        assert response["success"] is True
        assert "meta" in response
        assert response["meta"]["version"] == "response-v2"

    def test_error_response_has_required_fields(self):
        """Test error response includes required envelope fields."""
        from foundry_mcp.core.responses import error_response
        from dataclasses import asdict

        response = asdict(
            error_response(
                "Test error message",
                error_code="VALIDATION_ERROR",
                error_type="validation",
            )
        )

        assert "success" in response
        assert response["success"] is False
        assert "error" in response
        assert response["error"] == "Test error message"
        assert "meta" in response
        assert response["meta"]["version"] == "response-v2"


class TestFidelityHierarchyHelpers:
    """Ensure fidelity helper functions handle keyed hierarchy specs."""

    def test_phase_requirements_include_children(self, temp_project):
        from foundry_mcp.tools.documentation import _build_spec_requirements

        _, spec_data = temp_project
        output = _build_spec_requirements(spec_data, task_id=None, phase_id="phase-1")

        assert "Phase: Phase 1" in output
        assert "task-1-1" in output
        assert "task-1-2" in output

    def test_phase_artifacts_resolve_child_files(self, temp_project, monkeypatch):
        from foundry_mcp.tools.documentation import _build_implementation_artifacts

        project_path, spec_data = temp_project
        src_dir = project_path / "src"
        src_dir.mkdir()
        (src_dir / "task_one.py").write_text("print('ok')", encoding="utf-8")
        monkeypatch.chdir(project_path)

        output = _build_implementation_artifacts(
            spec_data,
            task_id=None,
            phase_id="phase-1",
            files=None,
            incremental=False,
            base_branch="main",
        )

        assert "src/task_one.py" in output
        assert "print('ok')" in output
